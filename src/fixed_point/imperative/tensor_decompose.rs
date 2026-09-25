//! Tensor decompositions: truncated SVD, Tucker/HOSVD, CP/ALS.
//!
//! Built on the existing `svd_decompose` (Golub-Kahan bidiagonalization) and
//! `Tensor` infrastructure. Multi-step computations (the Tucker core across
//! modes, the CP-ALS factors across iterations, every reconstruction) carry
//! their state at the compute tier and round to storage once at the end.
//!
//! **Use cases**:
//! - Weight compression (truncated SVD: 4096×4096 → rank-128 factors, 32× memory reduction)
//! - KV-cache compression (Tucker on batch × heads × seq × dim)
//! - Adapter merging (CP decomposition of LoRA deltas)

use super::{FixedPoint, FixedVector, FixedMatrix};
use super::tensor::Tensor;
use super::decompose::svd_decompose;
use super::compute_matrix::{compute_lu_decompose, ComputeMatrix};
use super::linalg::{compute_product, downscale_to_storage, exact_dot, upscale_to_compute, ComputeStorage};
use super::wide_acc::{
    acc, divide_to_storage_nearest, exact_dot_compute, narrow_product_to_compute, narrow_triple_nearest,
    widen_product, widen_storage, Wide,
};
use crate::fixed_point::universal::fasc::stack_evaluator::compute::{
    compute_add, compute_is_zero, make_compute_int, sqrt_at_compute_tier,
};
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// TRUNCATED SVD
// ============================================================================

/// Truncated SVD: A ≈ U_k Σ_k V_k^T where k << min(m,n).
///
/// Keeps only the top-k singular values and their corresponding vectors.
/// Memory: O(mk + k + nk) instead of O(m² + n + n²) for full SVD.
pub struct TruncatedSVD {
    /// Left singular vectors: m × k matrix.
    pub u: FixedMatrix,
    /// Top-k singular values (descending).
    pub sigma: FixedVector,
    /// Right singular vectors (transposed): k × n matrix.
    pub vt: FixedMatrix,
}

impl TruncatedSVD {
    /// Reconstruct the rank-k approximation: U_k Σ_k V_k^T.
    ///
    /// Each entry `sum_r u_ir σ_r vt_rj` is exact (`u_ir σ_r` at the compute
    /// tier, the triple product and the sum above it) and rounded to storage
    /// once. Panics if an entry exceeds storage.
    pub fn reconstruct(&self) -> FixedMatrix {
        const OVERFLOW: &str = "TruncatedSVD::reconstruct: entry exceeds storage";
        let m = self.u.rows();
        let n = self.vt.cols();
        let k = self.sigma.len();
        // u_ir σ_r, exact at 2F
        let mut u_sigma = Vec::with_capacity(m * k);
        for i in 0..m {
            for r in 0..k {
                u_sigma.push(exact_dot(&[self.u.get(i, r).raw()], &[self.sigma[r].raw()]).expect(OVERFLOW));
            }
        }
        let mut result = FixedMatrix::new(m, n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = <acc::Orient as Wide>::zero();
                for r in 0..k {
                    sum = sum.add_exact(widen_product(u_sigma[i * k + r], widen_storage(self.vt.get(r, j).raw()))).expect(OVERFLOW);
                }
                result.set(i, j, FixedPoint::from_raw(narrow_triple_nearest(sum).expect(OVERFLOW)));
            }
        }
        result
    }

    /// Compression ratio: original_elements / compressed_elements.
    pub fn compression_ratio(&self, m: usize, n: usize) -> f64 {
        let k = self.sigma.len();
        (m * n) as f64 / (m * k + k + k * n) as f64
    }
}

/// Compute truncated SVD keeping the top-k singular values.
///
/// If k >= min(m,n), returns the full SVD (no truncation).
pub fn truncated_svd(a: &FixedMatrix, k: usize) -> Result<TruncatedSVD, OverflowDetected> {
    let svd = svd_decompose(a)?;
    let full_k = svd.sigma.len();
    let k = k.min(full_k);

    // Extract top-k columns of U
    let m = svd.u.rows();
    let mut u_k = FixedMatrix::new(m, k);
    for i in 0..m {
        for j in 0..k {
            u_k.set(i, j, svd.u.get(i, j));
        }
    }

    // Extract top-k singular values
    let mut sigma_k = FixedVector::new(k);
    for i in 0..k {
        sigma_k[i] = svd.sigma[i];
    }

    // Extract top-k rows of Vt
    let n = svd.vt.cols();
    let mut vt_k = FixedMatrix::new(k, n);
    for i in 0..k {
        for j in 0..n {
            vt_k.set(i, j, svd.vt.get(i, j));
        }
    }

    Ok(TruncatedSVD { u: u_k, sigma: sigma_k, vt: vt_k })
}

/// Compute truncated SVD with automatic rank selection via singular value threshold.
///
/// Keeps all singular values > threshold. Uses the default threshold from
/// derived.rs if `threshold` is None.
pub fn truncated_svd_auto(a: &FixedMatrix, threshold: Option<FixedPoint>) -> Result<TruncatedSVD, OverflowDetected> {
    let svd = svd_decompose(a)?;

    let thresh = threshold.unwrap_or_else(|| {
        if svd.sigma.len() == 0 { return FixedPoint::one(); }
        let sigma_max = svd.sigma[0];
        let eps = super::linalg::convergence_threshold(sigma_max);
        eps.mul_count(a.rows().max(a.cols()))
    });

    let mut k = 0;
    for i in 0..svd.sigma.len() {
        if svd.sigma[i] > thresh { k += 1; } else { break; }
    }
    if k == 0 { k = 1; } // At least rank 1

    truncated_svd(a, k)
}

// ============================================================================
// TUCKER / HOSVD DECOMPOSITION
// ============================================================================

/// Tucker decomposition: T ≈ G ×₁ U₁ ×₂ U₂ ×₃ U₃ ...
///
/// G is a small core tensor, U_n are orthogonal factor matrices per mode.
/// HOSVD (Higher-Order SVD) computes the factors via SVD of mode unfoldings.
pub struct TuckerDecomposition {
    /// Core tensor of shape (r₁, r₂, ..., r_N) where r_n ≤ d_n.
    pub core: Tensor,
    /// Factor matrices: factors[n] is d_n × r_n.
    pub factors: Vec<FixedMatrix>,
}

impl TuckerDecomposition {
    /// Reconstruct the full tensor from core + factors.
    ///
    /// The mode products run at the compute tier (each entry an exact dot
    /// rounded to the compute tier) and the result is rounded to storage
    /// once. Panics if an entry exceeds the compute tier or storage.
    pub fn reconstruct(&self) -> Tensor {
        // T = G ×₁ U₁ ×₂ U₂ ... ×_N U_N
        // Mode-n product: contract core's n-th index with U_n's columns
        let chain = || -> Result<Tensor, OverflowDetected> {
            let mut data = to_compute(&self.core);
            let mut shape = self.core.shape().to_vec();
            for (n, u) in self.factors.iter().enumerate() {
                (data, shape) = mode_n_product_compute(&data, &shape, u, n)?;
            }
            to_storage_tensor(&data, &shape)
        };
        chain().expect("TuckerDecomposition::reconstruct: entry exceeds the compute tier or storage")
    }

    /// Compression ratio: original_elements / (core + factor) elements.
    pub fn compression_ratio(&self, original_shape: &[usize]) -> f64 {
        let orig: usize = original_shape.iter().product();
        let core_size: usize = self.core.shape().iter().product();
        let factor_size: usize = self.factors.iter().enumerate()
            .map(|(n, f)| original_shape[n] * f.cols())
            .sum();
        orig as f64 / (core_size + factor_size) as f64
    }
}

/// Compute Tucker decomposition via HOSVD.
///
/// `ranks[n]` specifies the truncation rank for mode n. If ranks[n] >= d_n,
/// that mode is not compressed. The core `T ×₁ U₁ᵀ ... ×_N U_Nᵀ` stays at the
/// compute tier across modes and is rounded to storage once.
pub fn tucker_decompose(t: &Tensor, ranks: &[usize]) -> Result<TuckerDecomposition, OverflowDetected> {
    let ndim = t.rank();
    assert_eq!(ranks.len(), ndim, "ranks must have one entry per tensor mode");

    let mut factors: Vec<FixedMatrix> = Vec::with_capacity(ndim);

    // Step 1: For each mode, compute SVD of mode-n unfolding
    for n in 0..ndim {
        let unfolded = mode_unfold(t, n);
        let k = ranks[n].min(unfolded.rows()).min(unfolded.cols());
        let tsvd = truncated_svd(&unfolded, k)?;
        factors.push(tsvd.u); // d_n × r_n factor matrix
    }

    // Step 2: Core tensor = T ×₁ U₁ᵀ ×₂ U₂ᵀ ... ×_N U_Nᵀ
    let mut data = to_compute(t);
    let mut shape = t.shape().to_vec();
    for (n, u) in factors.iter().enumerate() {
        (data, shape) = mode_n_product_compute(&data, &shape, &u.transpose(), n)?;
    }
    let core = to_storage_tensor(&data, &shape)?;

    Ok(TuckerDecomposition { core, factors })
}

// ============================================================================
// CP / ALS DECOMPOSITION
// ============================================================================

/// CP (Canonical Polyadic) decomposition: T ≈ Σ_r λ_r a₁_r ∘ a₂_r ∘ ... ∘ a_N_r
///
/// Decomposes a tensor into a sum of R rank-1 terms. Each term is an outer
/// product of vectors, weighted by λ_r.
pub struct CPDecomposition {
    /// Component weights (R values).
    pub weights: FixedVector,
    /// Factor matrices: factors[n] is d_n × R (columns are the rank-1 vectors).
    pub factors: Vec<FixedMatrix>,
}

impl CPDecomposition {
    /// Reconstruct the full tensor from CP factors.
    ///
    /// Each weighted rank-1 term is a product chain at the compute tier and
    /// the terms are summed there; every entry is rounded to storage once.
    /// Panics if an entry exceeds the compute tier or storage.
    pub fn reconstruct(&self, shape: &[usize]) -> Tensor {
        let rank = self.weights.len();
        let total: usize = shape.iter().product();
        let mut data = vec![make_compute_int(0); total];

        // For each rank-1 component
        for r in 0..rank {
            let w = self.weights[r];
            // Build rank-1 tensor as outer product of factor columns
            add_rank1_to_flat(&mut data, shape, &self.factors, r, w);
        }

        to_storage_tensor(&data, shape).expect("CPDecomposition::reconstruct: entry exceeds storage")
    }
}

/// Compute CP decomposition via Alternating Least Squares.
///
/// `rank`: number of rank-1 components (R).
/// `max_iter`: maximum ALS iterations.
/// `tol`: convergence tolerance (relative change in reconstruction error).
///
/// The factors stay at the compute tier across iterations: the Khatri-Rao
/// products, `VᵀV` and `T_(n) V` (exact dots rounded at the compute tier) and
/// the solve `factor VᵀV = T_(n) V` (compute-tier LU). The column norms are
/// compute-tier roots of exact sums; each normalized factor entry is one
/// division rounded to storage, each weight one rounding of the norms'
/// compute-tier product.
pub fn cp_decompose(
    t: &Tensor,
    rank: usize,
    max_iter: usize,
    _tol: FixedPoint,
) -> Result<CPDecomposition, OverflowDetected> {
    let ndim = t.rank();
    let shape = t.shape().to_vec();

    // Initialize factor matrices via first `rank` left singular vectors of mode-0 unfolding
    let mut factors: Vec<ComputeMatrix> = Vec::with_capacity(ndim);
    for n in 0..ndim {
        let unfolded = mode_unfold(t, n);
        let k = rank.min(unfolded.rows()).min(unfolded.cols());
        let svd = svd_decompose(&unfolded)?;
        let mut f = ComputeMatrix::new(shape[n], rank);
        for i in 0..shape[n] {
            for r in 0..rank {
                if r < k {
                    f.set(i, r, upscale_to_compute(svd.u.get(i, r).raw()));
                }
                // Remaining columns stay zero (will be refined by ALS)
            }
        }
        factors.push(f);
    }

    // ALS iterations
    for _iter in 0..max_iter {
        for n in 0..ndim {
            // Compute Khatri-Rao product of all factors except n
            let v = khatri_rao_except(&factors, n, &shape)?;
            // Unfolded tensor × V gives the new factor
            let unfolded = mode_unfold(t, n);
            let cols: Vec<Vec<ComputeStorage>> = (0..rank).map(|r| v.col_vec(r)).collect();
            // VᵀV (R × R): exact dots rounded at the compute tier
            let mut vtv = ComputeMatrix::new(rank, rank);
            for a in 0..rank {
                for b in 0..rank {
                    vtv.set(a, b, narrow_product_to_compute(exact_dot_compute(&cols[a], &cols[b])?)?);
                }
            }
            // Solve: factors[n] * VᵀV = T_(n) V, row by row (VᵀV is symmetric)
            let lu = match compute_lu_decompose(&vtv) {
                Ok(lu) => lu,
                // Singular VᵀV: skip update for this mode
                Err(_) => continue,
            };
            let mut next = ComputeMatrix::new(shape[n], rank);
            let mut solved = true;
            for i in 0..shape[n] {
                let row: Vec<ComputeStorage> = (0..unfolded.cols()).map(|c| upscale_to_compute(unfolded.get(i, c).raw())).collect();
                let mut rhs = Vec::with_capacity(rank);
                for col in &cols {
                    rhs.push(narrow_product_to_compute(exact_dot_compute(&row, col)?)?);
                }
                match lu.solve(&rhs) {
                    Ok(x) => for r in 0..rank { next.set(i, r, x[r]); },
                    // numerically singular VᵀV: skip update for this mode
                    Err(_) => { solved = false; break; }
                }
            }
            if solved {
                factors[n] = next;
            }
        }

        // Check convergence via factor norm change
        // (simplified: run all iterations, rely on max_iter for stopping)
    }

    // Extract weights: normalize factor columns, put norms into weights
    let mut weights = FixedVector::new(rank);
    let mut out: Vec<FixedMatrix> = (0..ndim).map(|n| FixedMatrix::new(shape[n], rank)).collect();
    for r in 0..rank {
        let mut norm_product = make_compute_int(1);
        for n in 0..ndim {
            let col = factors[n].col_vec(r);
            let col_norm = sqrt_at_compute_tier(narrow_product_to_compute(exact_dot_compute(&col, &col)?)?);
            if !compute_is_zero(&col_norm) {
                for i in 0..shape[n] {
                    out[n].set(i, r, FixedPoint::from_raw(divide_to_storage_nearest(lift_to_triple(col[i]), col_norm)?));
                }
                norm_product = compute_product(norm_product, col_norm)?;
            } else {
                for i in 0..shape[n] {
                    out[n].set(i, r, FixedPoint::from_raw(downscale_to_storage(col[i])?));
                }
            }
        }
        weights[r] = FixedPoint::from_raw(downscale_to_storage(norm_product)?);
    }

    Ok(CPDecomposition { weights, factors: out })
}

// ============================================================================
// HELPERS
// ============================================================================

/// Mode-n unfolding: reshape tensor into matrix with mode-n as rows.
///
/// Result has shape (d_n, product of all other dimensions).
fn mode_unfold(t: &Tensor, mode: usize) -> FixedMatrix {
    let shape = t.shape();
    let ndim = shape.len();
    let rows = shape[mode];
    let cols: usize = shape.iter().enumerate()
        .filter(|&(i, _)| i != mode)
        .map(|(_, &d)| d)
        .product();

    let mut result = FixedMatrix::new(rows, cols);

    // Build permutation: mode first, then others in order
    let mut perm: Vec<usize> = vec![mode];
    for i in 0..ndim {
        if i != mode { perm.push(i); }
    }

    // Iterate over all elements via multi-index
    let mut indices = vec![0usize; ndim];
    let total: usize = shape.iter().product();
    for flat in 0..total {
        // Compute multi-index from flat index
        let mut rem = flat;
        for d in (0..ndim).rev() {
            indices[d] = rem % shape[d];
            rem /= shape[d];
        }

        let row = indices[mode];
        // Column index: linearize all non-mode indices in order
        let mut col = 0;
        let mut stride = 1;
        for &p in perm[1..].iter().rev() {
            col += indices[p] * stride;
            stride *= shape[p];
        }

        result.set(row, col, t.get(&indices));
    }

    result
}

/// A compute raw (`2F` fractional bits) as an exact `3F` value, the numerator
/// scale of [`divide_to_storage_nearest`] over a `2F` denominator.
#[inline]
fn lift_to_triple(c: ComputeStorage) -> acc::Orient {
    widen_product(c, widen_storage(FixedPoint::one().raw()))
}

/// A tensor's entries at the compute tier (row-major, exact).
fn to_compute(t: &Tensor) -> Vec<ComputeStorage> {
    t.data().iter().map(|x| upscale_to_compute(x.raw())).collect()
}

/// Compute-tier entries rounded to storage once each (checked).
fn to_storage_tensor(data: &[ComputeStorage], shape: &[usize]) -> Result<Tensor, OverflowDetected> {
    let entries = data.iter().map(|c| downscale_to_storage(*c).map(FixedPoint::from_raw)).collect::<Result<Vec<_>, _>>()?;
    Ok(Tensor::from_data(shape, &entries))
}

/// Mode-n product at the compute tier: multiply a tensor (row-major compute
/// raws of shape `shape`) by a storage matrix along mode n.
///
/// T ×_n M: if T has shape (..., d_n, ...) and M is (r, d_n),
/// result has shape (..., r, ...). Each entry `sum_k M[i,k] T[...k...]` is an
/// exact dot rounded once to the compute tier, so a chain of mode products
/// rounds to storage only when the caller narrows the final tensor.
fn mode_n_product_compute(
    data: &[ComputeStorage],
    shape: &[usize],
    m: &FixedMatrix,
    mode: usize,
) -> Result<(Vec<ComputeStorage>, Vec<usize>), OverflowDetected> {
    let ndim = shape.len();
    let d_n = shape[mode];
    let r = m.rows(); // Output dimension for this mode

    assert_eq!(m.cols(), d_n, "Matrix cols must match tensor mode dimension");

    // row-major strides of the source
    let mut strides = vec![1usize; ndim];
    for d in (0..ndim.saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    // New shape: replace d_n with r
    let mut new_shape = shape.to_vec();
    new_shape[mode] = r;
    let m_rows: Vec<Vec<ComputeStorage>> = (0..r)
        .map(|i| (0..d_n).map(|k| upscale_to_compute(m.get(i, k).raw())).collect())
        .collect();

    let total: usize = new_shape.iter().product();
    let mut result = Vec::with_capacity(total);

    // For each element in the result
    let mut out_indices = vec![0usize; ndim];
    for flat in 0..total {
        let mut rem = flat;
        for d in (0..ndim).rev() {
            out_indices[d] = rem % new_shape[d];
            rem /= new_shape[d];
        }

        // Sum over mode dimension: result[...i...] = sum_k M[i,k] * T[...k...]
        let i = out_indices[mode];
        let base: usize = (0..ndim).filter(|&d| d != mode).map(|d| out_indices[d] * strides[d]).sum();
        let fiber: Vec<ComputeStorage> = (0..d_n).map(|k| data[base + k * strides[mode]]).collect();
        result.push(narrow_product_to_compute(exact_dot_compute(&m_rows[i], &fiber)?)?);
    }

    Ok((result, new_shape))
}

/// Khatri-Rao product of all factor matrices except mode n.
///
/// Result is a (product of d_i for i != n) × R matrix, where each column
/// is the element-wise (Hadamard) product of the corresponding columns
/// from all factors except n.
fn khatri_rao_except(factors: &[ComputeMatrix], skip: usize, shape: &[usize]) -> Result<ComputeMatrix, OverflowDetected> {
    let rank = factors[0].cols();
    let ndim = factors.len();

    // Product of all dimensions except skip
    let rows: usize = shape.iter().enumerate()
        .filter(|&(i, _)| i != skip)
        .map(|(_, &d)| d)
        .product();

    let mut result = ComputeMatrix::new(rows, rank);

    // For each column (rank component)
    for r in 0..rank {
        // Build the Khatri-Rao column via outer product of factor columns
        // Start with first non-skip factor
        let active_modes: Vec<usize> = (0..ndim).filter(|&i| i != skip).collect();

        for row in 0..rows {
            // Decompose row index into per-mode indices
            let mut rem = row;
            let mut val = make_compute_int(1);
            for &m in active_modes.iter().rev() {
                let idx = rem % shape[m];
                rem /= shape[m];
                val = compute_product(val, factors[m].get(idx, r))?;
            }
            result.set(row, r, val);
        }
    }

    Ok(result)
}

/// Add a weighted rank-1 component to a flat array of compute raws: the
/// product chain `w a_i b_j ...` at the compute tier, summed there.
fn add_rank1_to_flat(
    data: &mut [ComputeStorage],
    shape: &[usize],
    factors: &[FixedMatrix],
    r: usize,
    weight: FixedPoint,
) {
    let ndim = shape.len();
    let total = data.len();
    let mut indices = vec![0usize; ndim];

    for flat in 0..total {
        let mut rem = flat;
        for d in (0..ndim).rev() {
            indices[d] = rem % shape[d];
            rem /= shape[d];
        }

        let mut val = upscale_to_compute(weight.raw());
        for n in 0..ndim {
            val = compute_product(val, upscale_to_compute(factors[n].get(indices[n], r).raw()))
                .expect("CPDecomposition::reconstruct: term exceeds the compute tier");
        }
        data[flat] = compute_add(data[flat], val);
    }
}
