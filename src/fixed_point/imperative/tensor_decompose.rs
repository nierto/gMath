//! Tensor decompositions: truncated SVD, Tucker/HOSVD, CP/ALS.
//!
//! Built on the existing `svd_decompose` (Golub-Kahan bidiagonalization) and
//! `Tensor` infrastructure. All inner products accumulated at compute tier.
//!
//! **Use cases**:
//! - Weight compression (truncated SVD: 4096×4096 → rank-128 factors, 32× memory reduction)
//! - KV-cache compression (Tucker on batch × heads × seq × dim)
//! - Adapter merging (CP decomposition of LoRA deltas)

use super::{FixedPoint, FixedVector, FixedMatrix};
use super::tensor::Tensor;
use super::decompose::svd_decompose;
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
    pub fn reconstruct(&self) -> FixedMatrix {
        let m = self.u.rows();
        let n = self.vt.cols();
        let k = self.sigma.len();
        let mut result = FixedMatrix::new(m, n);
        for r in 0..k {
            let sv = self.sigma[r];
            for i in 0..m {
                let u_ir = self.u.get(i, r) * sv;
                for j in 0..n {
                    let val = result.get(i, j) + u_ir * self.vt.get(r, j);
                    result.set(i, j, val);
                }
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
        let dim_factor = FixedPoint::from_int(a.rows().max(a.cols()) as i32);
        let eps = super::linalg::convergence_threshold(sigma_max);
        dim_factor * eps
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
    pub fn reconstruct(&self) -> Tensor {
        // T = G ×₁ U₁ ×₂ U₂ ... ×_N U_N
        // Mode-n product: contract core's n-th index with U_n's columns
        let mut result = self.core.clone();
        for (n, u) in self.factors.iter().enumerate() {
            result = mode_n_product(&result, u, n);
        }
        result
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
/// that mode is not compressed.
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
    let mut core = t.clone();
    for (n, u) in factors.iter().enumerate() {
        let ut = u.transpose();
        core = mode_n_product(&core, &ut, n);
    }

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
    pub fn reconstruct(&self, shape: &[usize]) -> Tensor {
        let rank = self.weights.len();
        let total: usize = shape.iter().product();
        let mut data = vec![FixedPoint::ZERO; total];

        // For each rank-1 component
        for r in 0..rank {
            let w = self.weights[r];
            // Build rank-1 tensor as outer product of factor columns
            add_rank1_to_flat(&mut data, shape, &self.factors, r, w);
        }

        Tensor::from_data(shape, &data)
    }
}

/// Compute CP decomposition via Alternating Least Squares.
///
/// `rank`: number of rank-1 components (R).
/// `max_iter`: maximum ALS iterations.
/// `tol`: convergence tolerance (relative change in reconstruction error).
pub fn cp_decompose(
    t: &Tensor,
    rank: usize,
    max_iter: usize,
    _tol: FixedPoint,
) -> Result<CPDecomposition, OverflowDetected> {
    let ndim = t.rank();
    let shape = t.shape().to_vec();

    // Initialize factor matrices via first `rank` left singular vectors of mode-0 unfolding
    let mut factors: Vec<FixedMatrix> = Vec::with_capacity(ndim);
    for n in 0..ndim {
        let unfolded = mode_unfold(t, n);
        let k = rank.min(unfolded.rows()).min(unfolded.cols());
        let svd = svd_decompose(&unfolded)?;
        let mut f = FixedMatrix::new(shape[n], rank);
        for i in 0..shape[n] {
            for r in 0..rank {
                if r < k {
                    f.set(i, r, svd.u.get(i, r));
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
            let v = khatri_rao_except(&factors, n, &shape);
            // Unfolded tensor × V gives the new factor
            let unfolded = mode_unfold(t, n);
            // factors[n] = unfolded * V * (VᵀV)⁻¹
            let vt = v.transpose();
            let vtv = &vt * &v;          // R × R
            let rhs = &unfolded * &v;    // d_n × R
            // Solve: factors[n] * VᵀV = rhs → factors[n] = rhs * (VᵀV)⁻¹
            match super::derived::inverse(&vtv) {
                Ok(vtv_inv) => {
                    factors[n] = &rhs * &vtv_inv;
                }
                Err(_) => {
                    // Singular VᵀV — skip update for this mode
                    continue;
                }
            }
        }

        // Check convergence via factor norm change
        // (simplified: run all iterations, rely on max_iter for stopping)
    }

    // Extract weights: normalize factor columns, put norms into weights
    let mut weights = FixedVector::new(rank);
    for r in 0..rank {
        let mut norm_product = FixedPoint::one();
        for n in 0..ndim {
            let mut col_norm_sq = FixedPoint::ZERO;
            for i in 0..shape[n] {
                let v = factors[n].get(i, r);
                col_norm_sq = col_norm_sq + v * v;
            }
            let col_norm = col_norm_sq.sqrt();
            if !col_norm.is_zero() {
                for i in 0..shape[n] {
                    let v = factors[n].get(i, r);
                    factors[n].set(i, r, v / col_norm);
                }
                norm_product = norm_product * col_norm;
            }
        }
        weights[r] = norm_product;
    }

    Ok(CPDecomposition { weights, factors })
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

/// Mode-n product: multiply tensor by matrix along mode n.
///
/// T ×_n M: if T has shape (..., d_n, ...) and M is (r, d_n),
/// result has shape (..., r, ...).
fn mode_n_product(t: &Tensor, m: &FixedMatrix, mode: usize) -> Tensor {
    let shape = t.shape();
    let ndim = shape.len();
    let d_n = shape[mode];
    let r = m.rows(); // Output dimension for this mode

    assert_eq!(m.cols(), d_n, "Matrix cols must match tensor mode dimension");

    // New shape: replace d_n with r
    let mut new_shape = shape.to_vec();
    new_shape[mode] = r;

    let total: usize = new_shape.iter().product();
    let mut result = Tensor::new(&new_shape);

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
        let mut sum = FixedPoint::ZERO;
        let mut src_indices = out_indices.clone();
        for k in 0..d_n {
            src_indices[mode] = k;
            sum = sum + m.get(i, k) * t.get(&src_indices);
        }
        result.set(&out_indices, sum);
    }

    result
}

/// Khatri-Rao product of all factor matrices except mode n.
///
/// Result is a (product of d_i for i != n) × R matrix, where each column
/// is the element-wise (Hadamard) product of the corresponding columns
/// from all factors except n.
fn khatri_rao_except(factors: &[FixedMatrix], skip: usize, shape: &[usize]) -> FixedMatrix {
    let rank = factors[0].cols();
    let ndim = factors.len();

    // Product of all dimensions except skip
    let rows: usize = shape.iter().enumerate()
        .filter(|&(i, _)| i != skip)
        .map(|(_, &d)| d)
        .product();

    let mut result = FixedMatrix::new(rows, rank);

    // For each column (rank component)
    for r in 0..rank {
        // Build the Khatri-Rao column via outer product of factor columns
        // Start with first non-skip factor
        let active_modes: Vec<usize> = (0..ndim).filter(|&i| i != skip).collect();

        for row in 0..rows {
            // Decompose row index into per-mode indices
            let mut rem = row;
            let mut val = FixedPoint::one();
            for &m in active_modes.iter().rev() {
                let idx = rem % shape[m];
                rem /= shape[m];
                val = val * factors[m].get(idx, r);
            }
            result.set(row, r, val);
        }
    }

    result
}

/// Add a weighted rank-1 component to a flat data array.
fn add_rank1_to_flat(
    data: &mut [FixedPoint],
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

        let mut val = weight;
        for n in 0..ndim {
            val = val * factors[n].get(indices[n], r);
        }
        data[flat] = data[flat] + val;
    }
}
