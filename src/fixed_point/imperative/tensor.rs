//! L2B: Tensor types and operations.
//!
//! Generic rank-N tensors with flat storage, compute-tier contraction,
//! and index manipulation operations.
//!
//! **FASC-UGOD integration:** Contraction sums use `compute_tier_dot_raw`
//! for the inner accumulation — same pattern as matrix multiply. Each
//! contraction output element is a dot product at tier N+1 with single
//! downscale, giving 1 ULP regardless of the number of summed terms.

use super::FixedPoint;
use super::FixedVector;
use super::FixedMatrix;
use super::linalg::compute_tier_dot_raw;
use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;

// ============================================================================
// Tensor type
// ============================================================================

/// A generic rank-N tensor with fixed-point entries.
///
/// Stored as a flat `Vec<FixedPoint>` in row-major order (last index varies fastest).
/// Shape `[d₀, d₁, ..., d_{N-1}]` means the tensor has N indices,
/// each ranging from 0 to d_i - 1.
#[derive(Clone, Debug)]
pub struct Tensor {
    shape: Vec<usize>,
    strides: Vec<usize>,
    data: Vec<FixedPoint>,
}

impl Tensor {
    /// Create a zero-filled tensor with the given shape.
    pub fn new(shape: &[usize]) -> Self {
        let total = shape.iter().product::<usize>().max(1);
        let strides = Self::compute_strides(shape);
        Self {
            shape: shape.to_vec(),
            strides,
            data: vec![FixedPoint::ZERO; total],
        }
    }

    /// Create a tensor from a flat data slice and shape.
    ///
    /// Panics if `data.len() != product(shape)`.
    pub fn from_data(shape: &[usize], data: &[FixedPoint]) -> Self {
        let total: usize = shape.iter().product();
        assert_eq!(data.len(), total, "Tensor::from_data: shape/data mismatch");
        let strides = Self::compute_strides(shape);
        Self {
            shape: shape.to_vec(),
            strides,
            data: data.to_vec(),
        }
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Tensor rank (number of indices).
    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Shape: dimensions along each index.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Flat index from multi-index.
    fn flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len(), "Tensor: index rank mismatch");
        let mut idx = 0;
        for (i, &d) in indices.iter().enumerate() {
            assert!(d < self.shape[i], "Tensor: index {} out of bounds ({} >= {})", i, d, self.shape[i]);
            idx += d * self.strides[i];
        }
        idx
    }

    /// Get element by multi-index.
    pub fn get(&self, indices: &[usize]) -> FixedPoint {
        self.data[self.flat_index(indices)]
    }

    /// Set element by multi-index.
    pub fn set(&mut self, indices: &[usize], val: FixedPoint) {
        let idx = self.flat_index(indices);
        self.data[idx] = val;
    }

    /// Access the flat data slice.
    pub fn data(&self) -> &[FixedPoint] {
        &self.data
    }
}

// ============================================================================
// Conversions from existing types
// ============================================================================

impl From<FixedPoint> for Tensor {
    /// Rank-0 tensor (scalar).
    fn from(val: FixedPoint) -> Self {
        Tensor {
            shape: vec![],
            strides: vec![],
            data: vec![val],
        }
    }
}

impl From<&FixedVector> for Tensor {
    /// Rank-1 tensor from vector.
    fn from(v: &FixedVector) -> Self {
        let n = v.len();
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            data.push(v[i]);
        }
        Tensor {
            shape: vec![n],
            strides: vec![1],
            data,
        }
    }
}

impl From<&FixedMatrix> for Tensor {
    /// Rank-2 tensor from matrix (row-major).
    fn from(m: &FixedMatrix) -> Self {
        let (rows, cols) = (m.rows(), m.cols());
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(m.get(r, c));
            }
        }
        Tensor {
            shape: vec![rows, cols],
            strides: vec![cols, 1],
            data,
        }
    }
}

impl Tensor {
    /// Convert a rank-2 tensor to FixedMatrix.
    ///
    /// Panics if rank != 2.
    pub fn to_matrix(&self) -> FixedMatrix {
        assert_eq!(self.rank(), 2, "Tensor::to_matrix: rank must be 2");
        let (rows, cols) = (self.shape[0], self.shape[1]);
        FixedMatrix::from_fn(rows, cols, |r, c| self.get(&[r, c]))
    }

    /// Convert a rank-1 tensor to FixedVector.
    ///
    /// Panics if rank != 1.
    pub fn to_vector(&self) -> FixedVector {
        assert_eq!(self.rank(), 1, "Tensor::to_vector: rank must be 1");
        FixedVector::from_slice(&self.data)
    }

    /// Convert a rank-0 tensor to FixedPoint.
    ///
    /// Panics if rank != 0.
    pub fn to_scalar(&self) -> FixedPoint {
        assert_eq!(self.rank(), 0, "Tensor::to_scalar: rank must be 0");
        self.data[0]
    }
}

// ============================================================================
// Tensor contraction (compute-tier accumulation)
// ============================================================================

/// Contract two tensors over specified index pairs.
///
/// `pairs` is a list of (index_in_a, index_in_b) pairs to sum over.
/// The contracted dimensions must match in size.
///
/// **FASC-UGOD strategy:** Each output element is a dot product accumulated
/// at tier N+1 via `compute_tier_dot_raw`. For a contraction summing n terms,
/// this gives 1 ULP per output instead of n ULP from storage-tier accumulation.
///
/// **Special cases:**
/// - `contract(A[i,j], B[j,k], &[(1,0)])` = matrix multiply A×B
/// - `contract(A[i], B[i], &[(0,0)])` = dot product
pub fn contract(a: &Tensor, b: &Tensor, pairs: &[(usize, usize)]) -> Tensor {
    // Validate pairs
    for &(ai, bi) in pairs {
        assert!(ai < a.rank(), "contract: a index {} out of range (rank {})", ai, a.rank());
        assert!(bi < b.rank(), "contract: b index {} out of range (rank {})", bi, b.rank());
        assert_eq!(a.shape[ai], b.shape[bi],
            "contract: dimension mismatch at pair ({},{}): {} vs {}", ai, bi, a.shape[ai], b.shape[bi]);
    }

    let a_contracted: Vec<usize> = pairs.iter().map(|&(ai, _)| ai).collect();
    let b_contracted: Vec<usize> = pairs.iter().map(|&(_, bi)| bi).collect();

    // Output shape: free indices of A followed by free indices of B
    let mut out_shape = Vec::new();
    let mut a_free = Vec::new();
    for (i, &d) in a.shape.iter().enumerate() {
        if !a_contracted.contains(&i) {
            out_shape.push(d);
            a_free.push(i);
        }
    }
    let mut b_free = Vec::new();
    for (i, &d) in b.shape.iter().enumerate() {
        if !b_contracted.contains(&i) {
            out_shape.push(d);
            b_free.push(i);
        }
    }

    // Contracted dimension sizes
    let contract_dims: Vec<usize> = pairs.iter().map(|&(ai, _)| a.shape[ai]).collect();
    let contract_total: usize = contract_dims.iter().product::<usize>().max(1);

    let mut result = Tensor::new(&out_shape);
    if result.len() == 0 || contract_total == 0 {
        return result;
    }

    // Iterate over all output indices
    let out_total = result.len();
    let out_strides = Tensor::compute_strides(&out_shape);

    for out_flat in 0..out_total {
        // Decode output flat index into multi-index
        let mut out_idx = vec![0usize; out_shape.len()];
        let mut remainder = out_flat;
        for i in 0..out_shape.len() {
            out_idx[i] = remainder / out_strides[i];
            remainder %= out_strides[i];
        }

        // Split output indices into A-free and B-free parts
        let a_free_vals = &out_idx[..a_free.len()];
        let b_free_vals = &out_idx[a_free.len()..];

        // Accumulate the contraction sum at compute tier
        let mut a_raw_vals = Vec::with_capacity(contract_total);
        let mut b_raw_vals = Vec::with_capacity(contract_total);

        // Iterate over all contracted index combinations
        let mut contract_idx = vec![0usize; pairs.len()];
        for _ in 0..contract_total {
            // Build full A index
            let mut a_idx = vec![0usize; a.rank()];
            let mut af = 0;
            let mut ac = 0;
            for i in 0..a.rank() {
                if a_contracted.contains(&i) {
                    // Find which contraction pair this is
                    let pos = a_contracted.iter().position(|&x| x == i).unwrap();
                    a_idx[i] = contract_idx[pos];
                    ac += 1;
                    let _ = ac;
                } else {
                    a_idx[i] = a_free_vals[af];
                    af += 1;
                }
            }

            // Build full B index
            let mut b_idx = vec![0usize; b.rank()];
            let mut bf = 0;
            for i in 0..b.rank() {
                if b_contracted.contains(&i) {
                    let pos = b_contracted.iter().position(|&x| x == i).unwrap();
                    b_idx[i] = contract_idx[pos];
                } else {
                    b_idx[i] = b_free_vals[bf];
                    bf += 1;
                }
            }

            a_raw_vals.push(a.get(&a_idx).raw());
            b_raw_vals.push(b.get(&b_idx).raw());

            // Increment contracted multi-index (odometer)
            let mut carry = true;
            for k in (0..pairs.len()).rev() {
                if carry {
                    contract_idx[k] += 1;
                    if contract_idx[k] >= contract_dims[k] {
                        contract_idx[k] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
        }

        // Compute-tier dot product: single downscale for the entire sum
        let val = FixedPoint::from_raw(compute_tier_dot_raw(&a_raw_vals, &b_raw_vals));
        result.data[out_flat] = val;
    }

    result
}

// ============================================================================
// Tensor outer product
// ============================================================================

/// Outer (tensor) product: C_{i₁...iₐ j₁...jᵦ} = A_{i₁...iₐ} * B_{j₁...jᵦ}.
///
/// Result rank = rank(A) + rank(B).
/// Each element is a single product — no accumulation needed.
pub fn outer(a: &Tensor, b: &Tensor) -> Tensor {
    let mut out_shape = a.shape.clone();
    out_shape.extend_from_slice(&b.shape);

    let mut data = Vec::with_capacity(a.len() * b.len());
    for &av in &a.data {
        for &bv in &b.data {
            data.push(av * bv);
        }
    }

    Tensor::from_data(&out_shape, &data)
}

// ============================================================================
// Index operations
// ============================================================================

/// Transpose (reorder indices): T'_{perm[0], perm[1], ...} = T_{0, 1, ...}.
///
/// `perm` is a permutation of `0..rank`.
pub fn transpose(t: &Tensor, perm: &[usize]) -> Tensor {
    let rank = t.rank();
    assert_eq!(perm.len(), rank, "transpose: perm length must match rank");

    // Validate permutation
    let mut seen = vec![false; rank];
    for &p in perm {
        assert!(p < rank, "transpose: invalid permutation index {}", p);
        assert!(!seen[p], "transpose: duplicate permutation index {}", p);
        seen[p] = true;
    }

    // New shape
    let new_shape: Vec<usize> = perm.iter().map(|&p| t.shape[p]).collect();
    let mut result = Tensor::new(&new_shape);

    // Copy elements with permuted indices
    let mut idx = vec![0usize; rank];
    for flat in 0..t.len() {
        // Decode flat → multi-index in original
        let mut remainder = flat;
        for i in 0..rank {
            idx[i] = remainder / t.strides[i];
            remainder %= t.strides[i];
        }

        // Permute indices
        let new_idx: Vec<usize> = perm.iter().map(|&p| idx[p]).collect();
        result.set(&new_idx, t.data[flat]);
    }

    result
}

/// Trace: contract index `idx1` with index `idx2` (self-contraction).
///
/// The two indices must have the same dimension. Result has rank - 2.
/// Sum accumulated at compute tier.
pub fn trace(t: &Tensor, idx1: usize, idx2: usize) -> Tensor {
    assert_ne!(idx1, idx2, "trace: indices must differ");
    let rank = t.rank();
    assert!(idx1 < rank && idx2 < rank, "trace: index out of range");
    assert_eq!(t.shape[idx1], t.shape[idx2],
        "trace: dimensions must match ({} vs {})", t.shape[idx1], t.shape[idx2]);

    let trace_dim = t.shape[idx1];
    let (lo, hi) = if idx1 < idx2 { (idx1, idx2) } else { (idx2, idx1) };

    // Output shape: remove indices lo and hi
    let out_shape: Vec<usize> = t.shape.iter().enumerate()
        .filter(|&(i, _)| i != lo && i != hi)
        .map(|(_, &d)| d)
        .collect();

    let mut result = Tensor::new(&out_shape);
    let out_total = result.len();
    let out_strides = Tensor::compute_strides(&out_shape);

    for out_flat in 0..out_total {
        // Decode output index
        let mut out_idx = vec![0usize; out_shape.len()];
        let mut remainder = out_flat;
        for i in 0..out_shape.len() {
            if !out_strides.is_empty() {
                out_idx[i] = remainder / out_strides[i];
                remainder %= out_strides[i];
            }
        }

        // Build raw values for compute-tier dot: sum over trace dimension
        let mut vals = Vec::with_capacity(trace_dim);
        let ones: Vec<BinaryStorage> = vec![FixedPoint::one().raw(); trace_dim];

        for k in 0..trace_dim {
            // Build full index: insert k at positions lo and hi
            let mut full_idx = Vec::with_capacity(rank);
            let mut oi = 0;
            for i in 0..rank {
                if i == lo || i == hi {
                    full_idx.push(k);
                } else {
                    full_idx.push(out_idx[oi]);
                    oi += 1;
                }
            }
            vals.push(t.get(&full_idx).raw());
        }

        // Compute-tier sum: dot(vals, ones) = sum of vals
        let sum = FixedPoint::from_raw(compute_tier_dot_raw(&vals, &ones));
        result.data[out_flat] = sum;
    }

    result
}

/// Raise an index: T^i = g^{ij} T_j (contraction with metric inverse).
///
/// Contracts index `idx` of tensor T with the second index of `metric_inverse`.
pub fn raise_index(t: &Tensor, idx: usize, metric_inverse: &Tensor) -> Tensor {
    assert_eq!(metric_inverse.rank(), 2, "raise_index: metric must be rank 2");
    assert_eq!(t.shape[idx], metric_inverse.shape()[1],
        "raise_index: dimension mismatch");
    contract(metric_inverse, t, &[(1, idx)])
}

/// Lower an index: T_i = g_{ij} T^j (contraction with metric).
///
/// Contracts index `idx` of tensor T with the second index of `metric`.
pub fn lower_index(t: &Tensor, idx: usize, metric: &Tensor) -> Tensor {
    assert_eq!(metric.rank(), 2, "lower_index: metric must be rank 2");
    assert_eq!(t.shape[idx], metric.shape()[1],
        "lower_index: dimension mismatch");
    contract(metric, t, &[(1, idx)])
}

/// Symmetrize over specified indices: average over all permutations.
///
/// All specified indices must have the same dimension.
/// Result has the same shape as input.
pub fn symmetrize(t: &Tensor, indices: &[usize]) -> Tensor {
    if indices.len() <= 1 {
        return t.clone();
    }

    // Validate: all indexed dimensions equal
    let dim = t.shape[indices[0]];
    for &idx in &indices[1..] {
        assert_eq!(t.shape[idx], dim, "symmetrize: dimensions must match");
    }

    let mut result = Tensor::new(&t.shape);
    let n_perms = factorial(indices.len());
    let inv_n = FixedPoint::one() / FixedPoint::from_int(n_perms as i32);

    // Generate all permutations of the indices
    let perms = permutations(indices.len());

    let rank = t.rank();
    let mut idx = vec![0usize; rank];

    for flat in 0..t.len() {
        // Decode
        let mut remainder = flat;
        for i in 0..rank {
            idx[i] = remainder / t.strides[i];
            remainder %= t.strides[i];
        }

        // Collect permuted values for compute-tier accumulation
        let mut terms: Vec<BinaryStorage> = Vec::with_capacity(perms.len());
        let ones: Vec<BinaryStorage> = vec![FixedPoint::one().raw(); perms.len()];
        for perm in &perms {
            let mut permuted_idx = idx.clone();
            for (pi, &p) in perm.iter().enumerate() {
                permuted_idx[indices[pi]] = idx[indices[p]];
            }
            terms.push(t.get(&permuted_idx).raw());
        }
        // Compute-tier sum: dot(terms, ones) = sum of terms
        let sum = FixedPoint::from_raw(compute_tier_dot_raw(&terms, &ones));

        result.data[flat] = sum * inv_n;
    }

    result
}

/// Antisymmetrize over specified indices: signed average over permutations.
///
/// All specified indices must have the same dimension.
pub fn antisymmetrize(t: &Tensor, indices: &[usize]) -> Tensor {
    if indices.len() <= 1 {
        return t.clone();
    }

    let dim = t.shape[indices[0]];
    for &idx in &indices[1..] {
        assert_eq!(t.shape[idx], dim, "antisymmetrize: dimensions must match");
    }

    let mut result = Tensor::new(&t.shape);
    let n_perms = factorial(indices.len());
    let inv_n = FixedPoint::one() / FixedPoint::from_int(n_perms as i32);

    let perms = permutations(indices.len());
    let signs = perm_signs(indices.len());

    let rank = t.rank();
    let mut idx = vec![0usize; rank];

    for flat in 0..t.len() {
        let mut remainder = flat;
        for i in 0..rank {
            idx[i] = remainder / t.strides[i];
            remainder %= t.strides[i];
        }

        // Collect sign-adjusted values for compute-tier accumulation
        let mut signed_values: Vec<BinaryStorage> = Vec::with_capacity(perms.len());
        let ones: Vec<BinaryStorage> = vec![FixedPoint::one().raw(); perms.len()];
        for (pi, perm) in perms.iter().enumerate() {
            let mut permuted_idx = idx.clone();
            for (qi, &p) in perm.iter().enumerate() {
                permuted_idx[indices[qi]] = idx[indices[p]];
            }
            let val = t.get(&permuted_idx);
            if signs[pi] {
                signed_values.push(val.raw());
            } else {
                signed_values.push((-val).raw());
            }
        }
        // Compute-tier sum: dot(signed_values, ones) = signed sum
        let sum = FixedPoint::from_raw(compute_tier_dot_raw(&signed_values, &ones));

        result.data[flat] = sum * inv_n;
    }

    result
}

// ============================================================================
// Permutation helpers
// ============================================================================

fn factorial(n: usize) -> usize {
    (1..=n).product()
}

/// Generate all permutations of 0..n.
fn permutations(n: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut perm: Vec<usize> = (0..n).collect();
    heap_permute(&mut perm, n, &mut result);
    result
}

fn heap_permute(arr: &mut Vec<usize>, k: usize, result: &mut Vec<Vec<usize>>) {
    if k == 1 {
        result.push(arr.clone());
        return;
    }
    for i in 0..k {
        heap_permute(arr, k - 1, result);
        if k % 2 == 0 {
            arr.swap(i, k - 1);
        } else {
            arr.swap(0, k - 1);
        }
    }
}

/// Compute sign (even=true, odd=false) for each permutation.
fn perm_signs(n: usize) -> Vec<bool> {
    let perms = permutations(n);
    perms.iter().map(|perm| {
        // Count inversions
        let mut inv = 0;
        for i in 0..perm.len() {
            for j in (i + 1)..perm.len() {
                if perm[i] > perm[j] { inv += 1; }
            }
        }
        inv % 2 == 0
    }).collect()
}
