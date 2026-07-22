# Public API

Generated from source by `scripts/gen-public-api.rs` — do not edit by hand. Regenerate with: `rustc -O scripts/gen-public-api.rs -o /tmp/gen-public-api && /tmp/gen-public-api`

This is a **pragmatic source scan** of a curated set of surface files, not a compiler-verified export list. It lists `pub` free items and impl methods with the first line of their doc comment. `pub use` re-exports, `#[cfg(test)]` items, and `#[doc(hidden)]` items are omitted, as are impls of std/derive traits. See the header of `scripts/gen-public-api.rs` for exact scope and limitations.

## Canonical (g_math::canonical)

Modules: g_math::canonical

| Item | Kind | Summary |
| --- | --- | --- |
| `set_gmath_mode` | fn | Set compute:output mode. Examples: "binary:ternary", "auto:auto", "decimal:binary" |
| `reset_gmath_mode` | fn | Reset to default Auto:Auto mode |

**Re-exports** — signatures on [docs.rs](https://docs.rs/g_math):

| Item | Re-exported from |
| --- | --- |
| `LazyExpr` | `super::universal::fasc` |
| `LazyMatrixExpr` | `super::universal::fasc` |
| `DomainMatrix` | `super::universal::fasc` |
| `gmath` | `super::universal::fasc` |
| `gmath_parse` | `super::universal::fasc` |
| `ConstantId` | `super::universal::fasc` |
| `StackEvaluator` | `super::universal::fasc` |
| `StackValue` | `super::universal::fasc` |
| `evaluate` | `super::universal::fasc` |
| `evaluate_matrix` | `super::universal::fasc` |
| `evaluate_sincos` | `super::universal::fasc` |
| `evaluate_sinhcosh` | `super::universal::fasc` |
| `GmathMode` | `super::universal::fasc` |
| `ComputeMode` | `super::universal::fasc` |
| `OutputMode` | `super::universal::fasc` |
| `CompactShadow` | `super::universal::tier_types` |
| `ShadowConstantId` | `super::universal::tier_types` |

## FixedPoint

Modules: FixedPoint (re-exported at g_math::fixed_point)

**Re-exports** — signatures on [docs.rs](https://docs.rs/g_math):

| Item | Re-exported from |
| --- | --- |
| `OverflowDetected` | `crate::fixed_point::core_types::errors` |

### FixedPoint

| Method | Summary |
| --- | --- |
| `ZERO` | Zero constant. |
| `one` | One (1.0) in Q-format. |
| `from_raw` | Create from raw Q-format storage. |
| `raw` | Access the raw Q-format storage. |
| `from_int` | Create from an integer value. |
| `to_int` | Extract the integer part (floor toward negative infinity). |
| `abs` | Absolute value. |
| `is_negative` | Check if negative. |
| `is_zero` | Check if zero. |
| `from_f32` | Create from an f32 value. |
| `from_f64` | Create from an f64 value. |
| `to_f32` | Convert to f32 (lossy — for display/interop only). |
| `to_f64` | Convert to f64 (lossy — for display/interop only). |
| `from_str` | Parse from a decimal string (e.g., "3.14159"). |
| `exp` | e^x |
| `ln` | ln(x), x > 0 |
| `sqrt` | sqrt(x), x >= 0 |
| `sin` | sin(x) |
| `cos` | cos(x) |
| `sincos` | Fused (sin(x), cos(x)) — single range reduction, ~2× faster than separate calls. |
| `tan` | tan(x) = sin(x) / cos(x) — direct composition, no FASC |
| `atan` | atan(x) |
| `asin` | asin(x) = atan(x / sqrt(1 - x^2)), \|x\| <= 1 — direct composition |
| `acos` | acos(x) = pi/2 - asin(x), \|x\| <= 1 — direct composition |
| `sinh` | sinh(x) = (exp(x) - exp(-x)) / 2 — direct composition |
| `cosh` | cosh(x) = (exp(x) + exp(-x)) / 2 — direct composition |
| `sinhcosh` | Fused (sinh(x), cosh(x)) — single shared exp-pair evaluation at compute tier. |
| `tanh` | tanh(x) = (exp(2x) - 1) / (exp(2x) + 1) — direct composition |
| `asinh` | asinh(x) = ln(x + sqrt(x^2 + 1)) — direct composition |
| `acosh` | acosh(x) = ln(x + sqrt(x^2 - 1)), x >= 1 — direct composition |
| `atanh` | atanh(x) = ln((1+x)/(1-x)) / 2, \|x\| < 1 — direct composition |
| `pow` | x^y = exp(y * ln(x)) — direct composition |
| `atan2` | atan2(self=y, x) — direct binary engine |
| `try_exp` | Fallible e^x — returns `Err(TierOverflow)` if result exceeds storage tier. |
| `try_ln` | Fallible ln(x) — returns `Err(DomainError)` if x <= 0. |
| `try_sqrt` | Fallible sqrt(x) — returns `Err(DomainError)` if x < 0. |
| `try_sin` | Fallible sin(x). |
| `try_cos` | Fallible cos(x). |
| `try_sincos` | Fused sin+cos — single shared range reduction at compute tier. |
| `try_sinhcosh` | Fallible fused sinh+cosh — single shared exp-pair at compute tier. |
| `sincos_wide` | Fused sin+cos for wide-range angles that exceed storage-tier integer range. |
| `try_tan` | Fallible tan(x). |
| `try_atan` | Fallible atan(x). |
| `try_asin` | Fallible asin(x) — returns `Err(DomainError)` if \|x\| > 1. |
| `try_acos` | Fallible acos(x) — returns `Err(DomainError)` if \|x\| > 1. |
| `try_sinh` | Fallible sinh(x). |
| `try_cosh` | Fallible cosh(x). |
| `try_tanh` | Fallible tanh(x). |
| `try_asinh` | Fallible asinh(x). |
| `try_acosh` | Fallible acosh(x) — returns `Err(DomainError)` if x < 1. |
| `try_atanh` | Fallible atanh(x) — returns `Err(DomainError)` if \|x\| >= 1. |
| `try_pow` | Fallible x^y = exp(y * ln(x)). |
| `try_atan2` | Fallible atan2(self=y, x). |

## FixedVector

Modules: FixedVector

### FixedVector

A dynamically-sized vector of fixed-point values.

| Method | Summary |
| --- | --- |
| `new` | Create a zero-filled vector of the given dimension. |
| `from_f32_slice` | Create from a slice of f32 values. |
| `from_slice` | Create from a slice of FixedPoint values. |
| `len` | Number of components. |
| `dimension` | Alias for `len()`. |
| `is_empty` | Whether the vector is empty. |
| `dot` | Dot product of two vectors at compute tier (tier N+1). |
| `length_squared` | Squared length (self . self). |
| `length` | Length (Euclidean norm). |
| `length_fused` | Fused length — sqrt(Σ x_i²) entirely at compute tier. |
| `distance_to` | Fused Euclidean distance to another vector — sqrt(Σ (a_i - b_i)²) |
| `normalize` | Normalize in place (divide each component by length). |
| `normalized` | Return a normalized copy. |
| `map` | Apply a function to each component, returning a new vector. |
| `iter` | Iterator over components. |
| `iter_mut` | Mutable iterator over components. |
| `metric_distance_safe` | Metric distance between two vectors (Euclidean) at compute tier. |
| `dot_precise` | Compute-tier precise dot product. |
| `cross` | Cross product (3D vectors only). |
| `outer_product` | Outer product: u ⊗ v → Matrix where M[i][j] = u[i] * v[j]. |
| `as_slice` | Access the underlying data slice (for compute-tier operations). |

## FixedMatrix

Modules: FixedMatrix

### FixedMatrix

A row-major matrix of fixed-point values.

| Method | Summary |
| --- | --- |
| `new` | Create a zero-filled matrix. |
| `rows` | Number of rows. |
| `cols` | Number of columns. |
| `get` | Get element at (row, col). |
| `set` | Set element at (row, col). |
| `from_slice` | Create from a flat slice of values in row-major order. |
| `from_fn` | Create from a function: M[i][j] = f(i, j). |
| `identity` | Identity matrix of size n×n. |
| `diagonal` | Diagonal matrix from a vector. |
| `transpose` | Transpose: Aᵀ[i][j] = A[j][i]. |
| `trace` | Trace: sum of diagonal elements. |
| `is_square` | Whether this is a square matrix. |
| `row` | Extract a row as a FixedVector. |
| `col` | Extract a column as a FixedVector. |
| `mul_vector` | Matrix-vector multiply: y = A * x. |
| `submatrix` | Extract a submatrix starting at (row, col) with given dimensions. |
| `set_submatrix` | Insert a submatrix at position (row, col). |
| `kronecker` | Kronecker product: A ⊗ B. |
| `swap_rows` | Swap rows `i` and `j` in place. O(cols) time, zero allocation. |

## DecimalFixed

Modules: DecimalFixed

| Item | Kind | Summary |
| --- | --- | --- |
| `ParseError` | enum | Parse error for decimal string conversion |
| `compile_time_power_of_10` | fn | Compile-time power of 10 calculation |
| `DecimalFixed2` | type | Common decimal precision type aliases |
| `DecimalFixed3` | type |  |
| `DecimalFixed6` | type |  |
| `DecimalFixed9` | type |  |
| `Currency` | type | Financial precision (2 decimal places) |
| `HighPrecisionCurrency` | type | High precision financial (6 decimal places) |

### DecimalFixed

Exact decimal fixed-point arithmetic with configurable precision

| Method | Summary |
| --- | --- |
| `SCALE` | Scale factor: 10^DECIMALS computed at compile time |
| `MAX_VALUE` | Maximum safe value before overflow |
| `MIN_VALUE` | Minimum safe value before underflow |
| `ZERO` | Zero value |
| `ONE` | One value |
| `from_raw` | Create from raw scaled value (internal use) |
| `from_raw_checked` | Create from raw scaled value with overflow check |
| `from_integer` | Create from integer value (no decimal part) |
| `from_parts` | Create from parts: integer and fractional parts |
| `from_decimal_str_decimal` | Parse decimal string without float conversion |
| `integer_part` | Extract integer part |
| `fractional_part` | Extract fractional part as integer (e.g., 0.123 → 123 for DECIMALS=3) |
| `raw_value` | Get raw scaled value |
| `is_zero` | Check if value is zero |
| `is_negative` | Check if value is negative |
| `is_positive` | Check if value is positive |
| `abs` | Absolute value |
| `pure_decimal_multiply_decimal` | Pure decimal multiplication using base-10 arithmetic (eliminates binary contamination) |
| `pure_decimal_multiply_optimized_decimal` | PRODUCTION-OPTIMIZED: Pure decimal multiplication with 20-50x performance improvement |
| `multiply_exact_decimal` | Decimal multiplication using 256-bit intermediate to prevent truncation |
| `pure_decimal_add_decimal` | Pure decimal addition using optimized scaled integer arithmetic |
| `pure_decimal_subtract_decimal` | Pure decimal subtraction using optimized scaled integer arithmetic |
| `pure_decimal_negate_decimal` | Pure decimal negation using optimized scaled integer arithmetic |
| `pure_decimal_divide_decimal` | Pure decimal division using base-10 arithmetic (eliminates binary contamination) |
| `multiply_batch_decimal` | High-performance multiplication for batch operations |
| `to_f64_lossy` | Convert to f64 (lossy conversion for display/debugging) |
| `try_convert` | Convert to different decimal precision |
| `convert_with_rounding` | Force conversion to different decimal precision with rounding |
| `to_binary_q256` | Convert DecimalFixed to Q256.256 binary format (I512) |
| `from_binary_q256` | Create DecimalFixed from Q256.256 binary format (I512) |
| `exp` | `exp(x)` — native decimal exponential at full compute-tier precision. |
| `ln` | `ln(x)` — native decimal natural logarithm. Requires x > 0. |
| `sqrt` | `sqrt(x)` — native decimal square root. Requires x >= 0. |
| `sin` | `sin(x)` — native decimal sine. |
| `cos` | `cos(x)` — native decimal cosine. |
| `sincos` | `sincos(x)` — fused sine and cosine with single range reduction. |
| `tan` | `tan(x)` = sin(x)/cos(x), composed entirely at the compute tier. |
| `atan` | `atan(x)` — native decimal arctangent. |
| `atan2` | `atan2(y, x)` — native decimal two-argument arctangent. |
| `asin` | `asin(x)` = atan(x / sqrt(1 - x^2)), composed at the compute tier. |
| `acos` | `acos(x)` = pi/2 - asin(x), composed at the compute tier (single |
| `sinh` | `sinh(x)` — fused (exp(x) - exp(-x)) / 2 at the compute tier. |
| `cosh` | `cosh(x)` — fused (exp(x) + exp(-x)) / 2 at the compute tier. |
| `sinhcosh` | `sinhcosh(x)` — fused hyperbolic pair sharing one exp-pair evaluation at |
| `tanh` | `tanh(x)` — (exp(2x) - 1)/(exp(2x) + 1) at the compute tier, saturating to |
| `asinh` | `asinh(x)` = ln(x + sqrt(x^2 + 1)), composed at the compute tier. |
| `acosh` | `acosh(x)` = ln(x + sqrt(x^2 - 1)), composed at the compute tier. |
| `atanh` | `atanh(x)` = ln((1+x)/(1-x)) / 2, composed at the compute tier. |

## Fused operations

Modules: g_math::fixed_point::imperative::fused

| Item | Kind | Summary |
| --- | --- | --- |
| `sqrt_sum_sq` | fn | Fused sqrt(Σ x_i²) — norm of a slice, entirely at compute tier. |
| `euclidean_distance` | fn | Fused sqrt(Σ (a_i - b_i)²) — Euclidean distance, entirely at compute tier. |
| `euclidean_distance_squared` | fn | Fused Σ (a_i − b_i)² — squared Euclidean distance at compute tier, no sqrt (U1). |
| `dot` | fn | Fused Σ a_i·b_i — dot product entirely at compute tier (U1). |
| `mobius_denominator_sq` | fn | Fused squared Möbius denominator `\|1 − p̄q\|² = 1 − 2⟨p,q⟩ + \|p\|²·\|q\|²` (U1). |
| `softmax` | fn | Stable softmax entirely at compute tier. |
| `rms_norm_factor` | fn | Fused 1/sqrt(mean(x²) + eps) — RMSNorm scaling factor at compute tier. |
| `silu` | fn | Fused SiLU activation: x / (1 + exp(-x)) entirely at compute tier. |
| `softmax_mix` | fn | Fused softmax + weighted value mix, entirely at compute tier: |

## Linear algebra

Modules: g_math::fixed_point::imperative::decompose, g_math::fixed_point::imperative::derived, g_math::fixed_point::imperative::matrix_functions

| Item | Kind | Summary |
| --- | --- | --- |
| `lu_decompose` | fn | LU decomposition with partial pivoting (Doolittle, compute-tier). |
| `qr_decompose` | fn | QR decomposition via Householder reflections. |
| `cholesky_decompose` | fn | Cholesky decomposition for symmetric positive-definite matrices. |
| `EigenDecomposition` | struct | Result of symmetric eigenvalue decomposition: A = Q Λ Qᵀ. |
| `eigen_symmetric` | fn | Symmetric eigenvalue decomposition via the classical Jacobi method. |
| `SVDDecomposition` | struct | Result of SVD: A = U Σ Vᵀ. |
| `svd_decompose` | fn | SVD via Golub-Kahan bidiagonalization + implicit QR iteration. |
| `SchurDecomposition` | struct | Result of real Schur decomposition: A = Q T Qᵀ. |
| `schur_decompose` | fn | Real Schur decomposition via Hessenberg reduction + Francis implicit double-shift QR. |
| `frobenius_norm` | fn | Frobenius norm: \|\|A\|\|_F = sqrt(sum of squares of all entries). |
| `norm_1` | fn | 1-norm: max absolute column sum, accumulated at compute tier. |
| `norm_inf` | fn | Infinity-norm: max absolute row sum, accumulated at compute tier. |
| `least_squares` | fn | Least-squares solve: min \|\|Ax - b\|\|_2 via QR decomposition. |
| `inverse_spd` | fn | Inverse of a symmetric positive-definite matrix via Cholesky. |
| `condition_number_1` | fn | Condition number estimate: κ_1(A) = \|\|A\|\|_1 * \|\|A^{-1}\|\|_1. |
| `solve` | fn | Solve Ax = b using LU decomposition. |
| `solve_spd` | fn | Solve Ax = b for SPD matrix using Cholesky. |
| `determinant` | fn | Determinant via LU decomposition. |
| `inverse` | fn | Matrix inverse via LU decomposition. |
| `pseudoinverse` | fn | Moore-Penrose pseudoinverse: A⁺ = V Σ⁺ Uᵀ. |
| `pseudoinverse_with_threshold` | fn | Pseudoinverse with a user-specified threshold. |
| `rank` | fn | Numerical rank: count of singular values above threshold. |
| `condition_number_2` | fn | 2-norm condition number: κ₂(A) = σ_max / σ_min. |
| `nullspace` | fn | Nullspace basis: columns of V corresponding to near-zero singular values. |
| `matrix_exp` | fn | Matrix exponential: exp(A) via Padé [6/6] with scaling-and-squaring. |
| `matrix_sqrt` | fn | Matrix square root: A^{1/2} via Denman-Beavers iteration at compute tier. |
| `matrix_log` | fn | Matrix logarithm: log(A) via inverse scaling-and-squaring at compute tier. |
| `matrix_pow` | fn | Matrix power: A^p = exp(p * log(A)) for real scalar p. |

### LUDecomposition

Result of LU decomposition with partial pivoting: PA = LU.

| Method | Summary |
| --- | --- |
| `solve` | Solve Ax = b using forward then back substitution. |
| `determinant` | Determinant: det(A) = (-1)^num_swaps * product(U diagonal). |
| `refine` | Iterative refinement: improve solution accuracy by computing residual |
| `inverse` | Compute A^{-1} by solving AX = I column by column. |

### QRDecomposition

Result of QR decomposition via Householder reflections: A = QR.

| Method | Summary |
| --- | --- |
| `solve` | Solve Ax = b via R^{-1} Q^T b (back substitution). |

### CholeskyDecomposition

Result of Cholesky decomposition: A = LL^T.

| Method | Summary |
| --- | --- |
| `solve` | Solve Ax = b: forward (Ly = b), then back (L^T x = y). |
| `determinant` | Determinant: det(A) = product(L[i][i])^2. |

## Geometry

Modules: g_math::fixed_point::imperative::manifold, g_math::fixed_point::imperative::lie_group, g_math::fixed_point::imperative::curvature, g_math::fixed_point::imperative::projective, g_math::fixed_point::imperative::fiber_bundle

| Item | Kind | Summary |
| --- | --- | --- |
| `Manifold` | trait | A Riemannian manifold with fixed-point arithmetic. |
| `LieGroup` | trait | A Lie group with fixed-point arithmetic. |
| `differentiation_step` | fn | Optimal step size for central differences: h = 2^(-FRAC_BITS/3). |
| `MetricProvider` | trait | A metric function: given a point (as FixedVector of coordinates), returns |
| `christoffel` | fn | Compute Christoffel symbols Γ^k_{ij} at point p. |
| `riemann_curvature` | fn | Compute Riemann curvature tensor R^l_{ijk} at point p. |
| `ricci_tensor` | fn | Compute Ricci tensor Rᵢⱼ = R^k_{ikj} at point p. |
| `ricci_from_riemann` | fn | Compute Ricci tensor from a pre-computed Riemann tensor. |
| `scalar_curvature` | fn | Compute scalar curvature R = g^{ij} R_{ij} at point p. |
| `scalar_from_ricci` | fn | Compute scalar curvature from pre-computed Ricci tensor and inverse metric. |
| `sectional_curvature` | fn | Compute sectional curvature K(u, v) at point p. |
| `geodesic_integrate` | fn | Integrate the geodesic equation from an initial point and velocity. |
| `parallel_transport_ode` | fn | Parallel transport a tangent vector along a discrete curve. |
| `to_homogeneous` | fn | Convert affine coordinates [x₁, ..., xₙ] to homogeneous [x₁, ..., xₙ, 1]. |
| `from_homogeneous` | fn | Convert homogeneous coordinates [x₁, ..., xₙ, w] to affine [x₁/w, ..., xₙ/w]. |
| `is_at_infinity` | fn | Check if a homogeneous point is at infinity (last component ≈ 0). |
| `projective_transform` | fn | Apply a projective transformation H (n+1)×(n+1) to a point in affine coordinates. |
| `projective_transform_homogeneous` | fn | Apply a projective transformation to a homogeneous-coordinate point. |
| `compose_projective` | fn | Compose two projective transformations (matrix multiplication). |
| `cross_ratio_1d` | fn | Cross-ratio of 4 collinear points (a, b, c, d) in R¹. |
| `cross_ratio` | fn | Cross-ratio for 4 collinear points in R^n (using ratios of signed distances). |
| `stereo_project` | fn | Stereographic projection from S^n to R^n (north pole projection). |
| `stereo_unproject` | fn | Inverse stereographic projection from R^n to S^n. |
| `FiberBundle` | trait | A fiber bundle π: E → B with fiber F. |
| `BundleConnection` | trait | A connection on a fiber bundle — specifies how fibers relate along the base. |
| `apply_representation` | fn | Apply a transition function (group element) to a fiber element via |
| `change_chart` | fn | Change of chart for a section: ξ_β = g_{αβ} · ξ_α. |
| `vector_bundle_curvature` | fn | Compute the curvature 2-form of a vector bundle connection at a point. |

### EuclideanSpace

Flat Euclidean space R^n.

| Method | Summary |
| --- | --- |
| `dimension` |  |
| `inner_product` |  |
| `exp_map` |  |
| `log_map` |  |
| `distance` |  |
| `parallel_transport` |  |

### Sphere

The n-sphere S^n embedded as unit vectors in R^{n+1}.

| Method | Summary |
| --- | --- |
| `dimension` |  |
| `inner_product` |  |
| `exp_map` |  |
| `log_map` |  |
| `distance` |  |
| `parallel_transport` |  |

### HyperbolicSpace

Hyperbolic space H^n in the hyperboloid model.

| Method | Summary |
| --- | --- |
| `dimension` |  |
| `inner_product` |  |
| `norm` |  |
| `exp_map` |  |
| `log_map` |  |
| `distance` |  |
| `parallel_transport` |  |

### SPDManifold

The manifold of n×n symmetric positive-definite matrices.

| Method | Summary |
| --- | --- |
| `dimension` |  |
| `inner_product` |  |
| `exp_map` |  |
| `log_map` |  |
| `distance` |  |
| `parallel_transport` |  |

### Grassmannian

The Grassmann manifold Gr(k, n): k-dimensional subspaces of R^n.

| Method | Summary |
| --- | --- |
| `dimension` |  |
| `inner_product` |  |
| `exp_map` |  |
| `log_map` |  |
| `distance` |  |
| `parallel_transport` |  |

### StiefelManifold

The Stiefel manifold St(k, n): orthonormal k-frames in R^n.

| Method | Summary |
| --- | --- |
| `dimension` |  |
| `inner_product` |  |
| `exp_map` |  |
| `log_map` |  |
| `distance` |  |
| `parallel_transport` |  |

### ProductManifold

Product manifold M₁ × M₂: the Cartesian product of two manifolds.

| Method | Summary |
| --- | --- |
| `new` | Create a product manifold M₁ × M₂. |
| `dimension` |  |
| `inner_product` |  |
| `exp_map` |  |
| `log_map` |  |
| `distance` |  |
| `parallel_transport` |  |

### SO3

SO(3): Special orthogonal group of 3D rotations.

| Method | Summary |
| --- | --- |
| `hat_so3` | hat: ω = [wx, wy, wz] → 3×3 skew-symmetric matrix. |
| `vee_so3` | vee: extract ω from skew-symmetric matrix. |
| `rodrigues_exp` | Rodrigues exponential: ω (3-vector) → rotation matrix R. |
| `rodrigues_log` | Rodrigues logarithm: rotation matrix R → axis-angle ω. |
| `dimension` |  |
| `inner_product` |  |
| `exp_map` |  |
| `log_map` |  |
| `distance` |  |
| `parallel_transport` |  |
| `algebra_dim` |  |
| `matrix_dim` |  |
| `identity_element` |  |
| `compose` |  |
| `group_inverse` |  |
| `lie_exp` |  |
| `lie_log` |  |
| `hat` |  |
| `vee` |  |
| `adjoint` |  |
| `bracket` |  |
| `act` |  |

### SE3

SE(3): Special Euclidean group of 3D rigid body motions.

| Method | Summary |
| --- | --- |
| `hat_se3` | hat: ξ = [ωx, ωy, ωz, vx, vy, vz] → 4×4 se(3) matrix. |
| `vee_se3` | vee: extract 6-vector from 4×4 se(3) matrix. |
| `extract_rotation` | Extract R (3×3) from homogeneous matrix. |
| `extract_translation` | Extract t (3-vector) from homogeneous matrix. |
| `from_rt` | Build 4×4 homogeneous from R and t. |
| `se3_exp` | SE(3) exponential: ξ = [ω, v] → [[R, V·v], [0, 1]]. |
| `se3_log` | SE(3) logarithm: [[R, t], [0, 1]] → [ω, v]. |
| `dimension` |  |
| `inner_product` |  |
| `exp_map` |  |
| `log_map` |  |
| `distance` |  |
| `parallel_transport` |  |
| `algebra_dim` |  |
| `matrix_dim` |  |
| `identity_element` |  |
| `compose` |  |
| `group_inverse` |  |
| `lie_exp` |  |
| `lie_log` |  |
| `hat` |  |
| `vee` |  |
| `adjoint` |  |
| `bracket` |  |
| `act` |  |

### SOn

SO(n): General special orthogonal group.

| Method | Summary |
| --- | --- |
| `hat_son` | hat: vector → skew-symmetric n×n matrix. |
| `vee_son` | vee: skew-symmetric n×n matrix → vector. |
| `dimension` |  |
| `inner_product` |  |
| `exp_map` |  |
| `log_map` |  |
| `distance` |  |
| `parallel_transport` |  |
| `algebra_dim` |  |
| `matrix_dim` |  |
| `identity_element` |  |
| `compose` |  |
| `group_inverse` |  |
| `lie_exp` |  |
| `lie_log` |  |
| `hat` |  |
| `vee` |  |
| `adjoint` |  |
| `bracket` |  |
| `act` |  |

### GLn

GL(n): General linear group of invertible n×n matrices.

| Method | Summary |
| --- | --- |
| `hat_gln` | hat: n²-vector → n×n matrix (row-major). |
| `vee_gln` | vee: n×n matrix → n²-vector (row-major). |
| `dimension` |  |
| `inner_product` |  |
| `exp_map` |  |
| `log_map` |  |
| `distance` |  |
| `parallel_transport` |  |
| `algebra_dim` |  |
| `matrix_dim` |  |
| `identity_element` |  |
| `compose` |  |
| `group_inverse` |  |
| `lie_exp` |  |
| `lie_log` |  |
| `hat` |  |
| `vee` |  |
| `adjoint` |  |
| `bracket` |  |
| `act` |  |

### On

O(n): Orthogonal group — matrices with QᵀQ = I, det = ±1.

| Method | Summary |
| --- | --- |
| `dimension` |  |
| `inner_product` |  |
| `exp_map` |  |
| `log_map` |  |
| `distance` |  |
| `parallel_transport` |  |
| `algebra_dim` |  |
| `matrix_dim` |  |
| `identity_element` |  |
| `compose` |  |
| `group_inverse` |  |
| `lie_exp` |  |
| `lie_log` |  |
| `hat` |  |
| `vee` |  |
| `adjoint` |  |
| `bracket` |  |
| `act` |  |

### SLn

SL(n): Special linear group — n×n matrices with det = 1.

| Method | Summary |
| --- | --- |
| `hat_sln` | hat: (n²-1)-vector → traceless n×n matrix. |
| `vee_sln` | vee: traceless n×n matrix → (n²-1)-vector. |
| `project_traceless` | Project matrix onto sl(n) by removing trace: A - (tr(A)/n)·I. |
| `dimension` |  |
| `inner_product` |  |
| `exp_map` |  |
| `log_map` |  |
| `distance` |  |
| `parallel_transport` |  |
| `algebra_dim` |  |
| `matrix_dim` |  |
| `identity_element` |  |
| `compose` |  |
| `group_inverse` |  |
| `lie_exp` |  |
| `lie_log` |  |
| `hat` |  |
| `vee` |  |
| `adjoint` |  |
| `bracket` |  |
| `act` |  |

### EuclideanMetric

Flat Euclidean metric: g_ij = δ_ij.

| Method | Summary |
| --- | --- |
| `dimension` |  |
| `metric` |  |
| `metric_inverse` |  |

### SphereMetric

Sphere S^n metric in spherical coordinates.

| Method | Summary |
| --- | --- |
| `dimension` |  |
| `metric` |  |
| `christoffel_closed_form` | Exact Christoffel symbols for S² in (θ, φ) coordinates. |
| `scalar_curvature_closed_form` | Exact scalar curvature for S²: R = 2/r². |

### HyperbolicMetric

Hyperbolic space H^2 metric in the upper half-plane model.

| Method | Summary |
| --- | --- |
| `dimension` |  |
| `metric` |  |
| `christoffel_closed_form` | Exact Christoffel symbols for H² upper half-plane. |
| `scalar_curvature_closed_form` | Exact scalar curvature for H²: R = -2. |

### GeodesicOde

ODE system for the geodesic equation on a Riemannian manifold.

| Method | Summary |
| --- | --- |
| `eval` |  |

### Moebius

A Möbius transformation on the complex plane: z ↦ (az+b)/(cz+d).

| Method | Summary |
| --- | --- |
| `new` | Create a new Möbius transformation. |
| `identity` | Identity transformation: z ↦ z. |
| `apply` | Apply the transformation to a real value: (ax+b)/(cx+d). |
| `compose` | Compose two Möbius transformations: (self ∘ other)(z) = self(other(z)). |
| `inverse` | Inverse transformation: z ↦ (dz-b)/(-cz+a). |
| `determinant` | Determinant: ad - bc. Non-zero for valid Möbius transformation. |
| `to_matrix` | Convert to the corresponding 2×2 projective matrix [[a,b],[c,d]]. |

### MoebiusComplex

A Möbius transformation with complex coefficients: z ↦ (az+b)/(cz+d)

| Method | Summary |
| --- | --- |
| `new` | Create a new complex Möbius transformation. |
| `apply` | Apply to a complex number z = (re, im). |
| `compose` | Compose two complex Möbius transformations. |
| `inverse` | Inverse transformation. |

### TrivialBundle

A trivial fiber bundle E = B × F with the flat (product) connection.

| Method | Summary |
| --- | --- |
| `project` |  |
| `base_dim` |  |
| `fiber_dim` |  |
| `lift` |  |
| `local_trivialization` |  |
| `horizontal_lift` |  |
| `vertical_component` |  |
| `parallel_transport_along` |  |

### VectorBundle

A vector bundle with fiber R^k over a base manifold of dimension n.

| Method | Summary |
| --- | --- |
| `flat` | Create a vector bundle with flat connection. |
| `with_connection` | Create a vector bundle with given connection coefficients. |
| `project` |  |
| `base_dim` |  |
| `fiber_dim` |  |
| `lift` |  |
| `local_trivialization` |  |
| `horizontal_lift` |  |
| `vertical_component` |  |
| `parallel_transport_along` |  |

### PrincipalBundle

A principal G-bundle where the fiber is a Lie group.

| Method | Summary |
| --- | --- |
| `trivial` | Create a trivial principal bundle (all transitions are identity). |
| `transition` | Get the transition function g_{αβ}. |
| `set_transition` | Set a transition function g_{αβ} and automatically set g_{βα} = g_{αβ}⁻¹. |
| `verify_cocycle` | Verify the cocycle condition: g_{αβ} · g_{βγ} = g_{αγ} for all triples. |
| `project` |  |
| `base_dim` |  |
| `fiber_dim` |  |
| `lift` |  |
| `local_trivialization` |  |

## ODE

Modules: g_math::fixed_point::imperative::ode

| Item | Kind | Summary |
| --- | --- | --- |
| `OdeSystem` | trait | Right-hand side of an ODE system: dx/dt = f(t, x). |
| `ode_fn` | fn | Wrap a closure as an ODE system. |
| `OdePoint` | struct | A single point in an ODE solution trajectory. |
| `rk4_step` | fn | Perform a single RK4 step: x(t+h) from x(t). |
| `rk4_integrate` | fn | Integrate an ODE from t0 to t_end using fixed-step RK4. |
| `rk45_integrate` | fn | Integrate an ODE using adaptive Dormand-Prince RK45. |
| `HamiltonianSystem` | trait | A Hamiltonian system: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q. |
| `HamiltonianPoint` | struct | Result of a Hamiltonian integration step. |
| `verlet_step` | fn | Perform one Störmer-Verlet step. |
| `verlet_integrate` | fn | Integrate a Hamiltonian system using symplectic Störmer-Verlet. |
| `monitor_invariant` | fn | Monitor a conserved quantity during integration. |

### OdeFn

A boxed closure implementing OdeSystem for convenience.

| Method | Summary |
| --- | --- |
| `eval` |  |

### Rk45Config

Configuration for adaptive RK45 integration.

| Method | Summary |
| --- | --- |
| `new` | Default configuration with the given tolerance and initial step. |

## Tensors

Modules: g_math::fixed_point::imperative::tensor, g_math::fixed_point::imperative::tensor_decompose

| Item | Kind | Summary |
| --- | --- | --- |
| `contract` | fn | Contract two tensors over specified index pairs. |
| `outer` | fn | Outer (tensor) product: C_{i₁...iₐ j₁...jᵦ} = A_{i₁...iₐ} * B_{j₁...jᵦ}. |
| `transpose` | fn | Transpose (reorder indices): T'_{perm[0], perm[1], ...} = T_{0, 1, ...}. |
| `trace` | fn | Trace: contract index `idx1` with index `idx2` (self-contraction). |
| `raise_index` | fn | Raise an index: T^i = g^{ij} T_j (contraction with metric inverse). |
| `lower_index` | fn | Lower an index: T_i = g_{ij} T^j (contraction with metric). |
| `symmetrize` | fn | Symmetrize over specified indices: average over all permutations. |
| `antisymmetrize` | fn | Antisymmetrize over specified indices: signed average over permutations. |
| `truncated_svd` | fn | Compute truncated SVD keeping the top-k singular values. |
| `truncated_svd_auto` | fn | Compute truncated SVD with automatic rank selection via singular value threshold. |
| `tucker_decompose` | fn | Compute Tucker decomposition via HOSVD. |
| `cp_decompose` | fn | Compute CP decomposition via Alternating Least Squares. |

### Tensor

A generic rank-N tensor with fixed-point entries.

| Method | Summary |
| --- | --- |
| `new` | Create a zero-filled tensor with the given shape. |
| `from_data` | Create a tensor from a flat data slice and shape. |
| `rank` | Tensor rank (number of indices). |
| `shape` | Shape: dimensions along each index. |
| `len` | Total number of elements. |
| `get` | Get element by multi-index. |
| `set` | Set element by multi-index. |
| `data` | Access the flat data slice. |
| `to_matrix` | Convert a rank-2 tensor to FixedMatrix. |
| `to_vector` | Convert a rank-1 tensor to FixedVector. |
| `to_scalar` | Convert a rank-0 tensor to FixedPoint. |

### TruncatedSVD

Truncated SVD: A ≈ U_k Σ_k V_k^T where k << min(m,n).

| Method | Summary |
| --- | --- |
| `reconstruct` | Reconstruct the rank-k approximation: U_k Σ_k V_k^T. |
| `compression_ratio` | Compression ratio: original_elements / compressed_elements. |

### TuckerDecomposition

Tucker decomposition: T ≈ G ×₁ U₁ ×₂ U₂ ×₃ U₃ ...

| Method | Summary |
| --- | --- |
| `reconstruct` | Reconstruct the full tensor from core + factors. |
| `compression_ratio` | Compression ratio: original_elements / (core + factor) elements. |

### CPDecomposition

CP (Canonical Polyadic) decomposition: T ≈ Σ_r λ_r a₁_r ∘ a₂_r ∘ ... ∘ a_N_r

| Method | Summary |
| --- | --- |
| `reconstruct` | Reconstruct the full tensor from CP factors. |

## Balanced ternary

Modules: g_math::fixed_point::domains::balanced_ternary

| Item | Kind | Summary |
| --- | --- | --- |
| `pack_trits` | fn | Pack trits into bytes, 5 trits per byte using base-3 encoding. |
| `unpack_trits` | fn | Unpack bytes back to trits. |

**Re-exports** — signatures on [docs.rs](https://docs.rs/g_math):

| Item | Re-exported from |
| --- | --- |
| `UniversalTernaryFixed` | `ternary_types` |
| `TernaryTier` | `ternary_types` |
| `TernaryTier1` | `ternary_types` |
| `TernaryTier2` | `ternary_types` |
| `TernaryTier3` | `ternary_types` |
| `TernaryTier4` | `ternary_types` |
| `TernaryTier5` | `ternary_types` |
| `TernaryTier6` | `ternary_types` |
| `TernaryValue` | `ternary_types` |
| `TernaryRaw` | `ternary_types` |
| `SCALE_TQ8_8` | `ternary_types` |
| `SCALE_TQ16_16` | `ternary_types` |
| `SCALE_TQ32_32` | `ternary_types` |
| `add_ternary_tq8_8` | `ternary_addition` |
| `add_ternary_tq16_16` | `ternary_addition` |
| `add_ternary_tq32_32` | `ternary_addition` |
| `add_ternary_tq64_64` | `ternary_addition` |
| `subtract_ternary_tq8_8` | `ternary_addition` |
| `subtract_ternary_tq16_16` | `ternary_addition` |
| `subtract_ternary_tq32_32` | `ternary_addition` |
| `subtract_ternary_tq64_64` | `ternary_addition` |
| `add_ternary_tq64_64_checked` | `ternary_addition` |
| `subtract_ternary_tq64_64_checked` | `ternary_addition` |
| `add_ternary_tq128_128` | `ternary_addition` |
| `subtract_ternary_tq128_128` | `ternary_addition` |
| `add_ternary_tq256_256` | `ternary_addition` |
| `subtract_ternary_tq256_256` | `ternary_addition` |
| `multiply_ternary_tq8_8` | `ternary_multiplication` |
| `multiply_ternary_tq16_16` | `ternary_multiplication` |
| `multiply_ternary_tq32_32` | `ternary_multiplication` |
| `multiply_ternary_tq64_64` | `ternary_multiplication` |
| `multiply_ternary_tq64_64_checked` | `ternary_multiplication` |
| `multiply_ternary_tq128_128` | `ternary_multiplication` |
| `multiply_ternary_tq256_256` | `ternary_multiplication` |
| `divide_ternary_tq8_8` | `ternary_division` |
| `divide_ternary_tq16_16` | `ternary_division` |
| `divide_ternary_tq32_32` | `ternary_division` |
| `divide_ternary_tq64_64` | `ternary_division` |
| `divide_ternary_tq64_64_checked` | `ternary_division` |
| `divide_ternary_tq128_128` | `ternary_division` |
| `divide_ternary_tq256_256` | `ternary_division` |
| `negate_ternary_tq8_8` | `ternary_negation` |
| `negate_ternary_tq16_16` | `ternary_negation` |
| `negate_ternary_tq32_32` | `ternary_negation` |
| `negate_ternary_tq64_64` | `ternary_negation` |
| `negate_ternary_tq128_128` | `ternary_negation` |
| `negate_ternary_tq256_256` | `ternary_negation` |
| `OverflowDetected` | `crate::fixed_point::core_types::errors` |
| `TritQ1_9` | `trit_q1_9` |

### Trit

A balanced ternary digit: -1, 0, or +1

| Method | Summary |
| --- | --- |
| `from_i8` | Convert from i8. Returns Err if not in {-1, 0, 1}. |
| `as_i8` | Convert to i8 |

## TQ1.9 inference

Modules: g_math::tq19 _(feature: inference)_

| Item | Kind | Summary |
| --- | --- | --- |
| `SCALE` | const | TQ1.9 scale factor: 3^9 = 19683. |
| `MAX_RAW` | const | Maximum raw i16 value: (3^10 - 1) / 2. |
| `MIN_RAW` | const | Minimum raw i16 value. |
| `TRIT_DECODE_TABLE` | const | Pre-decoded trit table: maps each byte to 5 balanced trits in {-1, 0, +1}. |
| `tq19_dot` | fn | TQ1.9 dot product: `sum(weights[i] * activations[i]) / SCALE` |
| `tq19_dot_q2f` | fn | Wide-output TQ1.9 dot: the exact dot value at 2·FRAC_BITS fractional precision. |
| `trit_dot` | fn | Zero-multiply trit dot product for pre-decoded trits. |
| `packed_trit_dot` | fn | Packed trit dot product with per-block scale factor. |
| `packed_trit_matvec` | fn | Packed trit matrix-vector product with per-row scale factors. |
| `packed_trit_matvec_par` | fn | Row-parallel packed trit matvec. |
| `tq19_matvec` | fn | TQ1.9 matrix-vector product (sequential). |
| `tq19_matvec_batch` | fn | Batch TQ1.9 matvec with tiled accumulation. |
| `tq19_matvec_q2f` | fn | Wide-output TQ1.9 matvec (sequential) — 2·FRAC_BITS precision, one rounding. |
| `tq19_matvec_par` | fn | Row-parallel TQ1.9 matvec. |
| `tq19_matvec_q2f_par` | fn | Row-parallel wide-output TQ1.9 matvec — 2·FRAC_BITS precision, one rounding. |
| `tq19_matvec_q2f_batch_par` | fn | Row-parallel wide-output batch TQ1.9 matvec with tiled accumulation. |
| `tq19_matvec_batch_par` | fn | Row-parallel batch TQ1.9 matvec with tiled accumulation. |
| `NUM_PLANES` | const | Number of balanced-ternary digit planes in a TQ1.9 value. |
| `POW3` | const | Powers of three, 3^0 .. 3^9. |
| `SPARSE_DENSITY_PERCENT` | const | Density below which a plane is stored sparse (CSR) instead of dense packed. |
| `HYBRID_LOW_TRITS` | const | Number of low balanced-ternary digits fused into the 12-bit field. |
| `LOW_MOD` | const | 3^7 — modulus of the low part. |
| `LOW_BIAS` | const | Bias added to the balanced low remainder: biased = lo + 1093 ∈ [0, 2186]. |

**Re-exports** — signatures on [docs.rs](https://docs.rs/g_math):

| Item | Re-exported from |
| --- | --- |
| `RowScaledTQ19` | `rowscaled` |

### TQ19Matrix

Row-major TQ1.9 weight matrix.

| Method | Summary |
| --- | --- |
| `new` | Create from flat row-major data. |
| `from_fn` | Create from a generator function `f(row, col) -> i16`. |
| `rows` | Number of rows. |
| `cols` | Number of columns. |
| `data` | Raw weight data (row-major). |
| `row_slice` | Slice of weights for a single row. |
| `get` | Get weight at (row, col). |
| `matvec` | Matrix-vector product: `result[i] = sum_j(W[i][j] * x[j]) / SCALE` |
| `matvec_batch` | Batch matrix-vector: same weights applied to multiple activation vectors. |
| `matvec_fp` | Convenience: matvec returning `FixedPoint` values. |
| `matvec_par` | Row-parallel matvec. Each row computed independently via rayon. |
| `matvec_batch_par` | Row-parallel batch matvec. |
| `matvec_q2f` | Wide-output matvec: each row at 2·FRAC_BITS fractional precision with exactly one rounding. |
| `matvec_q2f_par` | Row-parallel wide-output matvec. See [`TQ19Matrix::matvec_q2f`]. |
| `matvec_q2f_batch_par` | Row-parallel wide-output batch matvec. See [`TQ19Matrix::matvec_q2f`]. |

### PlaneData

Storage for one trit plane.

| Method | Summary |
| --- | --- |
| `size_bytes` | Heap bytes used by this plane's storage. |

### PlanarTQ19

A TQ1.9 weight matrix decomposed into 10 balanced-ternary trit planes.

| Method | Summary |
| --- | --- |
| `from_tq19` | Decompose a [`TQ19Matrix`] into trit planes. |
| `to_tq19` | Reconstruct the original dense [`TQ19Matrix`] (lossless inverse). |
| `rows` | Number of rows. |
| `cols` | Number of columns. |
| `planes` | Access the plane storage (for serialization by consumers). |
| `from_parts` | Construct from raw parts (for deserialization by consumers). |
| `size_bytes` | Total heap bytes of plane storage. |
| `matvec` | Matrix-vector product: `result[i] = sum_j(W[i][j] * x[j]) / SCALE`. |
| `matvec_par` | Row-parallel matvec (rayon). One reconstruction buffer per worker. |
| `matvec_batch` | Batch matvec: same weights applied to multiple activation vectors. |
| `matvec_batch_par` | Row-parallel batch matvec: rows in parallel, reconstruction amortized |
| `matvec_q2f` | Wide-output matvec: each row at 2·FRAC_BITS precision, exactly one rounding. |
| `matvec_q2f_par` | Row-parallel wide-output matvec. See [`PlanarTQ19::matvec_q2f`]. |
| `matvec_q2f_batch_par` | Row-parallel wide-output batch matvec. See [`PlanarTQ19::matvec_q2f`]. |

### HybridTQ19

A TQ1.9 weight matrix in hybrid 12-bit + sparse-correction form.

| Method | Summary |
| --- | --- |
| `from_tq19` | Convert a dense [`TQ19Matrix`] (lossless; see `to_tq19`). |
| `to_tq19` | Reconstruct the original dense [`TQ19Matrix`] (lossless inverse). |
| `rows` | Number of rows. |
| `cols` | Number of columns. |
| `size_bytes` | Total heap bytes (packed low parts + CSR high corrections). |
| `num_high_corrections` | Number of nonzero high corrections (diagnostics). |
| `parts` | Raw parts accessor for consumer serialization: |
| `from_parts` | Construct from raw parts (consumer deserialization). The parts must |
| `matvec` | Matrix-vector product, bit-identical to [`TQ19Matrix::matvec`]. |
| `matvec_par` | Row-parallel matvec (rayon). One reconstruction buffer per worker. |
| `matvec_batch` | Batch matvec: each row reconstructed once, dotted per batch vector. |
| `matvec_batch_par` | Row-parallel batch matvec. |
| `matvec_q2f` | Wide-output matvec: each row at 2·FRAC_BITS precision, exactly one rounding. |
| `matvec_q2f_par` | Row-parallel wide-output matvec. See [`HybridTQ19::matvec_q2f`]. |
| `matvec_q2f_batch_par` | Row-parallel wide-output batch matvec. See [`HybridTQ19::matvec_q2f`]. |

## Serialization

Modules: g_math::fixed_point::imperative::serialization

| Item | Kind | Summary |
| --- | --- | --- |
| `MANIFOLD_TAG_EUCLIDEAN` | const | Manifold type tags for serialization. |
| `MANIFOLD_TAG_SPHERE` | const |  |
| `MANIFOLD_TAG_HYPERBOLIC` | const |  |
| `MANIFOLD_TAG_SPD` | const |  |
| `MANIFOLD_TAG_GRASSMANNIAN` | const |  |

### FixedPoint

| Method | Summary |
| --- | --- |
| `to_bytes` | Serialize to bytes with profile tag prefix (big-endian). |
| `from_bytes` | Deserialize from bytes with profile tag prefix. |
| `to_raw_bytes` | Serialize raw storage only (no profile tag), big-endian. |
| `from_raw_bytes` | Deserialize raw storage only (no profile tag), big-endian. |
| `profile_tag` | The profile tag byte for the current compilation profile. |
| `raw_byte_len` | Size in bytes of the raw storage (without profile tag). |
| `to_compact_bytes` | Encode a FixedPoint value in compact format. |
| `from_compact_bytes` | Decode a FixedPoint value from compact format. |

### FixedVector

| Method | Summary |
| --- | --- |
| `to_bytes` | Serialize to bytes. |
| `from_bytes` | Deserialize from bytes. |
| `to_compact_bytes` |  |
| `from_compact_bytes` |  |

### FixedMatrix

| Method | Summary |
| --- | --- |
| `to_bytes` | Serialize to bytes. |
| `from_bytes` | Deserialize from bytes. |

### Tensor

| Method | Summary |
| --- | --- |
| `to_bytes` | Serialize a tensor to bytes. |
| `from_bytes` | Deserialize a tensor from bytes. |

### ManifoldPoint

A serializable point on a manifold with its manifold type.

| Method | Summary |
| --- | --- |
| `euclidean` | Create a ManifoldPoint for Euclidean space R^n. |
| `sphere` | Create a ManifoldPoint for the n-sphere S^n. |
| `hyperbolic` | Create a ManifoldPoint for hyperbolic space H^n. |
| `spd` | Create a ManifoldPoint for the SPD manifold Sym⁺(n). |
| `grassmannian` | Create a ManifoldPoint for the Grassmannian Gr(k, n). |
| `to_bytes` | Serialize to bytes. |
| `from_bytes` | Deserialize from bytes. |

### FixedPointVisitor

| Method | Summary |
| --- | --- |
| `expecting` |  |
| `visit_bytes` |  |
| `visit_byte_buf` |  |

### VecVisitor

| Method | Summary |
| --- | --- |
| `expecting` |  |
| `visit_bytes` |  |
| `visit_byte_buf` |  |

### MatVisitor

| Method | Summary |
| --- | --- |
| `expecting` |  |
| `visit_bytes` |  |
| `visit_byte_buf` |  |

### TensorVisitor

| Method | Summary |
| --- | --- |
| `expecting` |  |
| `visit_bytes` |  |
| `visit_byte_buf` |  |

### MpVisitor

| Method | Summary |
| --- | --- |
| `expecting` |  |
| `visit_bytes` |  |
| `visit_byte_buf` |  |

