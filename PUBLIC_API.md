# Public API

Generated from source by `scripts/gen-public-api.rs`. Do not edit by hand. Regenerate with: `rustc -O scripts/gen-public-api.rs -o /tmp/gen-public-api && /tmp/gen-public-api`

This is a **pragmatic source scan** of a curated set of surface files, not a compiler-verified export list. It lists `pub` free items and impl methods with the first sentence of their doc comment. `pub use` re-exports, `#[cfg(test)]` items, and `#[doc(hidden)]` items are omitted, as are impls of std/derive traits. See the header of `scripts/gen-public-api.rs` for exact scope and limitations.

## Canonical (g_math::canonical)

Modules: g_math::canonical

| Item | Kind | Summary |
| --- | --- | --- |
| `set_gmath_mode` | fn | Set compute:output mode. |
| `reset_gmath_mode` | fn | Reset to default Auto:Auto mode |

**Re-exports**, signatures on [docs.rs](https://docs.rs/g_math):

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

**Re-exports**, signatures on [docs.rs](https://docs.rs/g_math):

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
| `raw_slice` | View a slice of values as their raw storage integers, without copying. |
| `from_raw_slice` | View a slice of raw storage integers as values, without copying. |
| `raw_slice_mut` | Mutable form of [`raw_slice`](Self::raw_slice). |
| `from_raw_slice_mut` | Mutable form of [`from_raw_slice`](Self::from_raw_slice). |
| `from_int` | Create from an integer value. |
| `try_from_int` | Create from an integer value, `Err(TierOverflow)` outside the range. |
| `to_int` | Extract the integer part (floor toward negative infinity). |
| `try_to_int` | The integer part (floor toward negative infinity), `Err(TierOverflow)` when it is outside the i32 range. |
| `abs` | Absolute value. |
| `is_negative` | Check if negative. |
| `is_zero` | Check if zero. |
| `from_f32` | Create from an f32 value, truncated toward zero to the profile's raw step. |
| `try_from_f32` | Create from an f32 value like `from_f32`, returning an error instead of panicking. |
| `from_f64` | Create from an f64 value, truncated toward zero to the profile's raw step. |
| `try_from_f64` | Create from an f64 value like `from_f64`, returning an error instead of panicking. |
| `to_f32` | Convert to f32: exact when the raw value fits 24 bits, else nearest-even. |
| `to_f64` | Convert to f64: exact when the raw value fits 53 bits, else nearest-even. |
| `from_str` | Parse from a decimal string (e.g., "3.14159", "1e-06"). |
| `try_from_str` | Parse from a string without panicking. |
| `exp` | e^x |
| `ln` | ln(x), x > 0 |
| `sqrt` | sqrt(x), x >= 0 |
| `sin` | sin(x) |
| `cos` | cos(x) |
| `sincos` | Fused (sin(x), cos(x)): single range reduction, ~2× faster than separate calls. |
| `tan` | tan(x) = sin(x) / cos(x): direct composition, no FASC |
| `atan` | atan(x) |
| `asin` | asin(x) = atan(x / sqrt(1 - x^2)), \|x\| <= 1: direct composition |
| `acos` | acos(x) = pi/2 - asin(x), \|x\| <= 1: direct composition |
| `sinh` | sinh(x) = (exp(x) - exp(-x)) / 2: direct composition |
| `cosh` | cosh(x) = (exp(x) + exp(-x)) / 2: direct composition |
| `sinhcosh` | Fused (sinh(x), cosh(x)): single shared exp-pair evaluation at compute tier. |
| `tanh` | tanh(x) = (exp(2x) - 1) / (exp(2x) + 1): direct composition |
| `asinh` | asinh(x) = ln(x + sqrt(x^2 + 1)): direct composition |
| `acosh` | acosh(x) = ln(x + sqrt(x^2 - 1)), x >= 1: direct composition |
| `atanh` | atanh(x) = ln((1+x)/(1-x)) / 2, \|x\| < 1: direct composition |
| `pow` | x^y = exp(y * ln(x)): direct composition |
| `atan2` | atan2(self=y, x): direct binary engine |
| `try_exp` | Fallible e^x: returns `Err(TierOverflow)` if result exceeds storage tier. |
| `try_ln` | Fallible ln(x): returns `Err(DomainError)` if x <= 0. |
| `inv_sqrt` | 1/√x computed at compute tier without materializing √x at storage. |
| `try_inv_sqrt` | Fallible 1/√x: `Err(DomainError)` if x <= 0, `Err(TierOverflow)` if the result does not fit the storage tier. |
| `try_sqrt` | Fallible sqrt(x): returns `Err(DomainError)` if x < 0. |
| `try_sin` | Fallible sin(x). |
| `try_cos` | Fallible cos(x). |
| `try_sincos` | Fused sin+cos: single shared range reduction at compute tier. |
| `try_sinhcosh` | Fallible fused sinh+cosh: single shared exp-pair at compute tier. |
| `sincos_wide` | Fused sin+cos for wide-range angles that exceed storage-tier integer range. |
| `sincos_wide_q64` | Fused sin+cos of a Q64.64 angle (`radians * 2^64`), rounded to storage. |
| `try_tan` | Fallible tan(x) = sin(x)/cos(x): `Err(DomainError)` if cos(x) is zero at the compute tier. |
| `try_atan` | Fallible atan(x). |
| `try_asin` | Fallible asin(x) = atan(x / sqrt(1 - x²)): `Err(DomainError)` if \|x\| > 1. |
| `try_acos` | Fallible acos(x) = π/2 - asin(x): `Err(DomainError)` if \|x\| > 1. |
| `try_sinh` | Fallible sinh(x) = (exp(x) - exp(-x)) / 2: `Err(TierOverflow)` when the result exceeds the storage tier (a ceiling exp means it already has, and the ceiling value would downscale to a plausible-wrong max). |
| `try_cosh` | Fallible cosh(x) = (exp(x) + exp(-x)) / 2: `Err(TierOverflow)` when the result exceeds the storage tier (a ceiling exp means cosh already has; the checked add alone cannot see it on wide profiles). |
| `try_tanh` | Fallible tanh(x) = (exp(2x) - 1) / (exp(2x) + 1). |
| `try_asinh` | Fallible asinh(x) = ln(x + sqrt(x² + 1)). |
| `try_acosh` | Fallible acosh(x) = ln(x + sqrt(x² - 1)): `Err(DomainError)` if x < 1. |
| `try_atanh` | Fallible atanh(x) = ln((1+x)/(1-x)) / 2: `Err(DomainError)` if \|x\| >= 1. |
| `try_pow` | Fallible x^y = exp(y * ln(x)). |
| `try_atan2` | Fallible atan2(self=y, x). |
| `try_add` | `self + rhs`, `Err(TierOverflow)` when the sum leaves storage. |
| `try_sub` | `self - rhs`, `Err(TierOverflow)` when the difference leaves storage. |
| `try_neg` | `-self`, `Err(TierOverflow)` for the storage minimum (no positive twin). |
| `try_mul` | `self * rhs`, rounded to nearest (ties toward +infinity) from the exact product; `Err(TierOverflow)` when it leaves storage. |
| `try_div` | `self / rhs`, rounded to nearest (ties toward +infinity) from the exact quotient; `Err(DivisionByZero)` or `Err(TierOverflow)`. |

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
| `length` | Length (Euclidean norm): the sum of squares and the root at the compute tier, one rounding (was `length_squared().sqrt()`: two roundings, and an overflow once the squared length left storage). |
| `length_fused` | Fused length: sqrt(Σ x_i²) entirely at compute tier. |
| `distance_to` | Fused Euclidean distance to another vector: sqrt(Σ (a_i - b_i)²) entirely at compute tier. |
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
| `as_mut_slice` | Mutable access to the underlying data slice. |
| `rotate_pairs` | Rotate the pairs `(v[i], v[i + half])`, `half = rotary_dim / 2`, by the angles whose sine and cosine are given: |

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
| `swap_rows` | Swap rows `i` and `j` in place. |

## DecimalFixed

Modules: DecimalFixed

| Item | Kind | Summary |
| --- | --- | --- |
| `DecimalRounding` | enum | How a `DecimalFixed` result that falls exactly halfway between two representable values is rounded. |
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
| `try_from_integer` | Create from integer value, `Err(TierOverflow)` when `int_val * 10^DECIMALS` leaves i128. |
| `from_parts` | Create from parts: integer and fractional parts |
| `from_decimal_str_decimal` | Parse decimal string without float conversion |
| `integer_part` | Extract integer part (truncated toward zero) |
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
| `try_add` | `self + other`, `Err(TierOverflow)` when the sum leaves i128. |
| `try_sub` | `self - other`, `Err(TierOverflow)` when the difference leaves i128. |
| `try_neg` | `-self`, `Err(TierOverflow)` for the raw minimum (no positive twin). |
| `try_mul` | `self * other` rounded half to even from the exact product, `Err(TierOverflow)` when it leaves i128. |
| `try_mul_with` | `self * other` rounded once, from the exact product, with the given tie rule. |
| `try_div` | `self / other` rounded half to even, `Err(DivisionByZero)` or `Err(TierOverflow)` (quotient beyond i128). |
| `try_div_with` | `self / other` rounded once, from the exact quotient, with the given tie rule (the same contract as [`try_div`](Self::try_div)). |
| `try_mul_div` | `self * num / den` with ONE rounding (half to even) from the exact value: the product is formed exactly in 256 bits and divided once. |
| `try_mul_div_with` | [`try_mul_div`](Self::try_mul_div) with the given tie rule. |
| `mul_div` | `self * num / den` with one rounding (half to even); see [`try_mul_div`](Self::try_mul_div). |
| `multiply_batch_decimal` | High-performance multiplication for batch operations |
| `to_f64_lossy` | Convert to f64 (lossy conversion for display/debugging) |
| `try_convert` | Convert to different decimal precision |
| `convert_with_rounding` | Force conversion to different decimal precision with rounding (half to even when digits are dropped). |
| `convert_with_rounding_mode` | Conversion to a different decimal precision; dropped digits round with the given tie rule. |
| `try_convert_with_rounding` | Conversion to a different decimal precision; dropped digits round with the given tie rule. |
| `to_binary_q256` | Convert DecimalFixed to Q256.256 binary format (I512) |
| `from_binary_q256` | Create DecimalFixed from Q256.256 binary format (I512) |
| `try_from_binary_q256` | Create DecimalFixed from Q256.256 binary format (I512), rounded half to even (the decimal rule), `Err(TierOverflow)` when the result leaves i128. |
| `exp` | `exp(x)`: native decimal exponential at full compute-tier precision. |
| `try_exp` | Fallible `exp(x)`, `Err(TierOverflow)` when the argument or the result is out of range. |
| `ln` | `ln(x)`: native decimal natural logarithm. |
| `try_ln` | Fallible `ln(x)`, `Err(DomainError)` for x <= 0. |
| `sqrt` | `sqrt(x)`: native decimal square root. |
| `try_sqrt` | Fallible `sqrt(x)`, `Err(DomainError)` for x < 0. |
| `sin` | `sin(x)`: native decimal sine. |
| `try_sin` | Fallible `sin(x)`, `Err(TierOverflow)` when the argument is outside the compute tier. |
| `cos` | `cos(x)`: native decimal cosine. |
| `try_cos` | Fallible `cos(x)`, `Err(TierOverflow)` when the argument is outside the compute tier. |
| `sincos` | `sincos(x)`: fused sine and cosine with single range reduction. |
| `try_sincos` | Fallible `sincos(x)`, `Err(TierOverflow)` when the argument is outside the compute tier. |
| `tan` | `tan(x)` = sin(x)/cos(x), composed entirely at the compute tier. |
| `try_tan` | Fallible `tan(x)`, `Err(DomainError)` when cos(x) is exactly 0 at the compute tier. |
| `atan` | `atan(x)`: native decimal arctangent. |
| `try_atan` | Fallible `atan(x)`, `Err(TierOverflow)` when the argument is outside the compute tier. |
| `atan2` | `atan2(y, x)`: native decimal two-argument arctangent. |
| `try_atan2` | Fallible `atan2(y, x)` with `self` as y, `Err(DomainError)` for atan2(0, 0). |
| `asin` | `asin(x)` = atan(x / sqrt(1 - x^2)), composed at the compute tier. |
| `try_asin` | Fallible `asin(x)`, `Err(DomainError)` for \|x\| > 1. |
| `acos` | `acos(x)` = pi/2 - asin(x), composed at the compute tier (single downscale, so no storage-tier cancellation). |
| `try_acos` | Fallible `acos(x)`, `Err(DomainError)` for \|x\| > 1. |
| `sinh` | `sinh(x)`: fused (exp(x) - exp(-x)) / 2 at the compute tier. |
| `try_sinh` | Fallible `sinh(x)`, `Err(TierOverflow)` when the argument or the result is out of range. |
| `cosh` | `cosh(x)`: fused (exp(x) + exp(-x)) / 2 at the compute tier. |
| `try_cosh` | Fallible `cosh(x)`, `Err(TierOverflow)` when the argument or the result is out of range. |
| `sinhcosh` | `sinhcosh(x)`: fused hyperbolic pair sharing one exp-pair evaluation at decimal compute tier. |
| `try_sinhcosh` | Fallible `sinhcosh(x)`, `Err(TierOverflow)` when the argument or either result is out of range. |
| `tanh` | `tanh(x)`: (exp(2x) - 1)/(exp(2x) + 1) at the compute tier, exactly ±1 for \|x\| > 2 * compute dp (where 1 - \|tanh x\| < e^(-4 dp) is below half a unit at the compute dp). |
| `try_tanh` | Fallible `tanh(x)`, `Err(TierOverflow)` only when the argument is outside the compute tier. |
| `asinh` | `asinh(x)` = sign(x) ln(\|x\| + sqrt(x^2 + 1)), composed at the compute tier. |
| `try_asinh` | Fallible `asinh(x)`, `Err(TierOverflow)` when the argument is outside the compute tier. |
| `acosh` | `acosh(x)` = ln(x + sqrt(x^2 - 1)), composed at the compute tier. |
| `try_acosh` | Fallible `acosh(x)`, `Err(DomainError)` for x < 1. |
| `atanh` | `atanh(x)` = ln((1+x)/(1-x)) / 2, composed at the compute tier. |
| `try_atanh` | Fallible `atanh(x)`, `Err(DomainError)` for \|x\| >= 1. |

## Fused operations

Modules: g_math::fixed_point::imperative::fused

| Item | Kind | Summary |
| --- | --- | --- |
| `sqrt_sum_sq` | fn | Fused sqrt(Σ x_i²): norm of a slice, entirely at compute tier. |
| `inv_sqrt_sum_sq` | fn | Fused 1/√(Σ vᵢ²): the reciprocal norm, entirely at compute tier. |
| `euclidean_distance` | fn | Fused sqrt(Σ (a_i - b_i)²): Euclidean distance, entirely at compute tier. |
| `euclidean_distance_squared` | fn | Fused Σ (a_i − b_i)²: squared Euclidean distance at compute tier, no sqrt (U1). |
| `dot` | fn | Fused Σ a_i·b_i: dot product entirely at compute tier (U1). |
| `quadratic_form` | fn | Fused quadratic form `v^T M v` with one rounding: every term is an exact triple product and the sum is narrowed once, to nearest. |
| `try_quadratic_form` | fn | Fallible twin of [`quadratic_form`]: `Err(TierOverflow)` where the result leaves the storage tier. |
| `mobius_denominator_sq` | fn | Fused squared Möbius denominator `\|1 − p̄q\|² = 1 − 2⟨p,q⟩ + \|p\|²·\|q\|²` (U1). |
| `softmax` | fn | Stable softmax entirely at compute tier. |
| `rms_norm_factor` | fn | Fused 1/sqrt(mean(x²) + eps): RMSNorm scaling factor at compute tier. |
| `rms_norm_factor_eps_wide` | fn | Fused 1/sqrt(mean(x²) + eps) with `eps` given in Q64.64 (`eps * 2^64`). |
| `silu` | fn | Fused SiLU activation: x / (1 + exp(-x)) entirely at compute tier. |
| `softmax_mix` | fn | Fused softmax + weighted value mix, entirely at compute tier: |
| `softmax_mix_values` | fn | [`softmax_mix`] without the observer weights: only the mixed output. |
| `softmax_mix_flat` | fn | [`softmax_mix`] over one contiguous value buffer: row `j` is `values_flat[j * dim..(j + 1) * dim]`. |
| `softmax_mix_flat_into` | fn | [`softmax_mix_flat`] writing into caller-provided slices: the mix into `out` and the observer weights into `weights`. |
| `softmax_mix_flat_values_into` | fn | [`softmax_mix_flat_into`] without the observer weights. |
| `softmax_mix_flat_values` | fn | [`softmax_mix_flat`] without the observer weights. |
| `sigmoid_mul` | fn | `x * sigmoid(gate)`: the sigmoid at the wide tier, the product exact, one rounding to storage (nearest, ties toward +infinity). |
| `sigmoid_mul_slice` | fn | [`sigmoid_mul`] element by element: `out[i] = x[i] * sigmoid(gate[i])`. |
| `sigmoid_mul_in_place` | fn | [`sigmoid_mul`] in place: `x[i] *= sigmoid(gate[i])`. |
| `entropy` | fn | Shannon entropy `-sum(w * ln(w))` in nats, the terms accumulated at the wide tier and the sum rounded to storage once. |
| `dot_many` | fn | One query against many keys stored in one contiguous buffer: element `k` of the result is `dot(query, keys_flat[k * dim..(k + 1) * dim])`, each accumulated at the compute tier and rounded to storage once, exactly as [`dot`] and `FixedVector::dot` round. |
| `dot_many_into` | fn | [`dot_many`] writing into a caller-provided slice: `out[k]` is the dot of the query with key `k`. |
| `rms_norm` | fn | RMS normalisation with a learned scale: `out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)`, each element rounded once (`eps` in Q64.64). |
| `rms_norm_in_place` | fn | [`rms_norm`] in place. |

## Certified intervals

Modules: Interval (re-exported at g_math::fixed_point), DecimalInterval (re-exported at g_math::fixed_point)

### Interval

A certified enclosure `[lo, hi]` of a real value, `lo <= hi`.

| Method | Summary |
| --- | --- |
| `point` | The degenerate interval `[x, x]`. |
| `new` | `[lo, hi]`. |
| `try_new` | `[lo, hi]`, or `Err(InvalidInput)` if `lo > hi`. |
| `lo` | Lower endpoint. |
| `hi` | Upper endpoint. |
| `width` | `hi - lo`. |
| `is_point` | `lo == hi`. |
| `contains` | `lo <= x <= hi`. |
| `contains_zero` | `lo <= 0 <= hi`. |
| `is_certainly_positive` | `lo > 0`: every value in the interval is positive. |
| `is_certainly_negative` | `hi < 0`: every value in the interval is negative. |
| `try_add` | `[a.lo + b.lo, a.hi + b.hi]`. |
| `try_sub` | `[a.lo - b.hi, a.hi - b.lo]`. |
| `try_neg` | `[-hi, -lo]`. |
| `try_mul` | Product: exact corner products at the compute tier, narrowed once. |
| `try_div` | Quotient; `Err(DivisionByZero)` if the divisor interval contains zero. |
| `try_sqrt` | Certified square root. |
| `try_dot` | Certified dot product of two point vectors, with one narrowing. |
| `try_dot_intervals` | Certified dot product of two interval vectors, with one narrowing. |
| `try_quadratic_form` | Certified quadratic form `v^T M v` for point inputs, with one narrowing. |
| `sqrt` | Certified square root; panics on a negative lower endpoint or overflow. |
| `dot` | Certified dot product; panics on overflow. |
| `quadratic_form` | Certified quadratic form; panics on overflow. |
| `dot_intervals` | Certified dot product of interval vectors; panics on overflow. |

### DecimalInterval

A certified enclosure `[lo, hi]` of a real value in the decimal domain, `lo <= hi`.

| Method | Summary |
| --- | --- |
| `point` | The degenerate interval `[x, x]`. |
| `new` | `[lo, hi]`. |
| `try_new` | `[lo, hi]`, or `Err(InvalidInput)` if `lo > hi`. |
| `lo` | Lower endpoint. |
| `hi` | Upper endpoint. |
| `width` | `hi - lo`. |
| `is_point` | `lo == hi`. |
| `contains` | `lo <= x <= hi`. |
| `contains_zero` | `lo <= 0 <= hi`. |
| `is_certainly_positive` | `lo > 0`: every value in the interval is positive. |
| `is_certainly_negative` | `hi < 0`: every value in the interval is negative. |
| `try_add` | `[a.lo + b.lo, a.hi + b.hi]`. |
| `try_sub` | `[a.lo - b.hi, a.hi - b.lo]`. |
| `try_neg` | `[-hi, -lo]`. |
| `try_mul` | Product: exact corner products at `2 * DECIMALS` places, narrowed once. |
| `try_div` | Quotient; `Err(DivisionByZero)` if the divisor interval contains zero. |
| `try_sqrt` | Certified square root. |
| `try_dot` | Certified dot product of two point vectors, with one narrowing. |
| `sqrt` | Certified square root; panics on a negative lower endpoint or overflow. |
| `dot` | Certified dot product; panics on overflow. |

## Certified and exact predicates

Modules: g_math::fixed_point::imperative::predicates

| Item | Kind | Summary |
| --- | --- | --- |
| `orient2d` | fn | Orientation of the triangle `a b c`: `Positive` if counterclockwise, `Negative` if clockwise, `Zero` if the three points are exactly collinear. |
| `orient3d` | fn | Orientation of the tetrahedron `a b c d`: `Positive` if `d` lies below the plane of `a b c` (the triangle seen counterclockwise from above), `Negative` if above, `Zero` if the four points are exactly coplanar. |
| `incircle` | fn | Whether `d` lies inside the circle through `a b c`: `Positive` if inside when `a b c` are counterclockwise (the sign flips with their orientation), `Negative` if outside, `Zero` if exactly on the circle. |
| `insphere` | fn | Whether `e` lies inside the sphere through `a b c d`: `Positive` if inside when `orient3d(a, b, c, d)` is `Positive` (the sign flips with their orientation), `Negative` if outside, `Zero` if exactly on the sphere. |
| `pd_verdict` | fn | Certified positive-definiteness verdict via interval Cholesky. |

### Sign

The sign of an exactly evaluated determinant.

| Method | Summary |
| --- | --- |
| `flip` | The sign of the negated quantity. |

### PdVerdict

The outcome of a certified positive-definiteness test.

| Method | Summary |
| --- | --- |
| `is_proven_positive_definite` | `true` only for [`PdVerdict::PositiveDefinite`]. |

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
| `matrix_sqrt` | fn | Matrix square root: A^{1/2} via Denman-Beavers iteration. |
| `matrix_log` | fn | Matrix logarithm: log(A) via inverse scaling-and-squaring. |
| `matrix_pow` | fn | Matrix power: A^p = exp(p * log(A)) for real scalar p. |

### LUDecomposition

Result of LU decomposition with partial pivoting: PA = LU.

| Method | Summary |
| --- | --- |
| `solve` | Solve Ax = b: forward then back substitution on the compute-tier factors, every sum exact, the solution rounded once. |
| `determinant` | Determinant: det(A) = (-1)^num_swaps * product(U diagonal), formed at the compute tier from the compute-tier factor and rounded once. |
| `refine` | Iterative refinement: the residual `b - Ax` exact at the compute tier, the correction solved at the compute tier, `x + dx` rounded once. |
| `inverse` | Compute A^{-1} by solving AX = I column by column at the compute tier, every entry rounded once. |

### QRDecomposition

Result of QR decomposition via Householder reflections: A = QR.

| Method | Summary |
| --- | --- |
| `solve` | Solve Ax = b via R^{-1} Q^T b on the compute-tier factors: `Q^T b` exact sums, back substitution at the compute tier, the solution rounded once (before 0.6.4 on the storage factors: up to 27 units on well-conditioned 4 x 4 systems). |

### CholeskyDecomposition

Result of Cholesky decomposition: A = LL^T.

| Method | Summary |
| --- | --- |
| `solve` | Solve Ax = b: forward (Ly = b), then back (L^T x = y), on the compute-tier factor with exact sums; the solution rounded once. |
| `determinant` | Determinant: det(A) = product(L[i][i])^2, formed at the compute tier and rounded once (before 0.6.4 a chain of storage products). |

### ComputeMatrix

| Method | Summary |
| --- | --- |
| `identity` |  |
| `dim` |  |
| `copy` |  |
| `add` |  |
| `sub` |  |
| `halve` |  |
| `mat_mul` |  |
| `scalar_mul` |  |
| `fraction` |  |
| `is_zero` |  |
| `norm_1` |  |
| `frobenius_below` |  |
| `step_below` |  |
| `solve` |  |
| `inverse` |  |

### WideMatrix

| Method | Summary |
| --- | --- |
| `identity` |  |
| `dim` |  |
| `copy` |  |
| `add` |  |
| `sub` |  |
| `halve` |  |
| `mat_mul` |  |
| `scalar_mul` |  |
| `fraction` |  |
| `is_zero` |  |
| `norm_1` |  |
| `frobenius_below` |  |
| `step_below` |  |
| `solve` |  |
| `inverse` |  |

## Geometry

Modules: g_math::fixed_point::imperative::manifold, g_math::fixed_point::imperative::lie_group, g_math::fixed_point::imperative::curvature, g_math::fixed_point::imperative::projective, g_math::fixed_point::imperative::fiber_bundle

| Item | Kind | Summary |
| --- | --- | --- |
| `Manifold` | trait | A Riemannian manifold with fixed-point arithmetic. |
| `LieGroup` | trait | A Lie group with fixed-point arithmetic. |
| `differentiation_step` | fn | Optimal step size for central differences: h = 2^(-FRAC_BITS/3). |
| `MetricProvider` | trait | A metric function: given a point (as FixedVector of coordinates), returns the metric tensor g_ij as an n×n FixedMatrix. |
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
| `BundleConnection` | trait | A connection on a fiber bundle: specifies how fibers relate along the base. |
| `apply_representation` | fn | Apply a transition function (group element) to a fiber element via matrix-vector multiplication (the fundamental representation). |
| `change_chart` | fn | Change of chart for a section: ξ_β = g_{αβ} · ξ_α. |
| `vector_bundle_curvature` | fn | Compute the curvature 2-form of a vector bundle connection at a point. |

### EuclideanSpace

Flat Euclidean space R^n.

| Method | Summary |
| --- | --- |
| `dimension` |  |
| `inner_product` |  |
| `norm` | \|\|v\|\|: exact sum of squares and root at the compute tier, one rounding. |
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
| `norm` | \|\|v\|\|: exact sum of squares and root at the compute tier, one rounding. |
| `exp_map` | `cos(theta) p + sin(theta) v / theta` with `theta = \|v\|`: the root, sin and cos, the quotient and the sum at the compute tier, one rounding per component. |
| `log_map` | `theta w / \|w\|` from [`Sphere::geodesic`]: angle, direction and scaling at the compute tier, one rounding per component. |
| `distance` | `atan2(\|p x q\|, p.q)` at the compute tier, one rounding. |
| `parallel_transport` | `v - <v, p+q> / (1 + <p,q>) (p + q)`: the products exact, the coefficient and each component formed at the compute tier with one rounding to storage. |

### HyperbolicSpace

Hyperbolic space H^n in the hyperboloid model.

| Method | Summary |
| --- | --- |
| `dimension` |  |
| `inner_product` |  |
| `norm` |  |
| `exp_map` | `cosh(theta) p + sinh(theta) v / theta` with `theta = \|v\|_L`: the root, the shared exp pair, the quotient and the sum at the compute tier, one rounding per component. |
| `log_map` | `d w / \|w\|_L` from [`HyperbolicSpace::geodesic`]: distance, direction and scaling at the compute tier, one rounding per component. |
| `distance` | `acosh(-<p,q> / sqrt(<p,p> <q,q>))` at the compute tier, one rounding. |
| `parallel_transport` | `v + <v,u>_L (sinh(d) p + (cosh(d) - 1) u)` with `u = w / \|w\|_L` the unit direction and `d` the distance of [`HyperbolicSpace::geodesic`]: everything at the compute tier, one rounding per component. |

### SPDManifold

The manifold of n×n symmetric positive-definite matrices.

| Method | Summary |
| --- | --- |
| `dimension` |  |
| `inner_product` | `tr(P⁻¹ U P⁻¹ V)`: the inverse and both products at the compute tier, the trace exact, one rounding. |
| `norm` | `sqrt(tr((P⁻¹ V)^2))` at the compute tier, one rounding. |
| `exp_map` | `P^1/2 expm(P^-1/2 V P^-1/2) P^1/2`: square root, inverse, products and exponential all at the compute tier, one rounding per entry. |
| `log_map` | `P^1/2 logm(P^-1/2 Q P^-1/2) P^1/2`, all at the compute tier, one rounding per entry. |
| `distance` | `\|\|logm(P^-1/2 Q P^-1/2)\|\|_F` (= `\|\|log_P(Q)\|\|_P`): the matrix log at the compute tier, its sum of squares exact, the root at the compute tier, one rounding. |
| `parallel_transport` | `E V Eᵀ` with `E = (Q P⁻¹)^1/2`: inverse, product, square root and transport at the compute tier, one rounding per entry. |

### Grassmannian

The Grassmann manifold Gr(k, n): k-dimensional subspaces of R^n.

| Method | Summary |
| --- | --- |
| `dimension` |  |
| `inner_product` |  |
| `norm` | \|\|V\|\|_F: exact sum of squares and root at the compute tier, one rounding. |
| `exp_map` | `Q V cos(Σ) Vᵀ + Δ V sinc(Σ) Vᵀ` for the thin SVD `Δ = U Σ Vᵀ`, with `Δ V` (= `U Σ`) and `σ_i = \|(Δ V)_i\|` formed at the compute tier from the exact tangent, sin and cos at the compute tier, and one rounding per entry. |
| `log_map` | `U diag(theta) Aᵀ` from [`Grassmannian::log_parts`], one rounding per entry. |
| `distance` | `sqrt(sum theta_i^2)` of the principal angles of [`Grassmannian::log_parts`]: sum exact, root at the compute tier, one rounding. |
| `parallel_transport` | Transport along the geodesic with `log_Q1(Q2) = U Θ Aᵀ`: `PT(Δ) = Δ - Q1 A sin(Θ) Uᵀ Δ + U (cos(Θ) - I) Uᵀ Δ`, the factors from [`Grassmannian::log_parts`] and the products at the compute tier, one rounding per entry. |

### StiefelManifold

The Stiefel manifold St(k, n): orthonormal k-frames in R^n.

| Method | Summary |
| --- | --- |
| `dimension` |  |
| `inner_product` |  |
| `norm` | \|\|V\|\|_F: exact sum of squares and root at the compute tier, one rounding. |
| `exp_map` |  |
| `log_map` | First-order log `Δ - Q sym(QᵀΔ)` with `Δ = Q' - Q`, at the compute tier, one rounding per entry. |
| `distance` | `\|\|log_Q(Q')\|\|_F`: the log at the compute tier, its sum of squares exact, the root at the compute tier, one rounding. |
| `parallel_transport` |  |

### ProductManifold

Product manifold M₁ × M₂: the Cartesian product of two manifolds.

| Method | Summary |
| --- | --- |
| `new` | Create a product manifold M₁ × M₂. |
| `dimension` |  |
| `inner_product` |  |
| `norm` | `sqrt(\|\|v1\|\|^2 + \|\|v2\|\|^2)` of the component norms: squares exact, sum and root at the compute tier, one rounding. |
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
| `exp_map` | log(exp(base) exp(tangent)) with both exps, the product and the log at the compute tier; each component rounded once. |
| `log_map` | log(exp(base)ᵀ exp(target)) entirely at the compute tier, rounded once. |
| `distance` | \|log_map\| from the compute-tier log, one rounding. |
| `parallel_transport` | vee(R_half [v]× R_halfᵀ) = R_half v with R_half = exp(log_map / 2), the whole chain at the compute tier and each component rounded once. |
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
| `bracket` | [ω₁, ω₂] = ω₁ × ω₂, each component one rounding of its exact value. |
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
| `exp_map` | log(exp(base) exp(tangent)) entirely at the compute tier, rounded once. |
| `log_map` | log(exp(base)⁻¹ exp(target)) entirely at the compute tier (the inverse [[Rᵀ, -Rᵀt], [0, 1]] included), rounded once. |
| `distance` | \|log_map\| from the compute-tier log, one rounding. |
| `parallel_transport` |  |
| `algebra_dim` |  |
| `matrix_dim` |  |
| `identity_element` |  |
| `compose` |  |
| `group_inverse` | [[Rᵀ, -Rᵀt], [0, 1]]: each translation entry one rounding of the exact negated sum. |
| `lie_exp` |  |
| `lie_log` |  |
| `hat` |  |
| `vee` |  |
| `adjoint` | Ad_g(ω, v) = (Rω, Rv + t × Rω): Rω and Rv exact at the compute tier, t × Rω exact at 3F, each component rounded once. |
| `bracket` | [(ω₁, v₁), (ω₂, v₂)] = (ω₁ × ω₂, ω₁ × v₂ - ω₂ × v₁), each component one rounding of its exact value. |
| `act` |  |

### SOn

SO(n): General special orthogonal group.

| Method | Summary |
| --- | --- |
| `hat_son` | hat: vector → skew-symmetric n×n matrix. |
| `vee_son` | vee: skew-symmetric n×n matrix → vector. |
| `dimension` |  |
| `inner_product` |  |
| `exp_map` | log(exp(base) exp(tangent)): both matrix exps, the product, the matrix log and its skew part at the compute tier, rounded once. |
| `log_map` | log(exp(base)ᵀ exp(target)) at the compute tier, rounded once. |
| `distance` | \|log_map\| from the compute-tier log, one rounding. |
| `parallel_transport` |  |
| `algebra_dim` |  |
| `matrix_dim` |  |
| `identity_element` |  |
| `compose` |  |
| `group_inverse` |  |
| `lie_exp` |  |
| `lie_log` | vee((log g - (log g)ᵀ) / 2) with the matrix log and the skew part at the compute tier, each component rounded once. |
| `hat` |  |
| `vee` |  |
| `adjoint` | vee(g ξ^ gᵀ), each component one rounding of its exact value. |
| `bracket` | vee(AB - BA), each component one rounding of its exact value. |
| `act` |  |

### GLn

GL(n): General linear group of invertible n×n matrices.

| Method | Summary |
| --- | --- |
| `hat_gln` | hat: n²-vector → n×n matrix (row-major). |
| `vee_gln` | vee: n×n matrix → n²-vector (row-major). |
| `dimension` |  |
| `inner_product` |  |
| `exp_map` | log(exp(base) exp(tangent)) with the exps, the product and the log at the compute tier, rounded once. |
| `log_map` | log(exp(base)⁻¹ exp(target)) with a compute-tier LU inverse (the storage LU inverse carried O(κ) units into the log before 0.6.4), rounded once. |
| `distance` | \|log_map\| from the compute-tier log, one rounding. |
| `parallel_transport` |  |
| `algebra_dim` |  |
| `matrix_dim` |  |
| `identity_element` |  |
| `compose` |  |
| `group_inverse` | g⁻¹ from a compute-tier LU, each entry rounded once. |
| `lie_exp` |  |
| `lie_log` |  |
| `hat` |  |
| `vee` |  |
| `adjoint` | vee(g ξ^ g⁻¹) with the inverse and both products at the compute tier, each component rounded once. |
| `bracket` | vee(AB - BA), each component one rounding of its exact value. |
| `act` |  |

### On

O(n): Orthogonal group: matrices with QᵀQ = I, det = ±1.

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

SL(n): Special linear group: n×n matrices with det = 1.

| Method | Summary |
| --- | --- |
| `hat_sln` | hat: (n²-1)-vector → traceless n×n matrix. |
| `vee_sln` | vee: traceless n×n matrix → (n²-1)-vector. |
| `project_traceless` | Project matrix onto sl(n) by removing trace: A - (tr(A)/n)·I. |
| `dimension` |  |
| `inner_product` |  |
| `exp_map` | log(exp(base) exp(tangent)), projected traceless, at the compute tier; rounded once. |
| `log_map` | log(exp(base)⁻¹ exp(target)) with a compute-tier LU inverse, projected traceless at the compute tier; rounded once. |
| `distance` | \|log_map\| from the compute-tier log, one rounding. |
| `parallel_transport` |  |
| `algebra_dim` |  |
| `matrix_dim` |  |
| `identity_element` |  |
| `compose` |  |
| `group_inverse` | g⁻¹ from a compute-tier LU, each entry rounded once. |
| `lie_exp` |  |
| `lie_log` | vee(project_traceless(log g)) with the matrix log and the projection at the compute tier, each component rounded once. |
| `hat` |  |
| `vee` |  |
| `adjoint` | vee(project_traceless(g ξ^ g⁻¹)) with the inverse, both products and the projection at the compute tier, each component rounded once. |
| `bracket` | vee(AB - BA), each component one rounding of its exact value. |
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
| `metric` | g = r² [[1, 0], [0, sin²θ]], each entry formed at the compute tier and rounded once (r² sin²θ was three storage roundings before 0.6.4). |
| `christoffel_closed_form` | Exact Christoffel symbols for S² in (θ, φ) coordinates. |
| `scalar_curvature_closed_form` | Exact scalar curvature for S²: R = 2/r², one rounding. |

### HyperbolicMetric

Hyperbolic space H^2 metric in the upper half-plane model.

| Method | Summary |
| --- | --- |
| `dimension` |  |
| `metric` | g = (1/y²) I, 1/y² formed at the compute tier and rounded once. |
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
| `determinant` | Determinant: ad - bc. |
| `to_matrix` | Convert to the corresponding 2×2 projective matrix [[a,b],[c,d]]. |

### MoebiusComplex

A Möbius transformation with complex coefficients: z ↦ (az+b)/(cz+d) where a,b,c,d,z are complex numbers represented as (real, imag) pairs.

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

CP (Canonical Polyadic) decomposition: T ≈ Σ_r λ_r a₁_r ∘ a₂_r ∘ ...

| Method | Summary |
| --- | --- |
| `reconstruct` | Reconstruct the full tensor from CP factors. |

## Balanced ternary

Modules: g_math::fixed_point::domains::balanced_ternary

| Item | Kind | Summary |
| --- | --- | --- |
| `pack_trits` | fn | Pack trits into bytes, 5 trits per byte using base-3 encoding. |
| `unpack_trits` | fn | Unpack bytes back to trits. |

**Re-exports**, signatures on [docs.rs](https://docs.rs/g_math):

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
| `SCALE_TQ10_10` | `ternary_types` |
| `SCALE_TQ20_20` | `ternary_types` |
| `SCALE_TQ40_40` | `ternary_types` |
| `add_ternary_tq10_10` | `ternary_addition` |
| `add_ternary_tq20_20` | `ternary_addition` |
| `add_ternary_tq40_40` | `ternary_addition` |
| `add_ternary_tq80_80` | `ternary_addition` |
| `subtract_ternary_tq10_10` | `ternary_addition` |
| `subtract_ternary_tq20_20` | `ternary_addition` |
| `subtract_ternary_tq40_40` | `ternary_addition` |
| `subtract_ternary_tq80_80` | `ternary_addition` |
| `add_ternary_tq80_80_checked` | `ternary_addition` |
| `subtract_ternary_tq80_80_checked` | `ternary_addition` |
| `add_ternary_tq160_160` | `ternary_addition` |
| `subtract_ternary_tq160_160` | `ternary_addition` |
| `add_ternary_tq320_320` | `ternary_addition` |
| `subtract_ternary_tq320_320` | `ternary_addition` |
| `multiply_ternary_tq10_10` | `ternary_multiplication` |
| `multiply_ternary_tq20_20` | `ternary_multiplication` |
| `multiply_ternary_tq40_40` | `ternary_multiplication` |
| `multiply_ternary_tq80_80` | `ternary_multiplication` |
| `multiply_ternary_tq80_80_checked` | `ternary_multiplication` |
| `multiply_ternary_tq160_160` | `ternary_multiplication` |
| `multiply_ternary_tq320_320` | `ternary_multiplication` |
| `divide_ternary_tq10_10` | `ternary_division` |
| `divide_ternary_tq20_20` | `ternary_division` |
| `divide_ternary_tq40_40` | `ternary_division` |
| `divide_ternary_tq80_80` | `ternary_division` |
| `divide_ternary_tq80_80_checked` | `ternary_division` |
| `divide_ternary_tq160_160` | `ternary_division` |
| `divide_ternary_tq320_320` | `ternary_division` |
| `negate_ternary_tq10_10` | `ternary_negation` |
| `negate_ternary_tq20_20` | `ternary_negation` |
| `negate_ternary_tq40_40` | `ternary_negation` |
| `negate_ternary_tq80_80` | `ternary_negation` |
| `negate_ternary_tq160_160` | `ternary_negation` |
| `negate_ternary_tq320_320` | `ternary_negation` |
| `OverflowDetected` | `crate::fixed_point::core_types::errors` |
| `TritQ1_9` | `trit_q1_9` |

### Trit

A balanced ternary digit: -1, 0, or +1

| Method | Summary |
| --- | --- |
| `from_i8` | Convert from i8. |
| `as_i8` | Convert to i8 |

## TQ1.9 inference

Modules: g_math::tq19, g_math::tq19::bits, g_math::tq19::quantize _(feature: inference)_

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
| `tq19_matvec_q2f` | fn | Wide-output TQ1.9 matvec (sequential): 2·FRAC_BITS precision, one rounding. |
| `tq19_matvec_par` | fn | Row-parallel TQ1.9 matvec. |
| `tq19_matvec_q2f_par` | fn | Row-parallel wide-output TQ1.9 matvec: 2·FRAC_BITS precision, one rounding. |
| `tq19_matvec_q2f_batch_par` | fn | Row-parallel wide-output batch TQ1.9 matvec with tiled accumulation. |
| `tq19_matvec_batch_par` | fn | Parallelizes across rows via rayon. |
| `NUM_PLANES` | const | Number of balanced-ternary digit planes in a TQ1.9 value. |
| `POW3` | const | Powers of three, 3^0 .. |
| `SPARSE_DENSITY_PERCENT` | const | Density below which a plane is stored sparse (CSR) instead of dense packed. |
| `HYBRID_LOW_TRITS` | const | Number of low balanced-ternary digits fused into the 12-bit field. |
| `LOW_MOD` | const | 3^7: modulus of the low part. |
| `LOW_BIAS` | const | Bias added to the balanced low remainder: biased = lo + 1093 ∈ [0, 2186]. |
| `TQ5_MAX` | const | Largest code: `(3^5 - 1) / 2`. |
| `HalfKind` | enum | Which 16-bit float format a bit pattern is in. |
| `decompose` | fn | `\|v\| = m * 2^e` with `m` the integer mantissa (implicit bit included) and the sign separately: `(negative, m, e)`. |
| `to_raw` | fn | `trunc(v * 2^frac_bits)` toward zero as an i128. |
| `to_q64_raw` | fn | The value at Q64.64: `trunc(v * 2^64)`. |
| `to_storage_raw` | fn | The value at the build's storage format: `trunc(v * 2^FRAC_BITS)` toward zero. |
| `to_fixed` | fn | The value as a `FixedPoint`, truncated toward zero; see [`to_storage_raw`]. |
| `to_tq19_raw` | fn | TQ1.9 raw value: `round(v * 3^9)`, half away from zero, unclamped (the caller clamps to `MAX_RAW` / `MIN_RAW`). |
| `quantize_tq19` | fn | Quantise to a `TQ19Matrix` with the global TQ1.9 scale: each element is `round(v * 3^9)`, half away from zero. |
| `quantize_tq19_rowscaled` | fn | Quantise to a row-scaled TQ1.9 matrix by exact rational rounding: each row quantises against its own largest magnitude, `q = round(v * MAX_RAW / max\|w\|)` (half away from zero), and carries the scale `round(max\|w\| * 3^9 * 2^32 / MAX_RAW)` in unsigned Q32.32. |
| `quantize_tq5_rowscaled` | fn | Quantise to five trits with a per-row scale by exact rational rounding: `s_r = max\|w\|_r / 121`, `code = round(w / s_r)` (half away from zero), `scale_q32 = round(s_r * 2^32)`. |

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
| `matvec_par` | Row-parallel matvec. |
| `matvec_batch_par` | Row-parallel batch matvec. |
| `matvec_batch_par_into` | [`matvec_batch_par`](Self::matvec_batch_par) writing into a caller-provided buffer, flat and batch-major: `out[b * rows + r]` is row `r` of the result for `batch[b]`. |
| `matvec_q2f` | Wide-output matvec: each row at 2·FRAC_BITS fractional precision with exactly one rounding. |
| `matvec_q2f_par` | Row-parallel wide-output matvec. |
| `matvec_q2f_batch_par` | Row-parallel wide-output batch matvec. |

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
| `matvec_par` | Row-parallel matvec (rayon). |
| `matvec_batch` | Batch matvec: same weights applied to multiple activation vectors. |
| `matvec_batch_par` | Row-parallel batch matvec: rows in parallel, reconstruction amortized across the batch within each row. |
| `matvec_batch_par_into` | [`matvec_batch_par`](Self::matvec_batch_par) writing into a caller-provided buffer, flat and batch-major: `out[b * rows + r]` is row `r` of the result for `batch[b]`. |
| `matvec_q2f` | Wide-output matvec: each row at 2·FRAC_BITS precision, exactly one rounding. |
| `matvec_q2f_par` | Row-parallel wide-output matvec. |
| `matvec_q2f_batch_par` | Row-parallel wide-output batch matvec. |

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
| `parts` | Raw parts accessor for consumer serialization: `(packed, hi_row_ptr, hi_cols, hi_vals)`. |
| `from_parts` | Construct from raw parts (consumer deserialization). |
| `matvec` | Matrix-vector product, bit-identical to [`TQ19Matrix::matvec`]. |
| `matvec_par` | Row-parallel matvec (rayon). |
| `matvec_batch` | Batch matvec: each row reconstructed once, dotted per batch vector. |
| `matvec_batch_par` | Row-parallel batch matvec. |
| `matvec_batch_par_into` | [`matvec_batch_par`](Self::matvec_batch_par) writing into a caller-provided buffer, flat and batch-major: `out[b * rows + r]` is row `r` of the result for `batch[b]`. |
| `matvec_q2f` | Wide-output matvec: each row at 2·FRAC_BITS precision, exactly one rounding. |
| `matvec_q2f_par` | Row-parallel wide-output matvec. |
| `matvec_q2f_batch_par` | Row-parallel wide-output batch matvec. |

### RowScaledTQ19

TQ1.9 matrix with one quantization scale per row.

| Method | Summary |
| --- | --- |
| `from_parts` | Construct from parts. |
| `rows` |  |
| `cols` |  |
| `data` |  |
| `scales_q32` |  |
| `size_bytes` | Bytes of weight + scale storage (2 B/weight + 8 B/row). |
| `matvec` | Row-scaled matvec: `out[r] = tq19_dot(row_r, x) × s_rel[r]`. |
| `matvec_par` | Row-parallel matvec. |
| `matvec_q2f` | Wide-output row-scaled matvec: each row at 2·FRAC_BITS precision. |
| `matvec_q2f_par` | Row-parallel wide-output matvec. |
| `matvec_q2f_batch_par` | Row-parallel wide-output batch matvec. |
| `matvec_batch_par_into` | [`matvec_batch_par`](Self::matvec_batch_par) writing into a caller-provided buffer, flat and batch-major: `out[b * rows + r]` is row `r` of the result for `batch[b]`. |
| `matvec_batch_par` | Row-parallel batch matvec (row weights stay in cache across the batch). |

### RowScaledTQ5

A row-major matrix of five-trit codes with a per-row scale.

| Method | Summary |
| --- | --- |
| `from_parts` | Construct from parts; `data.len() == rows * cols`, `scales_q32.len() == rows`, every code in `[-121, 121]` (the kernels' overflow bounds rest on it). |
| `rows` |  |
| `cols` |  |
| `data` |  |
| `scales_q32` |  |
| `size_bytes` | Bytes of weight + scale storage (1 B/weight + 8 B/row). |
| `matvec` | Matvec: `out[r] = floor(sum(code * x) * s_r / 2^32)`. |
| `matvec_par` | Row-parallel [`matvec`](Self::matvec): the same results. |
| `matvec_batch_par` | Batched matvec. |
| `matvec_batch_par_into` | [`matvec_batch_par`](Self::matvec_batch_par) writing into a caller-provided buffer, flat and batch-major: `out[b * rows + r]` is row `r` of the result for `batch[b]`. |
| `matvec_q2f` | Wide-output matvec at `2 * FRAC_BITS` fractional bits: `floor(sum(code * x) * s_r / 2^(32 - FRAC_BITS))`, one rounding. |
| `matvec_q2f_par` | Row-parallel [`matvec_q2f`](Self::matvec_q2f): the same results. |
| `write_to` | Serialize: `rows (u32) \| cols (u32) \| codes (i8 each) \| scales (u64 each)`, little-endian. |
| `read_from` | Deserialize (inverse of [`write_to`](Self::write_to)). |

### WeightBits

A matrix of weight bit patterns as read from a file: the float-free form projections are quantised from and embeddings are decoded from.

| Method | Summary |
| --- | --- |
| `from_le_bytes` | From the file's little-endian bytes. |
| `storage_raw` | Storage raw of element `i`; see [`to_storage_raw`]. |
| `tq19_raw` | TQ1.9 raw of element `i`; see [`to_tq19_raw`]. |
| `decompose` | `(negative, m, e)` of element `i`, for exact rational arithmetic. |

## Compute-tier transcendentals

Modules: g_math::compute_tier _(feature: inference)_

| Item | Kind | Summary |
| --- | --- | --- |
| `one` | fn | The value `1.0` at compute-tier scale (`1 << COMPUTE_FRAC_BITS`). |
| `ceiling` | fn | The compute tier's maximum value: the saturation ceiling for [`exp`]. |
| `from_fixed` | fn | Promote a `FixedPoint` (storage tier) to the compute tier. |
| `to_fixed` | fn | Round a compute-tier value to the nearest `FixedPoint` (single rounding). |
| `try_to_fixed` | fn | Round a compute-tier value to the nearest `FixedPoint`, or `None` on storage overflow. |
| `exp` | fn | `e^x` at the compute tier. |
| `ln` | fn | `ln(x)` at the compute tier. |
| `sqrt` | fn | `sqrt(x)` at the compute tier. |
| `sinhcosh` | fn | `(sinh(x), cosh(x))` at the compute tier from one shared exponential pair. |
| `sigmoid` | fn | `1 / (1 + e^-x)` at the compute tier. |
| `softplus` | fn | `ln(1 + e^x)` (softplus) at the compute tier. |
| `ln1p` | fn | `ln(1 + x)` at the compute tier. |

**Re-exports**, signatures on [docs.rs](https://docs.rs/g_math):

| Item | Re-exported from |
| --- | --- |
| `ComputeStorage` | `crate::fixed_point::universal::fasc::stack_evaluator` |
| `FRAC_BITS` | `crate::fixed_point::frac_config` |
| `COMPUTE_FRAC_BITS` | `crate::fixed_point::frac_config` |

## Wide tier (Q64.64)

Modules: g_math::wide

| Item | Kind | Summary |
| --- | --- | --- |
| `ONE_Q64` | const | `1.0` in Q64.64. |
| `PI_Q64` | const | `floor(π * 2^64)`. |
| `TWO_PI_Q64` | const | `floor(2π * 2^64)`. |
| `PI_HALF_Q64` | const | `floor(π/2 * 2^64)`. |
| `PI_Q32` | const | `floor(π * 2^32)`, the Q32.32 angle format of `FixedPoint::sincos_wide`. |
| `TWO_PI_Q32` | const | `floor(2π * 2^32)`. |
| `exp_q64` | fn | `e^x` for `x` in Q64.64. |
| `ln_q64` | fn | `ln(x)` for `x` in Q64.64; `None` when `x <= 0`. |
| `sin_q64` | fn | `sin(x)` for an angle `x` in radians, Q64.64. |
| `cos_q64` | fn | `cos(x)` for an angle `x` in radians, Q64.64. |
| `sincos_q64` | fn | `(sin(x), cos(x))` with one shared range reduction. |
| `try_from_str` | fn | Parse a decimal literal to a raw `i128` with `frac_bits` fractional bits. |
| `sigmoid_q64` | fn | `1 / (1 + e^-x)` for `x` in Q64.64, in `[0, 2^64]`. |
| `softplus_q64` | fn | `ln(1 + e^x)` for `x` in Q64.64. |
| `silu_q64` | fn | `x * sigmoid(x)` for `x` in Q64.64: [`sigmoid_q64`] times `x`, the product rounded once to nearest. |
| `sqrt_q64` | fn | `sqrt(x)` for `x` in Q64.64, correctly rounded: the nearest Q64.64 value to the exact root (an exact root is returned as is). |
| `sqrt_q64_to` | fn | `sqrt(x)` for `x` in Q64.64, correctly rounded to `frac_bits` fractional bits (`frac_bits <= 64`): the nearest value at that precision to the exact root, in one rounding. |
| `narrow_q64` | fn | A Q64.64 value rounded to `frac_bits` fractional bits (`frac_bits <= 64`), nearest with ties toward +infinity: the one narrowing after a wide-tier computation. |
| `mul_div_floor` | fn | `floor(a * b / d)` with the product formed exactly in 256 bits. |
| `mul_div_nearest` | fn | `a * b / d` rounded to the nearest integer, ties toward +infinity, with the product formed exactly in 256 bits. |
| `mul_div_floor_u128` | fn | `floor(a * b / d)` for unsigned operands, the product formed exactly. |
| `try_ratio_from_str` | fn | Parse a decimal literal to the exact fraction it denotes, in lowest terms: `(numerator, denominator)` with `denominator > 0`. |

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
| `profile_tag` | The profile tag byte for the current compilation profile and, on the realtime profile, its fractional split (see the module docs). |
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

