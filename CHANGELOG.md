# Changelog

All notable changes to gMath will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.6] - 2026-10-05

### Upgrading

One defect fix in the `inference` feature. Nothing else changes: every other
result is the same integer as in 0.6.5, and so is every quantiser output for
rows whose values lie within 2^100 of each other.

**Do you need to act?**

- **You call `tq19::quantize::quantize_tq5_rowscaled` or
  `quantize_tq19_rowscaled` on bfloat16 weights** → Upgrade. If a row holds a
  value near the bottom of the bfloat16 range (around 2^-124) beside ordinary
  weights, 0.6.5 either panicked or, in a release build, could return wrong
  codes for that row. Re-run any conversion made with 0.6.5 on data that may
  contain such values; conversions of rows without them are byte-identical.
- **Anything else** → Nothing to do.

### Fixed

- **Row-scaled quantisers on rows with a wide exponent gap.** `row_max`
  compared two values by shifting one mantissa left by the difference of
  their binary exponents, in 128 bits. bfloat16 exponents span 2^-133 to
  2^120, so two elements of one row can differ by far more than that; from a
  gap of a little over 100 bits the shifted product left the integer. In a debug build this
  panicked ("attempt to multiply with overflow"). In a release build the
  comparison could pick a tiny element as the row maximum, after which the
  codes of the ordinary elements were computed from a wrapped product:
  `quantize_tq5_rowscaled` then panicked in `RowScaledTQ5::from_parts`
  ("code outside [-121, 121]") or returned wrong codes that happened to be in
  range, and `quantize_tq19_rowscaled` returned wrong codes. The comparison
  now decides by the exponents alone once they differ by 11 or more (a
  mantissa is below 2^11, so the larger exponent is the larger value) and
  shifts by less than 11 bits otherwise. With a correct row maximum the
  per-element quotient needs no shift above 10 bits either, and that is now
  asserted. binary16 rows were never affected (their exponents span 40 bits).

  The 0.6.5 reference matrices had no row with a gap above 14 bits, which is
  why the gate passed. `scripts/generate_weight_bits_refs.py` now also emits
  a 32-row bfloat16 matrix from the exact-rational model: values near 2^-124
  before and after ordinary weights, a row of tiny values only, rows spanning
  2^-126 to 2^10 in both directions, subnormals beside 1.0, and 26 random
  rows over the whole range that has a 64-bit row scale. Both quantisers
  match the model on all 288 elements and 32 scales
  (`tests/weight_bits_validation.rs`); the two new tests fail on 0.6.5 in
  debug and in release.

## [0.6.5] - 2026-10-05

### Upgrading

Defect fixes and additions. No existing function changes a result in range:
every in-range value is bit-identical to 0.6.4. What changes is what happens
outside the range (values that used to wrap or come back plausible-but-wrong
are now a panic or a typed error), plus new functions.

**Using `DecimalFixed` for money?** Use 0.6.4 or later, and prefer this
release. Before 0.6.4 a value between -1 and 0 printed without its sign
(`-0.07` as `0.07`), the parser accepted a doubled sign (`"--5"` gave 5), and
the operators saturated on overflow and on division by zero instead of
failing. 0.6.5 adds the tie rule (`DecimalRounding`) and the one-rounding
`mul_div`.

**Do you need to act?**

- **You use `g_math::tq19` (feature `inference`)** → The dot products and
  matvecs (`tq19_dot`, `trit_dot`, `packed_trit_dot`, `TQ19Matrix::matvec*`,
  `PlanarTQ19` / `HybridTQ19` matvecs, `packed_trit_matvec*`) panic with
  "tq19: result exceeds storage range" where they used to wrap. Nothing to do
  unless you relied on wrapped outputs.
- **You serialize `FixedPoint` with `to_bytes` on the realtime profile at a
  `GMATH_FRAC_BITS` other than 16** → The tag byte is now `0x80 | FRAC_BITS`
  (0x8A at Q22.10) and `from_bytes` refuses the old `0x05`. Read bytes written
  by 0.6.4 or earlier with `FixedPoint::from_raw_bytes(&bytes[1..])`. Q16.16
  and every other profile write and read exactly the bytes they did before.
- **You enable more than one profile feature, or set `GMATH_PROFILE` to a
  value the crate does not know** → The build now fails with a message
  instead of picking a profile for you.
- **You pass a negative epsilon to `fused::rms_norm_factor`** → A negative
  `mean + eps` is `Err(DomainError)`; it used to be `Ok(0)`.

### Added

Decimal:

- **`DecimalRounding::{HalfEven, HalfUp}`** and the methods that take it:
  `DecimalFixed::try_mul_with`, `try_div_with`, `try_mul_div_with`,
  `convert_with_rounding_mode`, `try_convert_with_rounding`. `HalfUp` sends a
  tie away from zero (commercial rounding) and is applied to the magnitude, so
  `f(-x) == -f(x)`. The methods without a mode and the operators keep half to
  even: no existing result moves. The tie rule is a policy rather than
  arithmetic: half to even is unbiased over a series, while a counterparty
  that recomputes one field usually specifies half up. For an included-tax
  share `t * R / (100 + R)` the two rules differ on 1 amount in 12 at a 20%
  rate and 1 in 56 at 12%, and never at the other whole-percent rates tested.
- **`DecimalFixed::mul_div` / `try_mul_div`**: `self * num / den` with one
  rounding from the exact value (256-bit product, one division). `num` and
  `den` may carry a precision of their own. Staging the same computation as a
  product at six decimals, a division and a narrowing to two rounds three
  times and lands one unit away on 22 of 1,200,000 tax cases, all at rates
  with two decimal places.
- **`try_div` contract, documented**: one rounding, from the exact quotient
  `a * 10^DECIMALS / b`. It always worked this way; it is now a stated
  guarantee.

Core types:

- **`FixedPoint` is `#[repr(transparent)]`** over its raw storage integer, as
  a documented guarantee, with zero-copy slice views `FixedPoint::raw_slice`,
  `from_raw_slice`, `raw_slice_mut`, `from_raw_slice_mut`, and
  `FixedVector::as_mut_slice`.
- **`FixedVector::rotate_pairs(sin, cos, rotary_dim)`**: rotates the pairs
  `(v[i], v[i + rotary_dim / 2])`, leaving the tail untouched. Each output is
  rounded once from the exact `x0 * cos - x1 * sin` (or `x0 * sin + x1 * cos`):
  the correctly rounded value, where the operator expression rounds each
  product first and can be a unit off.

Fused operations:

- **`fused::dot_many(query, keys_flat, dim)`**: one query against many keys in
  one contiguous buffer, each result rounded as `dot` rounds.
- **`fused::softmax_mix_values`, `softmax_mix_flat`,
  `softmax_mix_flat_values`**: `softmax_mix` without the observer weights
  and/or over one contiguous value buffer. Same mixed output.
- **`fused::rms_norm` / `rms_norm_in_place`**: RMS normalisation with a
  learned scale, `x[i] * weight[i] / sqrt(mean(x^2) + eps)`, each output
  rounded once. The sum of squares is exact, the reciprocal root is taken on a
  radicand scaled into `[1, 4)` (full relative precision at any input size),
  and the exact triple product is rounded to storage once. This is not
  `x[i] * rms_norm_factor_eps_wide(..) * weight[i]`, which rounds the factor
  and both products to storage (at 10 fraction bits a factor of 0.05 alone is
  0.4% off). On realtime the work is done in 128-bit integers at Q64.64, so
  the epsilon enters exactly and the result does not depend on the compute
  tier's `2 * FRAC_BITS`. Gate: `tests/one_rounding_validation.rs`, mpmath
  references, 0 units on every profile and every gated realtime split.

Inference (feature `inference`):

- **`tq19::RowScaledTQ5`** (realtime): five-trit row-scaled matrices, one i8
  code in `[-121, 121]` per weight plus a Q32.32 scale per row. `matvec`,
  `matvec_par`, `matvec_batch_par`, `matvec_q2f`, `matvec_q2f_par`,
  `write_to` / `read_from`. AVX2 kernels on 16-bit halves of the activations
  with a scalar fallback; every path computes the same integer per row.
- **`tq19::bits`**: binary16 and bfloat16 bit patterns to fixed point by
  integer shifts (`decompose`, `to_raw` at any fractional width, `to_q64_raw`,
  `to_storage_raw`, `to_fixed`, `to_tq19_raw`, and the `WeightBits` matrix).
- **`tq19::quantize`**: `quantize_tq19`, `quantize_tq19_rowscaled`,
  `quantize_tq5_rowscaled` from `WeightBits`, by exact rational rounding.

### Changed (faster, same results)

- **`FixedVector::dot`** and the dot products inside the matrix operations, on
  the realtime profile with AVX2: for 32 or more elements the operands are
  bounded first, and when `len * max|a| * max|b| < 2^63` proves that no
  partial sum can overflow, the sum runs without the per-term check, which
  the compiler vectorises. Same integer; an input that does overflow still
  takes the checked loop and panics as before. Measured against 0.6.4: 1.3x
  faster at 32 elements, 2x at 128, 2.6x at 1024; unchanged below 32.
- **`fused::softmax_mix`** on the realtime profile at 15 or fewer fractional
  bits: the numerators accumulate in plain i64 where bounds prove it exact
  (`(e * v + 2^(F-1)) >> F` is the compute-tier product without the 128-bit
  intermediate). Measured: 2x to 4x faster; bit-identical to the 0.6.4 body,
  which is kept as the test reference.

### Fixed

- **tq19 narrowing wrapped.** The narrowing from the compute tier to storage
  in the TQ1.9 and packed-trit kernels was an `as` cast, the one place left
  where an infallible narrowing did not follow the crate's rule (panic, never
  wrap). On realtime `trit_dot(&[1, 1], &[2^30, 2^30])` returned `-2^31` and
  `tq19_dot(&[MAX_RAW; 8], &[i32::MAX; 8])` returned `-436426` for a true value
  of 25,769,367,350. All of them now panic when the result does not fit
  storage. In range they return the same integers as before.
- **`packed_trit_dot` narrowed before scaling.** The accumulated dot was
  narrowed to storage and then multiplied by the block scale, so a dot above
  the storage range wrapped even when the scaled result fit. On realtime and
  compact the scale is now applied to the accumulator itself (one exact
  product, one rounding to nearest); the result is checked.
- **AVX2 trit kernel and `i32::MIN`.** The realtime AVX2 path applied the trit
  sign in 32 bits, where negating `i32::MIN` gives `i32::MIN` back, so a row
  of eight or more elements containing that activation under a `-1` trit
  differed from the scalar path by `2^32`. The kernel now detects that lane
  and sums the row exactly; rows without it run the same instructions as
  before plus two per iteration (measured at parity with 0.6.4).
- **Realtime TQ1.9 rows longer than 65,536 columns** could overflow the i64
  accumulator with worst-case weights and activations. Such rows are now
  summed exactly and checked; shorter rows keep the unchecked SIMD loop
  (the bound `2^16 * 2^15 * 2^31 < 2^63` makes it safe).
- **`fused::rms_norm_factor` and `rms_norm_factor_eps_wide` with a negative
  radicand** returned `Ok(0)`: the square-root kernel answers a negative
  argument with a sentinel and only zero was checked. Now `Err(DomainError)`.
  A negative epsilon whose `mean + eps` is still positive stays a value.
- **Serialization tag did not record the fractional split.** Every realtime
  build wrote tag `0x05`, so `FixedPoint::from_bytes` in a Q22.10 build
  accepted Q16.16 bytes and read them at the wrong scale. See Upgrading. The
  vector, matrix, tensor and manifold-point formats never carried a tag; the
  module documentation now says so.
- **Conflicting profile selection was silent.** With two profile features
  enabled (which Cargo does whenever two crates in a build ask for different
  ones) `build.rs` picked by a fixed priority, and an unknown `GMATH_PROFILE`
  value fell through to the default profile. Both are now build errors.
  `GMATH_PROFILE` overriding a single enabled profile feature stays allowed
  and prints a build warning.
- **Top of the decimal UGOD ladder truncated.** A tier-6 product or quotient
  wider than 512 bits kept its low words, and a tier-5 quotient wider than 256
  bits kept its low 256. The product and the tier-6 quotient are now
  `Err(TierOverflow)` (the canonical layer falls back to exact rational), and
  the tier-5 quotient promotes to tier 6.
- **`Currency` and `HighPrecisionCurrency`** were listed as public types but
  reachable only through a hidden module path. They are re-exported at
  `g_math::fixed_point`.
- **`DecimalFixed::from_decimal_str_decimal`** reported `InvalidFormat` for a
  whitespace-only string; it is `EmptyString`, like the empty string.

### Documentation

- Every `Err` condition of `fused::softmax`, `fused::softmax_mix`,
  `fused::rms_norm_factor` and `fused::rms_norm_factor_eps_wide` is stated on
  the function, with the input sizes at which the realtime compute tier
  overflows for a given `GMATH_FRAC_BITS`.
- Decimal rounding, stated precisely: results are rounded half to even.
  Inside the decimal transcendentals and inside canonical decimal chains that
  stay at the compute tier, intermediate products and quotients carry guard
  digits and round half away from zero before the one half-even narrowing.
  The 0.6.4 notes said "half to even wherever it occurs", which is true of
  results and not of those intermediates. No behaviour changed.

### Validation

`tests/defects_065_validation.rs`: one regression per defect with exact
integer references, the loud path, and the in-range value. Runs on every
profile and at Q22.10 in the `fused-tq19-precision` workflow. A differential
probe linking 0.6.4 and 0.6.5 into one realtime Q22.10 binary found no
differing bit in range (4,160 dot cases against an exact integer reference,
every matrix form, 2,000 RMS-norm and softmax cases) and no slowdown in the
TQ1.9 matvec or the trit dot.

New gates for the additions: `tests/decimal_rounding_mode_validation.rs`
(1.2 million tax cases under both tie rules against an independent integer
reference, with the tie counts and the 22-case pin from an exact-rational
model; the 256-bit path), `tests/weight_bits_validation.rs` (every one of the
65,536 bit patterns of both formats against exact-rational checksums; the
quantisers against literal expectations), `tests/tq5_validation.rs` (every
kernel path against an i128 reference), `tests/layout_and_rotation_validation.rs`,
and in-crate tests holding the faster `dot` and `softmax_mix` paths equal to
the checked ones at raw extremes.

`tests/compute_tier_validation.rs` now has mpmath references for every
realtime split (`GMATH_FRAC_BITS` 2 to 30); its tables were Q16.16 only, so
the suite failed at any other split, and the split workflow never noticed
because it built without the `inference` feature. Measured at every split:
primitives 0 compute-tier units, composed forms at most 1, storage results
exact from 5 fractional bits up (at 2, 3 and 4 the second rounding can land
one unit away). The `realtime-splits` workflow now builds with `inference`.

## [0.6.4] - 2026-09-25

### Upgrading

Mostly additive (new functions, the `g_math::wide` module, `try_` twins), with
three kinds of behaviour change: an overflow that used to wrap or saturate now
panics (or returns an error from the new `try_` twin); results that were
rounded several times are now rounded once, so computed values move toward the
exact result; and several functions that returned wrong values now return the
right ones. Everyone on `^0.6` receives it on their next `cargo update`.

**Do you need to act?**

- **You rely on `FixedPoint` `+ - * /` or unary `-` wrapping on overflow** →
  They now panic ("FixedPoint: addition overflow", "division by zero", ...),
  like every other infallible path in the crate. Use `try_add`, `try_sub`,
  `try_mul`, `try_div`, `try_neg` for `Err(TierOverflow)` /
  `Err(DivisionByZero)`, or the canonical API (`gmath`/`evaluate`) for
  automatic promotion. In range the results are bit-identical to 0.6.3.
- **You rely on `DecimalFixed` operators saturating** at `i128::MAX` /
  `i128::MIN` (overflow, division by zero) → They now panic; the new
  `try_add`, `try_sub`, `try_mul`, `try_div`, `try_neg` return the error.
  `from_parts` with a fraction of `10^DECIMALS` or more panics instead of
  clamping; `integer_part`, `fractional_part`, `abs` and
  `convert_with_rounding` panic where they wrapped or saturated.
- **You call `FixedPoint::to_int` on values of 2^31 or more** (embedded and
  wider) → It wrapped; it now panics. `try_to_int` returns the error.
- **You call `FixedPoint::from_int` out of range** → It panics (it shifted
  without a check); `try_from_int` returns the error.
- **You persist or compare computed results bit for bit** → Solvers,
  factorizations, eigen-, singular-value and Schur decompositions, ODE
  integrators, geodesics, manifold and Lie-group maps, curvature, projective
  maps and tensor decompositions now round once instead of at every step, so
  their results move (toward the exact value; see "State carried at the
  compute tier"). Conversions (`from_str`, `from_f64`, `from_int`) and the
  scalar transcendentals are unchanged.
- **You use `sectional_curvature`, `StiefelManifold::distance`, or
  `Grassmannian::log_map` / `parallel_transport` with k >= 2** → They returned
  wrong values (see Fixed); results change.
- **You call `HyperbolicSpace` `distance` / `log_map` with points off the
  hyperboloid's upper sheet** → They now return `Err(DomainError)` instead of
  clamping the Minkowski product. `SPDManifold::inner_product` panics on a
  singular base point instead of silently using the identity metric.
- **You build `LUDecomposition`, `QRDecomposition` or
  `CholeskyDecomposition` with a struct literal** → They now carry private
  compute-tier factors; obtain them from `lu_decompose` / `qr_decompose` /
  `cholesky_decompose`. The public fields are unchanged.
- **You depend on how `DecimalFixed` rounds an exact tie** → Decimal rounding is half to even
  wherever it occurs (transcendental results, dp above the compute dp,
  `from_binary_q256`); it was half away from zero or truncation in places.

- **You call `FixedPoint::from_str` with plain decimals of up to 38 fraction
  digits** → Nothing changes: the results are bit-identical to 0.6.3 (checked
  against the old path on 20,000 random literals per profile). Exponent
  notation (`"1e-06"`, `"1.5e3"`) now parses instead of panicking, `.5` and
  `5.` are accepted, and literals with more than 38 fraction digits are rounded
  exactly instead of being truncated to 38 digits first.
- **You call `from_str` on realtime at `GMATH_FRAC_BITS=10` with 4 fraction
  digits and a magnitude of 214748.3648 or more** → 0.6.3 returned a wrapped,
  wrong value (`"467295.6470"` gave 37799.9); you now get the right one.
- **You pass a small RMSNorm epsilon on realtime** → `rms_norm_factor` is
  unchanged and still drops any epsilon below `2^-FRAC_BITS`; switch to
  `rms_norm_factor_eps_wide` to apply it at the compute tier.
- **You write hex or binary literals (`"0xFF"`, `"0b101"`) in `gmath()` or
  `from_str`** → They now denote their integer (`0xFF` is 255) on every path.
  0.6.3 stored the digits as raw storage bits, so `gmath("0xFF")` evaluated to
  `255 / 2^FRAC_BITS` alone, to `1 + 255/65536` inside `+ 1` on embedded, and
  its exact shadow said 255; `FixedPoint::from_str("0x10")` was raw 16.
- **You evaluate canonical decimals near or beyond your profile's range, or
  literals with more than 38 digits** → You now get the exact value, the
  nearest binary value, or `Err(TierOverflow)`, where 0.6.3 returned a wrapped
  or one-unit-off value (see Fixed).
- **You reach `exp_q64_64_native` / `ln_q64_64_native` through
  `fixed_point::domains`** → `g_math::wide::exp_q64` / `ln_q64` are the same
  functions with a documented contract (bit-identical; `ln_q64` returns
  `Option` instead of the `i128::MIN` sentinel). The hidden paths stay.

### Added

- **`fused::rms_norm_factor_eps_wide(values, eps_q64)`**: RMSNorm factor with
  epsilon given in Q64.64 and added at the compute tier (`2 x FRAC_BITS`
  fractional bits), rounded once to nearest, ties toward +infinity; exact on
  compact and wider. `rms_norm_factor` took epsilon at the storage tier, where
  `1e-5` and `1e-6` are zero at realtime Q22.10, so an all-zero input was
  `Err(DivisionByZero)` rather than `1/sqrt(eps)`. The realtime compute tier
  still bounds the result: at Q22.10, `1e-5` becomes `10 / 2^20` and an
  all-zero input gives 323.83 (exact `1/sqrt(1e-5)`: 316.23); at Q16.16 it
  gives 316.23.
- **`FixedPoint::try_from_str`**, and exponent notation in `from_str`: decimal
  literals are converted exactly with integer arithmetic (any digit count) and
  rounded once to nearest, ties toward +infinity. Hex, binary, ternary,
  fraction, repeating-decimal and named-constant literals still go through the
  canonical parser. `Err(ParseError)` / `Err(TierOverflow)` instead of a panic.
- **`g_math::wide`**: `exp_q64`, `ln_q64` (returns `Option`), `sin_q64`,
  `cos_q64`, `sincos_q64` over raw Q64.64 `i128`, the same engines and results
  on every profile; `ONE_Q64`, `PI_Q64`, `TWO_PI_Q64`, `PI_HALF_Q64`, `PI_Q32`,
  `TWO_PI_Q32`, each `floor(c * 2^f)`; `try_from_str(s, frac_bits)`, the exact
  literal parser for any Q format up to 127 fraction bits. Measured accuracy
  against mpmath in units of `2^-64`: `exp` 4 relative to the result, `ln` 55,
  `sin`/`cos` 3 for `|x| <= 2 pi`, growing as `0.34 |x|` beyond because range
  reduction subtracts multiples of the truncated `PI_HALF_Q64` (1390 below
  `2^12`, 8e-11 absolute below `2^32`).
- **`FixedPoint::sincos_wide_q64(angle_q64)`** (realtime and compact): the
  computation behind `sincos_wide` without truncating the angle to Q32.32;
  `sincos_wide(a) == sincos_wide_q64((a as i128) << 32)`.
- Gate `tests/wide_q64_validation.rs` (references
  `scripts/generate_wide_refs.py`: mpmath 120 digits, exact fractions for the
  literals), CI `wide-tier` on every profile plus Q22.10.
- **`try_add`, `try_sub`, `try_mul`, `try_div`, `try_neg`** on `FixedPoint`
  and `DecimalFixed`, and `FixedPoint::try_to_int`, `try_from_int`,
  `DecimalFixed::try_from_integer`, `try_from_binary_q256`: the fallible twins
  of operations that now panic instead of wrapping or saturating. Gates
  `tests/operator_overflow_validation.rs` (every profile and realtime split
  2 to 30) and `tests/narrowing_defects_validation.rs`.
- **`DecimalFixed` `try_` transcendentals**: `try_exp`, `try_ln`, `try_sqrt`,
  `try_sin`, `try_cos`, `try_sincos`, `try_tan`, `try_atan`, `try_atan2`,
  `try_asin`, `try_acos`, `try_sinh`, `try_cosh`, `try_sinhcosh`, `try_tanh`,
  `try_asinh`, `try_acosh`, `try_atanh`. `Err(DomainError)` outside the domain,
  `Err(TierOverflow)` when a result or intermediate leaves the range,
  `Err(PrecisionLimit)` when an in-domain argument rounds onto a domain
  boundary at a compute precision coarser than `D` (for example `ln(1e-38)` at
  38 decimals on embedded). They never panic (a sweep of 14,859 extreme calls
  per build under `catch_unwind`) and equal the infallible methods bit for bit
  whenever those return. Gate `tests/decimal_try_transcendentals_validation.rs`.

### Fixed

Silent wraps and rounding slips in the canonical (`gmath`) layer, all found
while checking the parser change; each is gated in
`tests/ugod_promotion_validation.rs`, `tests/wide_q64_validation.rs` or the
parser equivalence unit test:

- **Decimal literals wrapped on realtime and compact.** `parse_decimal` stored
  `value * 10^dp` in the storage-width decimal through a truncating cast. At
  `GMATH_FRAC_BITS=10`, `gmath("467295.6470")` was 37798.9, `"214748.3648"` was
  -214748.3648 and `"3000000.0001"` was -6477.1. Beyond the decimal storage the
  literal is now kept as the exact rational.
- **Decimal arithmetic results wrapped on every profile.** Decimal UGOD promoted
  correctly, then `decimal_to_storage` narrowed the promoted result back with
  truncating casts (`as i32`, `as_i128`, `as_i256`): at Q22.10
  `214748.3647 + 1` was -214747.3649. It is now checked, and add, subtract,
  multiply and divide fall back to the exact rational.
- **Decimal-to-binary coercion wrapped out-of-range values on every profile**
  (`decimal_to_binary_storage`; at Q22.10 `"-3745932.2"` in binary mode came
  back as an in-range value). It now returns `TierOverflow`.
- **Decimal compute results rounded toward zero when converted to binary**
  (`decimal_compute_to_binary_storage`), up to one unit off the nearest rule the
  rest of the crate uses. Now nearest, ties toward +infinity.
- **Literals past 38 digits.** The I256 fallback (integer part times `10^dp`
  beyond i128) truncated toward zero, one unit off on embedded, balanced and
  scientific; wide profiles dropped fraction digits past 38 (76 on scientific)
  before converting, which on balanced (one unit is 2.9e-39) can lose whole
  units; the scientific two-part path rounded ties away from zero; integer
  parts past i128 were a `ParseError` even where the profile's range holds
  them; on realtime and compact, digits past the 38-digit rational budget
  were dropped, which can turn a value just past a tie into a tie. All of
  these now use the exact converter behind `FixedPoint::try_from_str`.
- **Hex and binary literals** mixed three meanings and wrapped beyond the
  storage width (see Upgrading); they are now integers through the same range
  check as decimal integers.
- `sincos_wide`'s documentation example called a function that does not exist.

Defects that only showed at a realtime split other than Q16.16 (found by
running the whole suite at `GMATH_FRAC_BITS=10`, where 49 tests failed; 0 now),
plus the ones behind them that affect every profile:

- **`StackValue::to_rational` read realtime binary values with 16 fraction
  bits** whatever `GMATH_FRAC_BITS` was: at Q22.10 every shadowless binary value
  came out 64 times too small (`-1000 + 1/1` became -999/64).
- **Curvature finite differences hardcoded 16 fraction bits** on realtime: the
  step was 2.0 and the `1/(2h)` scale 64 times too large at Q22.10, so
  Riemann, Ricci and sectional curvature were wrong at any other split. The
  shift is now checked instead of wrapping on overflow, on every profile.
- **RK4 rounded h/6 to storage before using it**: at h = 0.01 on Q22.10 every
  step ran 20% long (a flat geodesic ended at 1.171, not 1); at Q16.16 the same
  test was 38.7 units off. Every Runge-Kutta stage is now formed at the compute
  tier from the exact coefficients and rounded once, on every profile.
- **Dormand-Prince (`rk45_integrate`) left the b*7 k7 term out of its
  4th-order solution**, so the error estimate carried a constant `h k / 40`
  instead of the local error; its tableau coefficients were also rounded to
  storage, and the step-growth test `err < tol / 32` rounded to zero for small
  tolerances. All three fixed (one extra function evaluation per step).
- **Verlet's half kicks used a floored h/2** for an odd raw step.
- **`geodesic_integrate` rounded T/N to storage**, so N steps missed T by up to
  N/2 units (0.977 instead of 1 at Q22.10, N = 100); step k now runs from
  round(kT/N) to round((k+1)T/N).
- **Matrix exponential and logarithm coefficients were rounded to storage**:
  the Pade coefficients were 21-digit decimals parsed at storage precision (b5
  and b6 were 0 at 10 fraction bits, and the wide profiles got 21 digits rather
  than their own), and the log series k/(k+1) likewise. Both are now exact
  fractions at the compute tier.
- **`downscale_to_storage` checked the range before adding the round bit**, so
  a value rounding up to 2^(W-1) passed the check and then overflowed (a panic
  in debug builds, a wrap in release), on every profile.

Realtime splits other than 16 and 10 (`GMATH_FRAC_BITS` 2 to 30; the suite now
gates 8 to 24 in full and the rest with correctness gates) exposed more of the
same classes, all fixed:

- **Silent wraps at narrow ranges.** `FixedPoint::from_int` shifted without a
  range check (now panics; new `try_from_int`); `make_compute_int` likewise;
  `compute_divide` narrowed its quotient with a truncating cast on every profile
  and returned the wrapped value as `Ok` (now `Err(TierOverflow)`); symbolic,
  decimal and decimal-compute values converted to the compute tier through
  unchecked casts (`atan(999999999)` at 24 fraction bits returned -pi/2);
  compute-tier sums in dot products, norms and the fused operations used plain
  additions (now checked). Library code that turned counts or constants into
  storage values (`from_int(720)` in the Rodrigues series, `from_int(n)` for a
  vector length or `n!`) no longer requires them to fit the storage range.
- **Rodrigues threshold** `0.001` was zero at 8 fraction bits, so `exp(0)` erred
  and `log(I)` panicked on SO(3) and SE(3); it is now at least one unit.
- **`rms_norm_factor`, `softmax`, `softmax_mix`** panicked instead of returning
  `Err(TierOverflow)` when a result left storage.

Wrong results and silent wraps found while moving state to the compute tier
(each gated in the test named with the change):

- **`sectional_curvature` returned rounding noise on every profile.** It
  contracted `R^l_ijk u^i v^j v^k w_l`, which is exactly zero because `R` is
  antisymmetric in its last two indices (the sphere of radius 1.5 gave 0.0,
  not 1/r^2 = 0.444). It now computes `<R(u,v)v, u>`.
- **`StiefelManifold::distance` returned the square root of the distance**
  (`frobenius_norm(..).sqrt()`, and `frobenius_norm` already takes the root).
- **`Grassmannian::log_map` and `parallel_transport`** paired the i-th largest
  sine with the i-th largest cosine, wrong for k >= 2 with distinct angles, and
  used the wrong singular vectors, which flipped the sign for k = 1 when
  `Q1^T Q2 < 0`.
- **SO(3) and SE(3) `lie_exp` returned `Err(DivisionByZero)` for small
  angles** on realtime (theta^2 rounded to zero at storage precision while
  theta was above the series threshold: `omega = [0.002, 0, 0]` at Q16.16).
  The SE(3) small-angle branch kept only the constants 1/2 and 1/6: 746 units
  at |omega| = 8.8e-6 on Q64.64, over 10^6 on Q128.128 and Q256.256.
- **Sphere and hyperbolic `exp_map` / `log_map` panicked at 24 fraction bits**
  for tangents shorter than 1/128 (1/theta left the storage range).
- **`FixedPoint::atan2` panicked on every realtime split** (`atan2(1, 1)`):
  the direct engine call passed the `Q(2F)` compute values to the Q64.64
  engine unscaled and truncated its angle. `try_atan2` was right; the two now
  agree bit for bit (`tests/try_direct_bypass_validation.rs`).
- **`cross_ratio` returned a false `DomainError`** at 8 to 12 fraction bits
  when a small nonzero denominator rounded to zero.
- **`FixedPoint::to_int` wrapped** at 2^31 on embedded and wider
  (`"3000000000"` gave -1294967296).
- **`DecimalFixed`**: transcendental results beyond i128 wrapped (`exp(50)` at
  19 decimals on embedded); large inputs were truncated into the realtime and
  compact compute tier (`sqrt(20)` at 18 decimals on realtime gave 1.2463);
  `from_binary_q256` wrapped (`2^300` gave a negative value); `from_integer`
  overflowed, `from_parts(1, 100)` gave 1.99, `integer_part` wrapped; the
  operators saturated on overflow and division by zero; `Display` dropped
  the sign of values in (-1, 0) (`-0.5` printed `0.5`); the parser accepted
  `"--5"` and `"1.+5"` and rejected fractions of 20 or more digits.
- **Decimal `exp` was inaccurate at large arguments and wrapped on
  realtime.** Its integer-power table was `e * e * ... * e` at the compute
  precision, multiplying `e`'s rounding by k: realtime `exp(22)` at 4 decimals
  was 446565 units off, `exp` past 22.9 wrapped (`exp(22.9451)` gave
  -9222509832.1468), embedded lost up to 267 units at 19 decimals, balanced
  490 at 38, scientific 20 at 77 with errors for large negative arguments;
  its `|x| > 30` path shifted without a check (compact `exp(44.28)` gave
  -1.7e28). It now reduces `x = n ln2 + r` at a wider working precision and
  rounds once: correctly rounded over the whole range on every profile
  (`tests/decimal_exp_range_validation.rs`, mpmath at 500 digits). `sinh`
  and `cosh` combine both exponentials at that precision.
- **Decimal `sin`/`cos` range reduction** used pi at the compute precision and
  an i64 quadrant count that wrapped: up to 19938 units off at 4 decimals on
  realtime (`sin(1e9)` had the wrong sign), garbage at 0 decimals on embedded,
  panics for large arguments on balanced and scientific. The reduction is now
  exact at the wider precision; results are correctly rounded.
- **Decimal `tanh`** formed `2x` before any check (wrapped on realtime for
  |x| > 4.6e9); **`asinh` / `acosh`** squared their argument past the compute
  tier (`asinh(1e30)` on embedded died in `sqrt`), and `asinh` of a negative
  argument cancelled; it is now odd by construction.
- **`DecimalFixed` `sin`, `cos`, `sincos`, `atan` and `tanh` panicked on the
  compute tier's minimum value**, as did the compact decimal wide multiply;
  `atan2` failed when `|y/x|` left the compute tier although the angle is
  representable (it now uses `sign(y) pi/2 - atan(x/y)` there). Behaviour
  change: `DecimalFixed::sqrt` of a negative argument below the compute
  resolution panics (it returned 0), like every other negative argument.
- **Canonical compute-tier arithmetic** (`gmath` chains of transcendental
  results, binary and decimal) wrapped on overflow; it now returns
  `Err(TierOverflow)` like the rest of the canonical layer.
- **`I256` and `I512` `Display`** printed values beyond i128 truncated (or as
  a saturated approximation); it is exact at every width.
- **Compute-tier arithmetic wrapped or truncated**: `compute_add`,
  `compute_subtract`, `compute_multiply` and `compute_negate` wrapped (and the
  multiply narrowed with unchecked casts), and every compute-tier quotient was
  truncated toward zero instead of rounded to nearest. `D256`, `D512`,
  `I256`, `I512`, `I1024` and `I2048` division by zero returned a saturated
  quotient (or 0); it now panics like integer division. The scientific `ln`
  engine's Q512.512 divide returned 0 for a zero divisor and narrowed its
  quotient unchecked (both unreachable from its table-factor callers, now
  asserted).

One rounding instead of two, and no overflow of intermediates that were never
results (owner decision: rework rather than document):

- `FixedVector::dot` and `dot_precise` **floored** the compute-tier sum; they now
  round to nearest, ties toward +infinity, like `mat_mul` and every other binary
  result (results move by at most one unit, toward the exact value).
- Norms and distances (`FixedVector::length`, `metric_distance_safe`,
  `frobenius_norm`, the Grassmann distance, the SO(3) angle) rounded the sum of
  squares to storage and then took the root: two roundings, amplified by
  1/(2|x|) for small norms, and an overflow once the SQUARED norm left storage.
  The root is now taken at the compute tier with one rounding.
- `qr_decompose` rounded ||x||^2 and v^T v to storage and R and Q after every
  reflection; both now stay at the compute tier (see "State carried at the
  compute tier" below).
- The Minkowski product of `HyperbolicSpace` rounded its spatial and temporal
  parts separately before they cancel; it is one compute-tier sum.
- `symmetrize` / `antisymmetrize` rounded the sum of n! terms and multiplied by
  a rounded 1/n!; they now divide the exact sum once.
- The Householder reflection shared by the SVD, Schur and QR formed the factor
  2 (v.w) / (v.v) at the compute tier; for a short v (the noise column of a
  rank-deficient matrix) it left the realtime compute tier at 24 fraction bits
  although every update fits (`svd_decompose([[1,2],[2,4]])` was
  `TierOverflow`). Each update is now one exact quotient, rounded once.
- **`matrix_exp` and `matrix_log` were accurate only up to Q32.32.** Pade [6/6]
  at ||B|| < 0.5 truncates at about 2^-55 and the 22-term log series at
  ||X|| < 0.25 near 2^-50, both amplified by the scaling: 21 units off on
  Q64.64 and about 2^31 units (29 of 38 digits) on Q128.128 and Q256.256. The
  scaling now follows the precision (12k >= F - 36.4 + log2 ||A|| extra
  halvings for exp, square roots to 2^-m with 22m >= F + 6 + s for log);
  realtime and compact keep the previous scaling. `matrix_sqrt`
  (Denman-Beavers) stopped at a step of sqrt(quantum) (a fixed 2^-8 on
  realtime), which `matrix_log`'s unscaling amplified to 122 units at 24
  fraction bits; it now iterates to one relative unit.

A new mpmath gate, `tests/one_rounding_validation.rs` (references from
`scripts/generate_one_rounding_refs.py`), measures these operations against
the correctly rounded result on every profile and at realtime 8, 10, 16 and 24
fraction bits. Worst errors in storage units, 0.6.3 then now: dot 1 to 0;
length, distance, Frobenius and Minkowski norms up to 29 to 0; symmetrize 4 to
10 to 0; QR R 3 to 23 to 0; SO(3) exp up to 12 to 1; matrix_exp up to about
2^31 to at most 1; matrix_log and matrix_sqrt at most 1.

### State carried at the compute tier

Multi-step computations kept their running state at storage precision and
rounded it after every step, so errors grew with the number of steps, the
matrix size or the iteration count. They now carry the state at the compute
tier (`2 x FRAC_BITS` fractional bits), form every sum of products exactly,
and round to storage once. Each is gated against mpmath or exact rationals on
every profile and at realtime 8, 10, 12, 16, 20 and 24 fraction bits; worst
errors in storage units, 0.6.3 then now:

| Operation | 0.6.3 | now |
| --- | --- | --- |
| LU solve / inverse / determinant (well-conditioned 2x2 to 4x4) | 114 / 42 / 70 | 1 / 1 / 1 |
| Cholesky solve / determinant; QR solve | 2 / 43; 27 | 1 / 1; 1 |
| `eigen_symmetric` values / vectors | 8 / past 2^30 | 0 / 1 |
| `svd_decompose` values / vectors | 25 / past 2^30 | 1 / 1 |
| `schur_decompose` eigenvalues | past 2^30 | 1 |
| RK4, Dormand-Prince, Verlet (same scheme in exact arithmetic) | 24 | 1 |
| `geodesic_integrate`, `parallel_transport_ode` | 1536 | 1 |
| Christoffel / Riemann / scalar curvature (same scheme) | 1e25 | 0 |
| Sphere, hyperbolic, Grassmann, SPD, Stiefel maps and distances | past 2^40 | 1 |
| SO(3) / SE(3) log near pi, manifold maps of SO(n), GL(n), SL(n) | past 10^6 | 3 |
| Fiber-bundle transport, projective maps, Moebius, cross ratio | 198 | 0 |
| Tucker / CP-ALS factors, pseudoinverse terms, `normalize` | 23 | 2 |
| `matrix_exp` / `matrix_log` on realtime at 8 fraction bits | 8 / 6 | 0 / 0 |

- **LU, Cholesky and QR** factor at the compute tier and keep those factors:
  `solve`, `inverse`, `determinant` and `refine` run on them. The public
  `l`, `u`, `q`, `r` fields are the same factors rounded once.
- **Jacobi, Golub-Kahan SVD and Francis Schur** carry the matrix being
  reduced and the accumulated transforms at the compute tier; an
  off-diagonal entry now counts as zero within `2^-(3F/2)` of its diagonal
  neighbours (was `2^-(2F/3)`). Householder vectors are scaled by a power of
  two before use (at 10 fraction bits a small column kept a few significant
  bits of `v.v` and the reflection lost orthogonality; a large column
  overflowed the realtime compute tier), and SVD and Schur form their shifts
  from the active block scaled up by a power of two (a block near `2^-8`
  froze the iteration at 8 and 10 fraction bits).
- **ODE integrators, `geodesic_integrate`, `parallel_transport_ode`,
  `VectorBundle::parallel_transport_along`** carry their state across steps
  at the compute tier; the user's right-hand side still sees storage values.
- **Manifolds** compute angles from exact products (`atan2` of the cross and
  dot parts on the sphere, the `ln` form of `acosh` on the hyperboloid,
  principal angles `atan2(|P_i|, |C_i|)` on the Grassmannian) instead of
  `acos`/`acosh` of a rounded cosine, which lost half the bits for close
  points; SPD maps keep `P^1/2`, its inverse, the products and `expm`/`logm`
  at the compute tier.
- **Lie groups**: SO(3)/SE(3) exp and log (the axis beyond 90 degrees from
  the symmetric part), group products, inverses (compute-tier LU), adjoints
  and brackets at the compute tier.
- **Curvature**: metric partials, the inverse metric, Christoffel symbols and
  their central differences at the compute tier (a storage rounding inside a
  central difference came out multiplied by `2^(k-1)`).
- **Projective and tensor code**: cross ratios, projective transforms,
  Moebius maps, `FixedVector::cross` and `normalize`, Tucker and CP-ALS
  (factors across iterations), SVD reconstruction, `pseudoinverse` and
  `condition_number_1`.

- **Matrix functions on realtime run at Q64.64.** The realtime compute tier
  holds `2F` fractional bits (16 at 8 fraction bits), and scaling and
  squaring amplified it: `matrix_exp` of a norm-7 matrix was 8 units off and
  `matrix_log` 6 at 8 fraction bits (1 at 10 to 16). `matrix_exp`,
  `matrix_log`, `matrix_sqrt`, `matrix_pow` and their compute-tier forms used
  by the Lie groups and manifolds now run on i128 values with 64 fractional
  bits on realtime (inputs widened exactly, results rounded once), and
  Denman-Beavers stops at `2^-(F + 8)` relative, compared at the working
  precision, on every profile (a stop at one storage unit left `2^-2F`, which
  the logarithm's unscaling amplified). Measured: 0 units on every profile and
  split, including norms up to 7 and SPD spectra from 1/8 to 60.
  `ComputeMatrix` products are one rounding of their exact sums (each product
  was rounded first). The public matrix functions return `Err(TierOverflow)`
  where a result leaves storage (they panicked).

Gates: `tests/one_rounding_validation.rs`, `tests/ode_compute_state_validation.rs`,
`tests/curvature_compute_tier_validation.rs`,
`tests/manifold_compute_tier_validation.rs`,
`tests/lie_fiber_compute_tier_validation.rs`,
`tests/projective_tensor_compute_tier_validation.rs`, each with its reference
generator in `scripts/`.

### Performance

- **Checked operators.** Release-mode cost per operation against 0.6.3,
  measured on one core: `+` and `-` unchanged; `*` 0.24 ns slower on realtime
  (1.69 to 1.93 ns), 0.44 ns on compact (1.78 to 2.22), 15% on balanced (15.9
  to 18.3), unchanged on embedded and scientific; `/` unchanged.
- **Decimal-domain multiplication on balanced and scientific** divided by
  10^77 / 10^154 with a bit-serial long division (2048 steps for I2048), once
  per multiply, which made canonical decimal transcendentals on scientific
  about 570 times slower than the binary engine. The division by the constant
  is now limb-wise (bit-identical, gated by a unit test).

## [0.6.3] - 2026-09-19

### Upgrading

A patch release: `FixedPoint`'s float conversions become exact wherever the
format allows, and values outside the profile's range are refused instead of
wrapped. Two functions are added; nothing is removed. Everyone on `^0.6`
receives it on their next `cargo update`.

**Do you need to act?**

- **You never convert `FixedPoint` to or from `f64` / `f32`** → Nothing changes
  for you.
- **You call `to_f64` / `to_f32`** → Results move to the exact value (or the
  nearest float). 0.6.2 was low in magnitude, always toward zero, by up to one
  unit of the last decimal digit it printed: in raw steps up to 1.0 at
  `GMATH_FRAC_BITS=10`, 6.6 at the default realtime split, 4.3 on compact and
  1.8 on embedded; on embedded and wider this shows only for values small
  enough for an f64 to resolve that digit. If you replay earlier runs bit for
  bit, `x.to_string().parse::<f64>()` gives exactly what 0.6.2's `to_f64`
  returned, and `format!("{:.10}", x).parse::<f32>()` what its `to_f32`
  returned, on every profile.
- **You call `from_f64` / `from_f32` with values inside your profile's range**
  → Nothing changes: the results are bit-identical to 0.6.2 (still truncated
  toward zero).
- **You pass values outside the range, or NaN or infinity** → 0.6.2 returned a
  wrapped, wrong value for out-of-range input (at `GMATH_FRAC_BITS=10`,
  `from_f64(3e6)` gave -1194304 and `from_f64(4194304.0)` gave 0). Now
  `from_f64` / `from_f32` panic, and the new `try_from_f64` / `try_from_f32`
  return `Err(TierOverflow)` (`Err(InvalidInput)` for NaN).

### Fixed

- **`to_f64` / `to_f32` were lossy by construction.** They printed the value as
  a decimal string with a capped digit count (realtime `floor(F log10 2)`: 3
  digits at `GMATH_FRAC_BITS=10`, 4 at the default; compact 9, embedded 19,
  balanced 38, scientific 77), cut the remaining digits, and parsed the string.
  The cut was worth up to 1.0 raw step at `GMATH_FRAC_BITS=10`, 6.6 at the
  default realtime split, 4.3 on compact and 1.8 on embedded, always toward
  zero. At `GMATH_FRAC_BITS=10`, 1024 consecutive raw values gave 1000 distinct
  f64 values, and `from_f64(x.to_f64())` came back one step lower in magnitude
  for 1016 of every 1024; every further round trip lost one more step. The float
  is now assembled from the raw integer's bits with integer operations: exact
  whenever the raw value has at most 53 (f32: 24) significant bits, which is
  every realtime value, and otherwise rounded to nearest with ties to even. For
  f32 on the balanced and scientific profiles, values below f32's normal range
  become subnormal or zero and (scientific only) values beyond it infinite. The
  conversion is 16 to 99 times faster (151,936 conversions, release build:
  realtime 17.3 ms to 0.51 ms, scientific 266 ms to 2.7 ms).
- **`from_f64` / `from_f32` wrapped out-of-range values** on every profile. The
  "value too large" panic only fired once the shift reached the full storage
  width; below that the magnitude was narrowed by an unchecked cast or shift. A
  magnitude outside the storage range is now `TierOverflow` (the minimum,
  exactly `-2^(W-1)` raw, is still accepted). Compared with the published 0.6.2
  on 200,000 random in-range inputs per profile, f64 and f32: identical.

### Added

- `FixedPoint::try_from_f64` and `FixedPoint::try_from_f32`.
- `tests/float_boundary_validation.rs`: to-float conversions bit-identical to
  references computed on exact rationals (round half to even, f64 cross-checked
  against CPython's correctly rounded division; 351 to 1630 raws per profile,
  including rounding ties, carries, the range ends and f32's subnormal and
  overflow boundaries); from-float truncation and range errors on about 400 f64
  and up to 400 f32 inputs per profile; exactness and round trips on about
  200,000 raws; sign symmetry and monotonicity; NaN, infinity and panics.
  References from `scripts/generate_float_boundary_refs.py`. CI workflow
  `float-boundary` on all five profiles plus realtime at `GMATH_FRAC_BITS=10`.

## [0.6.2] - 2026-09-16

### Upgrading

A patch release with no API change: three matrix decompositions could return
a wrong answer as if it were right, and now either return a correct one or an
error. Everyone on `^0.6` receives it on their next `cargo update`.

**Do you need to act?**

- **You do not call `svd_decompose`, `eigen_symmetric` or `schur_decompose`,
  nor anything built on them** (`pseudoinverse`, `rank`, `nullspace`,
  `condition_number_2`, the Grassmannian, Stiefel and SPD manifold maps,
  `truncated_svd`, `tucker_decompose`) → Nothing changes for you.
- **You call them** → Their results change, usually by a few ulp, and on some
  inputs from a wrong value to a right one. If you store, hash or compare
  against earlier results, regenerate them in one pass.
- **You `unwrap` their results** → Two errors are new. `Err(PrecisionLimit)`
  where an iteration used to run out of steps and return what it had, and
  `Err(TierOverflow)` where a norm or an entry beyond the storage range used to
  panic or wrap. Both mean the old `Ok` was not a correct answer.

**What was wrong, in plain terms**

1. **The SVD gave up silently** on matrices that are exactly rank-deficient,
   on every profile, and returned its unfinished state as the answer. It could
   also rotate one of its factors the wrong way, which changed the matrix it
   reconstructs without changing the singular values.
2. **The real Schur form was not a Schur form.** Its last step was skipped, 2×2
   blocks were never reduced, and some matrices made it loop until it gave up.
3. **The symmetric eigenvalue solver could believe it had converged** when the
   squares in its convergence test overflowed, and returned the diagonal
   unchanged.

Full detail and measurements follow below.

### Fixed

- **`svd_decompose` returned the unconverged diagonal as singular values** on
  exactly rank-deficient input, on every profile. An exact zero singular value
  is computed as a block of a few ulp of rounding noise; the purely relative
  convergence test (floored at one quantum) never passes on it, the iteration
  only advances the bottom unreduced block, and on exhausting its `30 n²`
  budget the loop returned `Ok`. Measured on 0.6.1: a rank-6 8×8 integer matrix
  on realtime at `GMATH_FRAC_BITS=10` returned singular values up to 15 percent
  off with a reconstruction error of 4.69; a rank-3 8×8 on embedded returned
  17.69, 14.69, 1.24 for 38.54, 30.96, 26.29 (reconstruction error 17.65). On
  the first matrix the failure appeared with 0.5.0 because the rounding
  unification changed its rounding noise; the defect is older, and the 0.4.x
  result for that matrix was itself not orthogonal (off by 0.40).
- **`svd_decompose` rotated U the wrong way** in interior zero-diagonal
  deflation: U took the rotation instead of its transpose. Singular values and
  orthogonality were unaffected, so only reconstruction could show it: a 3×3
  bidiagonal input with an interior zero reconstructed 2.0 off on realtime and
  on embedded.
- **`schur_decompose` did not return a real Schur form.** The Francis bulge
  chase stopped one step short and left an entry below the subdiagonal; 2×2
  blocks, including a 2×2 input, were never reduced; there were no exceptional
  shifts, so a cyclic permutation matrix stalled; and budget exhaustion
  returned `Ok`. A symmetric 3×3 on embedded returned `T[2,0] = 0.274` and a
  diagonal entry 7.0319 for the eigenvalue 7.0489. On realtime it panicked in
  `round_to_storage` on a random 8×8 integer matrix with entries in [-10, 10],
  squaring the shift column at storage precision.
- **`eigen_symmetric` squared off-diagonal entries at storage precision** in
  its convergence test: squares beyond the storage range wrapped and could pass
  at the first sweep. On `v (J - I)` 3×3 with `v = 592` at `GMATH_FRAC_BITS=10`
  it returned 592, -592, 0 for 1184, -592, -592 (the same on embedded with `v`
  near 5.2e18). On realtime a random symmetric 64×64 integer matrix came back
  as its own diagonal: reconstruction off by 20. Stagnation and sweep
  exhaustion returned `Ok` whether converged or not.

### Changed

- Transforms: rotation coefficients and Householder factors stay at the
  compute tier, and every transformed entry is narrowed once from an exact
  accumulator; shifts are formed at the compute tier from exact products.
  Coefficients rounded to storage precision injected about `|x|` ulp per
  transform, enough to keep rank-deficient inputs above any few-ulp convergence
  floor. The crate-internal storage-precision `givens` and
  `apply_givens_compute` helpers are removed.
- `svd_decompose` carries the bidiagonal's diagonal and superdiagonal at the
  compute tier through the whole QR iteration and narrows each singular value
  once at the end. A chase rounded to storage after every rotation loses a
  bulge smaller than one quantum, and the shift with it: on entries of a few
  hundred quanta the step then reproduces its input (or its input with signs
  flipped) and the iteration spends its budget. The Wilkinson shift is formed
  from the compute-tier values, and every product is rounded once from its
  exact value and fits-checked.
- Convergence: an off-diagonal entry is negligible within the tight relative
  bound `2^-(2F/3)` of its diagonal neighbours, floored at four quanta; a
  diagonal entry of at most four quanta is deflated and set to exactly zero. An
  iteration that stops improving is taken to be at its precision floor and
  deflates its smallest-backward-error entry only within the looser
  sqrt(quantum) relative bound; otherwise it continues until its budget runs
  out and returns `Err(PrecisionLimit)`. "Stops improving" means that no
  off-diagonal (and, for the SVD, no diagonal) entry of the active block has
  reached a new smallest magnitude for five iterations, and for Schur that the
  block has also run 30 iterations. The largest entry, or a count of
  iterations alone, is no such evidence: on scientific a rank-deficient 8×8
  deflated a zero singular value at iteration 5 while it was still shrinking
  (reconstruction 9.3e-48 instead of 4.8e-69), and on embedded a
  well-conditioned 10×10 Schur ended at the loose bound after an exceptional
  shift interrupted its convergence (backward error 100 times that of any
  other case in its corpus). On realtime that looser bound shifts
  by `min(8, F/2)` bits for `F = GMATH_FRAC_BITS`: the shared threshold's fixed
  shift of 8 is sqrt(quantum) only at Q16.16, and at Q22.10 it sat under the
  rounding floor of a Francis step on entries near one.
- `schur_decompose` returns exact zeros below the subdiagonal, splits 2×2
  blocks with real eigenvalues, and uses the LAPACK `dlahqr` exceptional shifts
  every 10 iterations without deflation (anchored alternately at the top and
  the bottom of the block). With shifts at 10 and 20 only, a non-normal 8×8
  with a repeated eigenvalue took 1232 of its 1920 iterations on compact; now
  36.
- `eigen_symmetric` updates the diagonal in Rutishauser's form
  (`a_pp + t a_pq`, `a_qq - t a_pq`) and forms `tan θ` without squaring `τ`.
- Measured on the new gate, largest error over its fixed cases, in ulp
  (singular values / symmetric eigenvalues / Schur eigenvalues): realtime
  17 / 212 / 888; realtime at `GMATH_FRAC_BITS=10` 10 / 5 / 288; compact
  3 / 4 / 105716; embedded 3 / 4 / 19728208; balanced 4 / 4 / 2051743236;
  scientific 3 / 5 / 4.1e18. Reconstruction errors follow the relative
  deflation bound, and a Schur eigenvalue's error is that backward error times
  the eigenvalue's condition number (up to 252 among the cases). A matvec
  through the SVD of the rank-6 matrix above: worst relative error 26.14 per
  mille at `GMATH_FRAC_BITS=10` (2172.97 on 0.6.1).
- Cost against 0.6.1, release build, min of repetitions, random integer
  matrices of 8×8, 32×32 and 64×64: SVD of full-rank matrices takes 1.08 to
  1.19 times as long on realtime and 1.34 to 1.54 times on embedded; symmetric
  eigenvalues 0.92 to 1.05 times. Rank-deficient SVD and Schur are faster
  (0.04 to 0.64 times) where 0.6.1 ran out its iteration budget, and 0.6.1
  panicked on realtime for Schur at every size and for rank-deficient SVD from
  32×32. The 64×64 symmetric case on realtime took 31 µs on 0.6.1 because it
  returned the diagonal unchanged (see Fixed); it now takes 90 ms.

### Added

- `tests/decomposition_convergence_validation.rs`: 17 SVD, 8 symmetric
  eigenvalue and 12 Schur cases (exactly rank-deficient matrices, interior zero
  diagonals, real 2×2 blocks, permutation and companion matrices, up to
  16×16), each checked for reconstruction, orthogonality, its spectrum against
  references and, for Schur, exact real Schur structure; plus large entries
  whose squares leave storage, and `TierOverflow` beyond range. References from
  `scripts/generate_decomposition_refs.py` (mpmath at 120 digits, cross-checked
  against exact characteristic-polynomial roots); bounds measured per profile.
  On 0.6.1 the gate fails on realtime and embedded.
- A seeded random corpus in the same gate: 64 cases per decomposition drawn
  from the failure classes above plus rectangular, scaled (entries up to 600)
  and small-valued (`k/64`) matrices, Gram and repeated spectra, signed
  permutations, hidden complex pairs and companion matrices, sizes 2 to 12
  (`tests/data/decomposition_random_refs.rs`, seed 20260916;
  `scripts/generate_decomposition_refs.py --random --seed S --cases N`
  regenerates any corpus, with the same mpmath references and cross-checks; a
  non-normal matrix with a repeated eigenvalue carries no Schur spectrum
  references, being possibly defective). Every case must decompose, meet the
  structure checks, and stay within bounds on spectrum and reconstruction error
  per unit of its largest entry and on orthogonality. A failing run lists every
  failing case and the command that regenerates its corpus;
  `DECOMP_CALIBRATE=1` prints every measurement instead of asserting. The
  bounds come from calibration corpora on other seeds, 52,736
  decompositions in all (512 cases per seed; 8 seeds on realtime at both
  `GMATH_FRAC_BITS` 16 and 10 and on compact, 4 on embedded and scientific, 3 on
  balanced, fewer for symmetric eigenvalues on the two widest), every one
  converged; the committed corpus and a fresh one pass on all six
  configurations. Bounds are four times the largest calibrated value (sixteen
  times for Schur spectrum and reconstruction on embedded and wider, where some
  small-valued cases deflate at the looser bound). Before this release's
  compute-tier chase and threshold changes the same calibration found
  `Err(PrecisionLimit)` from the SVD on realtime (2×2 to 8×8 matrices, among
  them a 2×2 with singular values 0.2134 and 0.0103 at `GMATH_FRAC_BITS=10`)
  and from Schur at `GMATH_FRAC_BITS=10` (a signed 10×10 permutation).
- A library unit test that budget exhaustion is `Err(PrecisionLimit)` for all
  three decompositions.
- CI workflow `linalg-decompositions` on all five profiles, plus realtime at
  `GMATH_FRAC_BITS=10`, running both gates on every push; and
  `linalg-decompositions-fresh`, weekly and on demand, which draws a new seed,
  generates its corpus with mpmath 1.3.0, and runs it on the same six
  configurations (the seed is printed; a manual run accepts one).

## [0.6.1] - 2026-08-30

### Upgrading

Additive, with one enclosure getting tighter. `Interval::quadratic_form`
now narrows once instead of once per stage: its width is at most 1 ulp on
every profile (0 when the value is representable), where 0.6.0 measured a
mean of 3.2 ulp at 7 dimensions and 5 to 64 ulp at 23. Endpoints move
inward, never outward, so a consumer test that pinned a width bound still
passes; a test that pinned the exact endpoints of a quadratic form will
move. No other function changes its result.

### Added

- **`fused::quadratic_form(v, m)`** and **`fused::try_quadratic_form`**: the
  scalar `v^T M v` with ONE rounding. Every term `v_i m_ij v_j` is an exact
  triple product at `3 * FRAC_BITS` fractional bits on the profile's widest
  accumulator (i128, I256, I512, I1024, I2048 by profile: the same
  accumulators the orient3d predicate uses), the sum is exact and checked,
  and the single narrowing rounds to nearest with ties toward positive
  infinity, the binary house rule. The result is the correctly rounded value
  of the form for the stored operands and always lies inside
  `Interval::quadratic_form(v, m)`, which narrows the same exact value
  outward. The two-stage scalar (`M v` rounded per row, then the dot product
  rounded again) that consumers built from `dot` rounds twice with an error
  that grows with `sum |v_i|`; this is the fix the geometric-validation
  consumer's width decomposition asked for. Storage overflow is a panic on
  the infallible form and `Err(TierOverflow)` on the `try_` twin.
- `I2048::checked_add` (the scientific accumulator had no checked addition).
- Gates: `tests/interval_enclosure.rs` asserts the interval is exactly
  `[floor, ceil]` of the exact value and the fused scalar exactly its nearest
  on 3000 random forms of dimension 2 to 7 (narrow profiles, i128 model),
  constructed exact ties of both signs on every profile, the identity pin
  `fused::quadratic_form(v, I) == fused::dot(v, v)`, and typed overflow on
  both paths. `tests/certified_geometry_refs_validation.rs` gains quadratic
  form references computed on VALUES with exact rationals (no raw-integer
  shifts in common with the kernel) and cross-checked against mpmath at 700
  digits, including operands near the storage maximum. Library unit tests
  `wide_acc::` join the width-budget guard in the certified-geometry CI.

### Changed

- `Interval::quadratic_form` uses the fused accumulation above and narrows
  once (floor and ceil of one exact value). Off-diagonal terms are paired,
  `(m_ij + m_ji) v_i v_j`, without assuming symmetry of `M`; that halves the
  wide multiplies. Cost on the embedded profile, release build, 2000 records,
  min of 3 rounds: fused scalar 714 ns per record at n = 7 and 6832 ns at
  n = 23, 85 to 89 percent of the two-stage scalar; the certified form 719
  and 6862 ns, 107 to 122 percent of the 0.6.0 two-stage interval and 86 to
  90 percent of the two-stage scalar. The two-stage interval form remains
  available by composition (`Interval::dot` per row, then `dot_intervals`)
  for anyone who wants its lower cost with the wider bracket.
- The exact-accumulator trait and the per-profile accumulator table moved
  out of `predicates.rs` into a private shared module (`wide_acc`), so the
  predicates, the intervals and the fused form share one width budget. The
  predicates' behaviour and API are unchanged.
- `CONTRACT.md` section 3: the enclosure clause no longer says "unreleased".

## [0.6.0] - 2026-08-30

### Upgrading

Everything in 0.6.0 is additive: new types (`Interval`, `DecimalInterval`),
a new module (`imperative::predicates` with `pd_verdict`, `PdVerdict`,
`Sign`, `orient2d`, `orient3d`, `incircle`, `insphere`), and new gates. No
existing function changes its signature or its rounding. The three defect
fixes listed under 0.5.1 below are included; if you are on `^0.5` you can
take them without adopting the new API by staying on 0.5.1.

### Added

- **Certified interval arithmetic**: `g_math::fixed_point::Interval`
  (module `imperative::interval`). An enclosure `[lo, hi]` that is sound by
  construction: every product of storage values is formed exactly at the
  compute tier, and every narrowing back to storage rounds the lower endpoint
  toward negative infinity and the upper toward positive infinity.
  Operations: `+ - * /` and `Neg` (with `try_*` twins), `sqrt` (an integer
  Newton candidate at the compute tier, certified a posteriori by the exact
  integer check `k^2 <= n < (k+1)^2` with a bounded correction that fails
  loudly rather than loop; no transcendental engine involved), `dot` and
  `quadratic_form`
  (exact accumulation, one narrowing per stage), and the certainty predicates
  `contains`, `contains_zero`, `is_certainly_positive`, `is_certainly_negative`,
  `width`, `is_point`. Endpoint arithmetic never wraps: storage overflow is a
  typed `TierOverflow`, and dividing by an interval containing zero is
  `DivisionByZero`. No transcendental is provided; their accuracy is measured
  at test points rather than proven over the domain, and an interval widened
  by a measured error would not be an enclosure. Design and measurements in
  `docs/design/CERTIFIED_INTERVALS.md`. Measured at Q64.64 on a 23-dimensional
  quadratic form: 5 to 64 ulp wide on values of order 10, zero enclosure
  failures, unmoved by ill-conditioning, at 106 to 113 percent of the scalar
  path.
- Directed narrowing points `downscale_to_storage_floor` and
  `downscale_to_storage_ceil` beside the existing nearest variant, crate
  internal. The scalar paths are untouched; one rounding rule per domain
  still holds on every scalar path.
- `tests/interval_enclosure.rs`: the permanent gate (directed rounding
  brackets the nearest scalar by at most one ulp including negatives and
  constructed ties, enclosure against exact integer references, the sqrt
  certificate on sampled inputs, typed errors), on every profile and in the
  narrow-profile CI matrix.
- **Certified decimal intervals**: `g_math::fixed_point::DecimalInterval<D>`
  over `DecimalFixed<D>`, the same design with `10^D` in place of `2^F`:
  exact products at `2D` places in the decimal-domain `D256`, directed
  narrowing from the truncating quotient and its remainder, `+ - * /`, `Neg`,
  `sqrt` certified a posteriori by `k^2 <= x * 10^D < (k+1)^2` with an
  integer Newton candidate, `dot`, and the certainty predicates. One code
  path on every profile. Where the scalar `DecimalFixed` operators saturate
  on overflow and on division by zero, the interval returns a typed error.
  Gate: `tests/decimal_interval_enclosure.rs` (floor/ceil models, exact
  halves against banker's rounding, the 256-bit intermediate path, the sqrt
  certificate, typed errors), every profile.

- **Certified positive-definiteness verdict**:
  `g_math::fixed_point::imperative::predicates::{pd_verdict, PdVerdict}`.
  Runs the Cholesky factorisation in certified interval arithmetic
  (`Interval::dot_intervals`, new, is the one addition it needed). Every
  pivot certainly positive proves the stored matrix positive definite, with
  no arbitrary precision; a pivot certainly at or below zero proves it is
  not, at that pivot; a pivot straddling zero returns `Inconclusive` with
  the straddling enclosure so the caller can decide, which turns blind
  diagonal regularisation into a documented decision taken after a
  diagnosis. Measured on dyadic SPD matrices: last-pivot width `2.6e-17` at
  n = 23 and `1.2e-15` at n = 50 on Q64.64, sixteen orders of magnitude
  below the pivot value. Gate: `tests/pd_verdict_validation.rs` (proven PD,
  proven not PD at the right pivot with exact zero pivots, negative and
  indefinite matrices, an inexact singular matrix never proven positive,
  consistency with `cholesky_decompose`, width bound), every profile.

- **Exact geometric predicates**: `predicates::{orient2d, orient3d, incircle,
  insphere}` returning `Sign::{Negative, Zero, Positive}`, never a `bool`.
  Each is the sign of a fixed-degree determinant evaluated in exact integer
  arithmetic on an accumulator selected per profile from the storage width
  (`2W+2`, `3W+3`, `4W+5`, `5W+6` bits), with every multiply asserting its
  bit-length budget so a violation is loud rather than wrong. The exact-zero
  cases (collinear, coplanar, cocircular, cospherical) are decided exactly,
  including one ulp either side. `incircle` and `insphere` are not compiled
  on the scientific profile (2053 and 2566 bits needed against 2048); the
  route for a future consumer there is the arbitrary-precision type behind
  the `infinite-precision` gate. Gate: `tests/exact_predicates_validation.rs`
  (hand-checkable configurations with all three signs, permutation parity,
  agreement with an exact i128 evaluation on random coordinates), every
  profile.

- **Independent reference gate for all of the above**:
  `tests/certified_geometry_refs_validation.rs` over
  `tests/data/certified_geometry_refs.rs`, generated by
  `scripts/generate_certified_geometry_refs.py` from Python exact integers
  and fractions and cross-checked against mpmath at 300 digits inside the
  generator (a generator aliasing bug was caught by that cross-check before
  any value reached Rust). Certified sqrt, product and quotient endpoints
  with operands within a factor of four of the storage maximum on every
  profile, the exact rational last Cholesky pivot the interval Cholesky must
  enclose at n = 23 and 50, predicate signs on configurations scaled to
  near the storage maximum, and decimal endpoints; raw values travel as
  exact little-endian bytes at the storage width so the wide profiles need
  no string parsing.
- New workflow `certified-geometry.yml`: the six certified-geometry gates and
  the width-budget unit tests on all five profiles on every push.


## [0.5.1] - 2026-08-30

A patch release: three defects in the published 0.5.0, found by the gates
built for the certified-arithmetic work in 0.6.0, fixed with no change to
the public API so that every `^0.5` user receives them. Two of the three
are on the scientific profile; the third is in the widest decimal integer
types. If you compute on the scientific profile with values above about
`2^200`, or subtract decimal values wider than 128 bits, your results were
wrong before and are correct now.

### Fixed

- **Scientific profile: `FixedPoint::sqrt` lost precision for large
  arguments.** The Q512.512 reciprocal-square-root engine ran its Newton
  iteration directly on the input; for large inputs `1/sqrt(x)` and its
  square are tiny and the Q512.512 grid left them only about 260
  significant bits, so the iteration converged to a fixed point of the
  truncated map: 0 ULP over the validated `[0.0001, 10000]` range, but
  `2^124` ulp at `2^254 + 12345 ulp` and outside the certified enclosure
  from roughly `2^200` upward. The engine now normalises the input to
  `m in [1, 4)` (exact shift; no bit of a Q256.256 input is ever discarded),
  iterates at full precision, certifies the result at the normalised scale
  by the exact integer check `r^2 <= m * 2^512 < (r+1)^2` with a bounded
  correction that panics rather than return an unverified value, and shifts
  back exactly. Modelled exactly in Python before porting: 3220 inputs
  spanning the whole Q256.256 range, storage results correctly rounded in
  every case. Found by the new scalar-containment assertion of the
  reference gate; a magnitude-ladder gate in `tests/interval_enclosure.rs`
  now checks every profile's scalar sqrt against the certified enclosure
  across its full range.
- **`D256::Sub` and `D512::Sub` never propagated a borrow.** The borrow test
  compared a `u128` difference against `u128::MAX`, which is never true, so
  `0 - 1` produced `2^64 - 1` and any subtraction needing a borrow across a
  64-bit word lost it. These are the operations behind the UGOD decimal
  tier 5 and 6 subtraction arms. Found by the decimal interval gate.
- **`Ord` on `I1024`, `I2048`, `D256` and `D512` ordered two negative
  values backwards.** The word comparison was reversed for negatives, but
  two's-complement negatives already order correctly as unsigned words, so
  `-2` compared greater than `-1`. `I256` and `I512` had been corrected
  earlier; these four had not. On the scientific profile the compute tier is
  `I1024`, so `<`, `min` and `max` between two negative compute-tier values
  there were wrong. Found by the decimal interval gate's negative-product
  sweep.
- Gate for both: `tests/wide_integer_sign_semantics.rs` (ordering of
  negatives beyond i128 and mixed-sign sorts on all six wide types, borrow
  chains through one and two words on `D256`/`D512`, and consistency with
  i128 on every sign combination).

## [0.5.0] - 2026-08-24

### ⚠ Please read before upgrading to 0.5.0

0.5.0 makes gMath's answers more accurate. The side effect: some results
now end in a different final digit than 0.4.x gave you. Nothing became
less precise: the new values are simply closer to the true answer.

**Do you need to act?**

- **You compute and display results** → No. You just get better answers.
- **You save results, hash them, or compare them against numbers an older
  version produced** → Yes. Old and new values won't always match on the
  last digit. Pick one version per dataset and regenerate in one go
  rather than mixing.
- **You need the same input to give the same answer on every machine** →
  Nothing changes. That guarantee is untouched.

**What changed, in plain terms**

1. **Multiply and divide now round to the nearest value.** Some of them
   used to just drop the leftover instead, and which way they dropped it
   depended on which profile you built. Roughly half of all
   multiplications and divisions that don't come out exact will end in a
   different last digit.
2. **Balanced-ternary values sit on a finer grid.** Each ternary tier now
   fits 25% more digits into the same amount of memory. The values mean
   the same thing, but the raw stored numbers are different: the same
   measurement written on a finer ruler. Ternary weights for inference
   (TQ1.9) are not affected.

**The rounding rules now, one per domain.** The same rule applies on
every path (direct calls, expressions, fused ops) and every profile:

| Domain | Rule |
| ------ | ---- |
| Binary | Nearest; a value exactly halfway goes up |
| Decimal | Exact when the result fits, otherwise banker's rounding (halfway goes to the even digit) |
| Balanced ternary | Nearest; multiplication can never land exactly halfway, and where halfway is possible it goes up |

One deliberate exception: the TQ1.9 `matvec_q2f` inference path still
truncates, because its published contract guarantees it reproduces the
narrow matvec bit for bit.

**The rest**: `exp`, `ln`, `sqrt`, `sin`, `cos`, every other
transcendental, and chained expressions return the same bits as 0.4.34;
their reference tests are unchanged and still pass. It is plain
arithmetic, and matrix work built on it, that can move by one digit.

If in doubt: stay on 0.4.34 and upgrade when you can regenerate stored
data in one pass.

---

### Changed: 0.5.0 rounding unification (breaking-precision)

One rounding rule per domain, identical on every path (imperative,
canonical/UGOD, fused, and coercions), replacing rules that differed by
operation, by path, and (imperatively) by profile. Gated permanently by
`tests/rounding_unification.rs` (cross-path bit-equality sweeps plus
constructed exact-tie inputs, per profile). All five profiles green,
including the scientific 18/18 transcendental 0-ULP gate.

- **Binary: nearest, ties toward +∞, everywhere.** Imperative multiply
  was floor (realtime/compact), banker's (embedded), truncation
  (balanced/scientific); imperative divide truncated on all profiles;
  UGOD tier divide was half-away. All now match the wide-downscale rule.
  This REPAIRS path independence for plain mul/div (measured divergence
  before: 48.7% of sampled products, 1 ulp).
- **Decimal: exact when representable; banker's where rounding occurs.**
  Canonical divide tiers 1–5 discovered to be exact-or-rational-fallback
  (kept: better than rounding); the tier-6 best-effort arm moved from
  truncation to banker's; `decimal_to_binary_storage` coercion unified
  from per-profile truncation/add-half to nearest ties-+∞.
- **Ternary: nearest.** Multiply and `div3` are tie-free (odd scale:
  the 0.4.33 contract theorem, now shipped: error halves to ≤ ½ ulp and
  `div3` becomes a true trit shift); divide and conversion-in round
  nearest with ties toward +∞, a documented tie asymmetry, with the sign
  threaded through `from_str`. (The pins moved with the tier resize
  below: `0.5` → raw 29525, `-0.5` → raw −29524.)
- Contracted exception: TQ1.9 `matvec_q2f` narrowing stays truncation
  per its published 0.4.31 bit-reproducibility contract.
- Consumer notice: results may move by up to 1 ulp on direct storage-tier
  multiplies/divides (an accuracy improvement: nearest ≥ floor/trunc);
  consumers freezing hashed or persisted outputs should pin versions
  across this boundary. Compound/tier-N+1 results are unchanged.

### Audited: unsigned widening-multiply call sites (0.5.0 item 0b)

- Every `mul_to_i512/i1024/i2048` call site enumerated and classified;
  positive-by-construction sites (exp/ln/sqrt internals) now carry
  debug_asserts so the invariant is machine-checked in every test build;
  two private UNSIGNED helpers that shadowed the sign-safe
  `multiply_i1024_q512_512` renamed `*_nonneg` (hazard removed); three
  genuinely sign-broken `y·ln(x)` multiplies fixed in
  `pow_tier_n_plus_1.rs`: dead code with zero external callers (pow is
  composed as exp(y·ln x) through the sign-safe path), and the module was
  removed later in this same release (see Removed, below). New `negative_operand_battery` test
  (odd/even symmetries bit-exact, negative-intermediate chains) on every
  profile.

### Changed: ternary tier resize: TQ10.10 … TQ320.320 (breaking, owner-approved)

Every balanced-ternary UGOD tier gains 25% more trits in the SAME storage
word: TQ8.8→TQ10.10 (i32), TQ16.16→TQ20.20 (i64), TQ32.32→TQ40.40
(i128), TQ64.64→TQ80.80 (I256), TQ128.128→TQ160.160 (I512),
TQ256.256→TQ320.320 (I1024). A trit carries log2(3) ≈ 1.585 bits; the
old counts (chosen to mirror the binary tier names) left ~20% of every
word unused. The new counts fill the words; same storage, same
instruction count, 25% more ternary precision digits, and a clean 2×
ladder (10→20→40→80→160→320).

- BREAKING for persisted/hashed ternary raws: every scale factor changed
  (Tier 1 scale is now 3^10 = 59,049). Pinned conversion constants moved
  with it: `0t0.5` → raw 29,525, `-0.5` → −29,524 (tie asymmetry
  preserved), Tier-1 window ±29,524.
- Public fn renames follow the formats (`multiply_ternary_tq10_10` etc.);
  `SCALE_TQ10_10/TQ20_20/TQ40_40` replace the old constants.
- The FASC binary→ternary coercion's tier-3 arm now runs its
  multiply-divide at I256 width: with the larger 3^40 scale, a binary
  raw numerator times the scale can exceed i128 (caught by the mode-
  routing suite: `binary:ternary` sin(1.0) overflowed the old i128 path).
- The same coercion now targets the PROFILE's own tier and scale
  (realtime tier 1 / 3^10, compact tier 2 / 3^20): the old code borrowed
  the tier-3 arm on narrow profiles, whose 3^40-scaled raws no longer
  fit their storage (routed `0t2 + 1/3` on compact went TierOverflow
  instead of exact).
- `from_tier_raw`'s tier-1/2 constructors used bare `as i32`/`as i64`
  casts: oversized raws silently wrapped; now checked `TierOverflow`
  (wrap-defect class).
- `from_str`'s Tier-3 fractional multiply is now checked: long
  fractions cascade to Tier 4 instead of overflowing (latent pre-resize
  hazard, window merely shifted by the resize).
- TQ1.9 (`g_math::tq19`, the packed inference weight format) is a
  separate type and is NOT affected.
- Rationale and word-capacity math: docs/design/BALANCED_TERNARY_CONTRACT.md §1b.

### Removed: dead `pow_tier_n_plus_1.rs` engine (owner-approved)

The dedicated pow engine was superseded before it was ever wired in:
shipping pow (FASC and imperative) composes exp(y·ln x) at the compute
tier, which the 0-ULP suites validate. The module had zero production
callers; its three latent sign bugs were found and fixed in the 0b audit
and are now moot. Its direct-engine tests and reference data
(`POW_REFS`, generator section) are removed with it; the composed pow
keeps its own mpmath 0-ULP gates (integer and fractional exponents) in
`fasc_ulp_validation`.

### Changed: fallible composed transcendentals bypass FASC (0.5.0 item 2)

The `try_*` variants of the composed transcendentals (`try_tan`,
`try_atan`, `try_asin`, `try_acos`, `try_sinh`, `try_cosh`, `try_tanh`,
`try_asinh`, `try_acosh`, `try_atanh`) are now direct compute-tier
compositions mirroring their infallible twins, instead of routing
through the FASC pipeline (LazyExpr tree + TLS evaluator + domain
dispatch) per call. Same engines, same formulas, one downscale; results
are bit-identical to the infallible methods on in-domain inputs, and the
0.4.27 error contract is unchanged (`DomainError` for |x|>1 asin/acos,
x<1 acosh, |x|≥1 atanh; `TierOverflow` on storage overflow; `asin(±1)`
= ±π/2 exactly; tanh saturates to exactly 1 at the exp ceiling). Gated
by `tests/try_direct_bypass_validation.rs`.

Two silent-wrong-value defects flushed out by the new gate (both in the
0.5.0 wrap class):

- The imperative π/2 constant on the realtime profile cast a Q64.64
  quantity straight to i64: π/2·2^64 wrapped negative, silently
  corrupting every imperative `acos` on that profile (the FASC path was
  unaffected, which is why nothing gated it). Now a rounded shift to the
  profile's compute scale.
- `sinh`/`cosh` at the exp overflow sentinel: the q128_128 exp engine's
  sentinel (`i128::MAX` at storage scale) equals the storage maximum, so
  it downscaled CLEANLY into a plausible-wrong result: `cosh(180)` on
  balanced returned ~storage-max/2 as `Ok`, and the FASC pipeline's own
  ceiling guard (`== compute max`) had the same blind spot on that
  profile. A shared per-profile sentinel predicate now guards both
  paths: infallible sinh/cosh panic, `try_` variants return
  `TierOverflow`, FASC cosh/tanh use the corrected check.

### Fixed: UGOD ladder top (0.5.0 item 1): exact or loud, never wrapped

Verdict of the promotion audit: mid-ladder promotion (binary tiers 1→4)
was always correct, but the TOP of every ladder wrapped silently.
Contract now gated by `tests/ugod_promotion_validation.rs` on every
profile (and in CI): arithmetic on representable inputs either returns
the EXACT value (via a wider tier or the symbolic domain) or fails
loud.

- Binary Tier-4/5 multiply truncated its wide product unchecked
  (`1e20 × 1e20` on balanced returned 1.318e38: the product mod 2^256);
  Tier-4/5/6 add/sub used bare wrapping operators (`9e18 + 9e18` on
  embedded returned 0.0). All top tiers now use checked arithmetic with
  4→5→6 promotion arms.
- Storage narrowing of promoted results (`binary_to_storage`) used bare
  casts; now fits-checked, and the FASC binary arms fall back to the
  exact rational path on `TierOverflow`: the true ladder top.
- Divide branched into per-tier code before distinguishing zero divisors
  from quotient overflow, mislabeling overflow as `DivisionByZero`; the
  zero check now happens once at ladder entry, and overflow falls back
  to the exact rational quotient.
- The symbolic ladder's multiply "promotion" retried at the same i128
  width (Huge×Huge could never reach the existing Massive/I256 tier);
  `divide_mixed_tiers` returned the target-tier attempt without
  escalating (a quotient can need a wider tier than either operand:
  9e18 ÷ 1e-9 = 9e27 needs Huge). Both now climb the ladder.
- FASC's symbolic/ternary → binary coercion (`to_binary_storage`)
  shifted i128 numerators before any range check: a symbolic 1e20
  coerced on embedded wrapped mod 2^64 into a PLAUSIBLE WRONG value.
  Replaced with a checked nearest-ties-+∞ conversion at tier N+1 width.
- Fractional literals beyond a narrow profile's decimal cap silently
  truncated at parse: `"0.000000001"` on realtime parsed to exactly 0,
  turning later divisions into division-by-zero. Such literals now fall
  back to the exact Symbolic domain (the fractional twin of the item-0
  integer fallback below).
- The scientific (Q256.256) formatter squeezed integer parts through
  i128, so any result ≥ 2^127 DISPLAYED as its value mod 2^128 even when
  the stored bits were exact. Integer parts now print digit-at-a-time at
  I256 width.
- A sqrt compute-tier multiply helper assumed non-negative operands, but
  Newton's 3 − S·y² factor goes negative on seed overshoot (caught by
  the new 0b debug_asserts on scientific); the helper is now sign-safe.

### Fixed

- **Realtime FASC decimal results lost half their digits at
  materialization** ("the cosine plateau that wasn't"): DecimalCompute
  values were materialized at a fixed `DECIMAL_STORAGE_MAX_DP − 2` in
  Display/to_decimal_string/to_rational: 2 harmless digits of slack on
  wide profiles, but dp 4 → 2 on realtime, so cos(0.1) = 0.9952 rendered
  as "1.00" and masqueraded as a sin/cos kernel plateau (the kernels were
  bit-perfect all along). Now adaptive: full MAX_DP first, stepping down
  only when the magnitude needs fewer decimals (checked, deterministic).
  Wide profiles regain their two withheld display digits; realtime passes
  its original strict tolerances again.
- UGOD binary tier divide mis-signed its rounding bump for exact
  quotients in (−1, 0) raw units (branched on `quotient < 0`, which is 0
  there): e.g. an exact −0.75-ulp quotient rounded to **+1** raw instead
  of −1. Fixed by deriving the sign from the operands; covered by the
  unification gate's sub-ulp regression case.
- Ternary Tier-4 negation used `saturating_neg` (the one tier silently
  absorbing the binary-MIN edge); now fail-loud like every other tier.

- Integer literals beyond the profile's binary integer range now fall
  back to the Symbolic domain instead of failing parse with `Overflow`
  (UGOD's ladder top never fails). Pre-fix, realtime could not parse
  `32768`+ at Q16.16, so `1000000 * 0.001` errored there while
  succeeding on every other profile. Regression-tested on every profile;
  the router/domain integration suites now also run on realtime+compact
  in CI, which is how this stayed hidden.

## [0.4.34] - 2026-08-14

### Added

- **Ternary routing column**: the fractal router's classifier has computed
  `TERNARY_BIT` since v0.4.0 with no table column consuming it; ternary-exact
  values (denominator 3^k: 1/3, 2/3, 100/3…) fell through to the symbolic
  rational fallback. `DomainChoice::Ternary` now exists and cross-domain
  **Add/Sub** of 3-adic operands routes into the balanced-ternary domain,
  where it is exact by construction (sums of 3-adic values stay 3-adic).
  **Mul/Div are deliberately excluded**: products multiply denominators past
  the tier scale, where ternary truncates while symbolic stays exact, and
  the 4-bit class mask cannot see exponents: routing them would let routing
  change results. Full reasoning: `docs/design/TERNARY_ROUTING_COLUMN.md`.
- Coercion failure falls back silently to the previous route: on narrow
  profiles a large 3-adic value can overflow ternary storage, and the
  router must never introduce a failure the old path did not have.
- The classifier now reads **Symbolic operands' denominators directly**
  (they carry no shadow: the rational itself is richer than any shadow).
  Side effect beyond ternary: symbolic operands with 2-adic/10-adic
  denominators can now coerce into Binary/Decimal on Add/Sub/Mul/Div where
  both operands are exact there: exactness preserved in every such case.
- Measured (embedded, release, full evaluate pipeline including literal
  parsing): routed `0t2 + 1/3` 334 ns vs rational-fallback `0t2 + 1/7`
  355 ns (~1.06×). The win at expression scale is modest: parsing
  dominates; the value is architectural: the 3-adic exactness class now
  reaches its domain and results stay in fixed-point form.

### Fixed

- `convert_to_ternary` (output-mode/coercion conversion) stored tier-3
  raws through an **unchecked** narrowing cast in its tier-3 and fallback
  arms: on realtime/compact, converting e.g. `1/3` with output mode
  `ternary` silently wrapped (same defect class as the 0.4.33
  `ternary_to_storage` fix). Both arms now use the checked conversion:
  loud `TierOverflow`, never a wrap. Pinned by test.

- `multiply_ternary_tq256_256` (Tier-6 balanced ternary) passed operands
  straight into the unsigned `mul_to_i2048`, so any negative operand
  produced a product with corrupted sign extension. Now sign-wrapped
  (magnitudes in, sign restored) per the widening-multiply convention.
- `I2048::Mul`'s I512 fast path had the same defect via `mul_to_i1024`,
  corrupting Tier-6 ternary division of negative values. Sign-wrapped the
  same way. Scientific-profile transcendental validation (18/18) re-run
  clean after the change.

### Documentation

- CONTRACT.md's rounding table was wrong on two rows and is now stated
  per-path, verified against source: binary multiply is round-half-even
  in the imperative `FixedPoint` kernel but ties-toward-+∞ in the
  canonical/UGOD tier path: **1 ULP apart on exact half-ULP ties**
  (measured; the one known exception to path independence, scheduled for
  unification in 0.5.0); `DecimalFixed<D>` is banker's (not half-away),
  and the canonical decimal domain's multiply is exactness-preserving.
  README and routing guide updated to carry the same caveat.

### Testing

- Wide-tier ternary coverage closing the 0.4.33 gaps: Tiers 2–6
  arithmetic against exact integer models with every sign combination
  (the adversarial axis for the unsigned-word defect class), wide
  `ternary_to_storage` arm unit tests (exact-or-loud on every profile),
  FASC-transcendental-on-ternary path equivalence, and the pin that `0t`
  literals cap at Tier 3. All five profiles green.

## [0.4.33] - 2026-08-11

### Added

- **Balanced ternary contract + dedicated validation** (closes the
  long-standing ternary-coverage gap): `docs/design/BALANCED_TERNARY_CONTRACT.md`
  specifies both shipping representations (native trits for packing and
  zero-multiply inference kernels; radix-3 scaled integers for the UGOD
  tiers, with TQ1.9 as the window-enforced hybrid), verified operation
  semantics, and the tie-free rounding theorem (3^m is odd, so
  round-to-nearest onto a 3-adic grid never ties; trit truncation IS
  round-to-nearest). Suites: `ternary_domain_validation` (trit-vector
  reference oracle, exhaustive small-range equivalence, boundary families,
  theorem tests) and `ternary_path_equivalence` (UGOD promotion at
  raw-overflow boundaries, canonical-vs-imperative equivalence over `0t`
  literals, cross-domain coercion neutrality, conversion pins). New
  `ternary-domain` CI workflow runs both on every push.
- `FixedPoint::inv_sqrt` / `try_inv_sqrt`: 1/√x at the compute tier
  (square root and reciprocal both at tier N+1, one rounding at the final
  downscale). `try_` returns `DomainError` for x ≤ 0, `TierOverflow` if
  the result exceeds storage. Reciprocal norms are the target use:
  one `inv_sqrt` plus N multiplies replaces N per-component divisions in
  normalization (division is ~200× a multiply at Q64.64).
- `fused::inv_sqrt_sum_sq`: 1/√(Σ vᵢ²) entirely at the compute tier; the
  reciprocal-norm form vector normalization actually wants. Panics on a
  zero vector (documented), matching the fused family's conventions.
  Both additions are purely additive: no existing output bits move.

### Fixed

- `UniversalTernaryFixed::from_str` dropped the sign of `-0.x` inputs
  (`"-0".parse::<i64>()` is 0): `-0.5` parsed as +0.5. The sign is now
  stripped once and the parsed magnitude negated: identical results for
  every input with a nonzero integer part, correct results for `-0.x`.
- `ternary_to_storage` narrowed tier raws with bare `as` casts: on the
  realtime/compact profiles a Tier-2+ ternary value silently wrapped
  (`0t3281` displayed as `-11.56021` on realtime). Conversion is now
  checked end-to-end and returns `TierOverflow` instead: wrap-defect
  class, same family as the 0.4.28–0.4.31 fixes. Note the asymmetry pinned
  by test: values reached by arithmetic stay at their operand tier
  (0t3280 + 0t1 is fine everywhere), while `from_str` window-gates
  literals upward (a bare `0t3281` literal errors loudly on realtime).

## [0.4.32] - 2026-08-01

### Added

- `g_math::compute_tier` (feature `inference`): public compute-tier
  (tier N+1) transcendentals over raw `ComputeStorage` values at
  2·FRAC_BITS fractional precision: `exp`, `ln`, `sqrt`, `sinhcosh`
  primitives plus `sigmoid`, `softplus`, `ln1p` compositions, with
  `from_fixed`/`to_fixed`/`try_to_fixed` conversions and the `one`/
  `ceiling` constants. These are the same engines every other API path
  uses (results are path-independent with the canonical and imperative
  surfaces (pinned by test)) exposed so wide-precision inference
  consumers no longer re-derive integer-only exp/sigmoid/softplus/ln1p
  on top of the storage-tier API. The format matches the wide-output
  `matvec_q2f` family (0.4.31), so those accumulators feed these
  functions directly with no conversion.
- Contract: `exp` saturates at `ceiling()` and never wraps; `ln`/`sqrt`/
  `ln1p` panic on domain violations; `to_fixed` panics (and
  `try_to_fixed` returns `None`) when a value does not fit storage;
  nothing wraps silently. `sigmoid` and `softplus` use sign-split stable
  forms whose intermediates cannot overflow for any input.
- Validation (`tests/compute_tier_validation.rs`): mpmath 60-digit
  references at q16_16 and q32_32, gated at measured maxima: storage
  level exact (0 LSB) for every function at both profiles; compute-tier
  raw kernel output within 0–4 ULP (primitives) / 1–5 ULP (compositions)
  of the true value. Plus path-independence, saturation, domain-panic,
  and symmetry-identity gates.
- Free `ln_at_compute_tier` kernel wrapper (crate-internal), completing
  the exp/sqrt/sinhcosh free-function set.

## [0.4.31] - 2026-07-22

### Added

- Wide-output `matvec_q2f` family for all four TQ1.9 forms (`TQ19Matrix`,
  `RowScaledTQ19`, `HybridTQ19`, `PlanarTQ19`), plus `tq19_dot_q2f`, gated
  to the q16_16/q32_32 profiles under `inference`. Each returns the exact
  row accumulator at 2·FRAC_BITS fractional precision with exactly one
  rounding (truncation toward zero of `acc·2^FRAC_BITS / SCALE`) instead
  of rounding to storage in the epilogue: for consumers whose signal sits
  below the storage rounding floor (e.g. fine-grained-MoE expert outputs).
  Inner loops, SIMD dispatch, and rayon parallelism are unchanged; zero
  cost on the narrow path.
- Narrowing contract, pinned by property tests on every gated profile:
  `q2f / (1 << FRAC_BITS)` (Rust truncating division) reproduces the
  narrow `matvec` bit-for-bit for `TQ19Matrix`/`HybridTQ19`/`PlanarTQ19`
  (nested truncation toward zero is exact). `RowScaledTQ19::matvec_q2f`
  applies the per-row scale to the wide dot: strictly more precise, so
  narrowing it may differ from the narrow path by ±1 storage LSB for
  non-unit scales (exact for unit scales); its wide value is pinned
  against an independent i128 oracle. Out-of-range wide outputs fail
  loud, never wrap.

## [0.4.30] - 2026-07-22

### Added

- `tq19::RowScaledTQ19` ("TQ1.9-R"): TQ1.9 with one quantization scale per
  row (Maniference O27 contribution). Per-row scales adapt the quantization
  step to each row's own max: measured on Mixtral-8x7B, matvec output error
  drops ~20× and wrong-expert routing drops 6.4% → 0.22% of tokens, at
  unchanged 2 bytes/weight plus one i128 multiply-shift per output element.
  Matvec reuses the existing SIMD `tq19_dot` verbatim. Gated to the q16_16
  and q32_32 profiles (wider profiles would need bigint scale arithmetic).
  Review hardening on merge: the scaled output now fails loud if it exceeds
  the storage range instead of wrapping, and an independent i128-oracle test
  (no shared code with the SIMD path) pins matvec and scale application.

## [0.4.29] - 2026-07-22

Root-cause fix for a latent exp-overflow corruption reported by the
Maniference project (their O26: Mixtral-8x7B expert gates reach ±70 and
tripped a path dense models never reach).

### Fixed

- `downscale_q64_to_q32` (the Q64.64 → compute-tier downscale used by the
  q16_16 and q32_32 exp/trig wrappers) wrapped oversized results via a plain
  `as i64` cast, including the exp overflow sentinel `i128::MAX`. It now
  saturates to `i64::MAX`/`i64::MIN`, so oversized results stay detectable
  and every later storage downscale reports them instead of materializing
  wrapped garbage. Measured pre-fix corruption at Q22.10: `fused::silu(-70)`
  returned `-70` (the gate value passed through unsquashed: a 200× residual
  spike in Maniference's Mixtral run) and `tanh(25)` returned `-1`.
- Ceiling guards for every exp consumer whose follow-up add could wrap (or
  panic in debug builds) on a saturated/sentinel exp:
  `fused::silu` returns 0 (correctly rounded for every such input),
  `FixedPoint::tanh` returns 1 (likewise), `FixedPoint::cosh` and the FASC
  cosh path fail loud (`cosh overflow` / `TierOverflow`), the FASC tanh path
  returns 1, and `sinhcosh_at_compute_tier` pins the cosh sum at the ceiling
  so materialization reports overflow. Softmax paths were audited and are
  safe (max-subtraction bounds exp inputs at 1).
- `tests/tq19_bench.rs` weight generator overflowed i32 at 4096×4096 in
  debug builds (pre-existing, test-infrastructure only).

### Added

- Regression tests: exp monotonicity property over the full storage range
  (the wrap broke monotonicity), silu deep-negative (−30 … −100, all
  profiles), tanh saturation/sentinel region = 1 (all profiles), and FASC
  try_tanh/try_cosh behavior on narrow profiles.

## [0.4.28] - 2026-07-18

### Fixed

- `round_to_storage` (shared downscale for the fused kernels, linalg dot,
  and decompositions): results exceeding the storage tier's range now
  panic (matching the infallible imperative transcendentals) instead of
  silently wrapping via the old shift-and-cast fallback. On narrow
  profiles the wrap produced garbage (e.g. a squared distance of 520,000
  returned as −4288 on Q16.16). Valid-range results are unaffected.
- Fused test suite made multi-profile compliant: test magnitudes fit the
  narrowest profile, and tolerances are representable at Q22.10 (the old
  `0.0001` tolerance rounded to 0 raw, failing identical values) and
  account for input quantization of non-representable decimal literals.

## [0.4.27] - 2026-07-11

### Fixed

- `FixedPoint::try_ln` / `FixedPoint::try_sqrt`: out-of-domain inputs
  (ln(x ≤ 0), sqrt(x < 0)) again return `OverflowDetected::DomainError` as
  documented. Since the v0.4.0 direct-engine-call rewrite these methods
  bypassed the FASC domain checks and misreported out-of-domain input as
  `TierOverflow` (the raw engine's MIN sentinel failing the storage
  downscale). The domain check now runs before the engine call on the
  direct path; valid inputs are unaffected.

## [0.4.26] - 2026-07-11 (unpublished)

The U1 consumer asks from gHyper/gFile (see their ROADMAPs): the fused
no-transcendental kernels that hyperbolic metric trees and Möbius-ratio
distance kernels score with.

### Added

- `fused::euclidean_distance_squared`: Σ (a−b)² at compute tier, no sqrt.
  The no-transcendental half of `euclidean_distance`: squared-space VP-tree
  scoring and Möbius-ratio numerators need only the squared value, and a
  fixed-point sqrt (~15 µs at Q64.64) immediately re-squared is the
  dominant waste in those kernels.
- `fused::dot`: Σ a·b at compute tier; replaces consumers' storage-tier
  hand-rolled accumulators (wrap-prone for large coordinates/dimensions).
- `fused::mobius_denominator_sq`: |1 − p̄q|² = 1 − 2⟨p,q⟩ + |p|²·|q|²
  fused end-to-end (one downscale). With `euclidean_distance_squared` this
  gives consumers the one-sqrt Poincaré kernel: r = √(dist²/den²).

## [0.4.25] - 2026-07-09

Hardening of the trit-plane inference formats and the fused attention op, and a
documentation overhaul to the Geodineum README standard.

> Note: the changelog was not maintained between 0.1.0 and 0.4.24; see the git
> history and `ROADMAP.md` for the intervening milestones (five profiles, TQ1.9,
> decimal transcendentals, fractal router, geometric extension).

### Fixed

- `fused::softmax_mix`: the `Σⱼ eⱼ·vⱼ` numerator and the exp-sum now accumulate
  with overflow detection and return `OverflowDetected::TierOverflow` instead of
  silently wrapping on long-context × large-activation inputs.
- `fused::softmax_mix`: value-row length mismatch is now a hard `assert!` (was a
  `debug_assert!`), so a ragged value matrix cannot silently mix wrong dimensions
  in release builds.

### Added

- `I1024::checked_add`: signed overflow-detecting addition (mirrors
  `I256`/`I512`), enabling overflow-safe compute-tier accumulation on the
  scientific profile.
- `softmax_mix` oracle tests (`tests/fused_ops_validation.rs`): exact-rational
  uniform-mean (long-n, the storage-floor survival property) plus mpmath 60-digit
  references for distinct-scores and near-one-hot mixes, validated on all five
  profiles.
- CI workflow `fused-tq19-precision.yml`: fused oracle and PlanarTQ19/HybridTQ19
  bit-exactness across all five profiles, plus the realtime Q22.10 floor branch.
- Documentation: `README.md` rewritten to the Geodineum README standard; per-layer
  guides under `docs/`; `CONTRACT.md` (integration/precision/determinism contract)
  and `CONTRACT.scn.md` (agent primer); generated `PUBLIC_API.md` with its
  regenerable extractor `scripts/gen-public-api.rs`.

### Changed

- `HybridTQ19` exhaustive split test tightened to the true invariant `hi ∈ [-13, 13]`.

## [0.1.0] - 2026-03-01

Initial open-source release.

### Core

- **FASC** (Fixed-Allocation Stack Computation) pipeline: `LazyExpr` tree builder with operator overloading, thread-local `StackEvaluator` with fixed-size workspace (4KB-64KB)
- **UGOD** (Universal Graceful Overflow Delegation): automatic 6-tier promotion across all domains, with symbolic rational as guaranteed-success fallback
- **Tier N+1** precision strategy: all transcendentals compute one tier above storage, single downscale at materialization
- **BinaryCompute chain persistence**: chained transcendentals stay at compute tier throughout, preventing cumulative precision loss
- **CompactShadow** precision preservation: 0-32 byte exact rational shadow on all non-symbolic values, propagated through arithmetic

### Domains

- **Binary fixed-point**: Q64.64 / Q128.128 / Q256.256 with 18 transcendental functions via tier N+1 computation
- **Decimal fixed-point**: exact base-10 arithmetic (0.1 + 0.2 = 0.3), 6-tier UGOD
- **Symbolic rational**: exact a/b arithmetic with 7-tier storage hierarchy (i8 to I512)
- **Balanced ternary**: base-3 fixed-point with 6-tier UGOD

### Transcendental Functions (18 total)

- **Dedicated algorithms**: exp, ln, sqrt, sin/cos, atan; each with tier N+1 table-driven implementations
- **FASC-composed**: tan, pow, asin, acos, atan2, sinh, cosh, tanh, asinh, acosh, atanh
- **AVX2 SIMD**: Q64.64 multiply hotpath with scalar fallback

### Mode Routing

- 25 compute:output combinations via `set_gmath_mode("binary:decimal")`
- Thread-local `Cell<GmathMode>` for zero-contention mode switching

### Profiles

- `GMATH_PROFILE=embedded`: Q64.64, 19 decimals, scalar
- `GMATH_PROFILE=performance`: Q64.64, 19 decimals, AVX2-optimized
- `GMATH_PROFILE=balanced`: Q128.128, 38 decimals
- `GMATH_PROFILE=scientific`: Q256.256, 77 decimals

### Build System

- Pure-Rust `build.rs` with zero external runtime dependencies
- Algorithmic constant generation: Machin's formula (pi), factorial series (e), continued fractions (sqrt2)
- 3-stage x 1024 entry lookup tables per tier for exp, ln, and trig
- Build cache: skip regeneration when source/profile unchanged

### Validation

- 60,860 arithmetic reference points (mpmath-verified, 4 domains x 4 operations)
- 16,974 transcendental reference points (18 functions x 1,000+ values)
- 288 mode routing test points (12 modes x 24 cases)
- 0 lossy results across all mode combinations

### Cross-Platform

- Bit-identical results across all architectures (x86, ARM, RISC-V)
- Zero floating-point contamination (f32/f64 forbidden in internal logic)
- Consensus-safe for blockchain, financial auditing, scientific reproducibility
