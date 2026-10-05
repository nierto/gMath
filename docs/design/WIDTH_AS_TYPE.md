# Width as a type: one crate, every precision, no build profile

**STATUS: SPECIFICATION, branch `width-types` (2026-10-05). Nothing implemented.**
Target release: 0.7.0. `main` keeps shipping 0.6.x until this branch passes the
acceptance gates in section 11.

## 1. Problem

Today the binary precision of the whole crate is one build-wide choice:
`GMATH_PROFILE` (or one of the mutually exclusive profile features) becomes
`cfg(table_format = "...")` in `build.rs`, and `GMATH_FRAC_BITS` becomes
`frac_config::FRAC_BITS`. Cargo builds one copy of `g_math` per binary, so every
crate in a dependency graph sees the same width. Two libraries that need
different widths cannot share a binary, and when two profile features are
enabled at once `build.rs` picks one by priority without an error, so one of
them computes at a width it never asked for.

Measured (findings, 2026-10-05): a stripped release binary pays 0.84 MB
(Q16.16) to 2.55 MB (Q256.256) for a representative workload; all five widths
linked together cost 6.3 MB, about the sum of the parts. Size is not the
obstacle. The build-global setting is.

## 2. Goal and non-goals

Goal: the width is part of the type. Any crate can use any supported width, any
number of widths can coexist in one binary, and enabling more widths never
changes a result anywhere.

Hard requirements, each with a gate in section 11:

- **R-BIT** Every operation on `Fixed<S, F>` is bit-identical to the same
  operation in today's crate built with the matching profile (and
  `GMATH_FRAC_BITS = F` for the 32-bit width). Moving to 0.7.0 opens no
  numerics epoch.
- **R-PERF** No operation is slower than in the matching profile build beyond
  measurement noise (criterion, 2 percent band, release, same machine).
- **R-FLEX** Every configuration supported today stays supported: Q(32-F).F for
  F in 2..=30, Q32.32, Q64.64, Q128.128, Q256.256, all transcendentals, FASC,
  decimal, ternary, intervals, predicates, the inference module.
- **R-ADD** Cargo features only add widths. No feature, environment variable or
  build script changes the numeric behaviour of a type that is already enabled.
- **R-STABLE** Stable Rust only. No `generic_const_exprs`.

Non-goals for 0.7.0: new fractional splits on the wide widths (for example
Q88.40 in i128). The design admits them (section 4.3) but each needs its own
mpmath gate, so they come after the migration, one at a time.

## 3. Shape of the public API

```rust
pub struct Fixed<S: Storage, const F: u32> { raw: S }   // #[repr(transparent)]

pub type Q16_16   = Fixed<i32, 16>;     // any Fixed<i32, F>, 2 <= F <= 30
pub type Q32_32   = Fixed<i64, 32>;
pub type Q64_64   = Fixed<i128, 64>;
pub type Q128_128 = Fixed<I256, 128>;
pub type Q256_256 = Fixed<I512, 256>;

pub struct FixedVector<T: FixedNum>  { data: Vec<T> }
pub struct FixedMatrix<T: FixedNum>  { rows: usize, cols: usize, data: Vec<T> }
pub struct Interval<T: FixedNum>     { lo: T, hi: T }
pub struct ComputeMatrix<T: FixedNum> { data: Vec<T::Compute> }
```

A consumer writes `type Real = g_math::Q64_64;` once. Another crate in the same
binary writes `type Real = g_math::Fixed<i32, 10>;`. Code generic over width is
ordinary Rust: `fn norm<T: FixedNum>(v: &FixedVector<T>) -> T`.

There is deliberately no default width. A default is the global knob again under
a new name; the one-line alias is the whole migration cost for a consumer.

The fallible surface is unchanged in shape: operators panic on overflow,
`try_*` twins return `Err` (0.6.4 contract), transcendentals keep their
`try_*` forms and error variants.

## 4. The width traits

### 4.1 `Storage` (sealed): what a raw word can do

Implemented for `i32, i64, i128, I256, I512` (and internally for the compute
and accumulator words `I1024, I2048`). It absorbs what is today spread over
`BinaryStorage` / `ComputeStorage` aliases, the `Wide` trait in
`imperative/wide_acc.rs`, and the near-uniform method surface of the wide
integers.

```rust
pub trait Storage: Copy + Eq + Ord + Send + Sync + 'static + sealed::Sealed {
    const BITS: u32;
    type Compute: Storage;          // i32->i64, i64->i128, i128->I256, I256->I512, I512->I1024
    type Orient: Storage;           // exact predicate accumulators (wide_acc::acc today)
    type Circle;                    // () where the width has no circle accumulator
    fn checked_add(self, o: Self) -> Option<Self>;
    fn checked_sub(self, o: Self) -> Option<Self>;
    fn checked_neg(self) -> Option<Self>;
    fn widen(self) -> Self::Compute;
    fn mul_exact(self, o: Self) -> Self::Compute;
    fn narrow(c: Self::Compute) -> Option<Self>;
    fn is_negative(self) -> bool;
    fn bit_length(self) -> u32;
    // ... the rest of today's Wide surface
}
```

The wide integer types get one uniform surface as part of this work
(`Shr<u32>` vs `Shr<usize>` today; `is_negative` missing on I1024/I2048).

### 4.2 `FixedNum` (sealed): what a width can compute

Implemented once per storage type, generic over `F`:
`impl<const F: u32> FixedNum for Fixed<i32, F>`, and for the four wide widths
with their single admitted `F`.

```rust
pub trait FixedNum: Copy + Ord + Send + Sync + 'static + sealed::Sealed {
    type Storage: Storage;
    type Compute: Storage;
    const FRAC_BITS: u32;
    const COMPUTE_FRAC_BITS: u32;   // 2F
    const DECIMAL_DIGITS: u32;      // today's MAX_DECIMAL_DIGITS
    const TAG: [u8; 2];             // serialization: storage code + F

    // conversion between tiers (today's compute.rs, nearest ties +inf)
    fn upscale(self) -> Self::Compute;
    fn downscale(c: Self::Compute) -> Result<Self, OverflowDetected>;
    fn downscale_floor(c: Self::Compute) -> Result<Self, OverflowDetected>;
    fn downscale_ceil(c: Self::Compute) -> Result<Self, OverflowDetected>;

    // compute-tier kernels (today's *_compute_tier_iN and direct_* wiring)
    fn exp_c(x: Self::Compute) -> Result<Self::Compute, OverflowDetected>;
    fn ln_c(x: Self::Compute) -> Result<Self::Compute, OverflowDetected>;
    fn sqrt_c(x: Self::Compute) -> Result<Self::Compute, OverflowDetected>;
    fn sincos_c(x: Self::Compute) -> (Self::Compute, Self::Compute);
    fn atan_c(x: Self::Compute) -> Self::Compute;
    fn atan2_c(y: Self::Compute, x: Self::Compute) -> Self::Compute;
    fn mul_c(a: Self::Compute, b: Self::Compute) -> Result<Self::Compute, OverflowDetected>;
    fn div_c(a: Self::Compute, b: Self::Compute) -> Result<Self::Compute, OverflowDetected>;
}
```

Everything above the kernels (operators, the 18 transcendentals and their
compositions, fused ops, linear algebra, decompositions, matrix functions,
manifolds, Lie groups, ODE, curvature, tensors, projective, fiber bundles,
intervals) is written once, generic over `T: FixedNum`, from the code that
exists today. The per-width `cfg` arms become the bodies of the five trait
impls.

### 4.3 Admitted fractional splits

`F` is checked at compile time with an inline const in every constructor path:

```rust
impl<S: Storage, const F: u32> Fixed<S, F> {
    const VALID: () = assert!(S::admits_frac_bits(F), "unsupported F for this storage");
}
```

0.7.0 admits exactly today's set: `i32` with 2..=30, `i64` with 32, `i128` with
64, `I256` with 128, `I512` with 256. `Fixed<i128, 40>` is a compile error, not
a runtime surprise. The 32-bit mechanism (upshift by `64 - 2F` into the Q64.64
engine, shift back) generalises to other splits on other storage words, which is
how splits like Q88.40 can be admitted later, each behind its own gate.

## 5. Kernels: one name, one meaning

The survey found the main hazard: today's wrappers `exp_binary_i128`,
`exp_binary_i256` and their siblings change meaning per profile
(`exp_binary_i128` is native Q64.64 on the Q64.64 profile and upscales to
Q128.128 on the Q128.128 profile). Under types, a function name has one meaning.

- The native engines keep their format names and are compiled for every enabled
  width: `exp_q64_64_native`, `exp_q128_128_native`, `exp_q256_256_native`,
  `exp_q512_512_native`, and the same for ln, sqrt, sincos, atan.
- The profile-dependent `*_binary_iN` and `*_compute_tier_iN` wrappers are
  deleted. Each `FixedNum` impl calls the native engine of its compute format
  directly: `Fixed<i32, F>` and `Fixed<i64, 32>` use Q64.64 (with the existing
  upshift for i32), `Q64_64` uses Q128.128, `Q128_128` uses Q256.256,
  `Q256_256` uses Q512.512. That is today's routing, written down once.
- The lookup tables are already compiled for every tier in every profile. With
  width features (section 9) they are included only for enabled widths.

## 6. Constants that depend on F

Associated consts on stable Rust can be computed from `F`
(`const ROUND_BIT: S = 1 << (F - 1)`), but types cannot (no
`generic_const_exprs`). That is enough: every F-dependent quantity today is a
value (`FRAC_ROUND_BIT`, `NATIVE_UPSHIFT`, `MAX_DECIMAL_DIGITS`, convergence
thresholds `2^-(3F/2)`, stagnation shifts, differentiation steps), and every
type that varies is a function of the storage word only (`Compute`, `Orient`,
`Circle`). Shifts by these consts compile to immediates exactly as the
`frac_config` consts do today, which is what R-PERF needs.

## 7. The canonical (FASC) layer

`LazyExpr` becomes `LazyExpr<T: FixedNum>`. Literals stay text until
evaluation, so the width is inferred from where the tree is used:

```rust
let e = gmath("1.5").exp() + gmath("0.1");
let v = evaluate::<Q64_64>(&e)?;          // or: let v: StackValue<Q64_64> = evaluate(&e)?;
```

- `StackValue<T>` carries `T::Storage` / `T::Compute` payloads. The UGOD ladder
  (`BinaryTier1..6`) is already width-independent; the width only sets the top
  tier, which today is a runtime value (`profile_max_binary_tier`).
- The thread-local evaluator cannot be generic (statics are not monomorphised).
  `Storage` gets `fn with_evaluator<R>(f: impl FnOnce(&mut StackEvaluator<Self>) -> R) -> R`,
  implemented per storage word by a small macro, one `thread_local!` each. The
  evaluator's buffer type depends only on the storage word; `F` is a const
  parameter of the evaluation, so all 29 splits of i32 share one buffer.
- `LazyExpr::Value` (feeding a result back) holds `StackValue<T>`, so a Q64.64
  result cannot be fed into a Q16.16 tree by accident; converting is explicit.
- Tier-1 arithmetic follows `F` on the 32-bit width today
  (`binary_multiplication.rs`, `binary_division.rs`); it takes `F` as a const
  parameter instead of reading `frac_config`.
- `DomainMatrix`, `LazyMatrixExpr`, `evaluate_matrix`, `evaluate_sincos` follow
  the same pattern. The `gmath!` macro emits an exact decimal value instead of a
  width-specific `StackValue::Decimal`.
- Mode routing (`set_gmath_mode`) is width-independent and unchanged.

## 8. Other modules

- **DecimalFixed<D>**: arithmetic is already profile-independent (i128 storage,
  D256 intermediates). Transcendentals pick their compute tier from the profile
  today (`DECIMAL_COMPUTE_DP` 9/19/38/77/154), which is why D > 9 is a
  `PrecisionLimit` on the realtime profile. Under types the decimal compute tier
  is a function of `D` alone: the narrowest tier whose working precision covers
  `D` with today's guard digits. Results are correctly rounded today, so for
  every D that some profile serves, the D-selected tier reproduces that
  profile's output; the gate checks it.
- **Interval, predicates**: generic over `T`. `incircle`/`insphere` require
  `T::Storage: Storage<Circle = Acc>`; on Q256.256 they are absent by trait bound
  (a compile error naming the reason) instead of by cfg.
- **compute_tier, wide**: `compute_tier` becomes methods on `T::Compute` through
  `FixedNum`. `wide` (Q64.64 kernels at native precision) stays as is; it was
  already profile-independent.
- **tq19 and the inference module**: today the 32- and 64-bit profiles only, AVX2
  kernels on 32 only. They become generic over `Fixed<i32, F>` and
  `Fixed<i64, 32>`; the AVX2 kernels work on any `F` because they operate on
  raw integers.
- **Serialization**: today's tag says which storage word, not which `F`, so
  bytes written at Q16.16 read back silently as Q22.10. The 0.7.0 format writes
  `[storage code, F]`. Legacy tags stay readable as "storage word, F unknown",
  accepted only by an explicit `from_bytes_legacy::<T>()`.
- **#[repr(transparent)]** on `Fixed`, with a documented guarantee, so
  `&[Fixed<S, F>]` and `&[S]` views are sound (`bytemuck`-style casts without the
  dependency: two `unsafe` fns with the layout argument in their docs).

## 9. Cargo features

`w32`, `w64`, `w128`, `w256`, `w512`, one per storage word, additive. A feature
compiles that width's `Storage`/`FixedNum` impls, engines and tables. Default:
all five. A consumer that wants a small build writes
`default-features = false, features = ["w32"]`. The profile features and
`GMATH_PROFILE` / `GMATH_FRAC_BITS` are removed in 0.7.0; `build.rs` no longer
emits `table_format`. Two crates enabling different widths get both; nothing
changes for either.

Binary size is pay-per-use regardless of features, because the linker drops
unused widths (measured: realtime plus Q64.64 together cost 1.8 MB above std).
Features bound compile time, not binary size.

## 10. Migration plan (each phase green before the next starts)

The trick that keeps every phase verifiable: until phase 7, `FixedPoint`
remains, defined as `Fixed<BinaryStorage, FRAC_BITS>` from the old cfg. The
entire existing test suite therefore keeps running per profile against the new
generic code, while new tests exercise all widths in one build.

| Phase | Work | Exit gate |
|---|---|---|
| 0 | Fixes that belong on `main` regardless (section 12), merged into the branch | main CI green |
| 1 | `Storage` trait, uniform wide-integer surface, engines renamed to format names, `*_binary_iN` meanings frozen | full sweep, 5 profiles + F = 8..24, unchanged |
| 2 | `Fixed<S, F>`, `FixedNum`, operators, conversions, 18 transcendentals; `FixedPoint` aliased | differential gate (11.1) on scalars, all widths |
| 3 | FixedVector/FixedMatrix/ComputeMatrix, linalg, decompositions, matrix functions | differential gate on linear algebra |
| 4 | Geometry layers, ODE, tensors, fused ops, compute_tier, Interval, predicates, tq19 | differential gate, certified-geometry gates |
| 5 | FASC generic (`LazyExpr<T>`, `StackValue<T>`, per-word evaluators), canonical API | differential gate on the FASC corpus |
| 6 | DecimalFixed compute tier by D; serialization v2 | decimal gates per D, legacy-read tests |
| 7 | Remove profiles, `frac_config`, `table_format`; tests generic over width; CI one build per toolchain | full gate set in one build |
| 8 | Docs (README, PUBLIC_API regen, CHANGELOG, CONTRACT, ROADMAP), consumer upgrade notes, 0.7.0 | release checklist |

## 11. Acceptance gates

### 11.1 Differential bit-identity (R-BIT)

`g_math 0.6.4` from crates.io is a renamed dev-dependency (`g_math_064`).
`0.6.4` still reads `GMATH_PROFILE` / `GMATH_FRAC_BITS`; the branch crate
ignores them. So one test binary built with `GMATH_PROFILE=p` compares the old
crate at profile `p` against the new crate's matching type on the same inputs:
every public scalar function, the linear algebra and geometry surface, FASC
expressions and decimal functions, on the existing mpmath corpora plus a seeded
random corpus. Required result: zero differing bits, including identical `Err`
variants. Run for the five profiles and for F = 2..30 on the 32-bit width.

### 11.2 Precision gates

The existing mpmath gates (`fasc_ulp_validation`, `one_rounding_validation`,
decimal, certified geometry, decomposition corpora, wide tier) become generic
over width and run for every width in one build.

### 11.3 Performance (R-PERF)

Criterion benches for the scalar operators, the 18 transcendentals, dot,
matvec, LU, SVD, softmax, tq19 matvec, FASC evaluate, per width, run against
`g_math_064` in the matching profile on the same machine. Any regression beyond
2 percent blocks the phase.

### 11.4 Size

The 2026-10-05 probe (`sizeprobe`, `allprobe`) rerun on the new crate: per-width
cost at most the 0.6.4 profile cost, all-widths cost at most the five-copy
measurement.

## 12. What changes on `main` before the branch lands

Defects and additive APIs that are independent of the width design ship from
`main` as 0.6.x and are merged into this branch, so consumers are not held
hostage to the refactor. Every item below was verified on 0.6.4 (5d0ecb2)
before being listed; the probes live in the 2026-10-05 session scratchpad.

### 12.1 Defects (found or confirmed 2026-10-05; fixed in 0.6.5 unless noted)

0.6.5 is the foundation this branch builds on. D1 to D5, D7 and D8 are fixed
there (gate `tests/defects_065_validation.rs`); the infallible tq19 forms
panic and no `try_` twins were added; D3 keeps `0x05` for Q16.16 and writes
`0x80 | F` for other realtime splits; D6 was resolved by correcting the
documentation, and making the compute-tier decimal intermediates half-even
stays open for 0.7.0 behind an output-difference measurement.

| # | Defect | Evidence | Fix |
|---|---|---|---|
| D1 | `tq19` narrowing wraps silently: `ops.rs narrow_to_storage` and `shift_right_frac_and_narrow` use `as` casts; scalar accumulators use plain `+` | `trit_dot(&[1,1], &[2^30, 2^30]) = -2^31` (true 2^31); `tq19_dot(&[MAX_RAW; 8], &[i32::MAX; 8]) = -436426` | checked narrowing, panic in infallible forms, `try_` twins; breaks the 0.4.28-0.4.31 rule otherwise |
| D2 | `fused::rms_norm_factor_eps_wide` (and the storage-eps form) with negative eps returns `Ok(0)` | `sqrt_binary_i64` returns the sentinel `i64::MIN` for negative input; the caller checks only for zero | `Err(DomainError)`; audit every caller of the sentinel |
| D3 | Serialization tag records the storage word but not `F`: Q16.16 bytes load silently as Q22.10 | `serialization.rs` `PROFILE_TAG` 0x05 for every split | new tag carrying `F`; legacy tag read only when the reader's `F` is 16 or through an explicit legacy call (owner decision: persisted data exists) |
| D4 | Two profile features enabled at once: `build.rs` picks by priority, no error | `build.rs` feature cascade | `compile_error` on conflicting profile features or on env/feature disagreement |
| D5 | Decimal UGOD tier 6 multiply and divide narrow I1024 to I512 with `as_i512` (low words) | `decimal_multiplication.rs`, `decimal_division.rs` | checked narrowing, rational fallback like the binary ladder top; probably unreachable through FASC today, fixed anyway |
| D6 | Docs say decimal rounding is half-even wherever it occurs; decimal compute-tier `mul`/`div`/`div_int` round half away from zero | `decimal_compute.rs` | make them half-even (moves FASC decimal-compute chains by at most one compute-tier unit before the final rounding; measure output change first) or correct the docs; owner decision |
| D7 | `Currency` / `HighPrecisionCurrency` listed public in PUBLIC_API.md but reachable only through a doc-hidden path | `fixed_point/mod.rs` re-exports | re-export at `g_math::fixed_point` |
| D8 | Undocumented `Err` conditions of `fused::softmax`, `softmax_mix`, `rms_norm_factor_eps_wide` | doc comments | state each condition with its width envelope (for example softmax_mix numerator overflow at `n * 2^(31+F) >= 2^63`) |

Minor: `DecimalFixed::convert_with_rounding` has no `try_` twin (panics on
upscale overflow); a whitespace-only decimal string reports `InvalidFormat`
rather than `EmptyString`; the tq19 module docs name a `parallel` feature that
does not exist.

### 12.2 Additive APIs requested by consumers, verified as sound

A1 to A9 shipped in 0.6.5 with the defect fixes (the five-trit matrices on
the realtime profile only; caller-provided output buffers not included). Each
was compared bit for bit with the downstream code it replaces. Under this
design they are ported in phase 4 like the rest of the fused and inference
surface.

| # | API | Why it is sound | Width relevance |
|---|---|---|---|
| A1 | Decimal rounding mode: `RoundingMode` for `DecimalFixed::try_mul_with`, `try_div_with`, `convert_with_rounding_mode`, `mul_div`; default stays half-even, so no existing result moves | tie direction is policy, not arithmetic; the exact quotient is already computed (one rounding), only the tie decision changes; half-away is applied to the magnitude so `f(-x) = -f(x)` survives | none: decimal arithmetic is profile-independent today |
| A2 | `DecimalFixed::mul_div(self, num, den)` with one rounding from the exact `a * b / c` | the two-step form (product at 6 dp, divide, narrow to 2 dp) differs from single rounding on 22 of 1,200,000 VAT cases, all at two-decimal rates, measured in an exact-rational model and on the crate | none |
| A3 | Documented contract: `try_div` rounds once from the exact quotient | verified end to end, including the D256 path | none |
| A4 | `#[repr(transparent)]` on `FixedPoint` and documented `FixedVector` layout, with safe slice views | one non-ZST field; size 4 / align 4 measured on Q22.10; adds a layout commitment, changes no value | carried into `Fixed<S, F>` (section 8) |
| A5 | Bounded unchecked dot: if `len * max|a| * max|b| < 2^(2·BITS-1)` sum without per-term checks, else the checked loop | no partial sum can overflow in any order under the bound; identical integer and identical panic otherwise; measured 3-4x faster per 128-element dot with native codegen | written once over `Storage::BITS` |
| A6 | `fused::softmax_mix` body without the i128 detour, a weights-free variant, a flat-buffer variant; `dot_many` over one key buffer | a port of the requested body matched today's on 3,000 of 3,000 random cases (outputs and weights); `(e*v + 2^9) >> 10` equals `compute_multiply(e, v << 10)` algebraically | generic over `F` |
| A7 | Five-trit row-scaled projection matrices beside `RowScaledTQ19` (i8 code in [-121, 121], Q32.32 row scale, matvec families, AVX2 with scalar fallback, stable byte layout) | same contract shape as `RowScaledTQ19`; independent of storage width except the q2f shift and the AVX2 split for 32-bit activations | belongs to the generic inference module |
| A8 | Exact bit-pattern decoders (bf16, binary16 to fixed raws) and quantisers for the ternary formats | integer shifts only; exhaustive over 65,536 patterns | generic over `F` |
| A9 | Fused RMS norm application (normalised weighted vector, in place and out of place), pair rotation (`rotate_pairs` with a rotary prefix), caller-buffer batch forms | compositions of existing contracted ops with one rounding each | generic over `F` |

Declined or held, with the reason:

- **Importing a non-correctly-rounded kernel bit-identically.** A requested
  Q.40 exp truncates at every step: at most 5 units at Q.40, correctly rounded
  on 2,651 of 4,000 inputs. It cannot sit under this crate's correctly-rounded
  contract. Measured alternative: `floor(wide::exp_q64(x << 24) >> 44)` equals
  the requested kernel's Q.20 output on 10,000,000 of 10,000,000 inputs (and both
  are correct there), so a consumer that only consumes Q.20 can switch without a
  numerics epoch. The sigmoid, softplus and silu compositions at Q.40 were not
  compared yet; each needs the same measurement before a decision.
- **A truncating decimal-literal parser.** The crate parses literals to
  nearest, ties toward +inf (0.6.4, bit-identical to `from_str`). A consumer
  parser that truncates differs on ordinary inputs ("0.0007" at F = 10: 1 vs 0).
  Adopting the crate parser is a numerics epoch for that consumer; the crate
  does not add a second rounding rule for literals.
- **A SiLU saturation request.** Stale: fixed in 0.4.29; `fused::silu` returns 0
  for every input from -20 down to the most negative raw at F = 10 and F = 16,
  and is correctly rounded on a 3,000-sample mpmath check. A consumer-side
  clamp at +-12 is exact only at F = 10 (it disagrees with the exact result on
  321,712 of 8.4 million raws at F = 16) and should be removed rather than
  generalised.
- **A public path for the realtime compute-tier exp.** Already exists:
  `g_math::compute_tier::exp` (feature `inference`) is bit-identical by
  construction and on 200,000 probed inputs.

## 13. How the inference module fits the width design

The inference kernels are where the width design pays off first: they already
combine a 32-bit storage split, a 64-bit compute tier, Q32.32 scales and Q64.64
wide kernels in one consumer, which is a multi-width program written by hand
today. Under section 4:

- Storage-side kernels (A5-A9, tq19, the five-trit format) are generic over
  `Fixed<i32, F>` and, where meaningful, `Fixed<i64, 32>`. The AVX2 paths are
  selected by `T::Storage = i32`, not by profile.
- A wide state tier (for example Q.40 in i128) becomes an admitted split of
  `i128` after 0.7.0, with its own mpmath gate. Until then the measured
  `wide::*_q64` plus explicit narrowing is the supported route.
- Kernels added on `main` before the branch lands are written against
  `BinaryStorage` / `FRAC_BITS` like the code around them, and ported to
  `T: FixedNum` in phase 4 with the differential gate proving the port changed
  no bit.
