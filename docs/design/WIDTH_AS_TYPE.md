# Width as a type: one crate, every precision, no build profile

**STATUS: IN PROGRESS, branch `width-types`, built on 0.6.5 (c81d3f0).**
Phase 0 done (0.6.5 published). Phases 1 and 2 implemented: see the phase
table in section 10 and the notes under sections 4 and 5. Target release: 0.7.0. `main` keeps shipping 0.6.x until this
branch passes the acceptance gates in section 11.

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

### 4.1 `WideInt`, `Storage`, `CircleStorage` (sealed): what a word can do

Three traits in `src/fixed_point/width.rs` (crate-private until phase 2 makes
the surface public). They absorb what was spread over the `BinaryStorage` /
`ComputeStorage` aliases, the `Wide` trait in `imperative/wide_acc.rs` and the
near-uniform method surface of the wide integers.

```rust
// every integer word: i32, i64, i128, I256, I512, I1024, I2048
pub trait WideInt: Copy + Eq + Ord + Send + Sync + 'static
    + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Neg<Output = Self>
    + sealed::Sealed
{
    const BITS: u32;
    fn zero() -> Self;
    fn one() -> Self;
    fn from_i64(v: i64) -> Self;
    fn is_negative(self) -> bool;
    fn bit_length(self) -> u32;
    fn checked_add_wide(self, o: Self) -> Option<Self>;
    fn checked_sub_wide(self, o: Self) -> Option<Self>;
    fn checked_neg_wide(self) -> Option<Self>;
    fn shl_bits(self, n: u32) -> Self;
    fn shr_bits(self, n: u32) -> Self;
    fn mul_exact(self, o: Self) -> Self;                       // budget asserted
    fn add_exact(self, o: Self) -> Result<Self, OverflowDetected>;
}

// the five words a value can be stored in: i32, i64, i128, I256, I512
pub trait Storage: WideInt {
    type Compute: WideInt;   // i32->i64, i64->i128, i128->I256, I256->I512, I512->I1024
    type Orient: WideInt;    // exact accumulator: i128, I256, I512, I1024, I2048
    fn widen(self) -> Self::Compute;
    fn mul_wide(self, o: Self) -> Self::Compute;               // exact product
    fn narrow(c: Self::Compute) -> Option<Self>;               // checked
    fn orient(self) -> Self::Orient;
    fn widen_product(a: Self::Compute, b: Self::Compute) -> Self::Orient;
}

// the four storage words with a second accumulator (not I512)
pub trait CircleStorage: Storage {
    type Circle: WideInt;
    fn circle(self) -> Self::Circle;
}
```

Two deviations from the first draft of this section, both forced by the
type chain ending at `I2048`:

- The integer surface (`WideInt`) is separate from `Storage`. `I1024` and
  `I2048` are compute and accumulator words only; they have no wider word, so
  they cannot implement a trait with `type Compute`.
- The circle accumulator is its own trait instead of `type Circle = ()` on the
  512-bit word. A bound `S: CircleStorage` states the requirement and the
  compiler names it when it is not met.

`Wide` in `wide_acc.rs` is now a re-export of `WideInt`; `exact_product`,
`widen_storage`, `widen_product` and the `acc` aliases delegate to `Storage`,
so the existing suite exercises the trait impls on every profile. The wide
integers gained the missing methods (`is_negative`, `checked_sub`,
`checked_neg` on `I1024` and `I2048`); the shift operators keep their
historical argument types (`I256 >> u32`, the others `usize`) and
`shl_bits` / `shr_bits` are the uniform form.

### 4.2 `FixedNum` (sealed): the kernels of one format

Implemented once per storage word in `src/fixed_point/fixed_num.rs`:
`impl<const F: u32> FixedNum for Fixed<i32, F>`, and for the four wide
widths with their single admitted `F`. All five impls compile in every build.

The trait works on raw words, not on `Fixed` values, so the profile-bound
layers that have not been made generic yet can call it through
`ProfileFormat` (the format the build profile selects). Its methods are
exactly the functions that used to carry one `cfg` arm per profile:

- between the tiers: `upscale_to_compute`, `downscale_to_storage` (nearest,
  ties toward +infinity), `downscale_to_storage_floor`, `_ceil`;
- storage tier: `one_storage`, `try_from_int`, `try_to_int`, `float_bits`,
  `truncated_raw`, `try_fixed_multiply`, `try_fixed_divide`;
- compute tier: `compute_checked_multiply`, `compute_checked_divide`,
  `make_compute_int`, `compute_mul_div_int`, `compute_div_count`,
  `compute_ceiling`, `exp_sentinel_reached`, and provided methods written
  once over `WideInt` (`compute_add`, `compute_subtract`, `compute_negate`,
  their checked forms, `compute_halve`, `sinhcosh_at_compute_tier`);
- the engine routing, twice: `direct_exp` ... `direct_atan2` and
  `compute_pi_half` as the imperative methods call them, and
  `exp_at_compute_tier` ... `pi_half_at_compute_tier` as the canonical
  evaluator calls them.

Every body was moved from its `cfg` arm by a script that specialises the
old function for one profile, not retyped, and the old free functions now
delegate to the trait. Three facts the move made visible, all preserved
as they were:

- **Two routings.** The imperative and canonical paths name the same engine
  everywhere except atan at Q64.64: the imperative `atan` uses the 55-term
  Q128.128 engine, the canonical path the 33-term one. Their compute-tier
  values differ (the engine digests differ); whether their rounded storage
  values ever differ has not been measured. Unifying them is a numerics
  decision for after 0.7.0, with that measurement first.
- **Two values of pi/2.** `compute_pi_half` (imperative acos/asin) is the
  constant of the format below, upscaled; `pi_half_at_compute_tier` is the
  compute tier's own constant. Same remark.
- **One subtraction form.** The 512-bit width computes `a - b` as
  `a + (-b)`, which panics for `b` at the compute minimum even when the
  difference is representable. It keeps that form (an override in its impl).

The scalar type is generic on top of the kernels
(`src/fixed_point/imperative/fixed_point.rs`):

```rust
impl<S: Storage, const F: u32> Fixed<S, F> where Self: FixedNum<Storage = S> {
    pub const ZERO: Self; pub fn one() -> Self; pub fn from_int(v: i32) -> Self;
    // conversions, the float boundary, operators and their try_ twins,
    // all 18 transcendentals with their try_ twins, sincos, sinhcosh, inv_sqrt
}
pub type FixedPoint = Fixed<BinaryStorage, { frac_config::FRAC_BITS }>;
```

`FixedPoint` is now an alias, so the whole crate above the scalar type and
the whole test suite run on the generic code without a change.

Still bound to the profile's format until phase 5, because they evaluate
through the canonical layer: `from_str` / `try_from_str`, `Display`,
`try_pow`, `try_atan2`. `sincos_wide` and `sincos_wide_q64` exist on
`Fixed<i32, F>` and `Fixed<i64, 32>`, as before.

Resolved in phase 3: the containers keep ONE type parameter with the
scalar as its default, `FixedVector<T = FixedPoint>` and
`FixedMatrix<T = FixedPoint>`, and every impl is written on the concrete
scalar, `impl<S: Storage, const F: u32> FixedVector<Fixed<S, F>> where
Fixed<S, F>: FixedNum<Storage = S>`. User code names a width as
`FixedVector<Q22_10>` and the methods come from that impl; nothing moved
onto a trait. Expression-position constructors whose type is not fixed by
an argument need an annotation (`let id: FixedMatrix = FixedMatrix::identity(n)`),
which a handful of tests needed. The compute-tier state uses
`ComputeMatrix<T: FixedNum = FixedPoint>` the same way.

Phase 4a settled the second half of the question, the user-implemented
traits. `Manifold`, `LieGroup`, `MetricProvider`, `OdeSystem`,
`HamiltonianSystem`, `FiberBundle` and `BundleConnection` take the scalar
as a type parameter with the profile's format as its default,
`pub trait Manifold<T: Scalar = FixedPoint>`, so `impl OdeSystem for Mine`
still compiles and `impl OdeSystem<Q22_10> for Mine` names a width. The
library's own types implement them for every format at once
(`impl<S, const F> Manifold<Fixed<S, F>> for Sphere`). `Scalar` is a sealed
trait implemented by every `Fixed<S, F>`: it carries the operators and
comparisons and, hidden, the three operations the traits' default methods
need (a square root, a matrix inverse, a vector difference), which is what
lets a default body be written against `T` while every inherent method
stays on `Fixed<S, F>`. A method with no scalar in its arguments, such as
`dimension()` on a non-generic manifold, needs the trait named at the call
site (`<Sphere as Manifold>::dimension(&s)`); a handful of tests did.
Per-format constants became functions of the storage width and the split:
the Rodrigues small-angle threshold is 2^F / 10^k rounded to nearest for k
chosen by the word (3, 5, 10, 20), identical to what `from_str` made of
the literals, and the curvature differentiation step is 2^-((F + 1) / 3)
on every format (the five profile arms were that formula).

### 4.3 Admitted fractional splits

`F` is checked at compile time with an inline const in every constructor path:

```rust
impl<S: Storage, const F: u32> Fixed<S, F> {
    const VALID: () = assert!(S::admits_frac_bits(F), "unsupported F for this storage");
}
```

0.7.0 admits exactly today's set: `i32` with 2..=30, `i64` with 32, `i128` with
64, `I256` with 128, `I512` with 256. `Fixed<i128, 40>` is a compile error, not
a runtime surprise.

As built: a combination without a `FixedNum` impl has no methods
(`Fixed<i128, 40>::one()` does not resolve). For the 32-bit word, where the
impl is generic over `F`, the range is asserted in the associated constant
`FRAC_BITS` and every split-dependent kernel reads the split through it, so
`Fixed<i32, 31>` fails at build time with "F must be between 2 and 30" at
its first use. Both are pinned by `compile_fail` doctests on `Fixed`. The
check fires in `cargo build` and `cargo test`; `cargo check` does not
instantiate the constant. The 32-bit mechanism (upshift by `64 - 2F` into the Q64.64
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

Phase 1 result. All 28 engines compile on every profile under one name each:
exp, ln, sqrt, sincos, atan and atan2 at Q64.64, Q128.128, Q256.256 and
Q512.512 (`*_native`), plus `atan_q128_128_compute`, `atan2_q128_128_compute`,
`atan_q256_256_compute` and `atan2_q256_256_compute`. The atan engines really
are two per format: `_native` carries the series terms for the format's own
precision (55 at Q128.128, 105 at Q256.256), `_compute` carries the terms
needed when that format is the compute tier of the storage format half its
width (33 and 65). They return different integers at the compute tier, so
they stay separate functions and each width keeps the one it uses today. The
sin/cos engines had the same duplication but with identical bodies; those
were merged. `engine_identity_tests` (in `transcendental/mod.rs`) pins a
digest of every engine's output on a fixed input set, and the same digests
hold on all five profiles.

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

Refined for the build (2026-10-06, after phase 4), in three sub-phases:

- **5a, the format's arms as a trait.** Every profile arm in the evaluator
  (decimal digit budgets, tier-raw widenings and narrowings, the literal
  parser's ranges, the decimal formatter, the compute-tier conversions of
  decimal values, the integer and halving helpers) becomes a method or
  constant of a crate-private trait `FascFormat: FixedNum`, with one impl
  per format produced by the cfg-specialiser (arms transplanted, not
  retyped); the old functions delegate through `ProfileFormat`. Zero
  behaviour change; gate = the full suite per profile.
- **5b, the typed value and the evaluator registry.** IMPLEMENTED
  (2026-10-06) with one change to the plan above: the evaluator is
  `StackEvaluator<T: FascFormat>` over `StackValue<T>` (the payload is
  `T::Storage` / `Compute<T>` directly), not a `RawValue<S>` buffer with `F`
  threaded through every method as a const parameter; that threading would
  have touched every call in the evaluator for a saving of one buffer per
  split. The thread-local is a registry `Vec<(TypeId, Rc<dyn Any>)>` with a
  one-entry last-used cache in front of it (`with_evaluator::<T>`), so the
  common path is one type comparison and a `RefCell` borrow; measured
  against 0.6.7 (realtime Q22.10, perf3 `fasc`): integer add 194 vs 207 ns,
  decimal mix 468 vs 466, division 204 vs 218, parse 67 vs 78, exp 1372 vs
  1381, display 384 vs 391; without the cache the registry scan cost 13 ns
  (+6 percent on the cheapest trees). `LazyExpr<T>`, `StackValue<T>` and
  `StackEvaluator<T>` default to the profile's format, so every existing
  call compiles unchanged; a result of one format cannot enter a tree of
  another. New entry points `evaluate_in`, `evaluate_sincos_in`,
  `evaluate_sinhcosh_in`, `gmath_parse_in`, `gmath_in`, `with_evaluator`;
  `FascFormat` is public and sealed (its items hidden). Prerequisites done
  in the same step: the decimal compute tier is generic over its compute
  word (`DecimalWord: WideInt` with `Narrow`, `Wide`, `Hp` and the digit
  budgets, one impl per word, every engine `<C: DecimalWord>`, a
  TypeId-keyed thread-local cache for the per-word constants; the profile
  names remain as aliases) and `Storage::Compute: DecimalWord<Narrow =
  Self>`; UGOD tier 1 takes its fraction bits as a parameter
  (`checked_mul_with_frac`, `multiply_with_tier1_frac`), the format
  supplies `TIER1_FRAC_BITS` (`F` on a 32-bit word, 16 elsewhere). The
  seven remaining profile arms (integer literal range, ternary raw, the
  two `to_rational` arms, the compute-tier divide, the decimal-compute
  narrowing, the scientific tier-5 ternary) are `FascFormat` methods. Gates:
  the full suite per profile, `tests/differential_065_validation.rs` FASC
  section (literals, 676 binary operations, 315 transcendental trees:
  value, error and panic identical to 0.6.7 on realtime Q22.10 and
  embedded), `tests/fasc_width_validation.rs` (six formats interleaved in
  one build: exact-domain results equal as rationals, transcendentals within
  4 storage units plus one decimal unit of the widest). Behaviour change:
  `gmath!` integer literals beyond the storage's integer range now fall
  back to the exact rational like the runtime parser (they wrapped);
  `StackEvaluator::new()` takes no profile.
- **5c, the matrix layer and the last scalar methods.** `DomainMatrix<T>`,
  `LazyMatrixExpr<T>`, `evaluate_matrix_in`, `evaluate_sincos_in`; the
  `gmath!` helpers emit exact decimal values; `from_str`, `Display`,
  `try_pow`, `try_atan2` on every `Fixed<S, F>` through `evaluate_in`.

## 8. Other modules

- **DecimalFixed<D>**: arithmetic is already profile-independent (i128 storage,
  D256 intermediates). Transcendentals pick their compute tier from the profile
  today (`DECIMAL_COMPUTE_DP` 9/19/38/77/154), which is why D > 9 is a
  `PrecisionLimit` on the realtime profile. Under types the decimal compute tier
  is a function of `D` alone: the narrowest tier whose working precision covers
  `D` with today's guard digits. Results are correctly rounded today, so for
  every D that some profile serves, the D-selected tier reproduces that
  profile's output; the gate checks it.
- **Geometry, dynamics, tensors** (phase 4a, done): traits generic over
  `T: Scalar`, structs that hold scalars generic with the default
  (`Tensor<T = FixedPoint>`, `OdePoint<T>`, `Moebius<T>`, `VectorBundle<T>`,
  ...), structs that hold only dimensions unchanged (`Sphere`, `SO3`, ...)
  with width-generic methods.
- **Interval, predicates** (phase 4c, done): `Interval<T = FixedPoint>`,
  `PdVerdict<T>`, the predicates generic over the format. `incircle` and
  `insphere` are bounded by `S: CircleStorage`, so on Q256.256 they are
  absent by trait bound (a compile error naming the reason) instead of by
  cfg. The directed division, the certified square root (now on
  `WideInt::isqrt`, Newton on the word types, the native `isqrt` on the
  primitives) and the storage-unit constants are functions of the word
  and the split. The profile-bound accumulator module survives only as
  the reference of its own tests.
- **compute_tier, wide** (phase 4b, done): `compute_tier` keeps its free
  functions, generic over the format: `exp::<T>(x)`, `sigmoid::<T>(x)`,
  `to_fixed::<T>(x)` on `Compute<T>` raws, `from_fixed(x)` inferring the
  format from its argument. `wide` (Q64.64 kernels at native precision)
  stays as is; it was already profile-independent.
- **fused** (phase 4b, done): every kernel generic over `Fixed<S, F>`. The
  word-specific fast paths (the i64 softmax numerators at F <= 15, the u128
  reciprocal root, the AVX2 dot of many keys, the Q64.64 sigmoid and
  entropy gates on the 32- and 64-bit words) are hidden methods of
  `Storage` with `None`/`false` defaults, so a kernel asks the word for its
  fast path and takes the generic body otherwise. They work on raw slices
  and are generic over the split, which keeps the arithmetic of 0.6.7 to
  the bit. A per-element hook returning `Option<Result<..>>` cost
  `rms_norm` 5 to 7 percent (the enum per element again, as with the add
  operator); the hook runs over the slice instead and `rms_norm` is 12
  percent faster than 0.6.7.

- **tq19 and the inference module** (phase 4d, done): activations are
  `&[Fixed<S, F>]` and results `Vec<Fixed<S, F>>` (owner decision: the
  format travels with the data, no turbofish, mixing splits is a compile
  error; raw buffers are viewed through `Fixed::from_raw_slice`). Weights,
  trit codes, Q32.32 scales and the wide outputs stay raw. Two public
  sealed traits in `tq19`: `TqWord` (every word; hidden hooks for the i32
  AVX2 and exact paths and the per-word scale rounding) and `WideOutput`
  (i32 and i64: the wide matvec and the row-scaled forms), so `matvec_q2f`
  and `RowScaledTQ19` are absent on the wide words by trait bound. The
  five-trit form is `Fixed<i32, F>` only (its raw kernels unchanged, the
  wide shift `32 - F`).
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
| 0 | Fixes that belong on `main` regardless (section 12), merged into the branch. DONE: 0.6.5 | main CI green |
| 1 | `Storage` trait, uniform wide-integer surface, engines renamed to format names, `*_binary_iN` meanings frozen. IMPLEMENTED | full sweep, 5 profiles + F = 8..24, unchanged |
| 2 | `Fixed<S, F>`, `FixedNum`, operators, conversions, 18 transcendentals; `FixedPoint` aliased. IMPLEMENTED | differential gate (11.1) on scalars, all widths |
| 3 | FixedVector/FixedMatrix/ComputeMatrix, linalg, decompositions, matrix functions. IMPLEMENTED (3a kernels, 3b containers, 3c compute matrix + decompositions + derived, 3d matrix functions + the 32-bit wide matrix) | differential gate on linear algebra |
| 4 | Geometry layers, ODE, tensors (4a, IMPLEMENTED), fused ops, compute_tier (4b, IMPLEMENTED), Interval, predicates (4c, IMPLEMENTED), tq19 (4d, IMPLEMENTED) | differential gate, certified-geometry gates |
| 5 | FASC generic (`LazyExpr<T>`, `StackValue<T>`, per-format evaluators), canonical API. 5a, 5b IMPLEMENTED; 5c open | differential gate on the FASC corpus |
| 6 | DecimalFixed compute tier by D; serialization v2 | decimal gates per D, legacy-read tests |
| 7 | Remove profiles, `frac_config`, `table_format`; tests generic over width; CI one build per toolchain | full gate set in one build |
| 8 | Docs (README, PUBLIC_API regen, CHANGELOG, CONTRACT, ROADMAP), consumer upgrade notes, 0.7.0 | release checklist |

## 11. Acceptance gates

### 11.1 Differential bit-identity (R-BIT)

`g_math 0.6.5` from crates.io is a renamed dev-dependency (`g_math_065`).
`0.6.5` still reads `GMATH_PROFILE` / `GMATH_FRAC_BITS`; the branch crate
ignores them. So one test binary built with `GMATH_PROFILE=p` compares the old
crate at profile `p` against the new crate's matching type on the same inputs:
every public scalar function, the linear algebra and geometry surface, FASC
expressions and decimal functions, on the existing mpmath corpora plus a seeded
random corpus. Required result: zero differing bits, including identical `Err`
variants. Run for the five profiles and for F = 2..30 on the 32-bit width.

As built for the scalar surface: `tests/differential_065_validation.rs`.
Inputs are raw integers of every magnitude (random bit length, random sign)
plus the extremes; each call's outcome is a value, an error variant or a
panic, and all three must match. It runs with the ordinary suite on every
profile and split. `tests/width_types_validation.rs` is the other half that
only this design allows: every width in one binary, the same generic code
checked for exact identities at ten formats, and each narrower format
compared with Q256.256 (at most one unit apart; measured: exact on 136 to
141 of 141 cases per format).

### 11.2 Precision gates

The existing mpmath gates (`fasc_ulp_validation`, `one_rounding_validation`,
decimal, certified geometry, decomposition corpora, wide tier) become generic
over width and run for every width in one build.

### 11.3 Performance (R-PERF)

Criterion benches for the scalar operators, the 18 transcendentals, dot,
matvec, LU, SVD, softmax, tq19 matvec, FASC evaluate, per width, run against
`g_math_065` in the matching profile on the same machine. Any regression beyond
2 percent blocks the phase.

Measured for the scalar layer after phase 2 (2026-10-06, against crates.io
0.6.7). Method that works: one binary per crate with the same harness, best
of 25 rounds, five alternating process runs, minimum per op. A harness that
links both crates into one binary is unusable at this resolution: code
alignment moved 0.6.7's own numbers by 20 to 60 percent between builds in
which 0.6.7 did not change, and sub-nanosecond operations quantise to whole
cycles. Two real costs were found and removed before parity was reached:

- The 32-bit engine wrappers had become generic functions instantiated in
  the consumer crate with a runtime shift; made generic over the split
  again so the upshift is an immediate (`fixed_num::c32`).
- The IEEE bit assembler and `float_bits` were private and inlined in 0.6.5
  and became cross-crate calls; marked inline (`to_f64` was 2x slower).

Result: transcendentals at parity or faster on realtime, compact and
embedded (sqrt, atan, asin 9 to 18 percent faster on the narrow profiles;
compact `sincos` 47 percent faster, it now uses the fused engine);
`to_f64` at parity or 11 to 27 percent faster; `mul`, `div`, `add`, `sub` at
parity; `from_int` / `to_int` within one cycle per call in either direction.
The inference kernels are measured when their phase lands.

Measured for the containers and linear algebra after phase 3 (2026-10-06,
realtime Q22.10, same method, scratch `perf3/src/bin/lin.rs`): dot of 3,
32 and 1024 elements 7, 13 and 0 percent faster; 64x64 matvec within 1.4
percent; 16x16 matmul 20 percent faster (the generic dot kernel); LU, QR,
Cholesky, Jacobi eigen, SVD, matrix exp and sqrt all within 2 percent
(LU 8x8 read +3.2 percent plain and -0.8 percent under LTO, the
measurement band). Two real costs were found and removed:

- The generic `Add`/`Sub` operators called `try_add(...).expect(...)`; in
  0.6.7 LLVM folded the `Result` to a single overflow branch, in the
  generic instantiation it materialised the enum per element (shift, cmov,
  compare, shift back: 64-element vector add +47 percent under LTO). The
  operators now match on the checked word directly; vector add is at
  parity under LTO and 38 percent faster without it.
- `transpose` through `from_fn` + `get` kept one bounds check per element
  that 0.6.7's codegen had hoisted; written with explicit indexing it is
  25 percent faster without LTO and at parity under LTO. Under LTO the
  harness first read +22 percent, which vanished once the call was
  isolated behind a never-inlined wrapper: inlining context inside the
  harness, not the library.

Phase 3 changed no scalar kernel; the scalar `div` re-read +7.7 percent at
Q22.10 without LTO, +2.8 under LTO, +1.6 at Q16.16 and -0.8 on compact,
which is the band of a 14 ns operation in this method.

### 11.4 Size

The 2026-10-05 probe (`sizeprobe`, `allprobe`) rerun on the new crate: per-width
cost at most the 0.6.5 profile cost, all-widths cost at most the five-copy
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
