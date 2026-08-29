# Certified Intervals: soundness argument, width model, and boundary

**STATUS: IMPLEMENTED 2026-08-29 (unreleased).** `g_math::fixed_point::Interval`
(module `imperative::interval`), gated by `tests/interval_enclosure.rs` on every
profile and in the narrow-profile CI matrix. This document records why the type
is allowed to say "certified", what it covers, what it excludes, and the
measurements the design rests on. The consumer that motivated it is not named
here; the public-crate IP rule applies.

## 1. The claim

An `Interval { lo, hi }` returned by any operation on the type contains the exact
mathematical result of that operation for every choice of operands inside the
input intervals. This is a soundness claim, not an accuracy claim. Its tightness
(how far apart `lo` and `hi` are) is measured and reported separately.

## 2. What the claim rests on

Two mechanisms, and only two.

**Exact products at the compute tier.** For storage values `a` and `b` with raw
representations `a_raw` and `b_raw` at scale `2^F`, the integer `a_raw * b_raw`
is the value `a * b` at scale `2^(2F)`, which is exactly the compute tier's
scale. The library already computes this product exactly (i32 x i32 in i64, i64
x i64 in i128, i128 x i128 in I256 via `mul_i128_to_i256`, and the unsigned
`mul_to_i512` / `mul_to_i1024` on magnitudes with the sign reapplied, the same
pattern the scalar `fixed_multiply` uses). Sums of such products accumulate with
`compute_checked_add`, so an accumulation that would leave the compute tier is a
typed `TierOverflow`, never a wrapped value.

**Directed narrowing.** The one place precision is lost is the narrowing from
compute tier back to storage. The existing `downscale_to_storage` computes
`val >> k` and conditionally adds one on the round bit. An arithmetic right
shift IS floor in two's complement, so `downscale_to_storage_floor` is the same
shift without the bump, and `downscale_to_storage_ceil` is the same shift plus
one whenever any discarded bit is set. Both are fits-checked exactly like the
nearest variant. The lower endpoint always takes floor, the upper always takes
ceil.

Given those two, the endpoint arguments are the standard ones:

| Operation | Endpoints | Why sound |
|---|---|---|
| `a + b` | `[a.lo + b.lo, a.hi + b.hi]` | storage addition is exact when it fits; checked |
| `a - b` | `[a.lo - b.hi, a.hi - b.lo]` | same |
| `-a` | `[-a.hi, -a.lo]` | checked negation |
| `a * b` | floor of the min, ceil of the max, over the four exact corner products | the product of two intervals is bounded by its corners; the extremes are chosen exactly at the compute tier and narrowed once each |
| `a / b`, `0 not in b` | min of the four corner floors, max of the four corner ceils | on the divisor's side of zero the quotient is monotone in each operand; each corner is `(floor, ceil)` from the truncating integer quotient and its remainder |
| `a / b`, `0 in b` | `Err(DivisionByZero)` | no enclosure exists |
| `sqrt(a)`, `a.lo >= 0` | `[floor(sqrt(a.lo)), ceil(sqrt(a.hi))]` | monotone; each endpoint carries the certificate in section 3 |
| `dot(u, v)` | floor and ceil of the exact accumulated sum | one narrowing |
| `quadratic_form(v, M)` | see section 4 | two narrowings |

Why `mul` narrows the extremes rather than each corner: choosing the min and max
among four exact compute-tier integers and narrowing those two is at least as
tight as narrowing all four and choosing among the results, and it is one
directed narrowing per endpoint rather than four.

## 3. The sqrt certificate

In raw units, `sqrt(x)_raw = isqrt(x_raw << F)`, and `n = x_raw << F` is
exactly the compute-tier representation of `x`. The integer square root is the
unique `k` with

    k^2 <= n < (k+1)^2

Both inequalities are checkable in exact compute-tier integers: `k` fits the
storage type (`k < 2^((W-1+F)/2) < 2^(W-1)` on every profile) so `k^2` is an
`exact_product`, and `n` is a compute-tier value. The implementation takes the
engine's tier N+1 result, narrows it by floor to a candidate, then moves the
candidate down while `k^2 > n` and up while `(k+1)^2 <= n`. The loop is the
certificate; the engine only makes it short (a `debug_assert` records that it
never moves more than two units). The upper endpoint is `k` if `k^2 == n`,
otherwise `k + 1`.

This is why sqrt is in the certified set and the transcendentals are not. An
algebraic function's result satisfies a polynomial identity that can be checked
after the fact. A transcendental's cannot, so its accuracy must be argued in
advance by error analysis of the engine.

## 4. The width model, and what was measured

Width is first-order error propagation with absolute values. If the narrowed
quantities are `q_k`, each enclosed to one ulp, then

    width(result) ~ sum_k |d(result) / d(q_k)| * 1 ulp

For the quadratic form `v^T M v`, computed as `(M v)_i` narrowed to one ulp
each and then `sum_i v_i (M v)_i` narrowed once more:

    width ~ (sum_i |v_i| + 1) ulp

Measured on 2026-08-29 at Q64.64, 23 dimensions, 2000 records, coordinates in
`[0, 1)`, a data-derived precision matrix regularised by `1e-6` on the diagonal,
50 held-out probes per case:

| Case | max width | mean width | `d^2` range | enclosure failures |
|---|---|---|---|---|
| well conditioned | 9 ulp | 5 ulp | 11.97 .. 62.43 | 0 |
| rank-deficient (collinear pair, near-constant column; `max |M| = 1/eps` to 13 digits) | 8 ulp | 5 ulp | 12.08 .. 47.99 | 0 |

Relative width `4e-20` to `3e-19`. Ill-conditioning did not widen the enclosure.
The same computation with a narrowing after every elementary operation, which is
how classical interval libraries execute, measured 183 to 212 ulp for the
direct form and 614 to 722 ulp for the whitened form: 12 to 24 times wider. Rounding once per compound operation is what makes the enclosure
tight, and gMath already does that everywhere for tier N+1 reasons.

Cost at the same setting: 106 to 113 percent of the scalar path for the whole
quadratic form. The O(n^2) stage, whose inputs are points and whose accumulator
is exact, measured 100 percent; the O(n) stage, whose inputs are intervals,
measured 186 percent. The textbook "about 2x" applies only to stages whose
inputs are already intervals.

A prediction that did not survive: the design expected the whitened form
`||L^-1 v||^2` (a sum of squares, no subtractive cancellation) to be the usable
one. It measured seven times WIDER than `v^T M v`, because its second stage
squares standardised coordinates (coefficient `2|y_i|`, around 1.6) where the
naive form multiplies by residuals with `|v_i| < 1`. The naive form also did
not cancel: its largest single term was 81 to 86 percent of the sum. Under
one narrowing per compound operation, cancellation inside the accumulator is
free and the rewrite only changes the sensitivities. `quadratic_form` therefore
implements the direct form.

The dependency problem (`x - x` evaluating to `[lo - hi, hi - lo]`) is bounded
by the width it starts from. Its sign-dependent cases need an interval that
straddles a critical point, which a one-ulp interval cannot, so `mul(y, y)` and
`sqr(y)` returned identical widths; its subtractive cases double a width that is
already negligible. Midpoint-radius and affine arithmetic are not needed for
this operation set and are not implemented.

Scope of the certificate: it encloses the arithmetic performed on the STORED
operands. A precision matrix built by a consumer with a rounding per product
carries its own construction error, and no certificate on the scoring path can
see it. The enclosure separates "outside the stored model" from "within the
arithmetic noise of scoring"; it is not an enclosure of a statistical quantity.

Full record with verbatim output and the measurement kernel: the findings
repository, `gmath-interval-width-and-cost-spike-2026-08-29`.

## 5. The boundary: what is deliberately absent

No transcendental interval is provided, not even labelled approximate. gMath's
exp, ln, sin, cos and atan are validated against mpmath at chosen test points to
zero ulp. That is evidence about those points. It is not a proof of a bound
over the domain, and an interval widened by a measured error would carry the
shape of a certificate with none of its content. A too-wide interval is visibly
useless; a too-narrow one is wrong while looking identical, and every consumer
decision built on it inherits the error. gMath retracted zero-ulp claims once
(0.4.23) for exactly this reason. A missing function cannot be misused; a type
name outlives its caveat.

Each transcendental joins the type only with a per-engine analytic error bound
(table entry error, truncation remainder, accumulation across stages, final
rounding) recorded in this document. Kantorovich's theorem gives the classical
route for the Newton-based engines; for sqrt the a posteriori identity made that
unnecessary.

Also absent, as explicit non-goals: IEEE 1788 decorations, reverse operations,
midpoint-radius and affine forms, and `inv_sqrt` (composable from `/` and
`sqrt`; a dedicated one is a future additive item if a consumer needs the
tighter single-narrowing version).

## 6. Where the type sits: imperative tier, binary domain, fixed storage tier

`Interval` is an imperative-tier type, at the same level of the architecture as
`FixedPoint`, `FixedVector`, `FixedMatrix` and the `fused` operations. It is
binary Q-format only, at the profile's storage tier, with intermediates at the
compute tier. It is NOT a FASC domain and NOT UGOD-tiered:

- **Not a FASC domain.** There is no `StackValue::Interval`, no `LazyExpr`
  variant, no routing-table column and no `CompactShadow` for it. A FASC
  interval would need directed rounding inside every domain's arithmetic and
  every cross-domain coercion (decimal to binary, symbolic to binary, ternary
  conversion), each with its own soundness argument, plus a router that knows
  intervals are binary-only. That is a second project with its own design
  document, and no consumer has asked for interval evaluation of mixed-domain
  expressions. It is recorded as not-now, not as never.
- **Not UGOD-tiered.** UGOD promotion lives in the FASC `StackValue` tiers.
  The imperative layer is fixed at the profile's storage tier by design
  (`FixedPoint` has no wider tier to promote into), and `Interval` inherits
  that. What the imperative layer does have is tier N+1 for intermediates,
  and `Interval` uses it fully: every product and every accumulation is
  exact at the compute tier. On storage overflow of a RESULT it returns
  `TierOverflow`, where the scalar `FixedPoint` operators wrap.

Other domains, for the record:

- **Symbolic (exact rational)** needs no interval. Its arithmetic is exact and
  UGOD promotes instead of rounding, so the enclosure of any result is the
  point `[x, x]` by construction. Nothing to build.
- **Decimal (`DecimalFixed<D>`)**: delivered the same day as
  `DecimalInterval<D>`. The same design with `10^D` in place of `2^F`: exact
  products at `2D` places in the decimal-domain `D256`, directed narrowing
  from the truncating quotient and whether its remainder is zero (the sign of
  the exact quotient comes from the operands, so the remainder's sign
  convention is never consulted), the same sqrt certificate on
  `isqrt(x_raw * 10^D)` with an integer Newton candidate. One code path on
  every profile. `divmod_d256_by_i128` saturates when the quotient leaves
  i128, so the interval uses `divmod_d256_by_d256` and checks the fit itself.
  Where the scalar `DecimalFixed` saturates (overflow, division by zero) the
  interval returns a typed error. No `quadratic_form`: there is no decimal
  matrix type. Gate: `tests/decimal_interval_enclosure.rs`.

  Building it found two pre-existing silent-wrong-value bugs in the wide
  integer types, fixed and gated by `tests/wide_integer_sign_semantics.rs`:
  `D256`/`D512` subtraction never propagated a borrow, and `Ord` on `I1024`,
  `I2048`, `D256`, `D512` ordered two negatives backwards. Every gate added in
  this track has found a real bug in code it merely depended on; that is the
  recurring lesson of the whole 0.5.0 cycle, and it argues for building the
  next gate before the next feature.
- **Ternary** has no imperative host type (the domain is FASC-only over
  engine functions), so a ternary interval arrives with the FASC-level
  interval, not before.

The consumer's scoring path is imperative (`quadratic_form` over
`FixedMatrix` / `FixedVector`), so the certified version has to sit where the
scalar version sits or it cannot be swapped in. That, and the simplicity of the
soundness argument when the narrowing points are explicit, decided the level.

## 7. Overflow is typed, never wrapped

Every endpoint operation goes through checked storage arithmetic and returns
`TierOverflow` on the storage boundary; `downscale_to_storage_ceil` checks its
bump as well, since a value whose floor is the storage maximum has no ceiling.
The infallible operators (`+ - * /`, `Neg`) panic where the `try_*` twins
return the error. This is stricter than the scalar `FixedPoint` operators, which
wrap on the narrow profiles; an enclosure that wraps is worse than no enclosure.

## 8. Per-profile accumulator facts used above

| Profile | storage W | F | exact product type | sqrt certificate `k` bound | `k^2` type |
|---|---|---|---|---|---|
| realtime Q16.16 | 32 | 16 (configurable) | i64 | `< 2^24` | i64 |
| compact Q32.32 | 64 | 32 | i128 | `< 2^48` | i128 |
| embedded Q64.64 | 128 | 64 | I256 | `< 2^96` | I256 |
| balanced Q128.128 | 256 | 128 | I512 | `< 2^192` | I512 |
| scientific Q256.256 | 512 | 256 | I1024 | `< 2^384` | I1024 |

All derived from `BinaryStorage` / `ComputeStorage` per `table_format`; no width
is hand-maintained per profile beyond the existing type aliases.

## 9. Test gate

`tests/interval_enclosure.rs`, all profiles:

- directed rounding: 20,000 products and 20,000 quotients on the narrow
  profiles against i128 floor/ceil models, asserting the interval equals
  `[floor, ceil]`, the nearest scalar lies inside, and the sweep exercised
  inexact and negative cases; constructed ties of both signs on every profile
- enclosure: interval x interval products checked at exact scale on corners and
  interior points; `dot` equals `[floor, ceil]` of the exact sum; the quadratic
  form encloses the exact value with width within the two-narrowing bound;
  `quadratic_form(v, I) == dot(v, v)` structurally; composed chains contain the
  scalar path on every profile
- sqrt: 10,000 sampled inputs with `k^2 <= n < (k+1)^2` checked in u128, ceil
  equal to floor iff exact, scalar inside; perfect-square and monotone pins
- errors: division by an interval containing zero, negative sqrt, inverted
  endpoints, and every storage-boundary overflow return the typed error
