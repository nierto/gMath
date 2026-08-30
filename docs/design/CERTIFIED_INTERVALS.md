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
`exact_product`, and `n` is a compute-tier value. The candidate `k` is an
integer Newton iteration on `n` at the compute tier (seed `2^ceil(bits/2)`
from above, iterate `(k + n/k) / 2` while it decreases; native `isqrt` on the
narrow profiles), which converges in O(log bits) steps and involves no
transcendental engine. The candidate is then verified against the two
inequalities, with at most two corrective steps each way; if the certificate
still fails the function panics. It cannot loop. The upper endpoint is `k` if
`k^2 == n`, otherwise `k + 1`.

The first implementation took its candidate from the Q-format sqrt engine and
walked it one unit at a time to the certificate. That was sound but unbounded,
and on the scientific profile the engine's candidate for inputs near the
storage maximum (`2^510`) was far enough off that the walk did not finish in an
hour. A certificate loop must be bounded; a candidate must not depend on the
accuracy of code the certificate exists to stand independent of. The
independent-reference gate now also asserts that the scalar `FixedPoint::sqrt`
lies inside the certified enclosure for every reference input on every
profile, so an inaccurate engine shows up as a failing test rather than a
hang. It did: the scientific Q512.512 engine was found to lose about 250 bits
of relative precision for large inputs (it ran its reciprocal Newton on the
raw input, where `1/sqrt(x)` and its square are too small for the grid), and
was repaired by normalising the input to `[1, 4)`, certifying the result at
the normalised scale with a bounded exact-integer correction, and shifting
back exactly. A magnitude-ladder gate now holds every profile's scalar sqrt
inside the certified enclosure across its full range. The certified interval
found the defect in the scalar it was built beside; that is the enclosure
doing its job.

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

### 4.1 One narrowing for the whole form (0.6.1)

The consumer's own decomposition of the measured width (dimension 7, 246
records) reproduced the model above to the second decimal: total 3.19 ulp =
2.20 (`sum_i |v_i|`, every row product inexact) + 0.99 (the final pair). The
width was a count of narrowings times a sensitivity, so the lever was the
count. Since 0.6.1 the quadratic form narrows once:

    d2_exact = sum_{i,j} v_i m_ij v_j     every term an exact triple product at 3F
    interval = [floor(d2_exact), ceil(d2_exact)]      width <= 1 ulp, 0 if exact
    scalar   = nearest(d2_exact), ties toward +inf     (fused::quadratic_form)

Each `v_i v_j` is the exact compute-tier product (`2W` bits); off-diagonal
pairs are folded into `(m_ij + m_ji) v_i v_j` at the compute width (no
symmetry assumed), so a form costs `n(n+1)/2` widening multiplies, each
`2W x 2W -> 4W` through the unsigned `mul_to_*` family on magnitudes with the
sign reapplied and the bit budget asserted. The accumulator is the orient3d
accumulator of the profile (section 10): a term is below `3W-2` bits and the
sum below `3W-2+2 log2 n`, against `4W` available. The scalar and the
interval narrow the same integer, so the scalar lies inside the bracket by
construction rather than by measurement.

Exact-integer model (Python, 400 random forms per row, coordinates below 1,
metric entries below 16) before the port: two-stage width mean 2.7 to 2.8 ulp
at n = 7 and 6.7 to 6.8 at n = 23 (max 10) on every profile; fused width
exactly 1 ulp on every inexact form, 0 on exact ones; widest partial sum at
about half the budget. Cost on the embedded profile (release build, 2000
records, min of 3 rounds, ns per record): fused scalar 714 (n = 7) and 6832
(n = 23), 85 to 89 percent of a two-stage scalar built from `fused::dot`;
certified form 719 and 6862, 107 to 122 percent of the 0.6.0 two-stage
interval and 86 to 90 percent of the two-stage scalar. A first port that
multiplied on the full accumulator width measured 141 to 151 percent of the
two-stage interval; routing the product through the compute-width `mul_to_*`
family (4 x 4 words instead of 8 x 8 on I512) is what brought it down.
The two-stage interval remains available by composition (`Interval::dot` per
row, then `dot_intervals`).

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

The fused quadratic form (section 4.1) accumulates at `3F` on the orient3d
accumulator of section 10: i128, I256, I512, I1024, I2048 for the five
profiles in the same order, with the narrowing to storage shifting by `2F`.

All derived from `BinaryStorage` / `ComputeStorage` per `table_format`; no width
is hand-maintained per profile beyond the existing type aliases and the
accumulator table shared with the predicates (`imperative/wide_acc.rs`).

## 9. Positive definiteness by interval Cholesky (`predicates::pd_verdict`)

The factorisation `L L^T = A` is run with `Interval` entries: pivot
`d_i = a_ii - sum_k L_ik^2` and column `L_ji = (a_ji - sum_k L_jk L_ik) / L_ii`,
each an exact compute-tier accumulation (`Interval::dot_intervals`: per term
the smallest corner product to the lower accumulator, the largest to the upper)
narrowed once. By induction on `i`, the exact pivots of the STORED matrix lie in
the pivot intervals. Sylvester's criterion in factored form then gives:

| Pivot interval | Verdict | Status |
|---|---|---|
| `lo > 0` for every pivot | `PositiveDefinite` | proven |
| `hi <= 0` at pivot `i`, earlier pivots `lo > 0` | `NotPositiveDefinite { pivot: i }` | proven |
| `lo <= 0 < hi` at pivot `i` | `Inconclusive { pivot: i, straddle }` | undecided, enclosure returned |

The division `L_ji / L_ii` is always by a certainly-positive interval, so it
never meets the zero-straddle refusal. `sqrt` of a certainly-positive pivot is
certainly positive at every profile (`floor(sqrt(1 ulp * 2^F)) = 2^(F/2) ulp`).

This is the first multi-step chain in the library, where interval entries
feed later intervals. Measured on dyadic `A^T A + I` (every input exact) at
Q64.64: last-pivot width `2.6e-17` (about 480 ulp) at n = 23, `1.2e-15`
(about 21,700 ulp) at n = 50, against pivot values 1.9 and 2.75. Growth is
polynomial in n as the sensitivity model predicts, and sixteen orders of
magnitude below the value at the larger dimension. The gate asserts the
width stays below the pivot value and prints the measurement.

`Inconclusive` is returned rather than escalated to an exact rational
fallback, by owner decision: the consumer chooses (regularise, report rank
deficiency, or escalate itself), and the enclosure tells it how close to
singular the stored matrix is. An escalation behind the `infinite-precision`
gate remains possible if a consumer wants it.

## 10. Exact predicates (`predicates::{orient2d, orient3d, incircle, insphere}`)

Each predicate is the sign of a determinant of fixed degree `d` in the input
coordinates, so its worst-case width is `d * W + c` bits for storage width `W`
(the width budget in the findings repository, `gmath-exact-predicate-width-
budget-2026-08-25`). The accumulator is selected per profile by `cfg` from
that budget:

| Profile | W | `Orient` (2W+2 / 3W+3) | `Circle` (4W+5 / 5W+6) |
|---|---|---|---|
| realtime | 32 | i128 (66 / 99) | I256 (133 / 166) |
| compact | 64 | I256 (130 / 195) | I512 (261 / 326) |
| embedded | 128 | I512 (258 / 387) | I1024 (517 / 646) |
| balanced | 256 | I1024 (514 / 771) | I2048 (1029 / 1286) |
| scientific | 512 | I2048 (1026 / 1539) | not compiled (2053 / 2566 > 2048) |

Arithmetic on the accumulator goes through one private trait: products are
formed on magnitudes with the sign reapplied, so the wide types' `Mul`
implementations are only ever entered with non-negative operands (where each
is a plain schoolbook or a truncated unsigned product, both exact), and every
product first asserts `bits(a) + bits(b) <= BITS - 1`. That assertion is the
width budget made loud; it cannot fire for inputs within the storage range, and
if the budget were ever wrong the predicate would panic rather than return a
wrong sign.

The lifted determinants are expanded along the lift column, so every product
pairs a degree-2 lift with a degree-2 (`incircle`) or degree-3 (`insphere`)
minor; this keeps the largest intermediate within the budget and within the
bit-length precondition on every profile.

Sign conventions follow Shewchuk: `orient2d` positive for a counterclockwise
triangle; `orient3d` positive when `d` lies below the plane of `a b c` seen
counterclockwise from above; `incircle` positive when `d` is inside the circle
of a counterclockwise `a b c`; `insphere` positive when `e` is inside the sphere
of a positively oriented `a b c d`. The circle predicates flip sign with the
orientation of their defining points, as the determinant does.

Why no filter: the interval-filter-then-exact pattern exists to avoid an
expensive exact path. Here the exact path is a handful of wide multiplies with
no allocation, chosen at compile time; a filter would cost more than it saves
for the small predicates. It stays available for the n-by-n case, which is what
`pd_verdict` is.

Why the scientific circle predicates are absent rather than widened: a
predicate returns a `Sign`, so the accumulator never leaves the function and
no downstream width is affected; the cost of widening would be an `I4096`
type, another hand-rolled wide integer, and this cycle showed what those carry
(section 6). The arbitrary-precision type already exists behind the
`infinite-precision` gate and is the right tool for a 2566-bit sign if a
consumer ever needs one at 77 digits.

## 11. Test gate

`tests/interval_enclosure.rs`, all profiles:

- directed rounding: 20,000 products and 20,000 quotients on the narrow
  profiles against i128 floor/ceil models, asserting the interval equals
  `[floor, ceil]`, the nearest scalar lies inside, and the sweep exercised
  inexact and negative cases; constructed ties of both signs on every profile
- enclosure: interval x interval products checked at exact scale on corners and
  interior points; `dot` equals `[floor, ceil]` of the exact sum; the quadratic
  form equals `[floor, ceil]` of the exact value (width at most 1 ulp, 0 exactly
  when representable) and `fused::quadratic_form` equals its nearest with ties
  toward +infinity, on 3000 random forms and on constructed ties of both signs;
  `quadratic_form(v, I) == dot(v, v)` and `fused::quadratic_form(v, I) ==
  fused::dot(v, v)` structurally; typed overflow on both paths; composed chains
  contain the scalar path on every profile
- sqrt: 10,000 sampled inputs with `k^2 <= n < (k+1)^2` checked in u128, ceil
  equal to floor iff exact, scalar inside; perfect-square and monotone pins
- errors: division by an interval containing zero, negative sqrt, inverted
  endpoints, and every storage-boundary overflow return the typed error

`tests/pd_verdict_validation.rs` and `tests/exact_predicates_validation.rs`
gate sections 9 and 10 on every profile; `tests/wide_integer_sign_semantics.rs`
gates the substrate fixes of section 6.

`tests/certified_geometry_refs_validation.rs` is the independent second
opinion: reference values from `scripts/generate_certified_geometry_refs.py`
(Python exact integers and fractions, mpmath at 300 digits as the in-generator
cross-check, no code shared with the Rust side), covering sqrt, product and
quotient endpoints with operands near the storage maximum on every profile,
containment of the exact rational last Cholesky pivot at n = 23 and 50,
predicate signs on configurations scaled to near the storage maximum, the
quadratic form's `[floor, ceil]` and nearest against references computed on
values with exact rationals (mpmath at 700 digits as the cross-check, since
the `3W`-bit sums exceed 300 digits on the scientific profile), and the
decimal endpoints. The other gates use i128 models written alongside the code;
this one does not, which is the point. All of these run on all five profiles
in `.github/workflows/certified-geometry.yml`.
