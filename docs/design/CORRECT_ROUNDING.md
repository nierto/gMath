# Correct rounding for chains and geometry

Status: implemented on branch `correct-rounding-chains` (2026-10-15; section 7
records what was built and measured). Owner request: "We want to fix the chains too
... using tier+1 where possible so there is no intermediate rounding", and
geometry "cut to 0". Builds on the full-range accuracy work (branch
`exp-full-range`): every single binary transcendental is correctly rounded
through a fast engine plus a sure path at about 1000 bits.

## 1. What is being promised

- A canonical expression tree whose result is a binary value is the
  correctly rounded value of the exact mathematical expression of its exact
  leaves: nearest, ties toward +infinity, one rounding. Leaves are exact:
  a decimal literal is its rational value, a binary value its raw, a named
  constant (pi, e, ...) the real number.
- The same for decimal results (`DecimalCompute` chains), nearest, ties to
  even (the decimal domain's rule).
- Every geometric primitive (manifold distance, exp/log maps, parallel
  transport, Lie group exp/log, curvature quantities, projective maps, ODE
  steps as defined functions of their inputs) returns each component
  correctly rounded from the exact formula on its stored inputs. "0" in the
  differential gate then means: equal to the correctly rounded value, with
  no tolerance.

What is not promised: a result that is exactly a tie at storage precision
and whose exactness cannot be shown within 1,900 bits is rounded as a tie
(toward +infinity). For algebraic and transcendental functions of rational
inputs this only happens when the value is exactly the tie.

## 2. Method: tracked fast pass, sure pass on demand (Ziv)

One evaluator, written once, generic over a numeric backend:

- `Compute`: the format's compute tier (2F fraction bits, the existing
  engines, chain persistence, no intermediate rounding to storage);
- `Big`: a binary floating value, mantissa on I2048 at a chosen precision
  (1,000 bits, then 1,900), exponent i64, with the sure algorithms of
  `transcendental/sure.rs` lifted to it.

Every value carries an error bound `2^b` in units of the backend's last
place, propagated by first-order rules (integer bit arithmetic, no floats):

| node | bound of the result |
|---|---|
| leaf | 0 if exact, else 1/2 unit |
| a + b, a - b | ea + eb + 1/2 |
| a * b | abs(a) eb + abs(b) ea + ea eb + 1/2 |
| a / b | (ea + abs(a/b) eb) / (abs(b) - eb) + 1/2 (undecided if abs(b) <= eb) |
| exp x | e^x (ex + engine) |
| ln x | ex / (abs(x) - ex) + engine |
| sqrt x | ex / (2 sqrt(x - ex)) + engine |
| sin, cos, atan, tanh, asinh | ex + engine (derivative at most 1) |
| tan x | (1 + tan^2) ex + engine |
| sinh, cosh | e^abs(x) ex + engine |
| asin, acos | ex / sqrt(1 - (abs(x) + ex)^2) + engine |
| acosh | ex / sqrt((x - ex)^2 - 1) + engine |
| atanh | ex / (1 - (abs(x) + ex)^2) + engine |
| pow(x, y) | as exp(y ln x) |

`engine` is each engine's measured maximum error at its backend
(Compute: from a corpus against the sure path, recorded per function and
format with a factor 4 of headroom; Big: the series truncation and rounding
bounds of the sure algorithms, a few units of 2^-p relative).

At the root: if the discarded bits lie farther than the bound from a
rounding boundary, the rounding is decided. Otherwise the tree is
evaluated again on `Big` at 1,000 bits, then 1,900 bits. Trigonometric
arguments beyond 2^24 and any bound that cannot be formed (division by a
value not separated from zero) send the evaluation straight to `Big`.

## 3. Big: the sure backend as a floating type

`BigFloat { neg, mant: I2048 (normalised, top bit at p - 1), exp: i64 }`.
Exact conversions from raws and decimal rationals (one rounded division);
add/sub with a sticky bit, mul through a 4096-bit product split into two
2048-bit halves (schoolbook on words), div by long division at p bits, sqrt
by integer square root, and exp, ln, sin/cos (reduction against pi held to
`p + exponent` bits, computed once per precision by Machin), atan (half-angle
steps plus series), the rest composed. Each function returns its value with
its error bound in units of 2^-p relative.

## 4. Where it plugs in

- Canonical: `evaluate` runs the existing evaluator (routing, exact
  domains, UGOD, the output mode); when the result is binary or decimal and
  came through a compute tier (a transcendental or an inexact division in
  the tree), it is replaced by the tracked evaluation of the same tree,
  rounded once. Exact-domain results (decimal and symbolic arithmetic) are
  untouched: they are already exact.
- Imperative single functions: the empirical margins of `Fixed::settle`
  become the Compute backend's engine bounds (same mechanism, documented
  constants).
- Geometry: each primitive's formula is written over the backend trait
  (vectors and matrices of backend values); the imperative function runs it
  on Compute, checks every component's bound, and re-runs on Big for the
  components that are undecided.

## 5. Phases and gates

| Phase | Work | Gate |
|---|---|---|
| CR1 | BigFloat and its functions, bounds | unit pins, every function vs mpmath at 1,000 bits on a corpus |
| CR2 | tracked evaluator over Compute and Big, canonical binary chains | random trees (depth 2 to 4 over the 17 functions, + - * /, decimal and binary leaves, pi and e) vs mpmath at seven formats, plus stress seeds |
| CR3 | decimal chains | the same trees with decimal output at D = 0..38 |
| CR4 | engine bounds measured and recorded; `settle` on them | the full-range gate and its stress seeds |
| CR5 | geometry primitives over the backend | per-primitive mpmath corpora at seven formats; differential gate without tolerance (new == correctly rounded) |
| CR6 | performance | fast-path cost within 5 percent of today where decided; sure-pass frequency and cost reported per function and format |

Each phase: the ten-build sweep before its commit, findings and docs in the
same cycle.

## 6. Risks recorded up front

- The engine bounds are measured, not proven. A miss outside the measured
  maximum by more than the factor 4 would misround silently; the stress
  corpora are the evidence, and the bounds are re-measured whenever an engine
  changes.
- Deep trees with cancellation can leave the Compute bound wide, so the
  sure pass runs often for them; that is a speed cost, not an accuracy one.
- The 4096-bit product is new code on the sure path; it gets its own exact
  unit tests (against products checked by the I2048 division).

## 7. As built (2026-10-15)

- `transcendental/mpf.rs`: `Mpf<C>`, the precision a capacity type (`C2` 128
  bits to `C65` 4160 bits), limbs on the stack (Knuth division, integer
  roots), bounds as a 32-bit mantissa and an exponent rounded up (a log2-only
  bound lost a bit per accumulation). Functions are written with the tracked
  operations, so their rounding bounds are derived; series truncation, the
  propagated input error and the ln start (one exp and `ln1p` of the
  residual, no assumption on the start) are added explicitly. Constants (ln 2,
  pi by Machin, Euler's gamma by Brent-McMillan) per capacity, once per
  thread. The 1000-bit `big.rs` stays as the test oracle.
- `fixed_point::exact`: the trait `Num` (primitives, composed functions with
  defaults), `Fast` (the compute tier with the imperative ladder's margins,
  calibrated per format on compute-precision arguments: `rel` 1 on the
  realtime word, 6 on wider words), and `settle` for formulas with several
  outputs.
- Canonical evaluator: `settle_tree` rounds every tree with an operation once
  (binary results to storage; decimal results at their storage places, half
  to even, the compute value kept when it already reads correctly); domain
  and range errors only from `Mpf`; where nothing decides, the compute-tier
  result stands.
- Geometry: Sphere and HyperbolicSpace (distance, log, exp, transport),
  SO(3) and SE(3) (exp, log, exp_map, log_map, distance, SO(3) transport) as
  formulas; degenerate inputs decided exactly on the compute tier first. The
  projective maps and tensor contractions were already exact sums rounded
  once (now gated). Not covered: SPD manifold and SO(n) (matrix functions,
  within one unit), curvature and ODE steps (user closures, finite
  differences: no exact target).
- Gates (no tolerance): chains 2,100 + 3 x 7,000, decimal chains 1,400 +
  14,000, geometry 9,660 + 32,200, engine-bound calibration 66,000, Mpf
  against the oracle at three capacities.
