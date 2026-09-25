# Transcendentals

18 functions, available on `LazyExpr`, `FixedPoint`, and `DecimalFixed`, each
computed at tier N+1 (one width above storage) with a single downscale at the end.

## What it is

The transcendental engines are the numerical core. Six are **dedicated**
algorithms (table-driven or Newton-Raphson); the remaining twelve are **composed**
from those six but still evaluated entirely at the wide compute tier, so a
composition rounds once rather than accumulating a rounding per step. The same 18
functions are reachable three ways: lazily via
[`gmath(...)`](README_ROUTING.md), directly on
[`FixedPoint`](README_IMPERATIVE.md), and natively on
[`DecimalFixed`](README_IMPERATIVE.md).

## Usage

```rust
use g_math::fixed_point::FixedPoint;

let x = FixedPoint::from_str("1.5");
let a = x.exp();
let (s, c) = x.sincos();     // fused: sin and cos from one range reduction
let (sh, ch) = x.sinhcosh(); // fused: sinh and cosh from one exp pair

// Fallible variants surface overflow instead of saturating.
let r = x.try_exp();          // Result<FixedPoint, OverflowDetected>
```

```rust
// Shared range reduction through the canonical layer, too.
use g_math::canonical::{gmath, evaluate_sincos};
let (s, c) = evaluate_sincos(&gmath("0.7")).unwrap();
```

## The 18 functions

**Dedicated engines** (table-driven or Newton-Raphson, at tier N+1):

| Function | Algorithm |
| -------- | --------- |
| `exp` | integer part by squaring + 3-stage table lookup + Taylor remainder |
| `ln` | multiplicative decomposition, 3-stage tables + Taylor |
| `sqrt` | integer Newton-Raphson |
| `sin`, `cos` | Cody-Waite range reduction + Horner Taylor (`sincos` fuses both) |
| `atan`, `atan2` | 3-level argument reduction + Taylor |

**Composed** from the dedicated engines, still at the wide tier:

| Function | Composition |
| -------- | ----------- |
| `tan` | sin / cos |
| `pow(x, y)` | exp(y·ln x) |
| `asin`, `acos` | atan(x/√(1−x²)), π/2 − asin |
| `sinh`, `cosh` | (eˣ ∓ e⁻ˣ)/2 (`sinhcosh` fuses both on one exp pair) |
| `tanh` | (e²ˣ−1)/(e²ˣ+1) |
| `asinh`, `acosh`, `atanh` | log forms |

## Public API

Transcendental methods appear on each surface's entry in the index:
**[FixedPoint](../PUBLIC_API.md#fixedpoint)**,
**[DecimalFixed](../PUBLIC_API.md#decimalfixed)**, and the canonical
**[LazyExpr](../PUBLIC_API.md#canonical-g_mathcanonical)** methods. Fused pairs are
`sincos`/`sinhcosh` (imperative) and `evaluate_sincos`/`evaluate_sinhcosh`
(canonical). Live signatures on [docs.rs](https://docs.rs/g_math).

With the `inference` feature, **[g_math::compute_tier](../PUBLIC_API.md#compute-tier-transcendentals)**
exposes the tier-N+1 engines directly over raw `ComputeStorage` values at
2·FRAC_BITS precision (`exp`/`ln`/`sqrt`/`sinhcosh` plus `sigmoid`/`softplus`/`ln1p`
compositions) for wide-precision consumers (e.g. chaining on the wide-output
`matvec_q2f` accumulators without an intermediate storage rounding). Results are
path-independent with the surfaces above: `to_fixed(compute_tier::exp(from_fixed(x)))`
is bit-identical to `x.exp()`.

On every profile, **[g_math::wide](../PUBLIC_API.md#wide-tier-q6464)** exposes
`exp_q64`, `ln_q64`, `sin_q64`, `cos_q64` and `sincos_q64` over raw Q64.64
`i128` values, with `PI_Q64`, `TWO_PI_Q64`, `PI_HALF_Q64`, `PI_Q32` and
`TWO_PI_Q32`, for quantities outside every storage tier (RoPE inverse
frequencies `theta^(-2i/d)`). These are the Q64.64 engines of the realtime and
compact compute tiers, with the same results on every profile. Measured
accuracy against mpmath (units of `2^-64`): `exp` 4 relative to the result,
`ln` 55, `sin`/`cos` 3 for `|x| <= 2π`, growing as `0.34 |x|` beyond (range
reduction by the truncated π/2: 1390 below `2^12`, 8e-11 absolute below
`2^32`). `exp_q64` saturates to `i128::MAX` for `x >= 41` and flushes to 0 below
`-40`; `ln_q64` returns `None` for `x <= 0`.

## Behaviour & limits

- On `FixedPoint` and `DecimalFixed`, every function also has a fallible `try_*`
  variant returning `Result<_, OverflowDetected>`: `DomainError` outside the
  domain, `TierOverflow` out of range, and on `DecimalFixed` `PrecisionLimit`
  when an in-domain argument rounds onto a domain boundary at its compute
  precision. The `try_` forms never panic and equal the infallible ones
  whenever those return.
- On `DecimalFixed`, transcendentals run natively in the decimal domain, no
  round-trip through binary. `exp` (and `sinh`, `cosh`, `tanh`) and `sin`/`cos`
  reduce their argument at a working precision wider than the compute tier
  (`x = n ln2 + r`; an exact quadrant count against pi/2 held to twice that
  precision), so they are correctly rounded over their whole range on every
  profile, gated by `tests/decimal_exp_range_validation.rs` (0.6.3: `exp(22)`
  on realtime was 446565 units off, `exp` past 22.9 wrapped there, and
  embedded lost up to 267 units at 19 decimals near the top of the range).
  A result beyond the range is `Err(TierOverflow)` from the `try_` form (every
  `DecimalFixed` transcendental has one, as on `FixedPoint`) and through the
  canonical layer, and a panic from the infallible method; `exp` far below
  zero is 0.
- Accuracy is defined by the test suite against mpmath references, not by slogans;
  see [the precision guide](README_PRECISION.md) and the validation methodology in
  **[CONTRACT.md](../CONTRACT.md)**. Inputs that are inexact in a *pinned*
  representation (e.g. `0.3` forced into binary) carry representation error into
  the result regardless of the engine's accuracy; the canonical
  [router](README_ROUTING.md) routes such literals to an exact domain
  automatically, and `DecimalFixed` computes decimals natively at 0 ULP.

## Disclaimer

This software is provided **"as is"**, without warranty of any kind, express or
implied. Use of this software is entirely at your own risk. In no event shall the
author or contributors be held liable for any damages arising from the use or
inability to use this software.

---

Built by **Niels Erik Toren** · [support & donations](../README.md#author--support).
