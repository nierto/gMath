# Transcendentals

17 functions, available on `LazyExpr`, `Fixed<S, F>` and `DecimalFixed`, each
correctly rounded: computed at tier N+1 (one width above storage), decided at
about 1000 bits when that result lies too close to a rounding boundary, and
rounded once.

## What it is

The transcendental engines are the numerical core. Six are **dedicated**
algorithms (table-driven or Newton-Raphson); the remaining twelve are **composed**
from those six but still evaluated entirely at the wide compute tier, so a
composition rounds once rather than accumulating a rounding per step. The same 17
functions are reachable three ways: lazily via
[`gmath(...)`](README_ROUTING.md), directly on
[`Fixed<S, F>`](README_IMPERATIVE.md), and natively on
[`DecimalFixed`](README_IMPERATIVE.md).

## Usage

```rust
use g_math::fixed_point::Q64_64;

let x = Q64_64::from_str("1.5");
let a = x.exp();
let (s, c) = x.sincos();     // fused: sin and cos from one range reduction
let (sh, ch) = x.sinhcosh(); // fused: sinh and cosh from one exp pair

// Fallible variants surface overflow instead of saturating.
let r = x.try_exp();          // Result<Q64_64, OverflowDetected>
```

```rust
// Shared range reduction through the canonical layer, too.
use g_math::canonical::{gmath, evaluate_sincos};
use g_math::fixed_point::Q64_64;
let (s, c) = evaluate_sincos::<Q64_64>(&gmath("0.7")).unwrap();
```

## The 17 functions

**Dedicated engines** (table-driven or Newton-Raphson, at tier N+1):

| Function | Algorithm |
| -------- | --------- |
| `exp` | integer part by squaring + 3-stage table lookup + Taylor remainder |
| `ln` | multiplicative decomposition, 3-stage tables + Taylor |
| `sqrt` | exact integer square root on words up to 256 bits (correctly rounded by construction); integer Newton-Raphson at Q256.256 |
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
**[Fixed](../PUBLIC_API.md#fixed)**,
**[DecimalFixed](../PUBLIC_API.md#decimalfixed)**, and the canonical
**[LazyExpr](../PUBLIC_API.md#canonical-g_mathcanonical)** methods. Fused pairs are
`sincos`/`sinhcosh` (imperative) and `evaluate_sincos`/`evaluate_sinhcosh`
(canonical). Live signatures on [docs.rs](https://docs.rs/g_math).

**[g_math::compute_tier](../PUBLIC_API.md#compute-tier-transcendentals)**
exposes the tier-N+1 engines directly over a format's compute word
(`Compute<T>`, `2F` fraction bits) (`exp`/`ln`/`sqrt`/`sinhcosh` plus `sigmoid`/`softplus`/`ln1p`
compositions) for wide-precision consumers (e.g. chaining on wide accumulators
without an intermediate storage rounding). These are the raw engines without
the correct-rounding decision: `to_fixed::<T>(compute_tier::exp::<T>(from_fixed(x)))`
equals `x.exp()` except where the engine's result lies within its error of a
rounding boundary.

In every build, **[g_math::wide](../PUBLIC_API.md#wide-tier-q6464)** exposes
`exp_q64`, `ln_q64`, `sin_q64`, `cos_q64` and `sincos_q64` over raw Q64.64
`i128` values, with `PI_Q64`, `TWO_PI_Q64`, `PI_HALF_Q64`, `PI_Q32` and
`TWO_PI_Q32`, for quantities outside every storage tier (RoPE inverse
frequencies `theta^(-2i/d)`). These are the Q64.64 engines that serve as the
compute tier of the 32-bit and Q32.32 formats. Measured
accuracy against mpmath (units of `2^-64`): `exp` 4 relative to the result,
`ln` 55, `sin`/`cos` 3 for `|x| <= 2π`, growing as `0.34 |x|` beyond (range
reduction by the truncated π/2: 1390 below `2^12`, 8e-11 absolute below
`2^32`). `exp_q64` saturates to `i128::MAX` for `x >= 41` and flushes to 0 below
`-40`; `ln_q64` returns `None` for `x <= 0`.

## Behaviour & limits

- On `Fixed` and `DecimalFixed`, every function also has a fallible `try_*`
  variant returning `Result<_, OverflowDetected>`: `DomainError` outside the
  domain, `TierOverflow` out of range, and on `DecimalFixed` `PrecisionLimit`
  when an in-domain argument rounds onto a domain boundary at its compute
  precision (not reachable from storage since the compute word follows
  `DECIMALS` with at least ten guard digits; up to 0.6.7 the Q16.16 and
  Q32.32 builds reached it). The `try_` forms never panic and equal the
  infallible ones whenever those return.
- On `DecimalFixed`, transcendentals run natively in the decimal domain, no
  round-trip through binary. `exp` (and `sinh`, `cosh`, `tanh`) and `sin`/`cos`
  reduce their argument at a working precision wider than the compute tier
  (`x = n ln2 + r`; an exact quadrant count against pi/2 held to twice that
  precision), so they are correctly rounded over their whole range, gated by
  `tests/decimal_exp_range_validation.rs` (0.6.3: `exp(22)` in a Q16.16 build
  was 446565 units off, `exp` past 22.9 wrapped there, and a Q64.64 build lost
  up to 267 units at 19 decimals near the top of the range). Since 0.7.0 the
  compute word follows `DECIMALS` (`Dp<D>: DecimalTier`), not a build profile,
  with a wider sure word when a rounding is ambiguous.
  A result beyond the range is `Err(TierOverflow)` from the `try_` form (every
  `DecimalFixed` transcendental has one, as on `Fixed`) and through the
  canonical layer, and a panic from the infallible method; `exp` far below
  zero is 0.
- On the binary `Fixed<S, F>` formats, every function is correctly rounded
  (nearest, ties toward +infinity) over its whole domain and output range,
  gated by `tests/exp_full_range_validation.rs` against mpmath at every
  format. The engine computes at twice the storage's fraction bits; where its
  result lies too close to a rounding boundary for its error to decide (near
  the top of a format's range, next to the singularities of `tan`, `asin`,
  `atanh`, `acosh`, and for `pow` with a large exponent), a sure path at about
  1000 bits decides it, and trigonometric arguments of 2^24 and above always
  take that path. A canonical tree is rounded once from the exact value of
  the whole tree (0.7.0; 0.6.7 misrounded 10 to 30 of 300 random chains per
  format). Up to 0.6.7 results near the top of the range could be off by one
  unit or more (Q32.32 `exp`; `pow` on every
  format, up to 9,048 units at Q22.10; the Q64.64 `exp` family up to 1.5e10
  units). The infallible `asin`, `acos`, `atanh` and `acosh` panic outside
  their domain (they returned a plausible value up to 0.6.7).
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
