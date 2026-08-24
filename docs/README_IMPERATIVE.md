# Imperative layer

Direct, `Copy`-semantics fixed-point types: one call is one engine invocation, no
lazy tree and no router dispatch.

## What it is

The imperative layer (`g_math::fixed_point`) is the fast path for code that
already knows its domain. `FixedPoint`, `FixedVector`, and `FixedMatrix` are plain
`Copy` value types whose operators map straight onto the binary Q-format engines,
no expression tree is built, and nothing is routed. `DecimalFixed<DECIMALS>` is
the decimal-domain scalar with its own native transcendental engines. Use this
layer in tight numeric kernels, matrix code, and ML inference; use the
[canonical layer](README_ROUTING.md) when you need cross-domain routing or chain
persistence.

> **Scope today: binary-only for vectors/matrices.** `FixedVector`/`FixedMatrix`
> and everything built on them ([linear algebra](README_LINALG.md),
> [geometry](README_GEOMETRY.md)) are binary. `DecimalFixed` covers decimal
> scalars but there is no decimal/ternary/symbolic vector or matrix surface yet:
> extending the imperative layer to every domain is a tracked roadmap item.

## Usage

```rust
use g_math::fixed_point::{FixedPoint, FixedVector, FixedMatrix};

let a = FixedPoint::from_str("0.5");
let b = a + FixedPoint::from_int(2);   // direct integer arithmetic, no tree
let e = a.exp();                        // direct engine call
let (s, c) = a.sincos();                // fused: one shared range reduction

let v = FixedVector::from_slice(&[a, b, e]);
let n = v.length_fused();               // norm accumulated at the compute tier

let id = FixedMatrix::identity(3);
let m = id.transpose();
```

Decimal scalars, stored exactly in base 10:

```rust
use g_math::fixed_point::DecimalFixed;

let price: DecimalFixed<2> = "19.99".parse().unwrap();  // FromStr, exact in base 10
let rate:  DecimalFixed<2> = "1.08".parse().unwrap();
let taxed = price * rate;                                // decimal-domain multiply (operator)
let root  = "2".parse::<DecimalFixed<9>>().unwrap().sqrt(); // native decimal transcendental
```

## Public API

- **[PUBLIC_API.md → FixedPoint](../PUBLIC_API.md#fixedpoint)**: `Copy` Q-format
  scalar: arithmetic, comparisons, `abs`, `from_str`/`from_int`/`from_raw`, all 18
  [transcendentals](README_TRANSCENDENTALS.md), `sincos`, `sinhcosh`, float
  conversions for interop, and a fallible `try_*` variant per transcendental.
- **[PUBLIC_API.md → FixedVector](../PUBLIC_API.md#fixedvector)**: `dot`, `length`,
  `length_fused`, `normalized`, `distance_to`, `cross`, `outer_product`, `map`,
  indexing, operators. Dot products accumulate at the compute tier.
- **[PUBLIC_API.md → FixedMatrix](../PUBLIC_API.md#fixedmatrix)**: `identity`,
  `diagonal`, `from_fn`, `from_slice`, `transpose`, `trace`, `row`/`col`,
  mat-mat / mat-vec multiply (compute-tier dots per entry), `kronecker`,
  `submatrix`.
- **[PUBLIC_API.md → DecimalFixed](../PUBLIC_API.md#decimalfixed)**: base-10
  scaled integer with a const-generic decimal-place count; full basic arithmetic
  and all 18 native transcendentals in the decimal domain, plus precision
  conversions and binary interop.

Live signatures on [docs.rs](https://docs.rs/g_math).

## Behaviour & limits

- Results match the [canonical layer](README_ROUTING.md) bit-for-bit for the same
  input and precision: the two paths share the compute-tier engines.
- A single imperative `.exp().sin()` materializes between the two calls; for chain
  persistence across transcendentals use the canonical layer
  (`gmath("x").exp().sin()`).
- `DecimalFixed` computes natively in the decimal domain (no binary round-trip),
  so its results are correctly rounded *in decimal*; see
  [the precision guide](README_PRECISION.md).

Rounding and determinism guarantees are in **[CONTRACT.md](../CONTRACT.md)**.

## Disclaimer

This software is provided **"as is"**, without warranty of any kind, express or
implied. Use of this software is entirely at your own risk. In no event shall the
author or contributors be held liable for any damages arising from the use or
inability to use this software.

---

Built by **Niels Erik Toren** · [support & donations](../README.md#author--support).
