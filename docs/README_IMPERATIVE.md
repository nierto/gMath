# Imperative layer

Direct, `Copy`-semantics fixed-point types: one call is one engine invocation, no
lazy tree and no router dispatch.

## What it is

The imperative layer (`g_math::fixed_point`) is the fast path for code that
already knows its domain. `Fixed<S, F>` (one type per format: `Q16_16`,
`Fixed<i32, 10>`, `Q64_64`, ...), `FixedVector<T>` and `FixedMatrix<T>` are plain
`Copy` value types whose operators map straight onto the binary Q-format engines,
no expression tree is built, and nothing is routed. `DecimalFixed<DECIMALS>` is
the decimal-domain scalar with its own native transcendental engines. Use this
layer in tight numeric kernels and matrix code; use the
[canonical layer](README_ROUTING.md) when you need cross-domain routing or chain
persistence.

> **Scope today: binary-only for vectors/matrices.** `FixedVector`/`FixedMatrix`
> and everything built on them ([linear algebra](README_LINALG.md),
> [geometry](README_GEOMETRY.md)) are binary. `DecimalFixed` covers decimal
> scalars but there is no decimal/ternary/symbolic vector or matrix surface yet:
> extending the imperative layer to every domain is a tracked roadmap item.

## Usage

```rust
use g_math::fixed_point::{FixedMatrix, FixedVector, Q64_64};

type Q = Q64_64;                        // any format: Fixed<i32, 10>, Q256_256, ...
let a = Q::from_str("0.5");
let b = a + Q::from_int(2);             // direct integer arithmetic, no tree
let e = a.exp();                        // direct engine call
let (s, c) = a.sincos();                // fused: one shared range reduction

let v = FixedVector::from_slice(&[a, b, e]);
let n = v.length_fused();               // norm accumulated at the compute tier

let id: FixedMatrix<Q> = FixedMatrix::identity(3);
let m = id.transpose();
assert_eq!(m.mul_vector(&v), v);
```

Decimal scalars, stored exactly in base 10:

```rust
use g_math::fixed_point::DecimalFixed;

let price: DecimalFixed<2> = "19.99".parse().unwrap();  // FromStr, exact in base 10
let rate:  DecimalFixed<2> = "1.08".parse().unwrap();
let taxed = price * rate;                                // decimal-domain multiply (operator)
let root  = "2".parse::<DecimalFixed<9>>().unwrap().sqrt(); // native decimal transcendental
```

Money: a proportional part with one rounding and an explicit tie rule (0.6.5):

```rust
use g_math::fixed_point::{Currency, DecimalRounding};

let total: Currency = "121.00".parse().unwrap();
let rate:  Currency = "21.00".parse().unwrap();           // percent
let base:  Currency = "121.00".parse().unwrap();          // 100 + rate
let vat = total.try_mul_div_with(rate, base, DecimalRounding::HalfUp).unwrap(); // 21.00
```

## Public API

- **[PUBLIC_API.md → Fixed](../PUBLIC_API.md#fixed)**: the `Copy` scalar of
  every format: arithmetic, comparisons, `abs`, `from_str`/`try_from_str`/`from_int`/`from_raw`, all 17
  [transcendentals](README_TRANSCENDENTALS.md) (correctly rounded), `sincos`, `sinhcosh`, float
  conversions for interop (`to_f64`/`to_f32`, `from_f64`/`from_f32` and their
  fallible `try_from_f64`/`try_from_f32`), and a fallible `try_*` variant per
  transcendental.
- **[PUBLIC_API.md → FixedVector](../PUBLIC_API.md#fixedvector)**: `dot`, `length`,
  `length_fused`, `normalized`, `distance_to`, `cross`, `outer_product`, `map`,
  indexing, operators. Dot products accumulate at the compute tier.
- **[PUBLIC_API.md → FixedMatrix](../PUBLIC_API.md#fixedmatrix)**: `identity`,
  `diagonal`, `from_fn`, `from_slice`, `transpose`, `trace`, `row`/`col`,
  mat-mat / mat-vec multiply (compute-tier dots per entry), `kronecker`,
  `submatrix`.
- **[PUBLIC_API.md → DecimalFixed](../PUBLIC_API.md#decimalfixed)**: base-10
  scaled integer with a const-generic decimal-place count; full basic arithmetic
  and all 17 native transcendentals in the decimal domain, plus precision
  conversions and binary interop.

Live signatures on [docs.rs](https://docs.rs/g_math).

## Behaviour & limits

- Results match the [canonical layer](README_ROUTING.md) bit-for-bit for the same
  input and format: both return the correctly rounded value of a single
  function.
- An imperative `.exp().sin()` rounds to storage between the two calls; the
  canonical layer rounds the whole chain once from its exact value
  (`evaluate::<Q>(&gmath("x").exp().sin())`).
- Float conversions are integer operations on the IEEE bits. `to_f64` is exact
  whenever the raw value has at most 53 significant bits (every 32-bit-word
  value; Q32.32 values below 2^21), so `from_f64(x.to_f64()) == x` there; wider values
  round to nearest with ties to even. `to_f32` does the same with 24 bits,
  subnormal or zero below f32's range and infinite above it. `from_f64` /
  `from_f32` truncate toward zero; NaN, infinity and values outside the
  format's range are an error from `try_from_f64` / `try_from_f32` and a panic
  from the infallible forms.
- `from_str` / `try_from_str` convert a decimal literal, with or without
  exponent notation (`"1e-06"`, `".5"`, `"5."`), exactly with integer arithmetic
  and round once to nearest, ties toward +∞, for any number of digits. Other
  literals (hex `0x..` and binary `0b..`, which denote their integer, ternary `0t..`, fractions, repeating decimals, named
  constants) go through the canonical parser as before. `try_from_str` returns
  `Err(ParseError)` or `Err(TierOverflow)`; `from_str` panics there.
- `sincos_wide(angle_q32_32)` and `sincos_wide_q64(angle_q64)` (the 32-bit
  formats and Q32.32) take angles beyond the storage range and round sin and cos once to
  storage; the Q64.64 form is the same computation without truncating the angle
  to Q32.32.
- The operators `+ - * /` and unary `-` on `Fixed` and `DecimalFixed`
  panic when the result leaves the storage range or on division by zero
  (0.6.3 wrapped the binary scalar and saturated `DecimalFixed`); `try_add`,
  `try_sub`, `try_mul`, `try_div`, `try_neg` return `Err(TierOverflow)` or
  `Err(DivisionByZero)` instead. In range the results are unchanged. These
  types have a fixed width and cannot promote; for automatic promotion to a
  wider tier or an exact rational use the [canonical layer](README_ROUTING.md).
  `to_int` / `try_to_int` and `from_int` / `try_from_int` follow the same rule.
- Multi-step routines on these types (solvers, decompositions, integrators,
  manifold and Lie-group maps, curvature) carry their state at the compute tier
  and round once; the closed-form sphere, hyperboloid, SO(3) and SE(3) maps
  are correctly rounded per component; see [the linear algebra
  guide](README_LINALG.md) and [the geometry guide](README_GEOMETRY.md).
- `DecimalFixed` rounds half to even by default. `try_mul_with`,
  `try_div_with`, `try_mul_div_with` and `convert_with_rounding_mode` take a
  `DecimalRounding` (`HalfEven`, or `HalfUp` for ties away from zero). Division
  is one rounding from the exact quotient; `mul_div` is one rounding from the
  exact `a * b / c`, where a product followed by a division would round twice.
- `Fixed<S, F>` is `#[repr(transparent)]` over its raw integer `S`:
  `Fixed::raw_slice` / `from_raw_slice` (and the `_mut` forms) view a
  slice as raw integers or back without copying.
- `FixedVector::rotate_pairs(sin, cos, rotary_dim)` rotates the pairs
  `(v[i], v[i + rotary_dim / 2])` and leaves the rest of the vector untouched;
  each output is rounded once from the exact two-product sum.
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
