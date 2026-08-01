# Balanced ternary & TQ1.9 inference

Base-3 arithmetic with digits {−1, 0, +1}, and the standalone 2-byte ternary
weight format used for zero-multiply neural-network inference.

## What it is

Two related surfaces. The **balanced-ternary domain**
(`g_math::fixed_point::domains::balanced_ternary`) provides base-3 fixed-point
arithmetic across six tier formats (TQ8.8 up to TQ256.256) plus trit packing
(5 balanced trits per byte). **TQ1.9** (`g_math::tq19`, feature `inference`) is a
decoupled, standalone format for neural-network weights: 1 integer trit + 9
fractional trits per weight (2 bytes, range ±1.5). Because weights are {−1, 0, +1}
at the trit level, a dot product needs no multiplications. `PlanarTQ19` and
`HybridTQ19` are lossless re-encodings of a `TQ19Matrix` that trade layout for
fewer weight bytes at bit-identical matvec results. `RowScaledTQ19`
(q16_16/q32_32) carries one quantization scale per row instead of the single
global step, adapting resolution to each row's own max at unchanged
2 bytes/weight.

All four forms also expose a **wide-output matvec** (`matvec_q2f`, `_par`,
`_batch_par`; q16_16/q32_32): the exact row accumulator at 2·FRAC_BITS
fractional precision with exactly one rounding, for consumers whose signal
sits below the storage rounding floor. `q2f / (1 << FRAC_BITS)` (truncating
division) reproduces the narrow `matvec` bit-for-bit for `TQ19Matrix`,
`HybridTQ19`, and `PlanarTQ19`; `RowScaledTQ19` applies its per-row scale to
the wide dot, which is strictly more precise and may differ from the narrow
path by at most 1 storage LSB for non-unit scales.

## Usage

TQ1.9 inference (requires `--features inference`):

```rust
use g_math::tq19::{TQ19Matrix, PlanarTQ19};
use g_math::fixed_point::FixedPoint;

// 2×3 weight matrix; raw TQ1.9 values, |raw| ≤ 29524. 19683 = 3^9 = weight 1.0.
let w = TQ19Matrix::new(2, 3, vec![19683, 0, -19683, 0, 19683, 19683]);
let acts: Vec<_> = [1, 2, 3].iter().map(|&x| FixedPoint::from_int(x).raw()).collect();

let dense = w.matvec(&acts);

// Lossless trit-plane re-encoding: same matvec, fewer weight bytes.
let planar = PlanarTQ19::from_tq19(&w);
assert_eq!(planar.matvec(&acts), dense);
```

## What's here

- **Balanced-ternary arithmetic** — add, subtract, multiply, divide, negate
  (checked and unchecked variants) across six tier formats; `pack_trits` /
  `unpack_trits` store 5 trits per byte. Ternary is also reachable through the
  [canonical layer](README_ROUTING.md) via `0t` literals or
  `set_gmath_mode("...:ternary")`; transcendentals on ternary values route through
  the binary engines.
- **TQ1.9** — `TQ19Matrix` with `matvec` / `matvec_batch` (and rayon `_par`
  variants), the `tq19_dot` / `trit_dot` / `packed_trit_dot` kernels (AVX2 on
  x86_64 with a scalar fallback), and the `PlanarTQ19` / `HybridTQ19` compressed
  weight forms — each bit-identical to dense `TQ19Matrix::matvec` on every profile.

## Public API

See **[PUBLIC_API.md → Balanced ternary](../PUBLIC_API.md#balanced-ternary)** and
**[→ TQ1.9 inference](../PUBLIC_API.md#tq19-inference)**, and
[docs.rs](https://docs.rs/g_math) (build with `--features inference` for the tq19
symbols).

## Behaviour & limits

- The tq19 module is gated behind the `inference` feature; it is standalone and
  does not depend on the routing or imperative layers.
- `TQ19Matrix` raw values must satisfy |raw| ≤ 29524 (the 10-digit balanced-ternary
  range); `PlanarTQ19::from_tq19` rejects out-of-range weights.
- `PlanarTQ19` / `HybridTQ19` are lossless: their matvec equals dense matvec
  bit-for-bit; they change only the weight byte layout.

The exactness contract is covered in **[CONTRACT.md](../CONTRACT.md)**.

## Disclaimer

This software is provided **"as is"**, without warranty of any kind, express or
implied. Use of this software is entirely at your own risk. In no event shall the
author or contributors be held liable for any damages arising from the use or
inability to use this software.

---

Built by **Niels Erik Toren** — [support & donations](../README.md#author--support).
