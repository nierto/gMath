# Balanced ternary

Base-3 arithmetic with digits {−1, 0, +1}.

## What it is

The **balanced-ternary domain**
(`g_math::fixed_point::domains::balanced_ternary`) provides base-3 fixed-point
arithmetic across six tier formats (TQ10.10 up to TQ320.320) plus trit packing
(5 balanced trits per byte).

The ternary weight formats for zero-multiply inference (TQ1.9 and its
compressed and row-scaled forms), the module `g_math::tq19` behind the
`inference` feature until 0.6.7, are the separate crate `g_math_inference`
from 0.7.0.

## What's here

- **Balanced-ternary arithmetic**: add, subtract, multiply, divide, negate
  (checked and unchecked variants) across six tier formats; `pack_trits` /
  `unpack_trits` store 5 trits per byte. Ternary is also reachable through the
  [canonical layer](README_ROUTING.md) via `0t` literals or
  `set_gmath_mode("...:ternary")`; transcendentals on ternary values route through
  the binary engines.

## Public API

See **[PUBLIC_API.md → Balanced ternary](../PUBLIC_API.md#balanced-ternary)** and
[docs.rs](https://docs.rs/g_math).

## Behaviour & limits

- Every infallible ternary operation panics on overflow, at every tier; the
  `_checked` forms return `Err` (the TQ80.80 and TQ320.320 add and subtract
  saturated or wrapped before 0.7.0).

The exactness contract is covered in **[CONTRACT.md](../CONTRACT.md)**.

## Disclaimer

This software is provided **"as is"**, without warranty of any kind, express or
implied. Use of this software is entirely at your own risk. In no event shall the
author or contributors be held liable for any damages arising from the use or
inability to use this software.

---

Built by **Niels Erik Toren** · [support & donations](../README.md#author--support).
