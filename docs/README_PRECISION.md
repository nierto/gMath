# Formats, precision & build

The formats, their storage and compute words, the rounding rules per domain,
how accuracy is validated, and how to build.

## Formats

A format is a type, `Fixed<S, F>`: a storage word `S` and `F` fraction bits.
Every format is compiled into every build and any number of them can be used
in one program (0.6.x chose one per build through `GMATH_PROFILE`; see
[UPGRADING_0.7.md](UPGRADING_0.7.md)).

| Type | Storage | Compute | Fraction bits | ~Decimal digits | Range |
| ---- | ------- | ------- | ------------- | --------------- | ----- |
| `Fixed<i32, F>`, e.g. `Q16_16` | i32 | i64 | any F in 2..=30 | 0.3 F | ±2^(31 - F) |
| `Q32_32` = `Fixed<i64, 32>` | i64 | i128 | 32 | 9 | ±2^31 |
| `Q64_64` = `Fixed<i128, 64>` | i128 | I256 | 64 | 19 | ±2^63 |
| `Q128_128` = `Fixed<I256, 128>` | I256 | I512 | 128 | 38 | ±2^127 |
| `Q256_256` = `Fixed<I512, 256>` | I512 | I1024 | 256 | 77 | ±2^255 |

The 32-bit word moves its binary point anywhere from Q30.2 to Q2.30:
`Fixed<i32, 10>` (Q22.10, range about ±2 million, 3 digits) and
`Fixed<i32, 24>` (Q8.24, range ±128, 7 digits) are as available as Q16.16.
Generic code takes the format as a parameter (`T: FixedNum`, or
`Fixed<S, F>` with `S: Storage`).

```rust
use g_math::fixed_point::{Fixed, Q16_16, Q256_256};

let a = Fixed::<i32, 10>::from_str("1234.5");   // Q22.10
let b = Q16_16::from_str("1234.5");
let c = Q256_256::from_str("1234.5");
assert_eq!(a.raw(), 1_264_128);                   // 1234.5 * 2^10, exact
assert_eq!(b.raw(), 80_904_192);                  // 1234.5 * 2^16, exact
assert!(c.sqrt().to_string().starts_with("35.1354521815"));
```

A clean release build compiles every format in about 35 seconds.

Lookup tables are checked in, so a default build is fast. `--features rebuild-tables`
regenerates them from `build.rs` (pure-Rust generation: π via Machin's formula,
e via factorial series, √2 via continued fractions, zero runtime dependencies)
and is slow.

## Rounding

Rounding is defined per domain, and it is deterministic (integer arithmetic, so
identical on every platform). The key property: everything beyond a lone
storage-tier multiply or divide (every transcendental, dot product,
decomposition, matrix chain, and fused op) computes at tier N+1 with double the
fractional bits and rounds to storage exactly once, so the wide→storage downscale
is the only rounding the result sees. The normative per-domain rounding table and
the full explanation are in the rounding contract,
**[CONTRACT.md §3](../CONTRACT.md#3-rounding-contract)**. The certified
interval types are the deliberate exception: they round each endpoint
outward (toward −∞ below, +∞ above) so that the bracket provably contains
the exact result; the scalar paths keep the per-domain rules and always
land inside the corresponding interval.

## Validation

Accuracy is defined by the test suite, not slogans:

- Reference values are generated with mpmath at 50–250 digit precision and
  embedded in the tests as exact strings, never computed with floats.
- Transcendentals are validated pointwise against those references at seven
  formats (Q22.10, Q16.16, Q8.24, Q32.32, Q64.64, Q128.128, Q256.256, and the
  32-bit splits 8, 12, 20 in CI), over the whole range including the top,
  with no tolerance: every result is the correctly rounded value. A fast
  engine decides when its result is clear of a rounding boundary by more
  than its measured error bound, and a sure path at about 1000 bits decides
  otherwise.
- Canonical chains (random expression trees of the 17 functions and the four
  operations) and the closed-form geometry maps are gated the same way.
- Linear algebra, manifolds, Lie groups, ODE, and tensor tests combine structural
  checks (PA=LU, QᵀQ=I, exp/log roundtrips) with concrete mpmath-validated
  numerical comparisons.
- Both user-reachable decimal paths (imperative `DecimalFixed` and canonical
  `gmath()`) are graded against [mootable/decimal-scaled](https://github.com/mootable/decimal-scaled)'s
  independent adversarial mpmath corpus in CI.

```bash
cargo test --release
```

## Limits worth knowing

- **Input representation.** Within a single fixed representation some values
  aren't exact: `0.3` and `1/3` repeat in binary, so the binary domain
  approximates them (up to half an ULP before any computation). The canonical
  `gmath(...)` router sidesteps this by classifying each literal and routing it to
  a domain where it *is* exact (decimal for `0.3`, symbolic for `1/3`) and
  preserving the exact value in a shadow; the limit applies only when the binary
  domain is pinned (the imperative `Fixed` path or a forced binary mode).
- **Conditioning.** Error in a solved system scales with the condition number of
  the matrix. An ill-conditioned system (e.g. a Hilbert matrix) amplifies input
  error by orders of magnitude in any finite precision; iterative refinement
  recovers the residual but not the lost input information.
- **Determinism.** Whatever the error is, it is the *same* error on every
  platform; results are bit-identical across architectures.

## Feature flags

| Flag | Effect |
| ---- | ------ |
| `infinite-precision` | BigInt tier for the symbolic rational domain (pulls in `num-bigint`) |
| `serde` | `Serialize`/`Deserialize` for `Fixed`, vectors, matrices, tensors and manifold points (as bytes) |
| `rebuild-tables` | regenerate lookup tables from `build.rs` |

There is no feature per format and none around core functionality: every
format, domain, transcendental, wide integer (I256 to I2048) and the tiered
overflow are always compiled. The ternary inference kernels are the separate
crate `g_math_inference`.

The full determinism and cross-format contract is in **[CONTRACT.md](../CONTRACT.md)**.

## Disclaimer

This software is provided **"as is"**, without warranty of any kind, express or
implied. Use of this software is entirely at your own risk. In no event shall the
author or contributors be held liable for any damages arising from the use or
inability to use this software.

---

Built by **Niels Erik Toren** · [support & donations](../README.md#author--support).
