# Precision, profiles & build

Storage widths, the fractional-bit split, the rounding rules per domain, how
accuracy is validated, and how to build.

## Profiles

Storage width is a compile-time choice via the `GMATH_PROFILE` environment
variable (default: `embedded`), or the equivalent Cargo feature.

| Profile | Format | Storage | Compute | ~Decimal digits | Intended use |
| ------- | ------ | ------- | ------- | --------------- | ------------ |
| `realtime` | Q16.16 | i32 | i64 | 4 | ML inference, real-time |
| `compact` | Q32.32 | i64 | i128 | 9 | mobile, game logic |
| `embedded` | Q64.64 | i128 | I256 | 19 | IoT, financial |
| `balanced` | Q128.128 | I256 | I512 | 38 | general services |
| `scientific` | Q256.256 | I512 | I1024 | 77 | research |

```bash
cargo build                           # embedded (default)
GMATH_PROFILE=scientific cargo build  # 77-digit precision
```

**Custom integer/fraction split (realtime only).** The realtime profile can move
the binary point inside the 32-bit storage word via `GMATH_FRAC_BITS` (valid range
2–30 fractional bits); the other profiles use fixed splits.

```bash
GMATH_FRAC_BITS=10 GMATH_PROFILE=realtime cargo build   # Q22.10 — wider range, 3 digits
GMATH_FRAC_BITS=24 GMATH_PROFILE=realtime cargo build   # Q8.24  — narrow range, 7 digits
```

**Switching profiles** compiles different code paths via `cfg` flags. Clear the
incremental cache first, or stale artifacts can crash:

```bash
rm -rf target/debug/incremental/
```

Lookup tables are checked in, so a default build is fast. `--features rebuild-tables`
regenerates them from `build.rs` (pure-Rust generation — π via Machin's formula,
e via factorial series, √2 via continued fractions, zero runtime dependencies)
and is slow.

## Rounding

Rounding is defined per domain, and it is deterministic (integer arithmetic, so
identical on every platform). The key property: everything beyond a lone
storage-tier multiply or divide — every transcendental, dot product,
decomposition, matrix chain, and fused op — computes at tier N+1 with double the
fractional bits and rounds to storage exactly once, so the wide→storage downscale
is the only rounding the result sees. The normative per-domain rounding table and
the full explanation are in the rounding contract,
**[CONTRACT.md §3](../CONTRACT.md#3-rounding-contract)**.

## Validation

Accuracy is defined by the test suite, not slogans:

- Reference values are generated with mpmath at 50–250 digit precision and
  embedded in the tests as exact strings — never computed with floats.
- Transcendentals are validated pointwise against those references on every
  profile; the wide-tier strategy means the final rounding step selects the
  nearest representable value for the storage format in the measured cases.
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

- **Input representation.** Values like `0.3` or `1/3` are repeating fractions in
  binary and carry up to half an ULP of representation error before any
  computation. No finite-precision system avoids this; the decimal and symbolic
  domains exist so you can pick a representation in which your inputs *are* exact.
- **Conditioning.** Error in a solved system scales with the condition number of
  the matrix. An ill-conditioned system (e.g. a Hilbert matrix) amplifies input
  error by orders of magnitude in any finite precision; iterative refinement
  recovers the residual but not the lost input information.
- **Determinism.** Whatever the error is, it is the *same* error on every
  platform — results are bit-identical across architectures.

## Feature flags

| Flag | Effect |
| ---- | ------ |
| `infinite-precision` | BigInt tier for the symbolic rational domain (pulls in `num-bigint`) |
| `serde` | `Serialize`/`Deserialize` for FixedPoint, vectors, matrices, tensors |
| `inference` | TQ1.9 ternary inference ops + rayon parallel matvec |
| `rebuild-tables` | regenerate lookup tables from `build.rs` |
| `realtime` / `compact` / `embedded` / `balanced` / `scientific` | select profile via Cargo feature instead of `GMATH_PROFILE` |
| `legacy-tests` | compile legacy test suites |

No feature gates around core functionality — all domains, transcendentals, wide
integers (I256/I512/I1024), and tiered overflow are always compiled.

The full determinism and cross-profile contract is in **[CONTRACT.md](../CONTRACT.md)**.

## Disclaimer

This software is provided **"as is"**, without warranty of any kind, express or
implied. Use of this software is entirely at your own risk. In no event shall the
author or contributors be held liable for any damages arising from the use or
inability to use this software.

---

Built by **Niels Erik Toren** — [support & donations](../README.md#author--support).
