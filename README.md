# g_math

**Author**: Niels Erik Toren — v0.4.23

Pure-Rust, zero-float, deterministic multi-domain fixed-point arithmetic.

`g_math` computes with scaled integers only. No `f32`/`f64` appears anywhere in the
arithmetic, validation, or comparison paths (float conversions exist solely as
user-convenience `from_f64`/`to_f64` wrappers). Every operation produces bit-identical
results on every architecture, which makes the crate suitable for blockchain consensus,
financial auditing, and reproducible scientific computation.

```toml
[dependencies]
g_math = "0.4"
```

## Key concepts

**Four numeric domains.** Values are routed to the representation that holds them best:

| Domain | Representation | Exact for |
| ------ | -------------- | --------- |
| Binary fixed-point | Q-format scaled integer (Q16.16 … Q256.256) | integers, powers of two |
| Decimal fixed-point | base-10 scaled integer | decimal literals like `0.1`, `19.99` |
| Balanced ternary | base-3 scaled integer, digits {-1, 0, +1} | powers of three; TQ1.9 weight storage |
| Symbolic rational | exact numerator/denominator pair | everything (`1/3`, repeating decimals) |

**Canonical API (FASC).** The user-friendly wrapper. `gmath("...")` builds a lazy
expression tree; `evaluate(...)` runs it through a thread-local stack evaluator with
fixed-size workspaces (FASC = Fixed Allocation Stack Computation). Parsing classifies
each literal and routes it to its natural domain — `"0.1"` becomes decimal, `"255"`
binary, `"1/3"` symbolic.

**Fractal topology router.** Cross-domain arithmetic (e.g. decimal + binary) is
dispatched by a router that classifies operands via shadow denominator factoring
(strip factors of 2 → binary-exact, 2 and 5 → decimal-exact, 3 → ternary-exact) and
coerces both sides to the optimal shared domain through a small const lookup table.
Only when no shared domain exists does arithmetic fall back to exact rational.

**Tier N+1 computation.** Every transcendental and every accumulating operation
(dot products, decompositions, matrix chains) computes one tier *wider* than the
storage format — Q64.64 storage computes at Q128.128 — and rounds down once at the
end. Intermediates between chained operations stay at the wide tier
(`sin(exp(x))` never narrows mid-chain).

**UGOD — tiered graceful overflow.** Arithmetic is attempted at the current tier;
on overflow it promotes to a wider tier and retries. The top of the ladder is the
symbolic rational domain, so overflow degrades into a wider or exact representation
instead of wrapping or failing silently.

**Shadow system.** A `CompactShadow` (0–32 bytes, stack-only) can ride alongside any
approximated value, carrying its exact rational identity (`1/3`) or a constant
reference (π, e, √2, φ, ln 2, ln 10, γ). The router reads shadows to classify
domain-exactness without reparsing.

## Quick start

```rust
use g_math::canonical::{gmath, evaluate};

let expr = (gmath("100") + gmath("50")) / gmath("3");
let value = evaluate(&expr).unwrap();
println!("{}", value);

// Transcendentals chain lazily; intermediates stay at the wide compute tier
let y = evaluate(&gmath("1.5").exp().sin()).unwrap();

// Runtime strings
use g_math::canonical::gmath_parse;
let parsed = gmath_parse("3.14159265358979323846").unwrap();

// Feed a result back into a new expression without reparsing
use g_math::canonical::LazyExpr;
let year0 = evaluate(&gmath("1000")).unwrap();
let year1 = evaluate(&(LazyExpr::from(year0) * gmath("1.05"))).unwrap();
```

Imperative path, for hot loops:

```rust
use g_math::fixed_point::{FixedPoint, FixedVector, FixedMatrix};

let a = FixedPoint::from_str("0.5");
let b = a + FixedPoint::from_int(2);  // direct integer arithmetic
let e = a.exp();                       // direct engine call, no expression tree
let (s, c) = a.sincos();               // fused: one shared range reduction
```

**Which to use:** the canonical API when domains are mixed (decimal money, user
input, symbolic fractions) or when chaining transcendentals; the imperative API
when you know you are in binary Q-format and the per-call overhead of the
expression pipeline matters (inner loops, matrix code, ML inference).

## Profiles

Storage width is a compile-time choice via the `GMATH_PROFILE` environment
variable (default: `embedded`).

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

**Custom integer/fraction split (realtime only).** The realtime profile lets you
move the binary point inside the 32-bit storage word via `GMATH_FRAC_BITS`:

```bash
GMATH_FRAC_BITS=10 GMATH_PROFILE=realtime cargo build   # Q22.10 — ±2M range, 3 digits
GMATH_FRAC_BITS=24 GMATH_PROFILE=realtime cargo build   # Q8.24  — ±127 range, 7 digits
```

Valid range is 2–30 fractional bits. The other profiles use fixed splits.

**Switching profiles:** each profile compiles different code paths via `cfg` flags.
Clear the incremental cache first, or stale artifacts will cause crashes:

```bash
rm -rf target/debug/incremental/
```

Pre-built lookup tables are checked in; a default build takes ~2 seconds.
`--features rebuild-tables` regenerates them from `build.rs` (~20 minutes,
pure-Rust generation: π via Machin's formula, e via factorial series, √2 via
continued fractions — zero runtime dependencies).

## API overview

### Canonical (`g_math::canonical`)

| Item | Purpose |
| ---- | ------- |
| `gmath("...")` | build a `LazyExpr` from a literal (deferred parsing) |
| `gmath_parse(&str)` | build from a runtime string, returns `Result` |
| `evaluate(&LazyExpr)` | evaluate → `Result<StackValue, _>` |
| `evaluate_sincos(&LazyExpr)` | sin and cos from one shared range reduction |
| `evaluate_sinhcosh(&LazyExpr)` | sinh and cosh from one shared exp pair |
| `evaluate_matrix(&LazyMatrixExpr)` | evaluate a matrix expression chain |
| `set_gmath_mode("compute:output")` / `reset_gmath_mode()` | force compute/output domains (`auto`, `binary`, `decimal`, `symbolic`, `ternary`) |
| `LazyExpr::from(StackValue)` | feed a result back into a new expression |

`LazyExpr` supports the basic operators `+`, `-`, `*`, `/`, unary `-`, plus the
18 transcendental methods listed below. Literals may be decimals (`"0.1"`),
integers, fractions (`"1/3"`), repeating decimals (`"0.333..."`), hex/ternary
(`"0x1F"`, `"0t10"`), or named constants (`"pi"`, `"e"`, `"sqrt2"`, `"phi"`).

`LazyMatrixExpr` is the matrix analog of scalar chain persistence: `Add`, `Sub`,
`Mul`, `ScalarMul`, `Transpose`, `Neg`, `Inverse`, `Exp`, `Log`, `Sqrt`, `Pow` —
the whole chain runs at the wide tier with a single downscale at
`evaluate_matrix()`. `DomainMatrix` holds per-element domain-tagged values for
mixed-domain matrices.

With `--features macros`, `gmath!("0.1")` pre-parses decimal and integer literals
at compile time; fractions, constants, and hex/ternary literals fall back to the
runtime function.

### Transcendentals

18 functions, available on `LazyExpr`, `FixedPoint`, and `DecimalFixed`.

Dedicated engines (table-driven or Newton-Raphson, computed at tier N+1):

| Function | Algorithm |
| -------- | --------- |
| `exp` | integer part by squaring + 3-stage table lookup + Taylor remainder |
| `ln` | multiplicative decomposition, 3-stage tables + Taylor |
| `sqrt` | integer Newton-Raphson |
| `sin`, `cos` | Cody-Waite range reduction + Horner Taylor (`sincos` fuses both) |
| `atan`, `atan2` | 3-level argument reduction + Taylor |

Composed from the dedicated engines, still at the wide tier:

| Function | Composition |
| -------- | ----------- |
| `tan` | sin/cos |
| `pow(x, y)` | exp(y·ln x) |
| `asin`, `acos` | atan(x/√(1−x²)), π/2 − asin |
| `sinh`, `cosh` | (eˣ ∓ e⁻ˣ)/2 (`sinhcosh` fuses both on one exp pair) |
| `tanh` | (e²ˣ−1)/(e²ˣ+1) |
| `asinh`, `acosh`, `atanh` | log forms |

On `FixedPoint`, every function also has a fallible `try_*` variant returning
`Result<_, OverflowDetected>`.

### Imperative (`g_math::fixed_point`)

- **`FixedPoint`** — `Copy` Q-format scalar. Arithmetic operators, comparisons,
  `abs`, `from_str`/`from_int`/`from_raw`, all 18 transcendentals, `sincos`,
  `sinhcosh`, float conversions for interop.
- **`FixedVector`** — `dot`, `length`, `length_fused`, `normalized`,
  `distance_to`, `cross`, `outer_product`, `map`, indexing, operators. Dot
  products accumulate at the compute tier.
- **`FixedMatrix`** — `identity`, `diagonal`, `from_fn`, `from_slice`,
  `transpose`, `trace`, `row`/`col`, mat-mat and mat-vec multiply (compute-tier
  dots per entry), `kronecker`, `submatrix`.

### Decimal (`DecimalFixed<DECIMALS>`)

Base-10 scaled integer with a const-generic decimal-place count. `0.1` is stored
exactly. Full basic arithmetic in pure decimal (add, subtract, multiply, divide,
negate, plus a batched multiply), and its own native transcendental engines
(all 18, plus fused `sincos` and `sinhcosh`) — no round-trip through binary, so
results are correctly rounded *in the decimal domain*. Conversions:
`try_convert`/`convert_with_rounding` between precisions,
`to_binary_q256`/`from_binary_q256`.

### Fused operations (`imperative::fused`)

Whole patterns computed at the wide tier with one downscale at the end:

| Function | Computes |
| -------- | -------- |
| `sqrt_sum_sq(&[x])` | √(Σ xᵢ²) |
| `euclidean_distance(&a, &b)` | √(Σ (aᵢ−bᵢ)²) |
| `softmax(&scores)` | numerically stable softmax |
| `rms_norm_factor(&x, eps)` | 1/√(mean(x²)+ε) |
| `silu(x)` | x/(1+e⁻ˣ) |

### Linear algebra (`imperative::decompose`, `derived`, `matrix_functions`)

- **Decompositions**: LU (Doolittle, partial pivoting), QR (Householder),
  Cholesky, SVD (Golub-Kahan), symmetric eigenvalues (Jacobi), Schur (Francis QR).
  Each returns a struct with `solve`/`determinant`/`inverse` where applicable,
  plus iterative refinement on LU.
- **Derived**: `frobenius_norm`, `norm_1`, `norm_inf`, `solve`, `solve_spd`,
  `determinant`, `inverse`, `inverse_spd`, `pseudoinverse`, `rank`, `nullspace`,
  `least_squares`, `condition_number_1`/`_2`.
- **Matrix functions**: `matrix_exp` (Padé + scaling-squaring), `matrix_sqrt`
  (Denman-Beavers), `matrix_log` (inverse scaling-squaring), `matrix_pow` —
  all chained through `ComputeMatrix` at the wide tier.

### Geometry (`imperative::{manifold, lie_group, curvature, projective, fiber_bundle}`)

- **Manifolds** (trait: `exp_map`, `log_map`, `distance`, `parallel_transport`,
  `inner_product`): Euclidean, Sphere, Hyperbolic (hyperboloid model), SPD,
  Grassmannian, Stiefel, products.
- **Lie groups** (trait adds `lie_exp`, `lie_log`, `hat`/`vee`, `adjoint`,
  `bracket`, `act`): SO(3) via closed-form Rodrigues, SE(3) via closed-form
  V-matrix, plus SO(n), GL(n), O(n), SL(n) via matrix exp/log.
- **Differential geometry**: Christoffel symbols, Riemann/Ricci/scalar/sectional
  curvature, geodesic integration, parallel transport along curves.
- **Projective**: homogeneous coordinates, projective transforms, cross-ratios,
  stereographic projection, Möbius transformations (real and complex).
- **Fiber bundles**: trivial, vector (connection coefficients, horizontal lift,
  parallel transport, curvature 2-form), principal (transition cocycles).

### ODE solvers (`imperative::ode`)

RK4 (fixed step), Dormand-Prince RK45 (adaptive), symplectic Störmer-Verlet
(energy-preserving, for Hamiltonian systems). Weighted sums accumulate at the
compute tier; step halving is an exact bit shift.

### Tensors (`imperative::tensor`, `tensor_decompose`)

Arbitrary-rank tensors: contraction, outer product, trace, index raising/lowering
via a metric, (anti)symmetrization. Decompositions: `truncated_svd`,
`tucker_decompose` (HOSVD), `cp_decompose` (ALS).

### Balanced ternary (`domains::balanced_ternary`)

Basic arithmetic (add, subtract, multiply, divide, negate — checked and
unchecked variants) across six tier formats from TQ8.8 up to TQ256.256, plus
trit packing: `pack_trits`/`unpack_trits` store 5 balanced trits {-1, 0, +1}
per byte. Ternary is also reachable through the canonical API via `0t` literals
or `set_gmath_mode("...:ternary")`; transcendentals on ternary values route
through the binary engines.

### TQ1.9 ternary inference (`g_math::tq19`, feature `inference`)

Standalone 2-byte balanced-ternary format for neural network weights: 1 integer
trit + 9 fractional trits, range ±1.5, ~4.3 decimal digits of uniform precision.
Because weights are {-1, 0, +1} at the trit level, dot products need no
multiplications.

- `TQ19Matrix` with `matvec`, `matvec_batch` (and rayon `_par` variants)
- `tq19_dot`, `trit_dot`, `packed_trit_dot` (5 trits/byte), `packed_trit_matvec`
- AVX2 SIMD on x86_64 with runtime detection and scalar fallback

### Serialization (`imperative::serialization`)

Profile-tagged big-endian encoding for `FixedPoint`, `FixedVector`,
`FixedMatrix`, `Tensor`, `ManifoldPoint` — compact, deterministic, suitable for
wire transport and consensus. Optional serde support behind `--features serde`.

## Rounding

Each domain has a defined rounding behavior:

| Domain | Multiply | Divide | Wide-tier downscale |
| ------ | -------- | ------ | ------------------- |
| Binary fixed-point | round-half-even (banker's) | round-half-away-from-zero | round-to-nearest, ties toward +∞ |
| Decimal fixed-point | round-half-away-from-zero | round-half-away-from-zero | round-half-away-from-zero |
| Balanced ternary | truncate toward zero | truncate toward zero | — (transcendentals route via binary) |

**The binary tie-breaking inconsistency is a development artifact, not a design
statement.** Each operation's rounding was chosen and validated independently
against reference values during iterative development, and the three rules were
never retroactively unified. It is documented here so nobody mistakes it for
numerical intent.

**Why it matters less than it looks: the downscale is the rounding that counts.**
Everything beyond a lone storage-tier multiply or divide — every transcendental,
dot product, decomposition, matrix chain, and fused op — runs at tier N+1 with
double the fractional bits, then rounds back to storage exactly once. In those
paths the per-op multiply/divide tie rules never fire; the single wide→storage
downscale (round-to-nearest) is the only rounding the result ever sees, and the
extra fractional bits absorb the intermediate error before that final round.
The mul/div tie rules apply only to direct storage-tier arithmetic, where all
three rules are round-to-nearest variants — they produce identical results
except on exact half-ULP ties, and each individual op stays within half an ULP
regardless of which rule breaks the tie.

All rounding is implemented in integer arithmetic and is therefore
deterministic across platforms — including the inconsistency itself.

## Precision and validation

The crate's accuracy claims are defined by its test suite, not slogans. The
approach:

- Reference values are generated with mpmath at 50–250 digit precision and
  embedded in the tests as exact strings — never computed with floats.
- Transcendentals are validated pointwise against those references on every
  profile. The wide-tier strategy means the final rounding step selects the
  nearest representable value for the storage format in the measured cases.
- Linear algebra, manifolds, Lie groups, ODE, and tensor tests combine
  structural checks (PA=LU, QᵀQ=I, exp/log roundtrips) with concrete
  mpmath-validated numerical comparisons.

```bash
cargo test --release
```

Honest limits worth knowing:

- **Input representation**: values like `0.3` or `1/3` are repeating fractions
  in binary and carry up to half an ULP of representation error before any
  computation happens. No finite-precision system avoids this; the decimal and
  symbolic domains exist precisely so you can pick a representation in which
  your inputs *are* exact.
- **Conditioning**: error in a solved system scales with the condition number
  of the matrix. An ill-conditioned system (e.g. Hilbert matrices) amplifies
  input error by orders of magnitude in any finite precision; iterative
  refinement recovers the residual but not the lost input information.
- **Determinism**: whatever the error is, it is the *same* error on every
  platform — results are bit-identical across x86_64, ARM, and RISC-V.

## Feature flags

| Flag | Effect |
| ---- | ------ |
| `infinite-precision` | BigInt tier for the symbolic rational domain (pulls in `num-bigint`) |
| `serde` | `Serialize`/`Deserialize` for FixedPoint, vectors, matrices, tensors |
| `inference` | TQ1.9 ternary inference ops + rayon parallel matvec |
| `macros` | `gmath!()` compile-time literal pre-parsing |
| `rebuild-tables` | regenerate lookup tables from `build.rs` (~20 min) |
| `realtime` / `compact` / `embedded` / `balanced` / `scientific` | select profile via Cargo feature instead of `GMATH_PROFILE` |
| `legacy-tests` | compile legacy test suites |

No feature gates around core functionality — all domains, transcendentals, wide
integers (I256/I512/I1024), and tiered overflow are always compiled.

## Author note

I build keystone libraries from first principles — this one because I needed
precise, deterministic fixed-point arithmetic and wanted the numeric domains to
coexist instead of collapsing everything into one representation. Use it, stress
it, break it, and tell me where it fails.

If you want to support the work:

| Currency | Address |
| -------- | ------- |
| Bitcoin (BTC) | bc1qwf78fjgapt2gcts4mwf3gnfkclvqgtlg4gpu4d |
| Ethereum (ETH) | 0xf38b517Dd2005d93E0BDc1e9807665074c5eC731 / nierto.eth |
| Monero (XMR) | 8BPaSoq1pEJH4LgbGNQ92kFJA3oi2frE4igHvdP9Lz2giwhFo2VnNvGT8XABYasjtoVY2Qb3LVHv6CP3qwcJ8UnyRtjWRZ5 |

## Disclaimer

This software is provided **"as is"**, without warranty of any kind, express or
implied. Use of this library is entirely at your own risk. In no event shall the
author or contributors be held liable for any damages arising from the use or
inability to use this software.

## License

Licensed under either of

* [Apache License, Version 2.0](LICENSE-APACHE)
* [MIT License](LICENSE-MIT)

at your option.
