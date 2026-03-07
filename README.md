# g_math | gMath

**Author**: Niels Erik Toren

Multi-domain fixed-point arithmetic for Rust.

`g_math` is a pure-Rust arithmetic crate built around a canonical expression pipeline:

`gmath(...) -> LazyExpr -> evaluate(...) -> StackValue`

Under that API, the crate routes values and operations across four numeric domains:

* Binary fixed-point
* Decimal fixed-point
* Balanced ternary fixed-point
* Symbolic rational

It also exposes imperative types such as `FixedPoint`, `FixedVector`, and `FixedMatrix`, but the canonical API is the primary public entry point.

## What g_math is trying to do

Most numeric systems are good at some things and bad at others.

* Binary fixed-point is fast and natural for many low-level operations.
* Decimal fixed-point is often preferable for decimal-facing arithmetic.
* Ternary is available as a first-class domain rather than a curiosity.
* Symbolic rational provides an exact fallback for values that should not be collapsed into an approximation too early.

`g_math` is an attempt to let those domains coexist in one library instead of forcing everything through a single representation.

## Current architecture

### 1. Canonical API

The primary API is the canonical expression pipeline:

```rust
use g_math::canonical::{gmath, evaluate};

let expr = gmath("1.5") + gmath("2.5");
let value = evaluate(&expr).unwrap();
println!("{}", value);
```

There is also `gmath_parse(...)` for runtime strings and `LazyExpr::from(...)` for feeding an evaluated value back into a new expression without reparsing.

### 2. FASC

FASC stands for **Fixed Allocation Stack Computation**.

In practical terms, it means:

* expressions are built as `LazyExpr` trees
* evaluation is deferred until `evaluate(...)`
* evaluation runs through a thread-local `StackEvaluator`
* the evaluator uses a fixed-size value stack and domain-aware dispatch

The important consequence is that the **evaluation engine** is stack-oriented and built around fixed workspace structures rather than a growable runtime evaluator.

This is the path the crate is organized around, and the one new users should start with.

### 3. UGOD — Universal Graceful Overflow Delegation

UGOD is the tiered overflow model.

Each major domain is aligned to a shared tier system. Operations are attempted at the current tier, and when a result cannot be represented there, the computation can promote upward. At the top end, symbolic rational is the exact fallback.

The current universal tier model is:

| Tier | Bits | Binary   | Decimal  | Ternary   | Symbolic  |
| ---- | ---- | -------- | -------- | --------- | --------- |
| 1    | 32   | Q16.16   | D16.16   | TQ8.8     | i16/u16   |
| 2    | 64   | Q32.32   | D32.32   | TQ16.16   | i32/u32   |
| 3    | 128  | Q64.64   | D64.64   | TQ32.32   | i64/u64   |
| 4    | 256  | Q128.128 | D128.128 | TQ64.64   | i128/u128 |
| 5    | 512  | Q256.256 | D256.256 | TQ128.128 | I256/U256 |
| 6    | 1024 | Q512.512 | D512.512 | TQ256.256 | I512/U512 |

At the architecture level:

* tiers 1-5 promote upward on overflow
* tier 6 overflows can fall back to rational arithmetic
* optional unbounded precision can extend symbolic arithmetic beyond the bounded native tiers

The goal is not to avoid overflow by pretending it never happens. The goal is to overflow **gracefully** into a larger or exact representation instead of failing silently.

### 4. Shadow system

`g_math` includes a compact shadow system for preserving exactness metadata alongside approximated values.

The public `CompactShadow` type can store:

* no shadow
* small rational shadows in progressively larger compact forms (2 to 32 bytes)
* a full rational shadow (i128/u128 numerator-denominator pair)
* references to known constants: pi, e, sqrt(2), phi, ln2, ln10, Euler's gamma

This lets an inexact domain value carry a compact rational companion when one exists.

Example idea:

* if a value is stored in a fixed-point domain as an approximation of `1/3`,
* a compact rational shadow can still preserve that exact fractional identity for later use.

In the current implementation, shadow arithmetic is propagated where possible. It is best understood as **exactness retention infrastructure**, not magical infinite memory.

### 5. Wider-tier transcendental computation

The crate implements 18 transcendental functions:

`exp`, `ln`, `sqrt`, `pow`, `sin`, `cos`, `tan`, `atan`, `atan2`, `asin`, `acos`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`

The current implementation computes each at a wider tier than the active storage tier and then rounds back down. Because the intermediate has more fractional bits than the storage format can represent, the final rounding step produces the nearest representable value at the storage tier.

The profile mapping is:

| Profile    | Storage   | Compute tier |
| ---------- | --------- | ------------ |
| Embedded   | Q64.64    | Q128.128     |
| Balanced   | Q128.128  | Q256.256     |
| Scientific | Q256.256  | Q512.512     |

That wider-tier strategy is one of the central design decisions in the crate.

## Profiles

Build profile selection is driven by `GMATH_PROFILE`. The default is **embedded** (Q64.64, 19 decimal digits).

| Profile      | Format    | Storage | Compute | Decimal digits |
| ------------ | --------- | ------- | ------- | -------------- |
| `embedded`   | Q64.64    | i128    | I256    | 19             |
| `balanced`   | Q128.128  | I256    | I512    | 38             |
| `scientific` | Q256.256  | I512    | I1024   | 77             |

```bash
cargo build                             # embedded (default)
GMATH_PROFILE=balanced cargo build      # 38-digit precision
GMATH_PROFILE=scientific cargo build    # 77-digit precision
```

**Important**: clear the incremental cache when switching profiles. Each profile compiles entirely different code paths via `cfg` flags. Stale artifacts cause build failures or runtime crashes.

```bash
rm -rf target/debug/incremental/        # Run BEFORE switching profiles
GMATH_PROFILE=scientific cargo build    # Now safe to build a different profile
```

Pre-built lookup tables are checked into the repository. A default build completes in about 2 seconds. To regenerate tables from scratch (about 20 minutes):

```bash
cargo build --features rebuild-tables
```

## Feature flags

| Flag | Default | Effect |
| ---- | ------- | ------ |
| `infinite-precision` | off | Adds BigInt tier 8 to the symbolic rational domain. Pulls in `num-bigint`, `num-traits`, `num-integer` as runtime dependencies. Without this flag, the rational domain caps at tier 7 (I512 numerator/denominator). |
| `rebuild-tables` | off | Regenerates all lookup tables (exp, ln, trig) from `build.rs` instead of using the checked-in pre-built tables. Takes about 20 minutes. |
| `legacy-tests` | off | Enables compilation of legacy test suites from earlier development phases. |
| `embedded` | off | Selects embedded profile via Cargo feature instead of environment variable. |
| `balanced` | off | Selects balanced profile via Cargo feature instead of environment variable. |
| `scientific` | off | Selects scientific profile via Cargo feature instead of environment variable. |

All other arithmetic — including transcendental functions, SIMD acceleration (AVX2 runtime-detected on x86_64), tiered overflow, and I256/I512/I1024 wide integer types — is always compiled. There are no feature gates around core functionality.

## Quick start

Add the crate:

```toml
[dependencies]
g_math = "0.1.1"
```

Basic use:

```rust
use g_math::canonical::{gmath, evaluate};

fn main() {
    let expr = (gmath("100") + gmath("50")) / gmath("3");
    let value = evaluate(&expr).unwrap();
    println!("{}", value);
}
```

Runtime parsing:

```rust
use g_math::canonical::{gmath_parse, evaluate};

fn main() {
    let input = "3.14159265358979323846";
    let parsed = gmath_parse(input).unwrap();
    let result = evaluate(&(parsed * gmath("2"))).unwrap();
    println!("{}", result);
}
```

Feeding values back into the expression system:

```rust
use g_math::canonical::{gmath, evaluate, LazyExpr};

fn main() {
    let year0 = evaluate(&gmath("1000")).unwrap();
    let year1 = evaluate(&(LazyExpr::from(year0) * gmath("1.05"))).unwrap();
    println!("{}", year1);
}
```

## Domain routing and mode control

The crate exposes a compute and output mode system.

You can set modes such as:

* `auto:auto` (default — routes each value to its natural domain)
* `binary:ternary` (compute in binary, output in ternary)
* `decimal:symbolic` (compute in decimal, output as symbolic rational)

Available domains: `auto`, `binary`, `decimal`, `symbolic`, `ternary` — any combination as `compute:output`.

Example:

```rust
use g_math::canonical::{set_gmath_mode, reset_gmath_mode, gmath, evaluate};

fn main() {
    set_gmath_mode("binary:ternary").unwrap();
    let value = evaluate(&(gmath("3") + gmath("7"))).unwrap();
    println!("{}", value);
    reset_gmath_mode();
}
```

## Canonical API surface

The primary public interface lives in `g_math::canonical`:

| Item | Purpose |
| ---- | ------- |
| `gmath("...")` | Build a `LazyExpr` from a string literal (deferred parsing) |
| `gmath_parse(&str)` | Build a `LazyExpr` from a runtime string (eager parsing, returns `Result`) |
| `evaluate(&LazyExpr)` | Evaluate an expression tree, returns `Result<StackValue, _>` |
| `LazyExpr` | Expression tree node — supports operator overloading and transcendental methods |
| `LazyExpr::from(StackValue)` | Feed a previous result back into a new expression |
| `StackValue` | Domain-tagged result — implements `Display`, carries shadow metadata |
| `set_gmath_mode("compute:output")` | Set compute and output domain routing |
| `reset_gmath_mode()` | Reset to `auto:auto` |

The imperative API (`FixedPoint`, `FixedVector`, `FixedMatrix`) is also available via `g_math::fixed_point` for mutable arithmetic workflows. Transcendentals on `FixedPoint` route through the FASC evaluator internally.

If you are new to the crate, start with `g_math::canonical`.

## Validation and tests

The published crate includes test suites for:

* arithmetic sweep validation (4 domains, 4 operations, 60k+ reference points)
* boundary stress testing
* compound operations (chained arithmetic, iterative accumulation)
* domain arithmetic validation
* error handling
* FASC ULP validation (18 transcendentals, validated against mpmath at 250+ digit precision)
* mode routing validation (12 modes x 24 test cases)
* transcendental ULP validation

Run the comprehensive suite:

```bash
cargo test --release --test comprehensive_benchmark -- --nocapture --test-threads=1
```

This README intentionally avoids broad numerical slogans. Stronger correctness claims belong in a dedicated validation document with exact definitions, scope, corpus size, and methodology.

## Design notes

This crate is opinionated.

It does not pretend all arithmetic should collapse into one representation. It does not assume floating point is the only practical route. It tries to preserve exactness when possible, promote gracefully when necessary, and keep the main API compact.

That is the wager.

## Author note

I write software like a builder from first principles, not a committee. This is a library I built because I needed a precise and deterministic fixed-point library.

Instead of focusing on front-end apps, I prefer to rebuild from first principles keystone libraries so these are future-proof and allow me to build software and paradigms that didn't exist before.

So yes, some of this project carries personal style, philosophy, and a slightly stubborn tone. That is intentional.

If this crate is useful to you, then use it, stress it, break it, and tell me where it fails. It could contain flaws but I have not found them myself. I validated all operations against mpmath — run the comprehensive test to see for yourself.

If you want to support the work:

| Currency | Address |
|----------|---------|
| Bitcoin (BTC) | bc1qwf78fjgapt2gcts4mwf3gnfkclvqgtlg4gpu4d |
| Ethereum (ETH) | 0xf38b517Dd2005d93E0BDc1e9807665074c5eC731 / nierto.eth |
| Monero (XMR) | 8BPaSoq1pEJH4LgbGNQ92kFJA3oi2frE4igHvdP9Lz2giwhFo2VnNvGT8XABYasjtoVY2Qb3LVHv6CP3qwcJ8UnyRtjWRZ5 |

Please star the project on GitHub if it was useful to you. Thank you sincerely.

I am building this in the middle of life, work, pressure, family, and limited time. That does not make the project weaker. It is the reason it exists at all. We don't do things because they are easy, but because they are hard.

## Disclaimer

This software is provided **"as is"**, without warranty of any kind, express or implied. Use of this library is entirely at your own risk. In no event shall the author or contributors be held liable for any damages, data loss, financial loss, or other consequences arising from the use or inability to use this software. By using gMath, you accept full responsibility for verifying its suitability for your use case.

See the license texts for the full legal terms.

## License

Licensed under either of

* [Apache License, Version 2.0](LICENSE-APACHE)
* [MIT License](LICENSE-MIT)

at your option.
