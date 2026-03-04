# gMath — Multi-Domain Fixed-Point Arithmetic Library

Zero-float, pure-Rust, consensus-safe precision arithmetic achieving **0 ULP** accuracy across all 18 transcendental functions on all computation profiles.

## Why gMath?

Most math libraries pick one number representation and accept its limitations. gMath routes each value to its **optimal domain** automatically: decimals stay exact in decimal, fractions stay exact in symbolic, powers of 2 stay exact in binary, and transcendentals compute at double-width precision. The result: **0 ULP** (Units in Last Place) across every operation, every profile.

I started building this because I was working on an idea which needed a reliable, robust and preferable precise fixedpoint library in Rust for high throughput. And because I believe that parts of the knowledge we already possess currently sits in the shade of our human comprehension. This is ground - already uncovered - yet to be explored by people and AI, often in close collaboration.

LLMs flipped the script for millions of people - including myself, no longer is technical expertise a barrier to creation, today it can be overcome. AI enables everyone, of every creed and background, to study faster, iterate and prototype continuously and find solutions quicker than ever before. 

I have caused my family considerable stress while developing this library - allnighters and generally obsessively bothering my wife with intricate math insights -  if you wish to help me atone for that, consider donating to any of the addresses I listed at the bottom of this README.

If you are more interested in the development story, I suggest you give my blogpost on it a read on nierto.com. I prioritize delivering this opensource software over finishing the story of how it came to be, so check back in in a fortnight and it will be available.

## THE VIBE IS STRONG WITH THIS ONE

Claude helped me provide substance to the vision I had for this library and I am forever grateful that it allowed me to build something that I can now share with the world. It is by no means finished, and there might be flawed, but it is only I and a legion of transformer-based LLMs that built this. All feedback and input is welcome!

## Quick Start

```rust
use g_math::canonical::{gmath, evaluate};

// Arithmetic — auto-routes to optimal domain
let sum = evaluate(&(gmath("1.5") + gmath("2.5"))).unwrap();           // 4.0 (exact)
let product = evaluate(&(gmath("1/3") * gmath("3"))).unwrap();         // 1.0 (exact — symbolic)
let decimal = evaluate(&(gmath("0.1") + gmath("0.2"))).unwrap();       // 0.3 (exact — decimal domain)

// 18 transcendental functions — all at 0 ULP
let e = evaluate(&gmath("1.0").exp()).unwrap();                        // 2.7182818284590452353...
let root2 = evaluate(&gmath("2.0").sqrt()).unwrap();                   // 1.4142135623730950488...
let pi_sin = evaluate(&gmath("3.14159265358979323846").sin()).unwrap(); // ~0
```

## Runtime Strings

`gmath()` accepts `&'static str` (string literals) for zero-cost deferred parsing. For runtime/dynamic strings, use `gmath_parse()`:

```rust
use g_math::canonical::{gmath, gmath_parse, evaluate, LazyExpr};

// Static strings — zero-cost, parsing deferred until evaluate()
let a = gmath("1.5");

// Runtime strings — eagerly parsed, full mode routing support
let user_input = String::from("3.14");
let b = gmath_parse(&user_input).unwrap();

// Both work identically in expressions
let result = evaluate(&(a + b)).unwrap();

// Read values from files, databases, user input, etc.
let values: Vec<String> = vec!["1.1".into(), "2.2".into(), "3.3".into()];
let mut sum = gmath_parse(&values[0]).unwrap();
for v in &values[1..] {
    sum = sum + gmath_parse(v).unwrap();
}
let total = evaluate(&sum).unwrap();
```

## The auto:auto Advantage

gMath's default mode (`auto:auto`) combines the strength of four computation domains to deliver results no single domain can match:

```
Input "0.1"  → Decimal domain  → 0.1 is EXACT (not 0.100000000000000001)
Input "1/3"  → Symbolic domain → 1/3 is EXACT (infinite precision rational)
Input "1024" → Binary domain   → power-of-2, exact in binary
Input "exp()" → Binary compute → Tier N+1 strategy, 0 ULP
```

Forcing all inputs through a single domain produces **approximate results** — our comprehensive benchmark proves this across 288 mode routing test points:

| Mode | Exact | Approx | Lossy | Why |
|------|-------|--------|-------|-----|
| **auto:auto** | **24** | **0** | **0** | Each input routed to its natural domain |
| symbolic:symbolic | 24 | 0 | 0 | Infinite precision (but no transcendentals) |
| binary:binary | 17 | 7 | 0 | 1/3 is approximate in binary |
| decimal:decimal | 14 | 10 | 0 | 1/3 is 0.333...333 (truncated) |
| ternary:ternary | 20 | 4 | 0 | Ternary representation limits |
| ternary:decimal | 16 | 8 | 0 | Double conversion loss |

## Canonical API

### Entry Points

```rust
use g_math::canonical::{gmath, gmath_parse, evaluate, LazyExpr};

// Static strings (compile-time literals) — zero-cost, deferred parsing
let expr = gmath("1.5");

// Runtime strings — eager parsing, returns Result
let expr = gmath_parse(&some_string)?;
```

### Input Formats

gMath recognizes several input syntaxes and routes each to the optimal domain:

| Syntax | Domain | Examples | Description |
|--------|--------|---------|-------------|
| `"1/3"`, `"22/7"` | Symbolic | `gmath("1/3")` | Fractions — exact rational arithmetic |
| `"0.333..."` | Symbolic | `gmath("0.142857...")` | Repeating decimals — converted to exact rational (1/7) |
| `"0x"`, `"0b"` | Binary | `gmath("0xFF")`, `gmath("0b1010")` | Hex and binary integer literals |
| `"0t"` | Ternary | `gmath("0t1.5")` | Balanced ternary fixed-point |
| `"1.5"`, `"0.001"` | Decimal | `gmath("19.99")` | Decimal point — exact in decimal domain |
| `"pi"`, `"e"`, `"sqrt2"` | Symbolic | `gmath("phi")`, `gmath("ln2")` | Named mathematical constants |
| `"42"`, `"1024"` | Binary | `gmath("1000000")` | Integers — binary fixed-point |

Named constants: `pi`, `e`, `sqrt2`, `sqrt3`, `phi`, `ln2`, `ln10`, `gamma` (also accepts uppercase and Unicode: `PI`, `π`, `φ`, `√2`, `√3`, `γ`).

### Arithmetic (operator overloading)

```rust
let a = gmath("1.5") + gmath("2.5");   // Addition
let b = gmath("3") * gmath("7");       // Multiplication
let c = gmath("10") - gmath("4");      // Subtraction
let d = gmath("22") / gmath("7");      // Division
let e = -gmath("42");                  // Negation

let result = evaluate(&a);  // Triggers computation
println!("{}", result.unwrap());  // Display with guaranteed decimals
```

### Transcendental Functions (18 total, 0 ULP)

```rust
// Exponential family
gmath("1.0").exp()          // e^x
gmath("2.0").ln()           // ln(x)
gmath("2.0").sqrt()         // √x
gmath("2.0").pow(gmath("3")) // x^y

// Trigonometric
gmath("0.5").sin()           // sin(x)
gmath("0.5").cos()           // cos(x)
gmath("0.5").tan()           // tan(x)

// Inverse trigonometric
gmath("0.5").atan()          // atan(x)
gmath("0.5").asin()          // asin(x)
gmath("0.5").acos()          // acos(x)
gmath("1.0").atan2(gmath("1.0"))  // atan2(y, x)

// Hyperbolic
gmath("1.0").sinh()          // sinh(x)
gmath("1.0").cosh()          // cosh(x)
gmath("0.5").tanh()          // tanh(x)

// Inverse hyperbolic
gmath("1.0").asinh()         // asinh(x)
gmath("2.0").acosh()         // acosh(x)
gmath("0.5").atanh()         // atanh(x)
```

### Chaining Results (zero precision loss)

```rust
use g_math::canonical::{gmath, evaluate, LazyExpr};

// Feed previous results back in — full precision preserved
let rate = gmath("1.05");
let mut balance = evaluate(&gmath("1000.00")).unwrap();

for year in 1..=5 {
    balance = evaluate(&(LazyExpr::from(balance) * gmath("1.05"))).unwrap();
    println!("Year {}: {}", year, balance);
}
```

### Mode Routing (compute:output control)

```rust
use g_math::canonical::{set_gmath_mode, reset_gmath_mode, gmath, evaluate};

// Force specific compute and output domains
set_gmath_mode("binary:decimal").unwrap();   // Compute in binary, output as decimal
let result = evaluate(&(gmath("1.5") + gmath("2.5")));

// Auto compute, specific output format
set_gmath_mode("auto:ternary").unwrap();     // Best compute domain, ternary output

// Reset to default (recommended)
reset_gmath_mode();  // Back to auto:auto
```

Available modes: `auto`, `binary`, `decimal`, `symbolic`, `ternary` — any combination as `compute:output`.

### Result Extraction

```rust
let val = evaluate(&(gmath("1/3") + gmath("2/3"))).unwrap();

// Display — guaranteed decimals per profile
println!("{}", val);           // "1.0000000000000000000" (19 digits on embedded)

// Exact rational form
let rational = val.to_rational().unwrap();

// Decimal string with custom precision
let s = val.to_decimal_string(10);  // "1.0000000000"

// Domain inspection
val.domain_type();  // Some(DomainType::Symbolic)
val.is_error();     // false
val.tier();         // precision tier (1-8)
```

### Error Handling

All fallible operations return `Result<_, OverflowDetected>`:

```rust
use g_math::canonical::{gmath, gmath_parse, evaluate};

match evaluate(&gmath("1.0").ln()) {
    Ok(value) => println!("{}", value),
    Err(e) => eprintln!("Error: {:?}", e),
}

// gmath_parse returns Result too — handles invalid input gracefully
match gmath_parse("not_a_number") {
    Ok(expr) => { /* use expr */ },
    Err(e) => eprintln!("Parse error: {:?}", e),
}
```

## Precision Profiles

| Profile | Format | Storage | Compute | Guaranteed Decimals | Transcendentals |
|---------|--------|---------|---------|--------------------|--------------------|
| Embedded | Q64.64 | `i128` | `I256` | **19** | 18/18 at 0 ULP |
| Balanced | Q128.128 | `I256` | `I512` | **38** | 18/18 at 0 ULP |
| Scientific | Q256.256 | `I512` | `I1024` | **77** | 18/18 at 0 ULP |

Select via environment variable before compilation:

```bash
GMATH_PROFILE=embedded cargo build      # IoT, embedded systems
GMATH_PROFILE=balanced cargo build      # Web services, general purpose
GMATH_PROFILE=scientific cargo build    # Research, ultra-precision
```

**Important: clear the incremental cache when switching profiles.** Each profile compiles entirely different code paths via `cfg` flags. Rust's incremental compilation cache can retain stale artifacts from the previous profile, causing build failures or crashes at runtime.

```bash
rm -rf target/debug/incremental/    # Run this BEFORE switching profiles
GMATH_PROFILE=scientific cargo build # Now safe to build a different profile
```

This is a one-time step per switch — rebuilds within the same profile are fine.

## Architecture

### ZASC — Zero-Allocation Stack Computation

Expressions build lazily as trees, then evaluate on a fixed-size stack workspace. No heap allocation on the hot path.

```
gmath("value") → LazyExpr (tree builder, operator overloading)
              → StackEvaluator (thread-local, 4KB-64KB workspace)
              → StackValue (domain-tagged result)
```

### UGOD — Universal Graceful Overflow Delegation

Automatic tier promotion on overflow. Operations start at the minimum required tier and promote upward as needed:

```
i8 → i16 → i32 → i64 → i128 → I256 → I512 → Symbolic Rational
```

Symbolic rational is the guaranteed-success fallback — no operation ever fails silently. For unbounded precision, enable the optional `infinite-precision` feature for BigInt tier 8.

### Tier N+1 Precision Strategy

All transcendentals compute at **one tier above storage**, then downscale:

```
Embedded:   Q64.64 (i128)  → compute at Q128.128 (I256)  → downscale to i128
Balanced:   Q128.128 (I256) → compute at Q256.256 (I512)  → downscale to I256
Scientific: Q256.256 (I512) → compute at Q512.512 (I1024) → downscale to I512
```

Result: the closest possible fixed-point integer to the true mathematical value — 0 ULP.

### BinaryCompute Chain Persistence

Chained transcendentals like `sin(ln(exp(x)))` stay at compute tier throughout — only one downscale at final materialization, preventing cumulative precision loss.

### Four Computation Domains

| Domain | Best For | Exactness | Example |
|--------|----------|-----------|---------|
| **Binary Fixed** | Transcendentals, integers, powers of 2 | 19-77 decimals | `42`, `1024`, `exp(1.0)` |
| **Decimal Fixed** | Financial, exact decimals | 0 ULP exact | `0.1`, `19.99`, `0.001` |
| **Balanced Ternary** | Base-3 computation, geometric symmetry | Exact in base 3 | `0t1.0`, `0t0.111` |
| **Symbolic Rational** | Fractions, repeating decimals, constants | Infinite precision | `1/7`, `0.333...`, `pi` |

## Performance (Q64.64 embedded, x86_64)

| Category | Operation | Throughput | Latency (avg) |
|----------|-----------|------------|---------------|
| Arithmetic | binary add | 3.9M ops/s | 255ns |
| Arithmetic | symbolic add | 4.4M ops/s | 228ns |
| Arithmetic | decimal mul | 3.6M ops/s | 274ns |
| Transcendental | exp | 1.0M ops/s | 980ns |
| Transcendental | sin / cos | 820K ops/s | 1.2us |
| Transcendental | ln | 1.0M ops/s | 971ns |
| Transcendental | sqrt | 46K ops/s | 21.7us |
| Transcendental | atan | 34K ops/s | 29.3us |
| Mode routing | auto:* | 4.0M ops/s | 246ns |

## Validation

The comprehensive benchmark suite validates across **60,000+ mpmath-verified reference points**:

```bash
GMATH_PROFILE=embedded cargo test --release --test comprehensive_benchmark -- --nocapture --test-threads=1
```

Coverage:
- **60,860 arithmetic points** across 4 domains x 4 operations (decimal, symbolic, binary, ternary, cross-domain)
- **16,974 transcendental points** across 18 functions x 1,000+ reference values, validated against mpmath at 250-digit precision
- **288 mode routing points** across 12 compute:output combinations x 24 test cases
- **0 lossy results** across all mode combinations — domain limitations produce approximations, never data loss

## Dependencies

**Zero runtime dependencies** by default. All arithmetic — including 18 transcendental functions — is implemented in pure Rust using native integers and custom wide types (I256, I512, I1024).

Optional: enable `--features infinite-precision` to activate BigInt tier 8 (pulls in `num-bigint`, `num-traits`, `num-integer`).

## Build System

Pre-built lookup tables are included in the crate — builds complete in **~2 seconds**. No table generation required.

If you want to verify or regenerate the tables from scratch:

```bash
cargo build --features rebuild-tables   # ~20-30 minutes, pure-Rust generation
```

Tables are generated algorithmically by `build.rs` with zero external data files:

- **Pi**: Machin's formula at 580-bit rational precision
- **Exp/Ln tables**: 3-stage x 1024 entries per tier
- **Trig coefficients**: Taylor series at arbitrary precision
- **Prime table**: 1,145 primes up to 10,000 (sieve of Eratosthenes)

## Cross-Platform Determinism

- **Bit-identical** results across all architectures (x86, ARM, RISC-V)
- **Zero floating-point contamination** — f32/f64 forbidden in all internal logic
- **Consensus-safe** for blockchain, financial auditing, scientific reproducibility

## Contributing

Contributions are welcome! Fork the repo, create a branch, and open a pull request. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code standards, and guidelines.

Bug reports and feature requests go to [GitHub Issues](https://github.com/nierto/gMath/issues).

## Author

Niels Erik Toren

## Support the Project

If gMath is useful to you, consider supporting its single dev by donating:

| Currency | Address |
|----------|---------|
| Bitcoin (BTC) | bc1qwf78fjgapt2gcts4mwf3gnfkclvqgtlg4gpu4d |
| Ethereum (ETH) | 0xf38b517Dd2005d93E0BDc1e9807665074c5eC731 | nierto.eth |
| Monero (XMR) | 8BPaSoq1pEJH4LgbGNQ92kFJA3oi2frE4igHvdP9Lz2giwhFo2VnNvGT8XABYasjtoVY2Qb3LVHv6CP3qwcJ8UnyRtjWRZ5 |

## Disclaimer

This software is provided **"as is"**, without warranty of any kind, express or implied. Use of this library is entirely at your own risk. In no event shall the author or contributors be held liable for any damages, data loss, financial loss, or other consequences arising from the use or inability to use this software. By using gMath, you accept full responsibility for verifying its suitability for your use case.

See the license texts for the full legal terms.

## License

Licensed under either of

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.
