# Canonical routing layer (FASC)

The user-friendly entry point: `gmath("…")` builds a lazy expression, `evaluate(…)`
runs it through a domain-routing stack evaluator.

## What it is

The canonical layer (`g_math::canonical`) is where a value is sent to the numeric
domain that holds it best, and where a chain of operations stays at the wide
compute tier until the final result is materialized. It trades a small amount of
dispatch overhead for two things the direct [imperative layer](README_IMPERATIVE.md)
does not do on its own: **cross-domain routing** and **chain persistence**. Reach
for it when domains are mixed (decimal money + integers + symbolic fractions),
when the input is user- or config-supplied, or when you are composing
transcendentals and want precision preserved across the whole chain.

## Usage

```rust
use g_math::canonical::{gmath, evaluate};

// Each literal is classified and routed: "0.1" → decimal, "3" → binary, "1/3" → symbolic.
let expr = (gmath("100") + gmath("50")) / gmath("3");
let value = evaluate(&expr).unwrap();
println!("{value}");

// Transcendentals chain lazily; intermediates never leave the wide compute tier.
let y = evaluate(&gmath("1.5").exp().sin()).unwrap();

// Runtime strings return a Result instead of panicking.
use g_math::canonical::gmath_parse;
let parsed = gmath_parse("3.14159265358979323846").unwrap();

// Feed a result back into a new expression without reparsing.
use g_math::canonical::LazyExpr;
let year0 = evaluate(&gmath("1000")).unwrap();
let year1 = evaluate(&(LazyExpr::from(year0) * gmath("1.05"))).unwrap();
```

## How it works

**Four numeric domains.** Values route to the representation that is exact for
them: binary fixed-point (integers, powers of two), decimal fixed-point (decimal
literals like `0.1`), balanced ternary (powers of three; TQ1.9 weights), and
symbolic rational (everything else, as an exact numerator/denominator pair). See
[the ternary guide](README_TERNARY.md) and [the precision guide](README_PRECISION.md)
for per-domain detail.

**FASC — Fixed Allocation Stack Computation.** `gmath()` builds a `LazyExpr` tree;
`evaluate()` walks it on a thread-local stack evaluator with fixed-size
workspaces, so the hot path does not allocate on the heap per operation.

**Tier N+1 chain persistence.** Every transcendental and every accumulating
operation computes one tier *wider* than the storage format and rounds down once
at the end. Between chained operations the intermediate stays at the wide tier —
`sin(exp(x))` never narrows mid-chain — so a chain rounds once, not once per step.

**UGOD — tiered graceful overflow.** Arithmetic is attempted at the current tier;
on overflow it promotes to a wider tier and retries. The top of the ladder is the
symbolic rational domain, so overflow degrades into a wider or exact
representation instead of wrapping.

**Fractal-topology router + shadow system.** Cross-domain arithmetic (e.g. decimal
+ binary) is dispatched by classifying each operand through shadow denominator
factoring — strip factors of 2 → binary-exact, 2 and 5 → decimal-exact, 3 →
ternary-exact — and coercing both sides to the optimal shared domain via a small
const lookup table. A `CompactShadow` (0–32 bytes, stack-only) can ride alongside
an approximated value carrying its exact rational identity or a constant reference
(π, e, √2, φ, ln 2, ln 10, γ); the router reads shadows to classify domain
exactness without reparsing. Only when no shared domain exists does arithmetic
fall back to exact rational. Ternary is a routing destination for
cross-domain **add/sub** of 3-adic values (denominator 3ᵏ), where routed
arithmetic is exact by construction; mul/div deliberately keep their
previous routes because products can leave the tier's exactness range,
and the router must never trade correctness for speed (the design
reasoning lives in `docs/design/TERNARY_ROUTING_COLUMN.md`). If a routed
value cannot fit ternary storage on a narrow profile, dispatch silently
falls back to the previous route.

**Mode routing.** `set_gmath_mode("compute:output")` forces the compute and output
domains independently (`auto`, `binary`, `decimal`, `symbolic`, `ternary`);
`reset_gmath_mode()` restores automatic routing.

## Public API

See **[PUBLIC_API.md → Canonical](../PUBLIC_API.md#canonical-g_mathcanonical)** for
the full symbol list, and [docs.rs](https://docs.rs/g_math) for live signatures.
`LazyExpr` supports `+`, `-`, `*`, `/`, unary `-`, and the 18
[transcendental methods](README_TRANSCENDENTALS.md). `LazyMatrixExpr` is the
matrix analog of chain persistence; `DomainMatrix` holds per-element
domain-tagged values for mixed-domain matrices.

## Behaviour & limits

- Correctness is **path-independent**: the router and the imperative surface
  compute through the same compute-tier engines and round the same way. Choose by
  cost and ergonomics, not by correctness.
- Routing adds dispatch overhead per operation relative to the imperative path;
  for known-domain hot loops use the [imperative layer](README_IMPERATIVE.md).
- Literals may be decimals (`"0.1"`), integers, fractions (`"1/3"`), repeating
  decimals (`"0.333..."`), hex/ternary (`"0x1F"`, `"0t10"`), or named constants
  (`"pi"`, `"e"`, `"sqrt2"`, `"phi"`).

The routing and determinism contract is in **[CONTRACT.md](../CONTRACT.md)**.

## Disclaimer

This software is provided **"as is"**, without warranty of any kind, express or
implied. Use of this software is entirely at your own risk. In no event shall the
author or contributors be held liable for any damages arising from the use or
inability to use this software.

---

Built by **Niels Erik Toren** — [support & donations](../README.md#author--support).
