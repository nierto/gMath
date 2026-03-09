# gMath Roadmap

Current version: **0.1.1**

This document tracks planned work and known gaps. Items are grouped by priority, not by timeline. Nothing here is promised — this is a working list for a solo-maintained project.

---

## Next: 0.2.0

### Compact profile — Q32.32 / D32.32 / TQ16.16 / i64 symbolic

Add a `compact` deployment profile using 64-bit storage with 128-bit compute tier.

| Domain | Storage | Compute | Decimals |
| ------ | ------- | ------- | -------- |
| Binary | Q32.32 (i64) | Q64.64 (i128) | 9 |
| Decimal | D32.32 (i64) | D64.64 (i128) | 9 |
| Ternary | TQ16.16 (i64) | TQ32.32 (i128) | 9 |
| Symbolic | i64 num/den | i128 intermediate | exact |

**Why**: Q64.64 is overkill for gamedev, audio DSP, embedded control, and many real-time applications where f32 is the current default. Q32.32 with i64 storage is a native word-size operation on all modern platforms and hits the sweet spot between precision and throughput.

**What exists**: `BinaryTier2`, `DecimalTier2`, `TernaryTier2` type definitions and `checked_*` arithmetic methods already exist. UGOD dispatch, stack evaluator wiring, transcendental support, and build system profile do not.

**Scope**:

- Wire Tier 2 into UGOD dispatch for all 4 domains (add/sub/mul/div/neg)
- Add `compact` profile to build.rs and Cargo.toml
- Add `BinaryStorage = i64` / `ComputeStorage = i128` type aliases under `#[cfg(table_format = "q32_32")]`
- Transcendentals: upscale Q32.32 to Q64.64, compute with existing tables, downscale (no new table generation needed)
- ULP validation against mpmath references at Q32.32 precision
- Update README, CHANGELOG, documentation

**Estimated effort**: ~2,500 lines, primarily UGOD dispatch wiring and test infrastructure.

### Domain arithmetic fixes

Known bugs in cross-domain operations:

- Symbolic rational addition produces incorrect results in some cases
- Decimal multiplication and division have edge-case failures

These were identified during development and deferred. They need investigation and targeted fixes before any new feature work.

### Stack evaluator modularization

The stack evaluator (`src/fixed_point/universal/fasc/stack_evaluator/`) was split into submodules (mod.rs, arithmetic.rs, compute.rs, conversion.rs, domain.rs, formatting.rs, parsing.rs, transcendentals.rs, tests.rs) but the dispatch logic is still dense. Further modularization would reduce cognitive load and make the compact profile integration cleaner.

---

## Future

### Fractal topology FASC integration

The fractal topology engine (`src/fixed_point/router/fractal_topology/`) is implemented but not wired into the FASC parse_literal() path. The engine uses 4D coordinate mapping and height-competition rules to recommend optimal domains based on input characteristics.

Integration would replace the current syntactic routing (pattern-matching on input strings) with geometric routing (domain selection based on empirical precision data). This is architecturally interesting but not urgent — the current syntactic routing works correctly.

### Ternary test coverage

Balanced ternary arithmetic is implemented with 6-tier UGOD but lacks a dedicated validation test suite comparable to the binary and decimal suites. The domain works but has not been stress-tested against reference values.

### Batch/vectorized API

The current API is scalar — one expression at a time. A batch API that processes arrays of values could genuinely benefit from AVX2 SIMD at lower tiers (8x Q32.32 values in a single 256-bit register). This would be a new API surface, not a change to the existing canonical pipeline.

### Q16.16 profile

If demand materializes for 4-decimal-digit fixed-point (retro gamedev, very constrained embedded), a `minimal` profile using Q16.16 (i32 storage, i64 compute) could be added. The pattern established by the compact profile would make this straightforward. Deferred because the value proposition at 4 decimals is thin — most users needing Q16.16 hand-roll it in 20 lines.

### Public API stabilization

Pre-1.0 cleanup: audit public exports, ensure `StackValue` methods are sufficient for all extraction needs, consider whether `FixedPoint`/`FixedVector`/`FixedMatrix` should remain public or be feature-gated.

---

## Non-goals

- **Floating-point interop beyond convenience**: `to_f64()`/`from_f64()` exist for user convenience. Internal float usage is architecturally forbidden.
- **Dynamic precision selection**: Profiles are compile-time. Runtime tier selection within a profile is handled by UGOD, but the base storage tier is fixed at build time.
- **GPU compute**: The library targets CPU determinism. GPU offload would compromise the cross-platform bit-identical guarantee.
