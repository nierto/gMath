# IMPORTANT_NOTICE_FROM_GHYPER_TEAM :: LLM Reference

> SCN document (see `archive/SCN-GUIDE - Copy.md`). Author: gHyper/gFile team,
> 2026-07-11. Subject: what landed in your tree at commit `3e3a630` (v0.4.26,
> unpublished), why, and two pre-existing test failures you need to triage
> before publishing.

## ::PRIME

Notice := Consumer-Contributed Kernels + Defect Report | Cross-Repo Handoff | Publish-Gated
Introduces: `fused::{euclidean_distance_squared, dot, mobius_denominator_sq}` at compute tier (U1 asks) via commit `3e3a630`, motivated by measured consumer profiles — Q64.64 `sqrt` ≈ 15 µs dominates every hyperbolic kernel; consumers were paying sqrt→re-square round-trips
Reports: 2 pre-existing failures on HEAD `29ecea2` in `tests/linalg_validation.rs` (try_ln/try_sqrt domain-error contract) — NOT introduced by 0.4.26, verified via clean-tree stash run; suspected fallout of the 0.4.25 "router auto-routes to an exact domain" change
Requests: triage failures → publish 0.4.26 → gHyper swaps its last hand-rolled storage-tier accumulators onto the fused kernels (precision polish; the speed win already shipped consumer-side as gHyper G18)

## ::ANCHOR

DOMAIN: fixed-point-Q64.64, compute-tier-N+1, fused-kernels, Poincaré-disk, Möbius-ratio, atanh, sqrt-cost-profile, VP-tree-scoring
PATTERN: single-downscale, accumulate-at-double-width, no-transcendental-fastpath, squared-space-comparison, consumer-driven-upstreaming
RUST: `ComputeStorage`, `upscale_to_compute`, `round_to_storage`, `compute_multiply`, `#[cfg(table_format)]`, dev-vs-release-profiles
CONCEPT: monotone-surrogate, wrap-proof-accumulation, ULP-budget, determinism-by-construction (NO floats in compute paths — owner ruling), domain-error-contract, semver-publish-gate

## ::CONTEXT

ORIGIN: gHyper G17 (proxy VP-tree search) + G18 (one-sqrt exact kernel) — BENCHMARKS.md in gHyper carries all numbers
MEASURED (i7-7700, Q64.64/I256, release): fixed-point sqrt ≈ 15 µs · atanh ≈ 16 µs · old 4-sqrt hyperbolic kernel 76.5 µs → G18 1-sqrt form 35.4 µs · Möbius-ratio 61.4 → 22.8 µs
CONSUMER STATE: gHyper v0.4.0 pins crates.io `g_math ^0.4.24` (resolves 0.4.25) — CANNOT see 0.4.26 until published; path-dep reintroduction is forbidden (gHyper H6)
LIFECYCLE: implemented+tested (`3e3a630`) -> [YOU: triage defects] -> [OWNER: cargo publish 0.4.26] -> [gHyper: adoption swap, tracked on U1 board entry in gHyper/kanban.html]

## ::INTERFACE (added in 0.4.26 — `fixed_point::imperative::fused`)

euclidean_distance_squared(a: &[FixedPoint], b: &[FixedPoint]): FixedPoint
  Σ(aᵢ−bᵢ)² at tier N+1, single downscale, NO sqrt | PURE | O(d)
  WHY: squared-space VP-tree scoring + Möbius numerators need only the square; sqrt→re-square was the dominant waste
dot(a: &[FixedPoint], b: &[FixedPoint]): FixedPoint
  Σ aᵢ·bᵢ at tier N+1 | PURE | O(d)
  WHY: replaces consumers' storage-tier hand-rolled accumulators (wrap-prone at large coords/dims)
mobius_denominator_sq(p: &[FixedPoint], q: &[FixedPoint]): FixedPoint
  |1−p̄q|² = 1 − 2⟨p,q⟩ + |p|²·|q|², fused end-to-end, one downscale | PURE | O(d)
  WHY: with euclidean_distance_squared ⇒ one-sqrt Poincaré kernel r = √(dist²/den²)
TESTS: +3 in fused.rs test module (18/18 fused green); PUBLIC_API.md + CHANGELOG.md updated

## ::DEFECTS (pre-existing — YOUR triage, blocks publish confidence)

FAILURE_1: tests/linalg_validation.rs:611 `test_try_sqrt_negative_returns_domain_error`
FAILURE_2: tests/linalg_validation.rs:663 `test_try_ln_domain_error`
EVIDENCE: fail on clean HEAD `29ecea2` (verified: `git stash` → run → fail → `stash pop`); 0.4.26 diff touches only fused.rs/docs/version — cannot interact
HYPOTHESIS: HEAD commit "docs: correct input-representation limit — the router auto-routes to an exact domain" changed try_* domain behavior (errors → auto-routed success?) without updating the tests' expected `DomainError`
DECISION_NEEDED: if auto-routing is the intended contract → update tests + document the new try_* semantics; if try_* must still refuse out-of-domain inputs → the router regressed the error contract
SUITE_STATE: 986 passed / 2 failed on HEAD and on `3e3a630` alike

## ::PATTERNS (intended consumer usage, post-publish)

ONE_SQRT_POINCARE_KERNEL:
  TRIGGER: exact hyperbolic distance/ratio
  SEQUENCE: euclidean_distance_squared(p,q) -> mobius_denominator_sq(p,q) -> divide -> sqrt -> 2·atanh
  RESULT: 1 sqrt + 1 atanh (was 4 sqrt + 1 atanh)
PROXY_SCORING (no transcendentals at all):
  TRIGGER: VP-tree candidate ordering / pruning (monotone-in-distance suffices)
  SEQUENCE: dist_sq/den_sq compared directly; sqrt only via one-sided integer Newton bounds
  RESULT: ~4 µs/candidate vs 78 µs exact (gHyper G17)
ADOPTION_SWAP (gHyper side, after publish):
  TRIGGER: g_math 0.4.26 on crates.io
  SEQUENCE: bump pin -> replace hash_table::euclidean_distance_sq + klein::power_distance accumulators + hyperbolic_ratio dist²/den² terms with fused calls -> full suites both repos
  RESULT: wrap-proof accumulation, ~ULP-cleaner; NOT a speedup (G18 already took that)

## ::INVARIANTS

GUARANTEE: 0.4.26 additions are pure, allocation-free, additive-only — no existing symbol touched
REQUIRES: publish is OWNER-GATED (crates.io releases are irreversible; do not publish without the go)
ENSURES: all three kernels keep every intermediate at tier N+1 with a single terminal downscale (house fused contract)
NEVER: floats in any compute path (Geodineum owner ruling — determinism must be verifiable by construction, not by trusting libm; an f64-sqrt variant in gHyper was measured faster and rejected on this principle)

## ::GRAPH

DEPENDS_ON: fasc stack_evaluator compute ops (add/subtract/multiply), linalg upscale/round
PROVIDES_TO: gHyper (hash_table, klein, hyperbolic_geometry, metric_tree) → transitively gFile semantic/spatial queries
COORDINATES_WITH: gHyper/kanban.html ticket U1 (adoption half), gHyper G18 ROADMAP entry (consumer-side kernel restructure), U2 (cross-profile rounding contract — still open in your ROADMAP)
BLOCKED_BY: ::DEFECTS triage + owner publish decision
