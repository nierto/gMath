# IMPORTANT_NOTICE_FROM_GHYPER_TEAM :: LLM Reference

> SCN document. Author: gHyper/gFile team, 2026-07-11 (second notice — the
> first was consumed and removed at `43953ec`, thank you for the fast
> DomainError triage). Subject: a ~24% sqrt/atanh performance regression
> between 0.4.24 and 0.4.27, isolated by measurement; plus the measured
> outcome of the U1 adoption swap so the kernels' intended audience is on
> record.

## ::PRIME

Notice := Perf-Regression Report + U1 Adoption Outcome | Consumer Measurement Channel | Lockfile-Held
Reports: transcendental call path (sqrt/atanh) ~24% slower 0.4.24 → 0.4.27 — identical consumer code, lockfile flip is the only variable; storage-tier mul/add is FLAT across versions (isolation)
Records: U1 fused-kernel adoption evaluated at all three gHyper sites and REJECTED by measurement — interior-bounded inputs make accumulator wrap impossible, so the ULP gain is unobservable while compute-tier overhead is 3.2× (ns sites) to +22% (µs sites); kernels stay, correctly aimed at unbounded-domain/high-dim consumers
Requests: perf-triage sqrt()/atanh() on the direct (non-try) path; consumers hold 0.4.24 lockfiles until restored

## ::ANCHOR

DOMAIN: fixed-point-Q64.64, sqrt, atanh, transcendental-dispatch, exact-domain-router, direct-engine-path, criterion-pair-cost
PATTERN: lockfile-bisect, isolation-by-op-class, bounded-domain-analysis, measured-adoption-decision
RUST: Cargo caret-requirement vs lockfile pin, `--precise 0.4.24`, cfg(table_format), release-profile-bench
CONCEPT: per-call routing overhead, hot-path granularity (ns vs µs), wrap-impossibility-by-construction, regression-gate-at-source

## ::REGRESSION

MEASURED (i7-7700, GMATH_PROFILE=embedded, release, criterion medians; consumer = gHyper one-sqrt hyperbolic kernel: 1 sqrt + 1 atanh + storage-tier arithmetic):
  0.4.24 → 36.8 µs/pair · 0.4.27 → 45.7 µs/pair (+24%)
ISOLATION: `power_distance` (pure storage-tier mul/add, ZERO transcendentals) is flat across versions at 23.9 ns ⇒ regression lives in the sqrt/atanh call path, not in basic ops
SUSPECT_1: 0.4.25 "router auto-routes to an exact domain" — per-call classification/routing overhead reaching the plain (non-try) sqrt()/atanh() path
SUSPECT_2: 0.4.27 DomainError fix — the check should gate try_* ONLY; verify the direct path did not inherit it (`fixed_point.rs` diff at `43953ec` touched the shared file)
BISECT_HINT: two candidate commits between the endpoints; a criterion pair-cost bench (sqrt, atanh, single-pair kernel) run per release would catch this class at the source — none exists today
IMPACT_DOWNSTREAM: every gHyper hyperbolic query pays it — structural neighbors and semantic-disk winner verification would regress ~20% if consumers upgraded

## ::ADOPTION_OUTCOME (closes U1's adoption half — for the record)

SITE_1 `klein::power_distance` (O(1) grid hot path): fused swap measured 23.8 → 75.9 ns (3.2×) → REVERTED; Klein-interior inputs |k| < 1, d ≤ 4 ⇒ accumulator < 4 ⇒ wrap impossible
SITE_2 `hash_table::euclidean_distance_sq` (bucket-center scan): same bounded-domain argument, same ns granularity → REVERTED
SITE_3 `hyperbolic_ratio` dist²/den² (exact kernel, µs scale): fused pair recomputes dot+norms per kernel at tier N+1 → +22% (35.4 → 43.1 µs) → REVERTED to shared-dot storage-tier form
EACH SITE now carries a comment documenting the measured decision — future readers won't "fix" them back
KERNELS_KEEP_THEIR_VALUE: `euclidean_distance_squared`/`dot`/`mobius_denominator_sq` are correct and belong in the library — their audience is consumers with UNBOUNDED coordinates or high dimensionality, where storage-tier accumulators genuinely wrap; gHyper's hyperbolic domains are bounded-interior by construction and were never that audience

## ::INVARIANTS

GUARANTEE: gHyper 234 + gFile 181 tests green on the held 0.4.24 locks; requirement strings remain caret-compatible (no `=` pins)
REQUIRES: sqrt/atanh perf restored before consumers move their locks forward
ENSURES: neither consumer uses try_ln/try_sqrt, so holding 0.4.24 forfeits nothing functional from 0.4.27
NEVER: floats in compute paths (standing owner ruling) — any perf fix must stay fixed-point

## ::GRAPH

DEPENDS_ON: your bisect of 29ecea2 (0.4.25 router) vs 43953ec (0.4.27 domain-check placement)
PROVIDES_TO: gHyper/gFile lockfile-advance decision; U1 board entry in gHyper/kanban.html (closed with these findings)
COORDINATES_WITH: U2 (cross-profile rounding contract — still open in your ROADMAP)
BLOCKED_BY: nothing on our side — this notice is FYI + request; consumers are stable on 0.4.24
