# The inference module as a gMath extension

Status: IMPLEMENTED 2026-10-17 (phase 7.5 of the width-as-type work; see
section 10). Sections 1 to 9 are the proposal of 2026-10-10 and the owner's
decisions of 2026-10-15, kept as written.

## 1. What is being decided

`g_math::tq19` (the TQ1.9 ternary weight formats, their dot products and
matvecs, the five-trit row-scaled form, the planar and hybrid repacks, the
bfloat16 / binary16 decoders, the two quantisers, the AVX2 kernels) ships
today inside the core crate behind the `inference` feature. The core's
mathematics is open under MIT OR Apache-2.0 and stays so. The question is
whether the inference layer should instead be a separate crate that depends
on the core through its public surface only, maintained and licensed on its
own, released in lockstep with the inference engine that drives it, and
serving as the worked example of how an extension is written.

Two facts frame the decision:

- Every published `g_math` from 0.4.25 to 0.6.7 carries `tq19` under the
  open licences. A licence cannot be withdrawn from code already published;
  anyone may keep building on 0.6.7's kernels. The extension changes what
  the *next* versions offer, and where the work that follows (new formats,
  new kernels, the engine-specific fast paths) lives.
- The module is already shaped like an extension. Phase 4d made it generic
  over `Fixed<S, F>` through two traits it defines itself (`TqWord`,
  `WideOutput`) and implemented for the core's five words; it reaches the
  core through `Fixed`, `Compute<T>`, `Storage`, `WideInt`, `OverflowDetected`
  and the raw-slice boundary (`Fixed::raw_slice`, `from_raw_slice`,
  `raw`, `from_raw`). Nothing it uses is `pub(crate)`. The only profile
  reads (`frac_config::FRAC_BITS`, `FixedPoint`) are in its own tests.

Measured on the tree (2026-10-10): 4,716 lines in `src/fixed_point/tq19/`
(ops 999, planar 858, hybrid 653, tq5 597, mod 468, rowscaled 368, simd 253,
bits 179, quantize 169) plus `compute_tier.rs` (172); ten integration test
files reference the two modules; `rayon` is the feature's only dependency.

## 2. The boundary

| Stays in the core (open) | Moves to the extension |
|---|---|
| `Fixed<S, F>`, `FixedNum`, the word traits, every engine, the canonical layer | `tq19` wholesale: `TQ19Matrix`, `PlanarTQ19`, `HybridTQ19`, `RowScaledTQ19`, `RowScaledTQ5`, `tq19_dot`, `trit_dot`, `packed_trit_dot`, the matvec family (plain, `_par`, `_batch`, `_batch_par`, `_q2f` forms), `TqWord`, `WideOutput`, `bits`, `quantize`, `simd`, the trit decode table and the `SCALE` / `MAX_RAW` / `MIN_RAW` constants |
| `fused` (softmax, silu, rms norm, dot_many, sigmoid_mul, entropy): general kernels, public since 0.3.0, consumed by the geometry crates too | the `rayon` dependency and the `_par` / `_batch_par` forms |
| `compute_tier` (exp, ln, sqrt, sinhcosh, sigmoid, softplus, ln1p, `from_fixed`, `to_fixed`, `one`, `ceiling` at 2F): the documented facade over the kernels, which the extension itself needs | the integration tests `tq19_validation`, `tq5_validation`, `weight_bits_validation`, `batch_into_validation`, `tq19_bench`, and the tq19 sections of `defects_065_validation`, `narrowing_defects_validation`, `width_types_validation`, `differential_065_validation` |
| the `inference` feature name disappears with the module | `docs/README_TERNARY.md`'s TQ1.9 half (the balanced-ternary domain half stays) |

Recommendation on the two borderline modules: `compute_tier` stays in the
core and loses its feature gate. It is 172 lines of wrappers that give the
kernels a documented, semver-bound name; the extension consumes it, the
engine consumes it directly for its activations, and it carries no inference
specific knowledge. `fused` stays for the same reason. If the owner wants
the engine's activation kernels (sigmoid and softplus at the compute tier)
closed as well, they are two functions that can be copied into the
extension without touching the core; the exp and ln engines they call are
the open mathematics either way.

## 3. The contract the extension consumes

An extension must not depend on `#[doc(hidden)]` items: they are not
semver-bound and phase 7 may rename them. The inference module today uses
these core items; all are public and documented except the ones marked.

| Item | Used for | Status |
|---|---|---|
| `Fixed<S, F>`: `raw`, `from_raw`, `raw_slice`, `from_raw_slice` (`#[repr(transparent)]`) | the activation and output buffers | public |
| `Compute<T>` | the wide accumulator and the `_q2f` outputs | public |
| `Storage`: `Compute`, `widen`, `narrow` | accumulate, narrow once | public, documented |
| `WideInt`: `one`, `bit`, `shr_bits`, `from_i128_checked`, `checked_add_wide` | the scale and round step | public, documented |
| `OverflowDetected` | the quantisers' errors | public |
| `I256`, `I512` | the `TqWord` impls for the wide words | public |
| `compute_tier::{from_fixed, to_fixed, one}` | the five-trit scale path | public |
| `FixedNum::from_raw_storage` (one use) | a raw word to a value | hidden; replace with `Fixed::from_raw` |

So the core needs no new API for the extraction. What it should gain is a
statement: a short `docs/EXTENSIONS.md` naming the contract (the items above
plus `FixedNum`'s documented methods) as the surface an extension may rely
on, and the rule that hidden items are not part of it.

## 4. The extension crate

- Name `g_math_inference` (working name), its own repository, `publish =
  false` in `Cargo.toml` so a publish cannot happen by accident, licence
  file of the owner's choosing (section 6). Depends on `g_math = "0.7"`
  (the first core without the module) and on `rayon` behind a `par` feature
  that is on by default.
- Layout mirrors the module: `src/lib.rs` (the constants, `TQ19Matrix`, the
  free dot products), `ops.rs` (`TqWord`, `WideOutput`, the word impls),
  `planar.rs`, `hybrid.rs`, `rowscaled.rs`, `tq5.rs`, `bits.rs`,
  `quantize.rs`, `simd.rs`. Every `crate::fixed_point::...` path becomes a
  `g_math::fixed_point::...` path; `super::` paths inside the module are
  unchanged. A script does the move (the phase 4d span rewriter pattern),
  and the result must be a byte-identical body apart from the paths, so the
  differential gate below is the whole proof.
- The traits stay local and sealed by the extension's own private marker,
  implemented for the core's `i32`, `i64`, `i128`, `I256`, `I512` (a local
  trait on foreign types is allowed; the core's sealed traits are bounds,
  never implemented outside). The word-specific fast paths remain the
  hidden hook methods with `None` / `false` defaults (phase 4b's pattern),
  so a new fast path is one impl, not a new public function.
- No `cfg(table_format)` anywhere: after phase 7 the extension is generic
  over the format by construction, and its tests pick formats explicitly
  (`Fixed<i32, 10>`, `Fixed<i64, 32>`, ...). This is why the extraction
  waits for phase 7: moving the module now would carry the profile reads
  in its tests across, and they would have to be removed twice.
- The engine depends on the extension by git tag (or a private registry),
  never by path, so the pair of versions it was built with is recorded in
  its lockfile; the extension's `CHANGELOG` names the core version each
  release was verified against.

## 5. Gates

1. Differential, both directions: `g_math_inference` evaluated against
   `g_math 0.6.7` with `inference` (the last open module) on the 4d corpus
   (`tq19_dot`, `trit_dot`, `packed_trit_dot`, the q2f forms, the four
   matrix forms with their batch and parallel variants, the five-trit form,
   both decoders, both quantisers; outcome = value | error | panic), and
   against the core's own copy for as long as the core still carries one.
   Zero differences is the exit criterion, as for every width phase.
2. The all-widths identities of `tests/width_types_validation.rs`'s
   inference section, moved as they are.
3. Perf: the phase 4d harness (one binary per crate, release, quiet
   machine, isolated calls) on the same matvec sizes; parity within the
   band, the hybrid matvec's two codegen states noted as in ROADMAP.
4. The core after removal: full sweep, `PUBLIC_API.md` regenerated, no
   `rayon` in the dependency tree without the feature, the `inference`
   feature gone from `Cargo.toml` and the `check-cfg` list, README and
   `docs/README_TERNARY.md` updated, CHANGELOG entry under 0.7.0 naming the
   move and the last open version.

## 6. Licensing mechanics (options, owner's call)

- Private git repository, access by invitation; the engine and licensees
  pull a tag over ssh. Simplest; the licence is a file in the repository
  and the terms of access. No crates.io presence.
- A private registry (any Cargo-compatible one); same visibility control
  with `cargo publish --registry` and index-based version resolution.
  Better once more than one consumer exists.
- crates.io with a proprietary `license-file`: technically possible, but
  crates.io publishes the source to everyone; the protection is legal only.
  Not recommended for the stated goal.

In every option the open history remains: 0.6.7's `tq19` is the floor
anyone can build on. What the extension protects is the work after it:
the formats, kernels and fast paths written for the engine, and the pace
at which they appear.

## 7. The extension pattern, as a guide for other authors

The extraction doubles as the reference for writing any gMath extension.
The rules it demonstrates:

1. Depend on the documented surface only: `Fixed`, `Compute`, the word
   traits as bounds, `compute_tier`, the raw-slice boundary. Never a hidden
   item, never `pub(crate)`.
2. Be generic over the format from the first line: `Fixed<S, F>` in
   signatures, a local sealed trait per word family for the parts that
   differ per word, hidden hook methods with safe defaults for fast paths.
3. Convert at the edge: the caller's buffers enter through
   `Fixed::from_raw_slice` and leave as `Vec<Fixed<S, F>>` (or through an
   `_into` form); no public function takes or returns a bare word.
4. Prove bit-identity, not plausibility: a differential gate against the
   previous version of the same code, outcome classes included.
5. Measure on a quiet machine with one binary per crate; a perturbed
   harness before believing a 2 percent reading.
6. Keep mpmath or exact-integer references for every numeric claim; the
   inference module's are integer oracles (i128 row dot) and the decoders'
   bit tables.

## 8. Schedule

After phase 7 and before the 0.7.0 release (phase 8), as "phase 7.5": one
move script, the two differential gates, the perf run, the core removal
with its docs, then the engine's switch. Estimated at one working day of
gate time on top of the mechanical move, because the module is already in
its extension shape.

## 9. Open questions for the owner

- Name of the crate and of its repository.
- Whether `compute_tier` stays open (recommended) or the engine's two
  activation functions move with the module.
- Which licensing mechanism (section 6).
- Whether the extension keeps the `inference` word in its name, given the
  engine's own naming.

### 9a. Decisions (owner sign-off, 2026-10-15)

| Question | Decision |
|---|---|
| How 0.7.0 ships | As g_math 0.7.0 (not a new crate name) |
| Crate name | `g_math_inference` |
| Keep the word "inference" | Keep it |
| Where `compute_tier` lives | Stays in the open core, without a feature gate |
| How licensees get the extension | crates.io with a proprietary licence file. Owner's reasoning: a company using it will do so for audit or legal reasons, so use becomes evident through audits and procedures; the protection is legal, by choice. |
| When the move happens | After phase 7c, before the 0.7.0 release |

Owner's note on timing: a 0.6.8 release may separate the module from the
0.6 line before 0.7.0. To settle with the owner before it is cut: removing
the `inference` feature in a patch release breaks every consumer that pins
"0.6" with that feature (Cargo resolves them to 0.6.8 on their next update).
A 0.6.8 that keeps the feature, marks it deprecated and points to
`g_math_inference` avoids that; the removal itself then lands in 0.7.0.
Either way the published 0.4.x to 0.6.7 releases keep the module under
their open licences.

## 10. As built (2026-10-17)

- The crate `g_math_inference` (its own private repository) holds the
  module's files unchanged apart from their paths (`crate::fixed_point` to
  `g_math::fixed_point`, `crate::tq19` to `crate`). It uses only documented
  core items: `Fixed`, `FixedNum`, `Compute`, `Storage`, `WideInt`, `I256`,
  `I512` and `imperative::OverflowDetected`. The core gained the exports of
  `Compute<T>` and `I256` .. `I2048` (they were hidden); nothing else.
- Differences from the proposal: `rayon` is a plain dependency (every
  consumer uses the parallel forms; no `par` feature); the crate is
  `publish = false` until the owner publishes it with 0.7.0, with a
  placeholder licence file until the owner's text replaces it; it depends on
  the core by path during development and on `g_math = "0.7"` at release.
- Tests moved with the module and match their counts in the core: 195 unit
  tests; tq19 138, tq5 18, weight_bits 48, batch_into 23, bench 30
  integration tests; the tq19 parts of three shared files (16, 1 and 8).
- Differential gate (`differential/` in the extension): every kernel against
  g_math 0.6.7 with `inference` at realtime F = 10 and 16, compact, embedded,
  balanced and scientific; 0 differences of value, error or panic.
- Core after removal: no `inference` feature, no `rayon` in the dependency
  tree, `compute_tier` without a gate; full suite 6702 tests (7180 minus the
  478 moved), differential against 0.6.7 12/12 at six builds.
- 0.6.8 (from v0.6.7) marks the module deprecated, naming the new crate;
  release order: g_math 0.7.0, g_math_inference 0.1.0, then 0.6.8.
