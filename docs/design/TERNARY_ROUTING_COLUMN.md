# Ternary Routing Column — Design Decisions (0.4.34)

Why the fractal router's ternary column is scoped the way it is. Companion
to `BALANCED_TERNARY_CONTRACT.md` (which proves the domain this column
routes into). Decisions ratified by the owner 2026-08-14.

## Context

Since v0.4.0 the router's classifier has computed `TERNARY_BIT` (shadow
denominator stripped of 3s), but no routing-table column consumed it:
ternary-exact values fell through to Binary — where they are *inexact* —
or to the symbolic rational fallback. The 0.4.33 contract suite plus the
post-0.4.33 gap-closing (tiers 1–6 adversarial coverage, two sign-extension
fixes, checked storage narrowing) made the domain formally defended;
this column makes it reachable.

Target class: values exact in ternary but not in binary/decimal —
denominator 3^k, k ≥ 1 (1/3, 2/3, 100/3…). Domain preference order
becomes **Binary > Decimal > Ternary > Symbolic**, ranked by cost:
integers keep routing Binary; only the 3-adic class moves. The win is
replacing rational-pair arithmetic (two multiplies + add + gcd
normalization per op on num/den pairs up to I512 — **not** BigInt, which
is feature-gated behind `infinite-precision` and absent from default
builds) with single integer adds on scaled raws.

## Decision 1 — the column covers add/sub (+neg) only, not mul/div

**This is a correctness argument, not caution.**

For operands representable at scale 3^F (denominators dividing 3^F):

- **add/sub**: the result's denominator divides lcm(3^a, 3^b) =
  3^max(a,b) ≤ 3^F. Routed addition is **always exact**. Same for
  subtraction and negation.
- **mul**: denominators multiply — 3^a · 3^b = 3^(a+b), which may exceed
  3^F. When it does, ternary **truncates toward zero** (contract §3)
  while today's symbolic path is **exact**. Routing such a multiply would
  make routing *change results* — violating the router's fail-safe
  invariant: *wrong routing costs performance, never correctness*.
- The static table indexes 4-bit exactness classes; it **cannot see
  denominator exponents**, so it cannot separate safe multiplies
  (a+b ≤ F) from unsafe ones. A compile-time table has no correct mul
  column to offer.
- **div** produces non-3-adic quotients in general; same argument,
  stronger.

Mul routing can become viable later via (a) a dispatch-time
shadow-exponent guard (route to ternary only when a+b ≤ F, checked
against the actual shadows, falling back otherwise), or (b) an explicit
owner re-scoping of the fail-safe invariant. Option (a) is the natural
follow-up; neither blocks this release. This scoping also **decouples the
column from the pending mul/div rounding decision** (toward-zero vs the
theorem's tie-free nearest) — that decision can now be taken on its own
schedule.

## Decision 2 — coercion failure falls back silently to the previous route

Gap-closing proved (and pinned by test) that narrow profiles cannot hold
Tier-2+ ternary raws in FASC storage: realtime caps at TQ8.8/i32,
compact at TQ16.16/i64. Coercing a large 3-adic value into ternary on
those profiles is a legitimate `TierOverflow`.

If the router propagated that error, routing would *introduce failures*
where the pre-column path succeeded — the other way to violate the
fail-safe invariant. So: on coercion failure, dispatch proceeds exactly
as if the column did not exist (rational fallback).

Implementation note: this costs nothing structural. The routing seam
(`try_route_coerce`) already returns `Option`; `None` *is* the rational
fallback. A failed `convert_to_ternary` maps to `None`.

## Scope honesty — what the column does NOT change

- **Same-domain pairs bypass the router.** `1/3 + 1/3` parses both
  operands Symbolic and takes the native symbolic path untouched. The
  column fires on **cross-domain pairs** (Ternary ⊕ Symbolic,
  Ternary ⊕ Binary-integer, Symbolic-3-adic ⊕ integer, …). Intercepting
  the symbolic same-domain fast path would tax every symbolic add with a
  classify; it stays deliberately out of scope (revisit only with a
  measured consumer case).
- Transcendentals on ternary operands keep routing to the binary engines
  (contract: ternary has no native transcendentals).
- Conversion exactness: `convert_to_ternary` computes `num·3^F / den` in
  exact integer arithmetic; for the routed class (den = 3^k, k ≤ F) the
  division is remainder-free — coercion is exact by construction. The
  same function truncates for non-3-adic dens, but the table never routes
  those to Ternary.

## Found during implementation

- `convert_to_ternary`'s tier-3 and fallback arms stored raws through the
  *unchecked* `to_binary_storage` — the same silent-wrap class fixed in
  0.4.33's `ternary_to_storage`, still live here and reachable today via
  `set_gmath_mode("…:ternary")` on narrow profiles. Fixed in this release
  (routed through the checked conversion); pinned by test.
- **Symbolic operands were invisible to the classifier**: `StackValue::
  Symbolic` carries no shadow (it IS exact), so `classify()` fell back to
  `SYMBOLIC_ONLY` and a symbolic 1/3 could never trigger the column. The
  classifier now reads the rational's own denominator — richer than any
  shadow. Deliberate side effect: symbolic operands with 2-adic/10-adic
  denominators can now coerce into Binary/Decimal wherever both operands
  are exact there; exactness is preserved in every such case by the
  intersection rule.
- A pre-existing, unrelated realtime defect surfaced while validating:
  integer literals beyond binary storage range (`32768`+ at Q16.16) fail
  PARSE with `Overflow` instead of falling back to Symbolic — bisected to
  ≤0.4.32, filed as ROADMAP 0.5.0 item 0. Not addressed here (out of the
  column's scope).

## Measured cost (embedded, release, full evaluate pipeline)

`0t2 + 1/3` (routed to ternary): **334 ns/eval**; `0t2 + 1/7` (rational
fallback): **355 ns/eval** — ~1.06×. Honest reading: at single-expression
scale the win is modest because literal parsing dominates both paths. The
column's value is architectural — the 3-adic exactness class finally
reaches the domain that represents it, results stay in fixed-point form
instead of rational pairs, and the seam is in place for the guarded mul
extension.

## Validation

- Routing-table pins: Add/Sub of ternary∩symbolic classes → `Ternary`;
  Mul/Div of the same classes → `Symbolic` (unchanged); integer∩integer →
  `Binary` (unchanged precedence).
- End-to-end: cross-domain 3-adic add routes into the ternary domain and
  equals the plain-literal computation exactly (router-difference zero).
- Fallback: a 3-adic pair whose coercion overflows narrow storage still
  evaluates correctly on every profile (exercises the silent fallback on
  realtime/compact, the ternary route on wide profiles).
- Wrap-fix pins for the two `convert_to_ternary` arms.
- Measured before/after cost of the routed case in the CHANGELOG.
