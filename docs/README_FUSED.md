# Fused operations

Whole computation patterns evaluated at the wide compute tier with a single
downscale at the end — no intermediate materialization.

## What it is

`g_math::fixed_point::imperative::fused` collects the accumulation patterns common
in numeric and ML code — norms, distances, softmax, normalization, activations —
and runs each entirely at tier N+1, materializing to storage exactly once. This
removes both the per-step rounding and the per-step materialization cost that a
naïve `dot(x, x).sqrt()` or a materialized-weights softmax would incur. It is the
fastest correct path for these specific shapes.

## Usage

```rust
use g_math::fixed_point::FixedPoint;
use g_math::fixed_point::imperative::fused;

let xs = [FixedPoint::from_str("3"), FixedPoint::from_str("4")];
let norm = fused::sqrt_sum_sq(&xs);                 // 5, accumulated wide

let scores = [FixedPoint::from_int(1), FixedPoint::from_int(2), FixedPoint::from_int(3)];
let weights = fused::softmax(&scores).unwrap();     // numerically stable

// Fused attention step: softmax(scores) · V without materializing the weights.
let v0 = [FixedPoint::from_int(1), FixedPoint::from_int(0)];
let v1 = [FixedPoint::from_int(0), FixedPoint::from_int(1)];
let v2 = [FixedPoint::from_int(2), FixedPoint::from_int(2)];
let values: [&[FixedPoint]; 3] = [&v0, &v1, &v2];
let (mixed, observer_weights) = fused::softmax_mix(&scores, &values).unwrap();
```

## Operations

| Function | Computes |
| -------- | -------- |
| `sqrt_sum_sq(&[x])` | √(Σ xᵢ²) |
| `euclidean_distance(&a, &b)` | √(Σ (aᵢ−bᵢ)²) |
| `softmax(&scores)` | numerically stable softmax |
| `softmax_mix(&scores, &values)` | softmax(scores) · V, weights never materialized to storage |
| `rms_norm_factor(&x, eps)` | 1/√(mean(x²)+ε) |
| `silu(x)` | x/(1+e⁻ˣ) |

`softmax_mix` exists because materializing softmax weights to storage tier before
the value mix imposes a 2^−FRAC_BITS resolution floor: under a low-fractional-bit
profile a small attention weight rounds to zero and its value row vanishes from
the mix. Keeping the weights at the compute tier through the accumulation removes
that floor. It returns the mixed output plus a storage-quantized copy of the
weights for observers (attention recording, diagnostics) — the observer copy is
not what the mix used.

## Public API

See **[PUBLIC_API.md → Fused operations](../PUBLIC_API.md#fused-operations)** and
[docs.rs](https://docs.rs/g_math). Fused ops that can overflow the compute tier on
pathological inputs (`softmax`, `softmax_mix`) return `Result<_, OverflowDetected>`
rather than wrapping silently.

## Behaviour & limits

- Same numeric result as the equivalent unfused sequence, minus the intermediate
  rounding — never worse, and strictly better where a materialization floor would
  otherwise bite.
- `softmax_mix` requires every value row to have the same length; a ragged matrix
  is a programming error and panics.
- These are binary-domain (`FixedPoint`) operations.

Determinism and rounding guarantees are in **[CONTRACT.md](../CONTRACT.md)**.

## Disclaimer

This software is provided **"as is"**, without warranty of any kind, express or
implied. Use of this software is entirely at your own risk. In no event shall the
author or contributors be held liable for any damages arising from the use or
inability to use this software.

---

Built by **Niels Erik Toren** — [support & donations](../README.md#author--support).
