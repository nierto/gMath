# Fused operations

Whole computation patterns evaluated at the wide compute tier with a single
downscale at the end, no intermediate materialization.

## What it is

`g_math::fixed_point::imperative::fused` collects the accumulation patterns common
in numeric and ML code (norms, distances, softmax, normalization, activations) 
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
| `inv_sqrt_sum_sq(&[x])` | 1/√(Σ xᵢ²): the reciprocal norm; one call + N multiplies replaces N per-component divisions in normalization |
| `euclidean_distance(&a, &b)` | √(Σ (aᵢ−bᵢ)²) |
| `softmax(&scores)` | numerically stable softmax |
| `softmax_mix(&scores, &values)` | softmax(scores) · V, weights never materialized to storage |
| `softmax_mix_values`, `softmax_mix_flat`, `softmax_mix_flat_values` | the same mix without the observer weights, and/or with the value rows in one contiguous buffer (0.6.5) |
| `dot_many(&query, &keys_flat, dim)` | one query against many keys in one buffer, each rounded as `dot` rounds (0.6.5) |
| `rms_norm(&x, &weight, eps_q64)`, `rms_norm_in_place` | `x[i] * weight[i] / sqrt(mean(x²)+ε)`, each output rounded ONCE from the exact product with a full-precision reciprocal root; not the same as multiplying by the stored factor below (0.6.5) |
| `rms_norm_factor(&x, eps)` | 1/√(mean(x²)+ε), ε a storage value |
| `rms_norm_factor_eps_wide(&x, eps_q64)` | the same with ε in Q64.64, added at the compute tier (0.6.4) |
| `silu(x)` | x/(1+e⁻ˣ) |
| `quadratic_form(&v, &m)` | vᵀMv with ONE rounding: exact triple products at 3·FRAC_BITS, nearest with ties toward +∞; the correctly rounded scalar, always inside `Interval::quadratic_form` (0.6.1) |
| `try_quadratic_form(&v, &m)` | the same, `Err(TierOverflow)` instead of a panic where the result leaves storage |

`rms_norm_factor` takes ε at the storage tier, so an ε below `2^-FRAC_BITS` is
zero: on realtime at Q22.10 both `1e-5` and `1e-6` vanish, and an all-zero input
is `Err(DivisionByZero)` instead of `1/√ε`. `rms_norm_factor_eps_wide` takes ε as a
Q64.64 integer (`g_math::wide::try_from_str("1e-5", 64)`) and rounds it once to
the compute tier (`2·FRAC_BITS` fractional bits; nearest, ties toward +∞; exact on
compact and wider). On realtime the compute tier still limits it: at Q22.10,
`1e-5` becomes `10/2^20` (`9.54e-6`) and an all-zero input gives 323.83, where the
exact `1/√1e-5` is 316.23.

Error conditions are listed on each function. In short: `softmax` can only
fail when the number of scores reaches `2^(63 - 2·FRAC_BITS)` on realtime
(never on wider profiles); `softmax_mix` returns `Err(TierOverflow)` when a
numerator leaves the compute tier, which on realtime needs
`n · max|v_raw| ≥ 2^(63 - FRAC_BITS)`; the RMS-norm factors return
`Err(DivisionByZero)` for empty input or a zero `mean + ε`, `Err(DomainError)`
for a negative one (0.6.5), and `Err(TierOverflow)` when the sum of squares
leaves the compute tier or the result leaves storage.

`softmax_mix` exists because materializing softmax weights to storage tier before
the value mix imposes a 2^−FRAC_BITS resolution floor: under a low-fractional-bit
profile a small attention weight rounds to zero and its value row vanishes from
the mix. Keeping the weights at the compute tier through the accumulation removes
that floor. It returns the mixed output plus a storage-quantized copy of the
weights for observers (attention recording, diagnostics): the observer copy is
not what the mix used.

## Public API

See **[PUBLIC_API.md → Fused operations](../PUBLIC_API.md#fused-operations)** and
[docs.rs](https://docs.rs/g_math). Fused ops that can overflow the compute tier on
pathological inputs (`softmax`, `softmax_mix`) return `Result<_, OverflowDetected>`
rather than wrapping silently.

## Behaviour & limits

- Same numeric result as the equivalent unfused sequence, minus the intermediate
  rounding, never worse, and strictly better where a materialization floor would
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

Built by **Niels Erik Toren** · [support & donations](../README.md#author--support).
