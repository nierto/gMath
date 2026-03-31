# g_math 0.3.90 — Inference Integration Guide

> For: Maniference dev team
> Addresses: Blockers 1-3 from GMATH-REALTIME-HANDOUT.md

## Upgrade

```toml
# Cargo.toml
[dependencies]
g_math = { version = "0.3.90", features = ["inference"] }
```

The `inference` feature enables:
- TQ1.9 module (`g_math::tq19`) with rayon row-parallel matvec
- All parallel variants (`matvec_par`, `matvec_batch_par`, `packed_trit_matvec_par`)

## Build

```bash
# Q8.24 realtime profile (recommended for inference)
GMATH_PROFILE=realtime GMATH_FRAC_BITS=24 cargo build --release

# IMPORTANT: clear incremental cache when switching profiles or FRAC_BITS
rm -rf target/debug/incremental/ target/release/incremental/
```

### Q8.24 vs Q16.16

| | Q16.16 (old default) | Q8.24 (recommended) |
|---|---|---|
| Integer range | +/-32,767 | +/-128 |
| Fractional precision | 4.8 digits | 7.2 digits |
| Headroom after /19683 | 0 bits | 7.3 bits |
| Embedding precision (0.002) | 7 bits | 15 bits |
| SIMD throughput | same | same |
| Storage type | i32 | i32 |

Activations are typically [-5, +5], so 8 integer bits is sufficient.
FRAC_BITS can be any value from 2 to 30. The default (when env var is not set) is 16.

## Blocker 1: RoPE Theta Overflow

`sincos_wide` takes the angle as a raw i64 in **fixed Q32.32 format** — integer range +/-2.1 billion, independent of FRAC_BITS. Computes sin/cos via native Q64.64 (hardware i128), narrows result to storage tier.

```rust
use g_math::fixed_point::imperative::FixedPoint;

// At model load — precompute RoPE frequency table using i64 arithmetic:
const Q32_SCALE: i64 = 1 << 32;

fn precompute_rope_freqs(theta: f64, head_dim: usize, max_seq_len: usize) -> Vec<Vec<(FixedPoint, FixedPoint)>> {
    let mut freq_table = Vec::with_capacity(max_seq_len);
    for pos in 0..max_seq_len {
        let mut pairs = Vec::with_capacity(head_dim / 2);
        for i in 0..(head_dim / 2) {
            // Compute frequency: 1.0 / theta^(2i/d)
            // Use f64 here — this is one-time precomputation, not inference hot path
            let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
            let angle = freq * pos as f64;

            // Convert angle to Q32.32 raw i64
            let angle_q32: i64 = (angle * Q32_SCALE as f64) as i64;

            // Compute (sin, cos) at full precision, narrowed to storage tier
            let (sin_val, cos_val) = FixedPoint::sincos_wide(angle_q32);
            pairs.push((sin_val, cos_val));
        }
        freq_table.push(pairs);
    }
    freq_table
}

// Per token — apply RoPE (just lookup + multiply, no transcendentals):
fn apply_rope(q: &mut [FixedPoint], k: &mut [FixedPoint], freqs: &[(FixedPoint, FixedPoint)]) {
    for (i, (sin_val, cos_val)) in freqs.iter().enumerate() {
        let q_even = q[2 * i];
        let q_odd = q[2 * i + 1];
        q[2 * i] = q_even * *cos_val - q_odd * *sin_val;
        q[2 * i + 1] = q_even * *sin_val + q_odd * *cos_val;
        // Same for k
        let k_even = k[2 * i];
        let k_odd = k[2 * i + 1];
        k[2 * i] = k_even * *cos_val - k_odd * *sin_val;
        k[2 * i + 1] = k_even * *sin_val + k_odd * *cos_val;
    }
}
```

`sincos_wide` handles theta=10M without overflow. Validated at 0 ULP for angles up to 1,000,000.

## Blocker 2: Precision Collapse

Q8.24 provides 7.3 bits of headroom after /19683 division (vs 0 for Q16.16).
No code changes needed — just set `GMATH_FRAC_BITS=24` at build time.

The TQ1.9 matvec SIMD kernel is unchanged — raw i32 multiplies are Q-format agnostic. The /SCALE narrowing automatically uses the correct FRAC_BITS.

## Blocker 3: F16 Embedding Decode

Q8.24 represents small values with 256x more precision than Q16.16:
```
0.002 * 2^16 =   131 raw (7 bits of precision)
0.002 * 2^24 = 33554 raw (15 bits of precision)
```

`FixedPoint::from_f64(0.002)` on Q8.24 retains 15 significant bits. No API change needed.

## Performance

All transcendentals now dispatch to native hardware i128 (was software-emulated I256):

| Function | Throughput (release) | Use in inference |
|----------|---------------------|------------------|
| exp | 1.5M ops/s | softmax, SiLU |
| sin/cos | 3.4-3.8M ops/s | RoPE |
| sqrt | 245K ops/s | RMSNorm |
| ln | 1.1M ops/s | entropy |

Q8.24 throughput is identical to Q16.16 — same types, same SIMD, just different bit interpretation.

## API Summary

```rust
use g_math::fixed_point::imperative::FixedPoint;
use g_math::tq19::{TQ19Matrix, tq19_dot, trit_dot};

// Transcendentals (unchanged API, faster dispatch)
let y = x.exp();           // SiLU: x * sigmoid(x)
let y = x.sqrt();          // RMSNorm
let (s, c) = x.sincos();  // When input fits in storage range

// Wide-range sincos (new — Blocker 1 fix)
let (s, c) = FixedPoint::sincos_wide(angle_q32_i64);

// TQ1.9 (now behind inference feature)
let result = TQ19Matrix::new(data, rows, cols).matvec(activations);
let result = TQ19Matrix::new(data, rows, cols).matvec_par(activations);  // rayon parallel
```

## Validation

Tested against mpmath at 50+ digit precision:
- Q16.16: 90 test points, 0 ULP across all 18 transcendentals
- Q8.24: 30 test points, 0-1 ULP across 6 core transcendentals
- sincos_wide: 0 ULP at angles 10K, 100K, 1M
- All 5 profiles (realtime/compact/embedded/balanced/scientific): 0 warnings, 0 failures
