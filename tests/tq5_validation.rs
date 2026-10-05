//! `RowScaledTQ5` (0.6.5), five-trit row-scaled matrices: every kernel path
//! against an i128 reference, the quantiser against its definition, and the
//! byte layout. Integer inputs only, no floats.
//!
//! The kernel takes 32-bit activations, so the suite runs on the realtime
//! profile (any GMATH_FRAC_BITS; only the wide output depends on the split).

#![cfg(all(feature = "inference", table_format = "q16_16"))]

use g_math::fixed_point::frac_config::FRAC_BITS;
use g_math::tq19::bits::{HalfKind, WeightBits};
use g_math::tq19::quantize::quantize_tq5_rowscaled;
use g_math::tq19::{RowScaledTQ5, TQ5_MAX};

/// SplitMix64: deterministic pseudo-random integers, no floats.
fn rng(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn random_matrix(seed: &mut u64, rows: usize, cols: usize) -> RowScaledTQ5 {
    let data: Vec<i8> = (0..rows * cols)
        .map(|_| (rng(seed) % (2 * TQ5_MAX as u64 + 1)) as i64 - TQ5_MAX as i64)
        .map(|v| v as i8)
        .collect();
    // scales up to ~0.05 (Q32.32): realistic rows have max|w| < 6 units of 121 codes
    let scales: Vec<u64> = (0..rows).map(|_| rng(seed) % (1u64 << 27) + 1).collect();
    RowScaledTQ5::from_parts(rows, cols, data, scales)
}

fn random_activation(seed: &mut u64, n: usize) -> Vec<i32> {
    (0..n)
        .map(|_| (rng(seed) % 200_000) as i32 - 100_000)
        .collect()
}

#[test]
fn matvec_matches_i128_reference() {
    let mut seed = 7u64;
    for &(rows, cols) in &[(1usize, 1usize), (3, 17), (64, 64), (37, 1000), (8, 4096)] {
        let m = random_matrix(&mut seed, rows, cols);
        let x = random_activation(&mut seed, cols);
        let got = m.matvec_par(&x);
        let wide = m.matvec_q2f_par(&x);
        assert_eq!(m.matvec(&x), got, "sequential form, {rows}x{cols}");
        assert_eq!(m.matvec_q2f(&x), wide, "sequential wide form, {rows}x{cols}");
        for r in 0..rows {
            let acc: i128 = (0..cols)
                .map(|c| m.data()[r * cols + c] as i128 * x[c] as i128)
                .sum();
            let want = ((acc * m.scales_q32()[r] as i128) >> 32) as i32;
            assert_eq!(got[r], want, "row {r} of {rows}x{cols}");
            let want_wide = ((acc * m.scales_q32()[r] as i128) >> (32 - FRAC_BITS)) as i64;
            assert_eq!(wide[r], want_wide, "wide row {r} of {rows}x{cols}");
        }
    }
}

#[test]
fn batch_equals_single() {
    let mut seed = 11u64;
    let m = random_matrix(&mut seed, 48, 300);
    let xs: Vec<Vec<i32>> = (0..5).map(|_| random_activation(&mut seed, 300)).collect();
    let refs: Vec<&[i32]> = xs.iter().map(|v| v.as_slice()).collect();
    let batch = m.matvec_batch_par(&refs);
    for (b, x) in xs.iter().enumerate() {
        assert_eq!(batch[b], m.matvec_par(x), "batch column {b}");
    }
}

#[test]
fn byte_layout_roundtrip() {
    let mut seed = 23u64;
    let m = random_matrix(&mut seed, 16, 128);
    assert_eq!(m.size_bytes(), 16 * 128 + 16 * 8);
    let mut buf = Vec::new();
    m.write_to(&mut buf).unwrap();
    // rows (u32) | cols (u32) | codes (i8) | scales (u64), little-endian
    assert_eq!(buf.len(), 8 + 16 * 128 + 16 * 8);
    assert_eq!(&buf[..8], &[16, 0, 0, 0, 128, 0, 0, 0]);
    assert_eq!(buf[8] as i8, m.data()[0]);
    assert_eq!(&buf[8 + 16 * 128..8 + 16 * 128 + 8], &m.scales_q32()[0].to_le_bytes());
    let back = RowScaledTQ5::read_from(&mut &buf[..]).unwrap();
    assert_eq!(back.data(), m.data());
    assert_eq!(back.scales_q32(), m.scales_q32());
    assert_eq!((back.rows(), back.cols()), (16, 128));
    // truncated input and an out-of-range code are refused
    assert!(RowScaledTQ5::read_from(&mut &buf[..buf.len() - 1]).is_err());
    buf[8] = 0x80; // -128
    assert!(RowScaledTQ5::read_from(&mut &buf[..]).is_err());
}

#[test]
#[should_panic(expected = "code outside")]
fn from_parts_refuses_an_out_of_range_code() {
    let _ = RowScaledTQ5::from_parts(1, 2, vec![0, 122], vec![1]);
}

#[test]
#[should_panic(expected = "exceeds storage range")]
fn result_past_storage_panics() {
    // 121 * (2^30 - 1) * 1.0 is far past i32
    let m = RowScaledTQ5::from_parts(1, 1, vec![121], vec![1u64 << 32]);
    let _ = m.matvec(&[(1 << 30) - 1]);
}

/// `i32::MIN` has no 32-bit magnitude: the split is refused and the scalar
/// path gives the exact value.
#[test]
fn i32_min_activations_are_exact() {
    let cols = 40;
    let data: Vec<i8> = (0..cols).map(|i| if i % 2 == 0 { 121 } else { -121 }).collect();
    let m = RowScaledTQ5::from_parts(1, cols, data.clone(), vec![3]);
    let x: Vec<i32> = (0..cols).map(|i| if i % 5 == 0 { i32::MIN } else { 77 - i as i32 }).collect();
    let acc: i128 = data.iter().zip(&x).map(|(&c, &v)| c as i128 * v as i128).sum();
    let want = ((acc * 3) >> 32) as i32;
    assert_eq!(m.matvec(&x), vec![want]);
    assert_eq!(m.matvec_par(&x), vec![want]);
    assert_eq!(m.matvec_batch_par(&[&x, &x, &x, &x, &x]), vec![vec![want]; 5]);
}

/// bf16 bits for value `M * 2^E` with M in [128, 255] (a normal bf16 mantissa) and sign.
fn bf16_bits(neg: bool, m: u32, e: i32) -> u16 {
    // value = 1.f x 2^exp with 7 fraction bits: M = 128 + f, E = exp - 7
    let exp = e + 7;
    let f = m - 128;
    ((neg as u16) << 15) | (((exp + 127) as u16) << 7) | f as u16
}

#[test]
fn quantiser_matches_definition() {
    // Two rows: values M x 2^E; row max is the largest |value|; codes = round_half_away(v * 121 / max).
    let entries: Vec<(bool, u32, i32)> = vec![
        (false, 200, -10),
        (true, 130, -9),
        (false, 255, -12),
        (true, 128, -14),
        (true, 170, -8),
        (false, 201, -9),
        (false, 129, -13),
        (false, 255, -8),
    ];
    let bits: Vec<u16> = entries
        .iter()
        .map(|&(n, m, e)| bf16_bits(n, m, e))
        .collect();
    let w = WeightBits {
        rows: 2,
        cols: 4,
        kind: HalfKind::BFloat16,
        bits,
    };
    let q = quantize_tq5_rowscaled(&w).unwrap();
    for r in 0..2 {
        let row = &entries[r * 4..(r + 1) * 4];
        // exact rationals: value = m * 2^e; compare on a common denominator 2^14
        let val = |&(n, m, e): &(bool, u32, i32)| -> i128 {
            let v = (m as i128) << (e + 14);
            if n {
                -v
            } else {
                v
            }
        };
        let mx = row.iter().map(|x| val(x).abs()).max().unwrap();
        for (c, x) in row.iter().enumerate() {
            let v = val(x);
            let num = v.abs() * 121;
            let want = ((2 * num + mx) / (2 * mx)) as i8 * if v < 0 { -1 } else { 1 };
            assert_eq!(q.data()[r * 4 + c], want, "row {r} col {c}");
        }
        // scale = round(max * 2^32 / 121) with max at 2^-14 units
        let want_scale = ((2 * (mx << 18) + 121) / (2 * 121)) as u64;
        assert_eq!(q.scales_q32()[r], want_scale, "scale row {r}");
        assert!(
            q.data()[r * 4..(r + 1) * 4].iter().any(|&c| c.abs() == 121),
            "row max maps to 121"
        );
    }
}

/// The split-activation kernel must equal the i128 reference at the extremes: activations
/// near the split's limit, rows longer than one accumulation block, and ragged lengths.
#[test]
fn extreme_activations_and_lengths_match_reference() {
    let mut seed = 99u64;
    for &(rows, cols) in &[
        (4usize, 1023usize),
        (4, 1024),
        (4, 1025),
        (3, 5000),
        (2, 17),
    ] {
        // small scales so the extreme activations stay inside the storage range
        let mut m = random_matrix(&mut seed, rows, cols);
        m = RowScaledTQ5::from_parts(
            rows,
            cols,
            m.data().to_vec(),
            (0..rows).map(|r| r as u64 + 1).collect(),
        );
        // patterns: alternating ±(2^30 - 1), all max codes, random large, and beyond the
        // split's range (forces the scalar fallback)
        let patterns: Vec<Vec<i32>> = vec![
            (0..cols)
                .map(|i| {
                    if i % 2 == 0 {
                        (1 << 30) - 1
                    } else {
                        -((1 << 30) - 1)
                    }
                })
                .collect(),
            (0..cols)
                .map(|_| ((rng(&mut seed) % (1u64 << 31)) as i64 - (1i64 << 30)) as i32)
                .collect(),
            (0..cols)
                .map(|i| if i % 3 == 0 { i32::MAX } else { i32::MIN + 1 })
                .collect(),
            (0..cols).map(|_| -65536).collect(),
        ];
        for x in &patterns {
            let got = m.matvec_par(x);
            for (r, &g) in got.iter().enumerate() {
                let acc: i128 = (0..cols)
                    .map(|c| m.data()[r * cols + c] as i128 * x[c] as i128)
                    .sum();
                let want = ((acc * m.scales_q32()[r] as i128) >> 32) as i32;
                assert_eq!(g, want, "row {r} of {rows}x{cols}");
            }
        }
    }
}

/// The batch kernel's fast path (register tile over low halves + sparse high-half correction) and every
/// fallback (dense high halves, values too large to split, partial token blocks, rows past the last full
/// group of 4, odd column counts) give the i128 reference's exact outputs.
#[test]
fn batch_paths_match_reference_on_realistic_activations() {
    let mut seed = 2026u64;
    // kinds: 0 small (|x| < 2^15: no high half), 1 sparse wide, 2 dense wide, 3 one element >= 2^30 (no split)
    let act = |seed: &mut u64, n: usize, kind: u8| -> Vec<i32> {
        (0..n)
            .map(|i| match kind {
                0 => (rng(seed) % 60_000) as i32 - 30_000,
                1 if rng(seed) % 50 == 0 => (rng(seed) % 4_000_000) as i32 - 2_000_000,
                1 => (rng(seed) % 60_000) as i32 - 30_000,
                2 => (rng(seed) % 4_000_000) as i32 - 2_000_000,
                _ if i == n / 2 => (1 << 30) + 5,
                _ => (rng(seed) % 60_000) as i32 - 30_000,
            })
            .collect()
    };
    for &(rows, cols) in &[
        (1usize, 1usize),
        (4, 16),
        (7, 17),
        (13, 1000),
        (64, 2049),
        (9, 4096),
    ] {
        // small row scales (< 2^-12): the dense-wide and unsplittable activations must not overflow the i32 output
        let base = random_matrix(&mut seed, rows, cols);
        let scales: Vec<u64> = (0..rows)
            .map(|_| rng(&mut seed) % (1u64 << 20) + 1)
            .collect();
        let m = RowScaledTQ5::from_parts(rows, cols, base.data().to_vec(), scales);
        for &bs in &[1usize, 3, 4, 5, 8, 13] {
            let xs: Vec<Vec<i32>> = (0..bs)
                .map(|k| act(&mut seed, cols, [0u8, 1, 0, 2, 1, 3, 0, 1][k % 8]))
                .collect();
            let refs: Vec<&[i32]> = xs.iter().map(|v| v.as_slice()).collect();
            let got = m.matvec_batch_par(&refs);
            for (k, x) in xs.iter().enumerate() {
                let single = m.matvec_par(x);
                for r in 0..rows {
                    let row = &m.data()[r * cols..(r + 1) * cols];
                    let acc: i128 = row
                        .iter()
                        .zip(x)
                        .map(|(&c, &v)| c as i128 * v as i128)
                        .sum();
                    let want = ((acc * m.scales_q32()[r] as i128) >> 32) as i32;
                    assert_eq!(
                        got[k][r], want,
                        "batch: rows {rows} cols {cols} bs {bs} token {k} row {r}"
                    );
                    assert_eq!(
                        single[r], want,
                        "single: rows {rows} cols {cols} token {k} row {r}"
                    );
                }
            }
        }
    }
}
