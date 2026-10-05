//! `g_math::tq19::bits` and `g_math::tq19::quantize` (0.6.5): weight bit
//! patterns to fixed point and to the ternary formats.
//!
//! References come from `scripts/generate_weight_bits_refs.py`, which works
//! on exact rationals from the definition of binary16 and bfloat16: an
//! exhaustive checksum over all 65,536 patterns of each format for the
//! decoders, and literal expectations for the quantisers. No floats.

#![cfg(feature = "inference")]

use g_math::fixed_point::OverflowDetected;
use g_math::tq19::bits::{decompose, to_q64_raw, to_raw, to_tq19_raw, HalfKind, WeightBits};
use g_math::tq19::quantize::quantize_tq19;

#[allow(dead_code)]
mod refs {
    use g_math::tq19::bits::HalfKind;
    include!("data/weight_bits_refs.rs");
}

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn fnv(mut h: u64, data: &[u8]) -> u64 {
    for &b in data {
        h = (h ^ b as u64).wrapping_mul(FNV_PRIME);
    }
    h
}

/// The generator's encoding of one result.
fn fold(h: u64, r: Result<i128, OverflowDetected>) -> u64 {
    match r {
        Ok(v) => fnv(fnv(h, &[0]), &v.to_le_bytes()),
        Err(OverflowDetected::InvalidInput) => fnv(h, &[0xE1]),
        Err(OverflowDetected::TierOverflow) => fnv(h, &[0xE2]),
        Err(e) => panic!("unexpected error {e:?}"),
    }
}

#[test]
fn to_raw_matches_the_exact_model_on_every_pattern() {
    let expected: [(HalfKind, [u64; 8]); 2] = [
        (HalfKind::Binary16, [
            refs::BINARY16_RAW_F2, refs::BINARY16_RAW_F10, refs::BINARY16_RAW_F16, refs::BINARY16_RAW_F24,
            refs::BINARY16_RAW_F30, refs::BINARY16_RAW_F32, refs::BINARY16_RAW_F64, refs::BINARY16_RAW_F100,
        ]),
        (HalfKind::BFloat16, [
            refs::BFLOAT16_RAW_F2, refs::BFLOAT16_RAW_F10, refs::BFLOAT16_RAW_F16, refs::BFLOAT16_RAW_F24,
            refs::BFLOAT16_RAW_F30, refs::BFLOAT16_RAW_F32, refs::BFLOAT16_RAW_F64, refs::BFLOAT16_RAW_F100,
        ]),
    ];
    for (kind, hashes) in expected {
        for (i, &f) in refs::RAW_SPLITS.iter().enumerate() {
            let mut h = FNV_OFFSET;
            for bits in 0u16..=0xFFFF {
                h = fold(h, to_raw(bits, kind, f));
            }
            assert_eq!(h, hashes[i], "{kind:?} at {f} fractional bits");
        }
    }
}

#[test]
fn to_tq19_raw_matches_the_exact_model_on_every_pattern() {
    for (kind, want) in [(HalfKind::Binary16, refs::BINARY16_TQ19), (HalfKind::BFloat16, refs::BFLOAT16_TQ19)] {
        let mut h = FNV_OFFSET;
        for bits in 0u16..=0xFFFF {
            h = fold(h, to_tq19_raw(bits, kind).map(|v| v as i128));
        }
        assert_eq!(h, want, "{kind:?}");
    }
}

#[test]
fn spot_values_and_structure() {
    for &(kind, bits, f, want) in refs::RAW_SPOTS {
        assert_eq!(to_raw(bits, kind, f), Ok(want), "{kind:?} {bits:#06x} F={f}");
    }
    // 1.0 in both formats
    assert_eq!(decompose(0x3F80, HalfKind::BFloat16), Ok((false, 0x80, -7)));
    assert_eq!(decompose(0x3C00, HalfKind::Binary16), Ok((false, 0x400, -10)));
    assert_eq!(to_tq19_raw(0x3F80, HalfKind::BFloat16), Ok(19_683));
    assert_eq!(to_tq19_raw(0xBC00, HalfKind::Binary16), Ok(-19_683));
    // signed zeros
    for kind in [HalfKind::Binary16, HalfKind::BFloat16] {
        assert_eq!(to_raw(0x0000, kind, 64), Ok(0));
        assert_eq!(to_raw(0x8000, kind, 64), Ok(0));
        assert_eq!(to_q64_raw(0x8000, kind), Ok(0));
    }
    // non-finite patterns
    assert_eq!(to_raw(0x7C00, HalfKind::Binary16, 10), Err(OverflowDetected::InvalidInput));
    assert_eq!(to_raw(0x7E00, HalfKind::Binary16, 10), Err(OverflowDetected::InvalidInput));
    assert_eq!(to_raw(0x7F80, HalfKind::BFloat16, 10), Err(OverflowDetected::InvalidInput));
    assert_eq!(to_tq19_raw(0xFFC0, HalfKind::BFloat16), Err(OverflowDetected::InvalidInput));
    // truncation is toward zero for both signs: 0.1 (bf16 0x3DCD) at 10 bits is 102.4...
    assert_eq!(to_raw(0x3DCD, HalfKind::BFloat16, 10), Ok(102));
    assert_eq!(to_raw(0xBDCD, HalfKind::BFloat16, 10), Ok(-102));
}

/// The storage form is `to_raw` at the build's split, refused when the
/// magnitude exceeds the largest storage value (both signs alike).
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
#[test]
fn storage_raw_is_to_raw_at_the_build_split() {
    use g_math::fixed_point::imperative::BinaryStorage;
    use g_math::tq19::bits::{to_fixed, to_storage_raw};
    #[cfg(table_format = "q16_16")]
    let f = g_math::fixed_point::frac_config::FRAC_BITS;
    #[cfg(table_format = "q32_32")]
    let f = 32;
    let (mut ok, mut refused) = (0u32, 0u32);
    for kind in [HalfKind::Binary16, HalfKind::BFloat16] {
        for bits in 0u16..=0xFFFF {
            let want = match to_raw(bits, kind, f) {
                Ok(v) if v.abs() <= BinaryStorage::MAX as i128 => Ok(v as BinaryStorage),
                Ok(_) => Err(OverflowDetected::TierOverflow),
                Err(e) => Err(e),
            };
            assert_eq!(to_storage_raw(bits, kind), want, "{kind:?} {bits:#06x}");
            assert_eq!(to_fixed(bits, kind).map(|v| v.raw()), want);
            if want.is_ok() { ok += 1 } else { refused += 1 }
        }
    }
    assert!(ok > 60_000 && refused > 250, "ok {ok} refused {refused}");
}

fn matrix(kind: HalfKind, bits: &[u16]) -> WeightBits {
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    WeightBits::from_le_bytes(refs::Q_ROWS, refs::Q_COLS, kind, &bytes).unwrap()
}

#[test]
fn weight_bits_from_le_bytes() {
    let w = matrix(HalfKind::BFloat16, refs::BFLOAT16_BITS);
    assert_eq!(w.bits, refs::BFLOAT16_BITS);
    assert_eq!((w.rows, w.cols), (refs::Q_ROWS, refs::Q_COLS));
    assert!(WeightBits::from_le_bytes(2, 2, HalfKind::Binary16, &[0u8; 7]).is_err());
    // extra bytes are ignored
    assert_eq!(WeightBits::from_le_bytes(1, 1, HalfKind::Binary16, &[0x00, 0x3C, 0xFF]).unwrap().bits, vec![0x3C00]);
}

#[test]
fn quantize_tq19_matches_the_exact_model() {
    for (kind, bits, big, want) in [
        (HalfKind::BFloat16, refs::BFLOAT16_BITS, refs::BFLOAT16_BIG_BITS, refs::BFLOAT16_TQ19_DATA),
        (HalfKind::Binary16, refs::BINARY16_BITS, refs::BINARY16_BIG_BITS, refs::BINARY16_TQ19_DATA),
    ] {
        let q = quantize_tq19(&matrix(kind, bits)).expect("in range");
        assert_eq!(q.data(), want, "{kind:?}");
        // more than 1% of the large matrix clamps: refused
        assert!(quantize_tq19(&matrix(kind, big)).is_none(), "{kind:?} big");
    }
    // a non-finite pattern is refused
    let mut bad = refs::BFLOAT16_BITS.to_vec();
    bad[7] = 0x7F80;
    assert!(quantize_tq19(&matrix(HalfKind::BFloat16, &bad)).is_none());
}

#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
#[test]
fn quantize_tq19_rowscaled_matches_the_exact_model() {
    use g_math::tq19::quantize::quantize_tq19_rowscaled;
    for (kind, bits, data, scales) in [
        (HalfKind::BFloat16, refs::BFLOAT16_BITS, refs::BFLOAT16_RS19_DATA, refs::BFLOAT16_RS19_SCALES),
        (HalfKind::BFloat16, refs::BFLOAT16_BIG_BITS, refs::BFLOAT16_BIG_RS19_DATA, refs::BFLOAT16_BIG_RS19_SCALES),
        (HalfKind::Binary16, refs::BINARY16_BITS, refs::BINARY16_RS19_DATA, refs::BINARY16_RS19_SCALES),
        (HalfKind::Binary16, refs::BINARY16_BIG_BITS, refs::BINARY16_BIG_RS19_DATA, refs::BINARY16_BIG_RS19_SCALES),
    ] {
        let q = quantize_tq19_rowscaled(&matrix(kind, bits)).unwrap();
        assert_eq!(q.data(), data, "{kind:?}");
        assert_eq!(q.scales_q32(), scales, "{kind:?}");
        // the all-zero row
        assert_eq!(q.scales_q32()[2], 0);
    }
}

#[cfg(table_format = "q16_16")]
#[test]
fn quantize_tq5_rowscaled_matches_the_exact_model() {
    use g_math::tq19::quantize::quantize_tq5_rowscaled;
    for (kind, bits, data, scales) in [
        (HalfKind::BFloat16, refs::BFLOAT16_BITS, refs::BFLOAT16_TQ5_DATA, refs::BFLOAT16_TQ5_SCALES),
        (HalfKind::BFloat16, refs::BFLOAT16_BIG_BITS, refs::BFLOAT16_BIG_TQ5_DATA, refs::BFLOAT16_BIG_TQ5_SCALES),
        (HalfKind::Binary16, refs::BINARY16_BITS, refs::BINARY16_TQ5_DATA, refs::BINARY16_TQ5_SCALES),
        (HalfKind::Binary16, refs::BINARY16_BIG_BITS, refs::BINARY16_BIG_TQ5_DATA, refs::BINARY16_BIG_TQ5_SCALES),
    ] {
        let q = quantize_tq5_rowscaled(&matrix(kind, bits)).unwrap();
        assert_eq!(q.data(), data, "{kind:?}");
        assert_eq!(q.scales_q32(), scales, "{kind:?}");
        // every non-zero row reaches the largest code
        for r in [0usize, 1, 3, 4] {
            assert!(q.data()[r * refs::Q_COLS..(r + 1) * refs::Q_COLS].iter().any(|c| c.abs() == 121), "row {r}");
        }
    }
}

/// The wide-gap matrix: bfloat16 rows that hold values near 2^-124 beside
/// ordinary weights (found in a real checkpoint), in both orders, a row of
/// tiny values only, and rows spanning the exponent range.
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
fn gap_matrix() -> WeightBits {
    let bytes: Vec<u8> = refs::BFLOAT16_GAP_BITS.iter().flat_map(|b| b.to_le_bytes()).collect();
    WeightBits::from_le_bytes(refs::GAP_ROWS, refs::Q_COLS, HalfKind::BFloat16, &bytes).unwrap()
}

/// 0.6.5 compared and divided by shifting a mantissa by the exponent gap,
/// which left 128 bits from a gap of a little over 100: the row maximum came out
/// wrong and the codes left their range.
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
#[test]
fn quantize_tq19_rowscaled_holds_a_wide_exponent_gap() {
    use g_math::tq19::quantize::quantize_tq19_rowscaled;
    let q = quantize_tq19_rowscaled(&gap_matrix()).unwrap();
    assert_eq!(q.data(), refs::BFLOAT16_GAP_RS19_DATA);
    assert_eq!(q.scales_q32(), refs::BFLOAT16_GAP_RS19_SCALES);
}

#[cfg(table_format = "q16_16")]
#[test]
fn quantize_tq5_rowscaled_holds_a_wide_exponent_gap() {
    use g_math::tq19::quantize::quantize_tq5_rowscaled;
    let q = quantize_tq5_rowscaled(&gap_matrix()).unwrap();
    assert_eq!(q.data(), refs::BFLOAT16_GAP_TQ5_DATA);
    assert_eq!(q.scales_q32(), refs::BFLOAT16_GAP_TQ5_SCALES);
    // the reported pattern: 2^-124-class elements beside ordinary weights
    // quantise to zero, in either position, and the rest keep their codes
    assert_eq!((q.data()[0], q.data()[2 * refs::Q_COLS - 1]), (0, 0));
    for r in [0usize, 1] {
        assert!(q.data()[r * refs::Q_COLS..(r + 1) * refs::Q_COLS].iter().any(|c| c.abs() == 121), "row {r}");
    }
}
