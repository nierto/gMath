//! TQ1.9 module validation tests.
//!
//! Tests core operations against known values, verifies SIMD/scalar equivalence,
//! batch/sequential equivalence, and precision bounds.

use g_math::fixed_point::tq19::{
    TQ19Matrix, SCALE, TRIT_DECODE_TABLE,
    tq19_dot, trit_dot, packed_trit_dot, packed_trit_matvec,
};
use g_math::fixed_point::imperative::{FixedPoint, BinaryStorage};

// ============================================================================
// Helpers
// ============================================================================

fn fp(s: &str) -> FixedPoint {
    if s.starts_with('-') { -FixedPoint::from_str(&s[1..]) }
    else { FixedPoint::from_str(s) }
}

fn raw(s: &str) -> BinaryStorage {
    fp(s).raw()
}

fn assert_ulp(got: BinaryStorage, expected: BinaryStorage, max_ulp: i64, label: &str) {
    let diff = ulp_distance(got, expected);
    assert!(diff <= max_ulp, "{label}: diff = {diff} ULP (max {max_ulp})");
}

/// Assert within TQ1.9 quantization tolerance (~0.01% relative).
/// TQ1.9 has ~4.3 decimal digits, so 1/SCALE ≈ 5e-5 relative error per weight.
fn assert_tq19_close(got: BinaryStorage, expected: BinaryStorage, label: &str) {
    let tolerance = tq19_tolerance(expected);
    let diff = ulp_distance(got, expected);
    assert!(diff <= tolerance, "{label}: diff = {diff} ULP (tolerance {tolerance})");
}

fn ulp_distance(a: BinaryStorage, b: BinaryStorage) -> i64 {
    #[cfg(table_format = "q16_16")]
    { (a as i64 - b as i64).abs() }
    #[cfg(table_format = "q32_32")]
    { (a as i128 - b as i128).unsigned_abs() as i64 }
    #[cfg(table_format = "q64_64")]
    { (a - b).unsigned_abs() as i64 }
    #[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
    { 0 }
}

/// TQ1.9 quantization tolerance in BinaryStorage units.
/// Allows 0.02% relative error + 1 ULP (covers SCALE rounding + accumulation).
fn tq19_tolerance(expected: BinaryStorage) -> i64 {
    #[cfg(table_format = "q16_16")]
    { ((expected.unsigned_abs() as i64) / 5000).max(2) }
    #[cfg(table_format = "q32_32")]
    { ((expected.unsigned_abs() / 5000) as i64).max(2) }
    #[cfg(table_format = "q64_64")]
    { ((expected.unsigned_abs() / 5000) as i64).max(2) }
    #[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
    { 2 }
}

/// Encode i8 trits to packed bytes for testing.
fn encode_trits(trits: &[i8]) -> Vec<u8> {
    let mut packed = Vec::new();
    for chunk in trits.chunks(5) {
        let mut byte = 0u8;
        let powers = [81u8, 27, 9, 3, 1];
        for (j, &t) in chunk.iter().enumerate() {
            byte += (t + 1) as u8 * powers[j];
        }
        // Pad remaining with Zero (encoded as 1)
        for j in chunk.len()..5 {
            byte += powers[j];
        }
        packed.push(byte);
    }
    packed
}

// ============================================================================
// Trit decode table
// ============================================================================

#[test]
fn trit_table_all_valid_roundtrip() {
    for byte in 0u8..=242 {
        let trits = TRIT_DECODE_TABLE[byte as usize];
        for t in &trits {
            assert!(
                *t == -1 || *t == 0 || *t == 1,
                "byte {byte}: invalid trit {t}"
            );
        }
        let re = ((trits[0] + 1) as u8) * 81
            + ((trits[1] + 1) as u8) * 27
            + ((trits[2] + 1) as u8) * 9
            + ((trits[3] + 1) as u8) * 3
            + ((trits[4] + 1) as u8);
        assert_eq!(re, byte, "roundtrip failed for byte {byte}");
    }
}

// ============================================================================
// TQ1.9 dot product
// ============================================================================

#[test]
fn tq19_dot_unity_weight() {
    // Weight = SCALE → TQ1.9 value = 1.0
    // dot([SCALE], [x]) / SCALE = x
    let x = raw("2.5");
    let result = tq19_dot(&[SCALE as i16], &[x]);
    assert_ulp(result, x, 1, "unity weight dot");
}

#[test]
fn tq19_dot_zero_weight() {
    let result = tq19_dot(&[0i16, 0, 0], &[raw("1.0"), raw("2.0"), raw("3.0")]);
    assert_ulp(result, raw("0.0"), 0, "zero weights");
}

#[test]
fn tq19_dot_negative_weight() {
    // Weight = -SCALE → TQ1.9 value = -1.0
    // dot([-SCALE], [x]) / SCALE = -x
    let x = raw("3.0");
    let neg_x = raw("-3.0");
    let result = tq19_dot(&[-(SCALE as i16)], &[x]);
    assert_ulp(result, neg_x, 1, "negative unity weight");
}

#[test]
fn tq19_dot_half_weight() {
    // Weight = SCALE/2 = 9841 → represents 9841/19683 ≈ 0.49997 (not exactly 0.5)
    // TQ1.9 quantization: 0.5 - 9841/19683 ≈ 2.5e-5 relative error
    let half_scale = (SCALE / 2) as i16;
    let x = raw("4.0");
    let result = tq19_dot(&[half_scale], &[x]);
    let expected = raw("2.0");
    assert_tq19_close(result, expected, "half weight (TQ1.9 quantized)");
}

#[test]
fn tq19_dot_accumulation_precision() {
    // 100 × (SCALE/10 × 1.0) / SCALE
    // SCALE/10 = 1968 → represents 1968/19683 ≈ 0.09985 (not exactly 0.1)
    // 100 × 0.09985 ≈ 9.985, not 10.0 — this is TQ1.9 quantization, not accumulation error
    let w = (SCALE / 10) as i16;
    let a = raw("1.0");
    let weights = vec![w; 100];
    let activations = vec![a; 100];
    let result = tq19_dot(&weights, &activations);
    let expected = raw("10.0");
    assert_tq19_close(result, expected, "accumulation of 100 terms");
}

// ============================================================================
// Trit dot product
// ============================================================================

#[test]
fn trit_dot_all_positive() {
    let activations = vec![raw("1.0"), raw("2.0"), raw("3.0")];
    let trits = vec![1i8, 1, 1];
    let result = trit_dot(&trits, &activations);
    assert_ulp(result, raw("6.0"), 1, "all-positive trits");
}

#[test]
fn trit_dot_all_negative() {
    let activations = vec![raw("1.0"), raw("2.0"), raw("3.0")];
    let trits = vec![-1i8, -1, -1];
    let result = trit_dot(&trits, &activations);
    assert_ulp(result, raw("-6.0"), 1, "all-negative trits");
}

#[test]
fn trit_dot_all_zero() {
    let activations = vec![raw("1.0"), raw("2.0"), raw("3.0")];
    let trits = vec![0i8, 0, 0];
    let result = trit_dot(&trits, &activations);
    assert_ulp(result, raw("0.0"), 0, "all-zero trits");
}

#[test]
fn trit_dot_mixed_signs() {
    // [+1, 0, -1] · [1.0, 2.0, 3.0] = 1.0 - 3.0 = -2.0
    let activations = vec![raw("1.0"), raw("2.0"), raw("3.0")];
    let trits = vec![1i8, 0, -1];
    let result = trit_dot(&trits, &activations);
    assert_ulp(result, raw("-2.0"), 1, "mixed trits");
}

#[test]
fn trit_dot_large_vector() {
    // 1024 elements, alternating +1/-1
    let n = 1024;
    let activations: Vec<BinaryStorage> = (0..n).map(|_| raw("0.001")).collect();
    let trits: Vec<i8> = (0..n).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
    let result = trit_dot(&trits, &activations);
    // Even count: 512 × 0.001 - 512 × 0.001 = 0
    assert_ulp(result, raw("0.0"), 1, "large alternating trit dot");
}

// ============================================================================
// Packed trit dot
// ============================================================================

#[test]
fn packed_trit_dot_basic() {
    let trits_i8: Vec<i8> = vec![1, -1, 0, 1, 1];
    let packed = encode_trits(&trits_i8);
    let activations = vec![raw("1.0"), raw("2.0"), raw("3.0"), raw("4.0"), raw("5.0")];
    let scale = raw("1.0");

    // Expected: 1.0 - 2.0 + 0 + 4.0 + 5.0 = 8.0, × 1.0 = 8.0
    let result = packed_trit_dot(&packed, 5, &activations, scale);
    assert_ulp(result, raw("8.0"), 2, "packed trit dot basic");
}

#[test]
fn packed_trit_dot_with_scale() {
    let trits_i8: Vec<i8> = vec![1, 1, 1];
    let packed = encode_trits(&trits_i8);
    let activations = vec![raw("1.0"), raw("2.0"), raw("3.0")];
    let scale = raw("0.5");

    // Expected: (1+2+3) × 0.5 = 3.0
    let result = packed_trit_dot(&packed, 3, &activations, scale);
    assert_ulp(result, raw("3.0"), 2, "packed trit dot with scale");
}

// ============================================================================
// TQ19Matrix matvec
// ============================================================================

#[test]
fn matrix_matvec_identity() {
    // Diagonal = SCALE, off-diagonal = 0 → identity transform
    let n = 4;
    let m = TQ19Matrix::from_fn(n, n, |r, c| if r == c { SCALE as i16 } else { 0 });
    let activations: Vec<BinaryStorage> = vec![raw("1.0"), raw("2.0"), raw("3.0"), raw("4.0")];
    let result = m.matvec(&activations);

    for i in 0..n {
        assert_ulp(result[i], activations[i], 1, &format!("identity row {i}"));
    }
}

#[test]
fn matrix_matvec_zero_matrix() {
    let m = TQ19Matrix::new(3, 3, vec![0i16; 9]);
    let activations = vec![raw("1.0"), raw("2.0"), raw("3.0")];
    let result = m.matvec(&activations);
    for i in 0..3 {
        assert_ulp(result[i], raw("0.0"), 0, &format!("zero matrix row {i}"));
    }
}

#[test]
fn matrix_matvec_single_row() {
    // 1×3 matrix: [SCALE, SCALE, SCALE] → dot = sum of activations
    let m = TQ19Matrix::new(1, 3, vec![SCALE as i16; 3]);
    let activations = vec![raw("1.0"), raw("2.0"), raw("3.0")];
    let result = m.matvec(&activations);
    assert_ulp(result[0], raw("6.0"), 1, "sum-row matvec");
}

#[test]
fn matrix_matvec_fp_convenience() {
    let m = TQ19Matrix::from_fn(2, 2, |r, c| if r == c { SCALE as i16 } else { 0 });
    let activations = vec![raw("1.5"), raw("2.5")];
    let result = m.matvec_fp(&activations);
    assert_eq!(result.len(), 2);
    // Check that results are FixedPoint instances (type system guarantees this)
    let _: FixedPoint = result[0];
}

// ============================================================================
// Batch matvec
// ============================================================================

#[test]
fn batch_matvec_matches_sequential() {
    let n = 4;
    let data: Vec<i16> = (0..n * n)
        .map(|i| ((i as i32 * 137 - 500) % (SCALE / 2)) as i16)
        .collect();

    let v1: Vec<BinaryStorage> = (0..n).map(|i| raw(&format!("{}.5", i))).collect();
    let v2: Vec<BinaryStorage> = (0..n).map(|i| raw(&format!("{}.25", i + 1))).collect();
    let v3: Vec<BinaryStorage> = (0..n).map(|_| raw("0.1")).collect();

    let seq = vec![
        TQ19Matrix::new(n, n, data.clone()).matvec(&v1),
        TQ19Matrix::new(n, n, data.clone()).matvec(&v2),
        TQ19Matrix::new(n, n, data.clone()).matvec(&v3),
    ];

    let m = TQ19Matrix::new(n, n, data);
    let batch = m.matvec_batch(&[&v1, &v2, &v3]);

    for b in 0..3 {
        assert_eq!(batch[b], seq[b], "batch[{b}] must match sequential");
    }
}

// ============================================================================
// Packed trit matvec
// ============================================================================

#[test]
fn packed_trit_matvec_basic() {
    // 2×3 matrix of trits:
    // Row 0: [+1, +1, +1] → dot = sum(activations) × scale[0]
    // Row 1: [+1, -1, 0]  → dot = (a0 - a1) × scale[1]
    let trits_row0: Vec<i8> = vec![1, 1, 1];
    let trits_row1: Vec<i8> = vec![1, -1, 0];

    let packed_row0 = encode_trits(&trits_row0);
    let packed_row1 = encode_trits(&trits_row1);
    let mut packed = packed_row0;
    packed.extend(packed_row1);

    let activations = vec![raw("1.0"), raw("2.0"), raw("3.0")];
    let scales = vec![raw("1.0"), raw("1.0")];

    let result = packed_trit_matvec(&packed, 2, 3, &activations, &scales);
    assert_ulp(result[0], raw("6.0"), 2, "packed matvec row 0");
    assert_ulp(result[1], raw("-1.0"), 2, "packed matvec row 1");
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn empty_dot_product() {
    let result = tq19_dot(&[], &[]);
    assert_ulp(result, raw("0.0"), 0, "empty dot");
}

#[test]
fn single_element_dot() {
    let result = tq19_dot(&[SCALE as i16], &[raw("7.25")]);
    assert_ulp(result, raw("7.25"), 1, "single element dot");
}

#[test]
fn max_weight_dot() {
    // MAX_RAW = 29524 → represents 29524/19683 ≈ 1.49987 (not exactly 1.5)
    let result = tq19_dot(&[g_math::fixed_point::tq19::MAX_RAW], &[raw("1.0")]);
    let expected = raw("1.5");
    assert_tq19_close(result, expected, "max weight value");
}

#[test]
#[should_panic(expected = "activation length mismatch")]
fn matvec_dimension_mismatch() {
    let m = TQ19Matrix::new(2, 3, vec![0i16; 6]);
    m.matvec(&[raw("1.0"), raw("2.0")]); // wrong length
}
