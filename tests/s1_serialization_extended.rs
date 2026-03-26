//! S1 extended serialization tests: Tensor, ManifoldPoint, compact encoding.
//!
//! Tests verify:
//! 1. Tensor to_bytes/from_bytes roundtrip (rank-0, rank-1, rank-2, rank-4)
//! 2. ManifoldPoint to_bytes/from_bytes roundtrip (all 5 manifold types)
//! 3. Compact encoding roundtrip (zero, small, negative, large values)
//! 4. Compact encoding saves bytes vs full encoding
//! 5. Compact FixedVector roundtrip
//!
//! All tests run on the active profile (embedded/balanced/scientific).

use g_math::fixed_point::{FixedPoint, FixedVector, FixedMatrix};
use g_math::fixed_point::imperative::tensor::Tensor;
use g_math::fixed_point::imperative::ManifoldPoint;

fn fp(s: &str) -> FixedPoint {
    if s.starts_with('-') { -FixedPoint::from_str(&s[1..]) }
    else { FixedPoint::from_str(s) }
}

fn tight() -> FixedPoint {
    #[cfg(table_format = "q16_16")]
    { fp("0.01") }
    #[cfg(table_format = "q32_32")]
    { fp("0.0001") }
    #[cfg(not(any(table_format = "q16_16", table_format = "q32_32")))]
    { fp("0.000000001") }
}

fn assert_fp(got: FixedPoint, exp: FixedPoint, tol: FixedPoint, name: &str) {
    let d = (got - exp).abs();
    assert!(d < tol, "{}: got {}, expected {}, diff={}", name, got, exp, d);
}

// ============================================================================
// Tensor serialization
// ============================================================================

#[test]
fn test_tensor_roundtrip_rank0() {
    let t = Tensor::from_data(&[], &[fp("3.14")]);
    let bytes = t.to_bytes();
    let t2 = Tensor::from_bytes(&bytes).unwrap();
    assert_eq!(t2.rank(), 0);
    assert_fp(t2.to_scalar(), fp("3.14"), tight(), "rank-0 roundtrip");
}

#[test]
fn test_tensor_roundtrip_rank1() {
    let v = FixedVector::from_slice(&[fp("1"), fp("2"), fp("3")]);
    let t = Tensor::from(&v);
    let bytes = t.to_bytes();
    let t2 = Tensor::from_bytes(&bytes).unwrap();
    assert_eq!(t2.rank(), 1);
    assert_eq!(t2.shape(), &[3]);
    let v2 = t2.to_vector();
    for i in 0..3 {
        assert_fp(v2[i], v[i], tight(), &format!("rank-1[{}]", i));
    }
}

#[test]
fn test_tensor_roundtrip_rank2() {
    let m = FixedMatrix::from_slice(2, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("4"), fp("5"), fp("6"),
    ]);
    let t = Tensor::from(&m);
    let bytes = t.to_bytes();
    let t2 = Tensor::from_bytes(&bytes).unwrap();
    assert_eq!(t2.rank(), 2);
    assert_eq!(t2.shape(), &[2, 3]);
    let m2 = t2.to_matrix();
    for r in 0..2 {
        for c in 0..3 {
            assert_fp(m2.get(r, c), m.get(r, c), tight(), &format!("rank-2[{},{}]", r, c));
        }
    }
}

#[test]
fn test_tensor_roundtrip_rank4() {
    // Rank-4 tensor (like Riemann curvature, 2×2×2×2)
    let mut t = Tensor::new(&[2, 2, 2, 2]);
    t.set(&[0, 1, 0, 1], fp("1.5"));
    t.set(&[1, 0, 1, 0], fp("-1.5"));

    let bytes = t.to_bytes();
    let t2 = Tensor::from_bytes(&bytes).unwrap();
    assert_eq!(t2.rank(), 4);
    assert_eq!(t2.shape(), &[2, 2, 2, 2]);
    assert_fp(t2.get(&[0, 1, 0, 1]), fp("1.5"), tight(), "rank-4[0,1,0,1]");
    assert_fp(t2.get(&[1, 0, 1, 0]), fp("-1.5"), tight(), "rank-4[1,0,1,0]");
    assert_fp(t2.get(&[0, 0, 0, 0]), fp("0"), tight(), "rank-4[0,0,0,0]");
}

#[test]
fn test_tensor_invalid_bytes() {
    assert!(Tensor::from_bytes(&[]).is_err());
    // rank=2 but not enough shape bytes
    assert!(Tensor::from_bytes(&[2, 0]).is_err());
}

// ============================================================================
// Compact encoding
// ============================================================================

#[test]
fn test_compact_zero() {
    let zero = FixedPoint::ZERO;
    let bytes = zero.to_compact_bytes();
    assert_eq!(bytes.len(), 1, "Zero should be 1 byte");
    let (decoded, consumed) = FixedPoint::from_compact_bytes(&bytes).unwrap();
    assert!(decoded.is_zero(), "Decoded zero should be zero");
    assert_eq!(consumed, 1);
}

#[test]
fn test_compact_one() {
    let one = FixedPoint::one();
    let compact = one.to_compact_bytes();
    let full = one.to_bytes();
    // Compact should be smaller than full (one has many leading zero bytes in abs)
    assert!(compact.len() <= full.len(),
        "Compact {} bytes should be <= full {} bytes", compact.len(), full.len());

    let (decoded, _) = FixedPoint::from_compact_bytes(&compact).unwrap();
    assert_fp(decoded, one, tight(), "compact one roundtrip");
}

#[test]
fn test_compact_negative() {
    let val = fp("-2.5");
    let bytes = val.to_compact_bytes();
    let (decoded, _) = FixedPoint::from_compact_bytes(&bytes).unwrap();
    assert_fp(decoded, val, tight(), "compact negative roundtrip");
}

#[test]
fn test_compact_small_value() {
    // Very small value — should save the most bytes
    let small = fp("0.00001");
    let compact = small.to_compact_bytes();
    let full = small.to_bytes();
    assert!(compact.len() < full.len(),
        "Small value: compact {} < full {} bytes", compact.len(), full.len());

    let (decoded, _) = FixedPoint::from_compact_bytes(&compact).unwrap();
    assert_fp(decoded, small, tight(), "compact small roundtrip");
}

/// Q16.16/Q32.32: 999999999999 overflows both profiles.
#[test]
#[cfg(not(any(table_format = "q16_16", table_format = "q32_32")))]
fn test_compact_large_value() {
    // Large value — uses more of the storage, less savings
    let large = fp("999999999999");
    let compact = large.to_compact_bytes();
    let (decoded, _) = FixedPoint::from_compact_bytes(&compact).unwrap();
    assert_fp(decoded, large, tight(), "compact large roundtrip");
}

#[test]
fn test_compact_various_values() {
    let values = vec![
        fp("0"), fp("1"), fp("-1"), fp("0.5"), fp("-0.5"),
        fp("100"), fp("-100"), fp("0.000001"), fp("-0.000001"),
        fp("3.14159265358979"), fp("-2.71828182845904"),
    ];
    for val in &values {
        let compact = val.to_compact_bytes();
        let (decoded, consumed) = FixedPoint::from_compact_bytes(&compact).unwrap();
        assert_eq!(consumed, compact.len(), "consumed should match compact length");
        assert_fp(decoded, *val, tight(), &format!("compact roundtrip {}", val));
    }
}

#[test]
fn test_compact_vector_roundtrip() {
    let v = FixedVector::from_slice(&[fp("0"), fp("1"), fp("0"), fp("0"), fp("0.001")]);
    let compact = v.to_compact_bytes();
    let full = v.to_bytes();

    // Sparse vector should compress well — 3 zeros out of 5 elements
    assert!(compact.len() < full.len(),
        "Sparse vector: compact {} < full {} bytes", compact.len(), full.len());

    let (decoded, consumed) = FixedVector::from_compact_bytes(&compact).unwrap();
    assert_eq!(consumed, compact.len());
    assert_eq!(decoded.len(), v.len());
    for i in 0..v.len() {
        assert_fp(decoded[i], v[i], tight(), &format!("compact vector[{}]", i));
    }
}

/// Q16.16/Q32.32: test uses 1e9 which overflows both profiles.
#[test]
#[cfg(not(any(table_format = "q16_16", table_format = "q32_32")))]
fn test_compact_savings_report() {
    println!("\n========================================");
    println!("Compact Encoding — Savings Report");
    println!("========================================\n");

    let test_cases: Vec<(&str, FixedPoint)> = vec![
        ("zero", fp("0")),
        ("one", fp("1")),
        ("small 0.001", fp("0.001")),
        ("tiny 0.000001", fp("0.000001")),
        ("pi", fp("3.14159265358979")),
        ("large 1e9", fp("1000000000")),
        ("negative -42.5", fp("-42.5")),
    ];

    for (name, val) in &test_cases {
        let full = val.to_bytes().len();
        let compact = val.to_compact_bytes().len();
        let savings = if full > 0 { 100.0 * (1.0 - compact as f64 / full as f64) } else { 0.0 };
        println!("  {:20} full={:3}B  compact={:3}B  savings={:.0}%",
            name, full, compact, savings);
    }

    // Sparse vector savings
    let sparse = FixedVector::from_slice(&[
        fp("0"), fp("0"), fp("0"), fp("0"), fp("0"),
        fp("0"), fp("0"), fp("0"), fp("0"), fp("1"),
    ]);
    let full = sparse.to_bytes().len();
    let compact = sparse.to_compact_bytes().len();
    println!("\n  10-elem sparse vec   full={:3}B  compact={:3}B  savings={:.0}%",
        full, compact, 100.0 * (1.0 - compact as f64 / full as f64));
}

// ============================================================================
// ManifoldPoint serialization
// ============================================================================

#[test]
fn test_manifold_point_euclidean_roundtrip() {
    let coords = FixedVector::from_slice(&[fp("1"), fp("2"), fp("3")]);
    let mp = ManifoldPoint::euclidean(3, coords.clone());
    let bytes = mp.to_bytes();
    let mp2 = ManifoldPoint::from_bytes(&bytes).unwrap();
    assert_eq!(mp2.manifold_tag, 0x01);
    assert_eq!(mp2.params, vec![3]);
    for i in 0..3 {
        assert_eq!(mp2.coordinates[i], coords[i], "euclidean coord[{}]", i);
    }
}

#[test]
fn test_manifold_point_sphere_roundtrip() {
    let coords = FixedVector::from_slice(&[fp("0.5"), fp("0.5"), fp("0.7071067811865475")]);
    let mp = ManifoldPoint::sphere(2, coords.clone());
    let bytes = mp.to_bytes();
    let mp2 = ManifoldPoint::from_bytes(&bytes).unwrap();
    assert_eq!(mp2.manifold_tag, 0x02);
    assert_eq!(mp2.params, vec![2]);
    for i in 0..3 {
        assert_fp(mp2.coordinates[i], coords[i], tight(), &format!("sphere coord[{}]", i));
    }
}

#[test]
fn test_manifold_point_hyperbolic_roundtrip() {
    let coords = FixedVector::from_slice(&[fp("1.5"), fp("0.3"), fp("0.7")]);
    let mp = ManifoldPoint::hyperbolic(2, coords.clone());
    let bytes = mp.to_bytes();
    let mp2 = ManifoldPoint::from_bytes(&bytes).unwrap();
    assert_eq!(mp2.manifold_tag, 0x03);
}

#[test]
fn test_manifold_point_spd_roundtrip() {
    // 2×2 SPD: upper triangle [a, b, c] → [[a,b],[b,c]]
    let coords = FixedVector::from_slice(&[fp("2"), fp("0.5"), fp("3")]);
    let mp = ManifoldPoint::spd(2, coords.clone());
    let bytes = mp.to_bytes();
    let mp2 = ManifoldPoint::from_bytes(&bytes).unwrap();
    assert_eq!(mp2.manifold_tag, 0x04);
    assert_eq!(mp2.params, vec![2]);
}

#[test]
fn test_manifold_point_grassmannian_roundtrip() {
    // Gr(2, 4): 4×2 matrix as column-major vector (8 entries)
    let coords = FixedVector::from_slice(&[
        fp("1"), fp("0"), fp("0"), fp("0"),  // col 0
        fp("0"), fp("1"), fp("0"), fp("0"),  // col 1
    ]);
    let mp = ManifoldPoint::grassmannian(2, 4, coords.clone());
    let bytes = mp.to_bytes();
    let mp2 = ManifoldPoint::from_bytes(&bytes).unwrap();
    assert_eq!(mp2.manifold_tag, 0x05);
    assert_eq!(mp2.params, vec![2, 4]); // k=2, n=4
    assert_eq!(mp2.coordinates.len(), 8);
}

#[test]
fn test_manifold_point_cross_node_determinism() {
    // Simulate: manifold point → serialize → ship → deserialize
    // The coordinates must be BIT-IDENTICAL after the roundtrip.
    let coords = FixedVector::from_slice(&[
        fp("0.5773502691896257645"),
        fp("0.5773502691896257645"),
        fp("0.5773502691896257645"),
    ]);
    let mp = ManifoldPoint::sphere(2, coords);

    let wire = mp.to_bytes();
    let received = ManifoldPoint::from_bytes(&wire).unwrap();

    // Bit-identical comparison (not tolerance — EXACT)
    for i in 0..3 {
        assert_eq!(mp.coordinates[i], received.coordinates[i],
            "ManifoldPoint wire transport must be bit-identical, coord {}", i);
    }
    assert_eq!(mp.manifold_tag, received.manifold_tag);
    assert_eq!(mp.params, received.params);
}

#[test]
fn test_manifold_coordinate_transport() {
    // Simulate: manifold coordinates → serialize → deserialize → verify
    // This is the exact flow for distributed resolution.

    // "Domain projection" result: a point on S^2
    let coords = FixedVector::from_slice(&[
        fp("0.5773502691896257645"),
        fp("0.5773502691896257645"),
        fp("0.5773502691896257645"),
    ]);

    // Serialize
    let wire_bytes = coords.to_bytes();
    // Deserialize on another node
    let received = FixedVector::from_bytes(&wire_bytes).unwrap();

    // Exact match — deterministic wire format
    for i in 0..3 {
        assert_eq!(coords[i], received[i],
            "Wire format must be bit-identical, component {}", i);
    }
}

#[test]
fn test_tensor_shard_transport() {
    // Simulate: model tensor shard → serialize → ship to node → deserialize
    let shard = Tensor::from_data(&[2, 3], &[
        fp("0.1"), fp("-0.2"), fp("0.3"),
        fp("-0.4"), fp("0.5"), fp("-0.6"),
    ]);

    let wire = shard.to_bytes();
    let received = Tensor::from_bytes(&wire).unwrap();

    assert_eq!(received.shape(), shard.shape());
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(shard.get(&[i, j]), received.get(&[i, j]),
                "Tensor shard [{},{}] must be bit-identical", i, j);
        }
    }
}
