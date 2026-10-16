//! Public functions the 2026-10-16 coverage audit found never executed by the
//! suite (llvm-cov over the full suite; finding
//! gmath-public-api-coverage-audit-2026-10-16): least squares, the
//! homogeneous projective transform, index raising and lowering, truncated
//! SVD with automatic rank, the Lie groups' actions, inner products and
//! dimensions, interval and verdict predicates, FixedVector::from_f32_slice,
//! FixedMatrix::swap_rows, the SE(3) hat / vee maps, the tensor
//! decompositions' compression ratios, and (outside the format suite)
//! DecimalFixed's batch multiply, float conversion and predicates, the wide
//! ternary add / subtract tiers and the serde impls. The tq19 part moved to
//! the `g_math_inference` crate with the module.
//!
//! References: `tests/data/api_gaps_refs.rs` from
//! `scripts/generate_api_gap_refs.py` (exact sums of products, or mpmath at
//! 120 digits). Inputs are dyadic with <= 8 fraction bits and |x| <= 8, exact
//! on every format; each reference is parsed at the build's split, so it is
//! the correctly rounded result and every error below is in storage units.

#[macro_use]
mod common;

macro_rules! suite {
    ($m:ident, $S:ty, $F:literal) => {
        mod $m {
            type FixedPoint = Fixed<$S, $F>;
            const FRAC_BITS: u32 = $F;
            type FixedMatrix<T = FixedPoint> = g_math::fixed_point::FixedMatrix<T>;
            type FixedVector<T = FixedPoint> = g_math::fixed_point::FixedVector<T>;
            type Tensor<T = FixedPoint> = g_math::fixed_point::imperative::tensor::Tensor<T>;
            use g_math::fixed_point::imperative::derived::least_squares;
            use g_math::fixed_point::imperative::lie_group::{GLn, LieGroup, On, SE3, SLn, SO3, SOn};
            use g_math::fixed_point::imperative::manifold::{HyperbolicSpace, Manifold};
            use g_math::fixed_point::imperative::predicates::pd_verdict;
            use g_math::fixed_point::imperative::projective::projective_transform_homogeneous;
            use g_math::fixed_point::imperative::tensor::{lower_index, raise_index};
            use g_math::fixed_point::imperative::tensor_decompose::{truncated_svd, truncated_svd_auto, tucker_decompose};
            use g_math::fixed_point::{Fixed, Interval};

            #[allow(dead_code)]
            mod refs {
                include!("data/api_gaps_refs.rs");
            }

            fn fp(s: &str) -> FixedPoint { FixedPoint::from_str(s) }
            fn fps(v: &[&str]) -> Vec<FixedPoint> { v.iter().map(|s| fp(s)).collect() }
            fn vecs(v: &[&str]) -> FixedVector { FixedVector::from_slice(&fps(v)) }
            fn mat(r: usize, c: usize, v: &[&str]) -> FixedMatrix { FixedMatrix::from_slice(r, c, &fps(v)) }

            /// |got - want| in storage units, saturating at 2^30.
            fn units(got: FixedPoint, want: FixedPoint) -> i32 {
                let half = fp("0.5");
                let mut unit = FixedPoint::one();
                for _ in 0..FRAC_BITS { unit = unit * half; }
                let cap = FixedPoint::try_from_int(1 << 30).ok();
                match got.try_sub(want).and_then(|d| d.abs().try_div(unit)) {
                    Ok(q) if cap.map_or(true, |c| q < c) => q.to_int(),
                    _ => 1 << 30,
                }
            }

            fn check(name: &str, worst: i32, bound: i32) {
                println!("{name}: worst {worst} units (bound {bound})");
                assert!(worst <= bound, "{name}: {worst} units > {bound}");
            }

            #[test]
            fn least_squares_against_mpmath() {
                let mut worst = 0;
                for (m, n, a, b, x) in refs::LEAST_SQUARES {
                    let got = least_squares(&mat(*m, *n, a), &vecs(b)).expect("full column rank");
                    for i in 0..*n { worst = worst.max(units(got[i], fp(x[i]))); }
                }
                // measured 0 at every format (2026-10-16); a regression shows here first
                check("least_squares", worst, 0);
            }

            #[test]
            fn projective_transform_is_the_correctly_rounded_product() {
                let mut worst = 0;
                for (h, p, r) in refs::PROJECTIVE {
                    let got = projective_transform_homogeneous(&mat(3, 3, h), &vecs(p));
                    for i in 0..3 { worst = worst.max(units(got[i], fp(r[i]))); }
                }
                check("projective_transform_homogeneous", worst, 0);
            }

            #[test]
            fn index_maps_keep_the_index_in_place() {
                let (mut lo, mut hi) = (0, 0);
                for (shape, t, idx, g, r) in refs::INDEX_MAP {
                    let d = shape[*idx];
                    let t = Tensor::from_data(shape, &fps(t));
                    let g = Tensor::from_data(&[d, d], &fps(g));
                    for (k, f) in [(0, lower_index::<$S, $F> as fn(&Tensor, usize, &Tensor) -> Tensor), (1, raise_index::<$S, $F>)] {
                        let got = f(&t, *idx, &g);
                        assert_eq!(got.shape(), *shape, "index map moved the index: shape {:?} from {:?}", got.shape(), shape);
                        let w = got.data().iter().zip(r.iter()).map(|(a, b)| units(*a, fp(b))).max().unwrap();
                        if k == 0 { lo = lo.max(w) } else { hi = hi.max(w) }
                    }
                }
                check("lower_index", lo, 0);
                check("raise_index", hi, 0);
            }

            #[test]
            fn truncated_svd_auto_finds_the_rank() {
                let mut worst = 0;
                for (m, n, a, rank, sigma) in refs::TSVD_AUTO {
                    let a = mat(*m, *n, a);
                    let t = truncated_svd_auto(&a, None).expect("svd");
                    assert_eq!(t.sigma.len(), *rank, "automatic rank");
                    for i in 0..*rank { worst = worst.max(units(t.sigma[i], fp(sigma[i]))); }
                    let t = truncated_svd_auto(&a, Some(fp("0.125"))).expect("svd");
                    assert_eq!(t.sigma.len(), *rank, "rank with a threshold");
                    let t = truncated_svd_auto(&a, Some(fp("1000"))).expect("svd");
                    assert_eq!(t.sigma.len(), 1, "at least rank 1");
                }
                check("truncated_svd_auto sigma", worst, 0);
            }

            #[test]
            fn compression_ratios_are_exact() {
                let a = mat(4, 3, refs::TSVD_AUTO[0].2);
                let t = truncated_svd(&a, 2).expect("svd");
                // (4 * 3) / (4 * 2 + 2 + 2 * 3)
                assert_eq!(t.compression_ratio(4, 3), 12.0 / 16.0);
                let x = Tensor::from_data(&[2, 3, 2], &fps(refs::INDEX_MAP[6].1));
                let tk = tucker_decompose(&x, &[1, 2, 1]).expect("tucker");
                // 12 / (core 1*2*1 + factors 2*1 + 3*2 + 2*1)
                assert_eq!(tk.compression_ratio(&[2, 3, 2]), 12.0 / 12.0);
            }

            #[test]
            fn group_actions_and_inner_products() {
                let mut worst = 0;
                for (g, p, r) in refs::ACT3 {
                    let (g, p) = (mat(3, 3, g), vecs(p));
                    let outs = [GLn { n: 3 }.act(&g, &p), On { n: 3 }.act(&g, &p), SLn { n: 3 }.act(&g, &p), SO3.act(&g, &p), SOn { n: 3 }.act(&g, &p)];
                    for got in outs.iter() { for i in 0..3 { worst = worst.max(units(got[i], fp(r[i]))); } }
                }
                check("act (GL, O, SL, SO(3), SO(n))", worst, 0);
                let mut worst = 0;
                for (g, p, r) in refs::ACT_SE3 {
                    let got = SE3.act(&mat(4, 4, g), &vecs(p));
                    for i in 0..3 { worst = worst.max(units(got[i], fp(r[i]))); }
                }
                check("act SE(3)", worst, 0);
                let mut worst = 0;
                for (u, v, r) in refs::INNER {
                    let (u, v) = (vecs(u), vecs(v));
                    let base = FixedVector::new(u.len());
                    let n = u.len();
                    let mut got = vec![GLn { n: 3 }.inner_product(&base, &u, &v), On { n: 3 }.inner_product(&base, &u, &v), SLn { n: 3 }.inner_product(&base, &u, &v), SOn { n: 4 }.inner_product(&base, &u, &v)];
                    if n == 3 { got.push(SO3.inner_product(&base, &u, &v)); }
                    if n == 6 { got.push(SE3.inner_product(&base, &u, &v)); }
                    for g in got { worst = worst.max(units(g, fp(r))); }
                }
                check("inner_product", worst, 0);
            }

            #[test]
            fn group_dimensions_and_identities() {
                for n in [2usize, 3, 4] {
                    assert_eq!((LieGroup::<FixedPoint>::algebra_dim(&GLn { n }), LieGroup::<FixedPoint>::matrix_dim(&GLn { n }), Manifold::<FixedPoint>::dimension(&GLn { n })), (n * n, n, n * n));
                    assert_eq!((LieGroup::<FixedPoint>::algebra_dim(&On { n }), LieGroup::<FixedPoint>::matrix_dim(&On { n }), Manifold::<FixedPoint>::dimension(&On { n })), (n * (n - 1) / 2, n, n * (n - 1) / 2));
                    assert_eq!((LieGroup::<FixedPoint>::algebra_dim(&SLn { n }), LieGroup::<FixedPoint>::matrix_dim(&SLn { n }), Manifold::<FixedPoint>::dimension(&SLn { n })), (n * n - 1, n, n * n - 1));
                    assert_eq!((LieGroup::<FixedPoint>::algebra_dim(&SOn { n }), LieGroup::<FixedPoint>::matrix_dim(&SOn { n }), Manifold::<FixedPoint>::dimension(&SOn { n })), (n * (n - 1) / 2, n, n * (n - 1) / 2));
                    assert_eq!(Manifold::<FixedPoint>::dimension(&HyperbolicSpace { dim: n }), n);
                }
                assert_eq!((LieGroup::<FixedPoint>::algebra_dim(&SO3), LieGroup::<FixedPoint>::matrix_dim(&SO3), Manifold::<FixedPoint>::dimension(&SO3)), (3, 3, 3));
                assert_eq!((LieGroup::<FixedPoint>::algebra_dim(&SE3), LieGroup::<FixedPoint>::matrix_dim(&SE3), Manifold::<FixedPoint>::dimension(&SE3)), (6, 4, 6));
                assert_eq!(LieGroup::<FixedPoint>::identity_element(&SO3), FixedMatrix::identity(3));
                assert_eq!(LieGroup::<FixedPoint>::identity_element(&SE3), FixedMatrix::identity(4));
            }

            #[test]
            fn se3_hat_and_vee() {
                let xi = vecs(&["0.25", "-1.5", "2", "3.125", "-0.5", "7"]);
                let h = SE3::hat_se3(&xi);
                let want = mat(4, 4, &["0", "-2", "-1.5", "3.125", "2", "0", "-0.25", "-0.5", "1.5", "0.25", "0", "7", "0", "0", "0", "0"]);
                assert_eq!(h, want);
                assert_eq!(SE3::vee_se3(&h), xi);
            }

            #[test]
            fn swap_rows_and_from_f32_slice() {
                let mut m = mat(3, 2, &["1", "2", "3", "4", "5", "6"]);
                m.swap_rows(0, 2);
                assert_eq!(m, mat(3, 2, &["5", "6", "3", "4", "1", "2"]));
                m.swap_rows(1, 1);
                assert_eq!(m, mat(3, 2, &["5", "6", "3", "4", "1", "2"]));
                // dyadic f32 values are exact at every format; others truncate like from_f32
                let exact = [1.5f32, -0.25, 3.0, -7.75, 0.00390625];
                let v = FixedVector::from_f32_slice(&exact);
                assert_eq!(v, vecs(&["1.5", "-0.25", "3", "-7.75", "0.00390625"]));
                let inexact = [0.1f32, -0.3, 2.7];
                let v = FixedVector::from_f32_slice(&inexact);
                for (i, x) in inexact.iter().enumerate() { assert_eq!(v[i], FixedPoint::from_f32(*x)); }
            }

            #[test]
            fn interval_and_verdict_predicates() {
                let neg = Interval::new(fp("-3"), fp("-0.5"));
                let straddle = Interval::new(fp("-1"), fp("2"));
                let pos = Interval::new(fp("0.25"), fp("4"));
                assert!(neg.is_certainly_negative() && !straddle.is_certainly_negative() && !pos.is_certainly_negative());
                assert!(!Interval::new(fp("-1"), FixedPoint::ZERO).is_certainly_negative());
                let pd = mat(2, 2, &["4", "1", "1", "3"]);
                let indef = mat(2, 2, &["1", "2", "2", "1"]);
                assert!(pd_verdict(&pd).unwrap().is_proven_positive_definite());
                assert!(!pd_verdict(&indef).unwrap().is_proven_positive_definite());
            }
        }
    };
}

for_each_format!(suite);

mod decimal {
    use g_math::fixed_point::{DecimalFixed, DecimalInterval};

    #[allow(dead_code)]
    mod refs {
        include!("data/api_gaps_refs.rs");
    }

    macro_rules! dec_cases {
        ($name:ident, $d:literal, $mul:path, $f64:path) => {
            #[test]
            fn $name() {
                type D = DecimalFixed<$d>;
                let inputs: Vec<(D, D)> = $mul.iter().map(|(a, b, _)| (D::from_raw(*a), D::from_raw(*b))).collect();
                let mut out = vec![D::ZERO; inputs.len()];
                D::multiply_batch_decimal(&inputs, &mut out);
                for (i, (_, _, r)) in $mul.iter().enumerate() {
                    assert_eq!(out[i].raw_value(), *r, "multiply_batch_decimal case {i}");
                    assert_eq!(out[i], inputs[i].0 * inputs[i].1);
                }
                // binary64 of raw / 10^D: correctly rounded (nearest, ties to even)
                let bad: Vec<_> = $f64.iter().filter(|(raw, bits)| D::from_raw(*raw).to_f64_lossy().to_bits() != *bits).collect();
                assert!(bad.is_empty(), "to_f64_lossy not correctly rounded on {} of {}: first {:?}", bad.len(), $f64.len(), bad.first());
                assert!(D::from_raw(1).is_positive() && !D::ZERO.is_positive() && !D::from_raw(-1).is_positive());
                assert_eq!(D::MAX_VALUE, i128::MAX / D::SCALE);
                assert_eq!(D::MIN_VALUE, i128::MIN / D::SCALE);
                let neg = DecimalInterval::new(D::from_raw(-5), D::from_raw(-1));
                let pos = DecimalInterval::new(D::from_raw(1), D::from_raw(5));
                let straddle = DecimalInterval::new(D::from_raw(-1), D::from_raw(1));
                assert!(neg.is_certainly_negative() && !pos.is_certainly_negative() && !straddle.is_certainly_negative());
                assert!(pos.is_certainly_positive() && !neg.is_certainly_positive() && !straddle.is_certainly_positive());
            }
        };
    }
    dec_cases!(decimal_2, 2, refs::DEC_MUL_2, refs::DEC_F64_2);
    dec_cases!(decimal_4, 4, refs::DEC_MUL_4, refs::DEC_F64_4);
    dec_cases!(decimal_9, 9, refs::DEC_MUL_9, refs::DEC_F64_9);
    dec_cases!(decimal_18, 18, refs::DEC_MUL_18, refs::DEC_F64_18);
}

mod ternary {
    use g_math::fixed_point::domains::balanced_ternary::{
        add_ternary_tq320_320, add_ternary_tq80_80, subtract_ternary_tq320_320, subtract_ternary_tq80_80, Trit,
    };
    use g_math::fixed_point::{I1024, I256};

    #[test]
    fn wide_tiers_add_and_subtract_exactly() {
        let (a, b) = (I256::from_i128(170141183460469231731687303715884105727), I256::from_i128(-98765432109876543210));
        assert_eq!(add_ternary_tq80_80(a, b), a.checked_add(b).unwrap());
        assert_eq!(subtract_ternary_tq80_80(a, b), a.checked_sub(b).unwrap());
        assert_eq!(add_ternary_tq80_80(a, a), a.checked_add(a).unwrap());
        let (c, d) = (I1024::from_i128(i128::MAX), I1024::from_i128(i128::MIN));
        assert_eq!(add_ternary_tq320_320(c, d), I1024::from_i128(-1));
        assert_eq!(subtract_ternary_tq320_320(c, d), c.checked_sub(d).unwrap());
    }

    #[test]
    #[should_panic(expected = "overflow")]
    fn tq80_add_overflow_panics() { let _ = add_ternary_tq80_80(I256::max_value(), I256::from_i128(1)); }

    #[test]
    #[should_panic(expected = "overflow")]
    fn tq80_subtract_overflow_panics() { let _ = subtract_ternary_tq80_80(I256::min_value(), I256::from_i128(1)); }

    #[test]
    #[should_panic(expected = "overflow")]
    fn tq320_add_overflow_panics() { let _ = add_ternary_tq320_320(I1024::max_value(), I1024::from_i128(1)); }

    #[test]
    #[should_panic(expected = "overflow")]
    fn tq320_subtract_overflow_panics() { let _ = subtract_ternary_tq320_320(I1024::min_value(), I1024::from_i128(1)); }

    #[test]
    fn trit_as_i8() {
        assert_eq!([Trit::Neg.as_i8(), Trit::Zero.as_i8(), Trit::Pos.as_i8()], [-1, 0, 1]);
        for v in [-1i8, 0, 1] { assert_eq!(Trit::from_i8(v).unwrap().as_i8(), v); }
    }
}

#[cfg(feature = "serde")]
mod serde_impls {
    use g_math::fixed_point::imperative::ManifoldPoint;
    use g_math::fixed_point::imperative::{MANIFOLD_TAG_EUCLIDEAN, MANIFOLD_TAG_GRASSMANNIAN, MANIFOLD_TAG_HYPERBOLIC, MANIFOLD_TAG_SPD, MANIFOLD_TAG_SPHERE};
    use g_math::fixed_point::imperative::tensor::Tensor;
    use g_math::fixed_point::{FixedMatrix, FixedVector, Q16_16, Q256_256, Q64_64};
    use serde::de::value::{BytesDeserializer, Error as DeError, SeqDeserializer};
    use serde::de::Deserialize;
    use serde::ser::{self, Impossible, Serialize};

    /// A serializer that accepts exactly one byte string and returns it.
    struct Bytes;
    #[derive(Debug)]
    struct E(String);
    impl std::fmt::Display for E { fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str(&self.0) } }
    impl std::error::Error for E {}
    impl ser::Error for E { fn custom<T: std::fmt::Display>(m: T) -> Self { E(m.to_string()) } }
    macro_rules! no { ($($f:ident($($t:ty),*)),*) => { $(fn $f(self, $(_: $t),*) -> Result<Vec<u8>, E> { Err(E("only bytes".into())) })* } }
    impl ser::Serializer for Bytes {
        type Ok = Vec<u8>; type Error = E;
        type SerializeSeq = Impossible<Vec<u8>, E>; type SerializeTuple = Impossible<Vec<u8>, E>;
        type SerializeTupleStruct = Impossible<Vec<u8>, E>; type SerializeTupleVariant = Impossible<Vec<u8>, E>;
        type SerializeMap = Impossible<Vec<u8>, E>; type SerializeStruct = Impossible<Vec<u8>, E>;
        type SerializeStructVariant = Impossible<Vec<u8>, E>;
        fn serialize_bytes(self, v: &[u8]) -> Result<Vec<u8>, E> { Ok(v.to_vec()) }
        no!(serialize_bool(bool), serialize_i8(i8), serialize_i16(i16), serialize_i32(i32), serialize_i64(i64), serialize_u8(u8), serialize_u16(u16), serialize_u32(u32), serialize_u64(u64), serialize_f32(f32), serialize_f64(f64), serialize_char(char), serialize_str(&str), serialize_none(), serialize_unit(), serialize_unit_struct(&'static str), serialize_unit_variant(&'static str, u32, &'static str));
        fn serialize_some<T: ?Sized + Serialize>(self, _: &T) -> Result<Vec<u8>, E> { Err(E("only bytes".into())) }
        fn serialize_newtype_struct<T: ?Sized + Serialize>(self, _: &'static str, _: &T) -> Result<Vec<u8>, E> { Err(E("only bytes".into())) }
        fn serialize_newtype_variant<T: ?Sized + Serialize>(self, _: &'static str, _: u32, _: &'static str, _: &T) -> Result<Vec<u8>, E> { Err(E("only bytes".into())) }
        fn serialize_seq(self, _: Option<usize>) -> Result<Self::SerializeSeq, E> { Err(E("only bytes".into())) }
        fn serialize_tuple(self, _: usize) -> Result<Self::SerializeTuple, E> { Err(E("only bytes".into())) }
        fn serialize_tuple_struct(self, _: &'static str, _: usize) -> Result<Self::SerializeTupleStruct, E> { Err(E("only bytes".into())) }
        fn serialize_tuple_variant(self, _: &'static str, _: u32, _: &'static str, _: usize) -> Result<Self::SerializeTupleVariant, E> { Err(E("only bytes".into())) }
        fn serialize_map(self, _: Option<usize>) -> Result<Self::SerializeMap, E> { Err(E("only bytes".into())) }
        fn serialize_struct(self, _: &'static str, _: usize) -> Result<Self::SerializeStruct, E> { Err(E("only bytes".into())) }
        fn serialize_struct_variant(self, _: &'static str, _: u32, _: &'static str, _: usize) -> Result<Self::SerializeStructVariant, E> { Err(E("only bytes".into())) }
    }

    /// Serialize as bytes, then deserialize from those bytes as a byte string
    /// (binary formats) and as a sequence of u8 (self-describing formats
    /// that write bytes as an array, e.g. JSON).
    fn round_trip<T: Serialize + for<'de> Deserialize<'de>>(v: &T, to_bytes: impl Fn(&T) -> Vec<u8>) {
        let bytes = v.serialize(Bytes).unwrap();
        assert_eq!(bytes, to_bytes(v), "serializes as its to_bytes");
        let from_bytes = T::deserialize(BytesDeserializer::<DeError>::new(&bytes)).unwrap();
        assert_eq!(to_bytes(&from_bytes), bytes);
        let seq: SeqDeserializer<_, DeError> = SeqDeserializer::new(bytes.clone().into_iter());
        let from_seq = T::deserialize(seq).unwrap();
        assert_eq!(to_bytes(&from_seq), bytes);
        assert!(T::deserialize(BytesDeserializer::<DeError>::new(&bytes[..bytes.len() - 1])).is_err(), "truncated bytes rejected");
    }

    #[test]
    fn every_form_round_trips() {
        let a = Q64_64::from_str("-3.25");
        round_trip(&a, |x| x.to_bytes());
        let b = Q16_16::from_str("7.5");
        round_trip(&b, |x| x.to_bytes());
        let c = Q256_256::from_str("0.1");
        round_trip(&c, |x| x.to_bytes());
        let v = FixedVector::from_slice(&[a, Q64_64::from_str("2"), Q64_64::from_str("0.125")]);
        round_trip(&v, |x| x.to_bytes());
        let m = FixedMatrix::from_slice(2, 2, &[a, a, Q64_64::ZERO, Q64_64::from_str("1")]);
        round_trip(&m, |x| x.to_bytes());
        let t = Tensor::from_data(&[2, 1, 2], &[a, a, a, Q64_64::from_str("5")]);
        round_trip(&t, |x| x.to_bytes());
        for p in [ManifoldPoint::euclidean(3, v.clone()), ManifoldPoint::sphere(2, v.clone()), ManifoldPoint::hyperbolic(2, v.clone())] {
            round_trip(&p, |x: &ManifoldPoint<Q64_64>| x.to_bytes());
        }
    }

    #[test]
    fn manifold_tags() {
        assert_eq!([MANIFOLD_TAG_EUCLIDEAN, MANIFOLD_TAG_SPHERE, MANIFOLD_TAG_HYPERBOLIC, MANIFOLD_TAG_SPD, MANIFOLD_TAG_GRASSMANNIAN], [1, 2, 3, 4, 5]);
        let v = FixedVector::from_slice(&[Q64_64::from_str("1"), Q64_64::ZERO]);
        assert_eq!(ManifoldPoint::euclidean(2, v.clone()).to_bytes()[0], MANIFOLD_TAG_EUCLIDEAN);
        assert_eq!(ManifoldPoint::sphere(1, v.clone()).to_bytes()[0], MANIFOLD_TAG_SPHERE);
        assert_eq!(ManifoldPoint::hyperbolic(1, v).to_bytes()[0], MANIFOLD_TAG_HYPERBOLIC);
    }
}
