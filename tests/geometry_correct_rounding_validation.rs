//! The closed-form manifold maps against mpmath
//! (`scripts/generate_geometry_refs.py`, 400 digits): Sphere and
//! HyperbolicSpace distance, log_map, exp_map and parallel_transport on
//! stored inputs. Every output component must be the exact value of the map
//! on those inputs, rounded once (nearest, ties toward +infinity): no
//! tolerance. The tally per format is printed with `--nocapture`.

#[macro_use]
mod common;

macro_rules! suite {
    ($m:ident, $S:ty, $F:literal) => {
        mod $m {
            #![allow(dead_code, unused_imports)]
            type FixedPoint = g_math::fixed_point::Fixed<$S, $F>;
            const BITS: u32 = <$S as g_math::fixed_point::WideInt>::BITS;
            const FRAC_BITS: u32 = $F;
            use g_math::fixed_point::{FixedVector, WideInt};
            use g_math::fixed_point::imperative::manifold::{HyperbolicSpace, Manifold, Sphere};
            use g_math::fixed_point::imperative::lie_group::{LieGroup, SE3, SO3};
            use g_math::fixed_point::FixedMatrix;
            use g_math::fixed_point::imperative::projective::{self, Moebius};

            fn flat(m: &FixedMatrix<FixedPoint>) -> Vec<FixedPoint> {
                let mut out = Vec::new();
                for i in 0..m.rows() { for j in 0..m.cols() { out.push(m.get(i, j)); } }
                out
            }

            fn square(v: &FixedVector<FixedPoint>) -> FixedMatrix<FixedPoint> {
                let n = if v.len() == 16 { 4 } else { 3 };
                FixedMatrix::from_slice(n, n, v.as_slice())
            }

            mod data {
                #[allow(unused_imports)]
                use super::{BITS, FRAC_BITS};
                include!("data/geometry_refs.rs");
            }

            fn raw(s: &str) -> $S {
                let (neg, digits) = match s.strip_prefix('-') { Some(r) => (true, r), None => (false, s) };
                let ten = <$S as WideInt>::from_i128(10);
                let mut v = <$S as WideInt>::from_i128(0);
                for ch in digits.chars() { v = v * ten + <$S as WideInt>::from_i128(ch.to_digit(10).expect("digit") as i128); }
                if neg { -v } else { v }
            }

            fn vector(s: &str) -> FixedVector<FixedPoint> {
                FixedVector::from_slice(&s.split(',').map(|t| FixedPoint::from_raw(raw(t))).collect::<Vec<_>>())
            }

            #[test]
            fn closed_form_maps_are_correctly_rounded() {
                let (sph, hyp) = (Sphere { dim: 2 }, HyperbolicSpace { dim: 2 });
                let mut tally: Vec<(&str, usize, usize)> = Vec::new();
                let mut failures = Vec::new();
                for &(op, ins, want) in data::REFS {
                    let v: Vec<&str> = ins.split(';').collect();
                    let (p, q, t) = (vector(v[0]), vector(v[1]), vector(v[2]));
                    let want: Vec<$S> = want.split(',').map(raw).collect();
                    let got: Result<Vec<$S>, _> = std::panic::catch_unwind(|| {
                        let r = match op {
                            "s_dist" => sph.distance(&p, &q).map(|x| vec![x]),
                            "s_log" => sph.log_map(&p, &q).map(|x| x.as_slice().to_vec()),
                            "s_exp" => sph.exp_map(&p, &t).map(|x| x.as_slice().to_vec()),
                            "s_tr" => sph.parallel_transport(&p, &q, &t).map(|x| x.as_slice().to_vec()),
                            "h_dist" => hyp.distance(&p, &q).map(|x| vec![x]),
                            "h_log" => hyp.log_map(&p, &q).map(|x| x.as_slice().to_vec()),
                            "h_exp" => hyp.exp_map(&p, &t).map(|x| x.as_slice().to_vec()),
                            "h_tr" => hyp.parallel_transport(&p, &q, &t).map(|x| x.as_slice().to_vec()),
                            "so3_exp" => SO3.lie_exp(&p).map(|m| flat(&m)),
                            "so3_log" => SO3.lie_log(&square(&p)).map(|x| x.as_slice().to_vec()),
                            "so3_expmap" => Manifold::exp_map(&SO3, &p, &q).map(|x| x.as_slice().to_vec()),
                            "so3_logmap" => Manifold::log_map(&SO3, &p, &q).map(|x| x.as_slice().to_vec()),
                            "so3_dist" => Manifold::distance(&SO3, &p, &q).map(|x| vec![x]),
                            "so3_tr" => Manifold::parallel_transport(&SO3, &p, &q, &t).map(|x| x.as_slice().to_vec()),
                            "se3_exp" => SE3.lie_exp(&p).map(|m| flat(&m)),
                            "se3_log" => SE3.lie_log(&square(&p)).map(|x| x.as_slice().to_vec()),
                            "se3_expmap" => Manifold::exp_map(&SE3, &p, &q).map(|x| x.as_slice().to_vec()),
                            "se3_logmap" => Manifold::log_map(&SE3, &p, &q).map(|x| x.as_slice().to_vec()),
                            "se3_dist" => Manifold::distance(&SE3, &p, &q).map(|x| vec![x]),
                            "mob" => Moebius::new(p[0], p[1], p[2], p[3]).apply(p[4]).map(|x| vec![x]),
                            "st_proj" => projective::stereo_project(&p).map(|x| x.as_slice().to_vec()),
                            "st_unproj" => Ok(projective::stereo_unproject(&p).as_slice().to_vec()),
                            "cross" => projective::cross_ratio_1d(p[0], p[1], p[2], p[3]).map(|x| vec![x]),
                            other => panic!("unknown op {other}"),
                        };
                        r.map(|xs| xs.iter().map(|x| x.raw()).collect::<Vec<_>>())
                    }).map_err(|_| ()).and_then(|r| r.map_err(|_| ()));
                    let i = match tally.iter().position(|t| t.0 == op) { Some(i) => i, None => { tally.push((op, 0, 0)); tally.len() - 1 } };
                    tally[i].1 += 1;
                    if got.as_ref().ok() != Some(&want) {
                        tally[i].2 += 1;
                        if failures.len() < 12 { failures.push(format!("{op} {ins}: got {:?}, want {:?}", got, want)); }
                    }
                }
                println!("geometry Q{}.{}: op, cases, not correctly rounded", BITS - FRAC_BITS, FRAC_BITS);
                for t in &tally { println!("  {:7} {:4} {:4}", t.0, t.1, t.2); }
                assert!(!data::REFS.is_empty());
                assert!(failures.is_empty(), "Q{}.{}: {} of {} not correctly rounded, e.g. {:#?}", BITS - FRAC_BITS, FRAC_BITS,
                    tally.iter().map(|t| t.2).sum::<usize>(), data::REFS.len(), failures);
            }
        }
    };
}

for_each_format!(suite);
suite!(q8_24, i32, 24);
