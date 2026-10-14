//! Transcendental Functions: Fixed-Point Implementations
//!
//! 18 transcendental functions across 6 engines, computed at tier N+1 for full storage-tier precision.
//!
//! Dedicated engines: `exp`, `ln`, `sqrt`, `sin_cos`, `atan`. (pow is composed as exp(y·ln x) at the compute tier: the dead dedicated engine was removed in 0.5.0.)
//! FASC-composed: `tan`, `asin`, `acos`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `atan2`.
//!
//! | Profile    | Format    | Decimals | ULP |
//! |------------|-----------|----------|-----|
//! | Embedded   | Q64.64    | 19       | 0   |
//! | Balanced   | Q128.128  | 38       | 0   |
//! | Scientific | Q256.256  | 77       | 0   |
//!
//! Strategy: compute at tier N+1 (double precision), downscale with rounding.
//!
//! Naming. A function called `*_q64_64_native`, `*_q128_128_native`,
//! `*_q256_256_native` or `*_q512_512_native` is an engine: it takes and
//! returns values in that one format, is compiled on every profile and
//! returns the same integers on every profile (`engine_identity_tests`
//! below pins a digest of each). The atan engines exist twice at Q128.128 and
//! Q256.256: `_native` carries enough series terms for the format's own
//! precision, `_compute` carries the terms needed when the format is the
//! compute tier of the storage format half its width. The profile-bound
//! wrappers (`*_binary_iN`, `*_compute_tier_iN`, `pi_half_iN`) are gone since
//! 0.7.0: a caller names the engine of the format it computes in.

pub mod exp_tier_n_plus_1;
pub mod ln_tier_n_plus_1;
pub mod sqrt_tier_n_plus_1;
pub mod sin_cos_tier_n_plus_1;
pub mod atan_tier_n_plus_1;
// ln decomposition functions are internal to ln_tier_n_plus_1.rs (not public API)

// traits.rs and exp.rs archived — FASC uses exp_tier_n_plus_1 directly


#[cfg(test)]
mod engine_identity_tests {
    //! One name, one meaning: every engine returns the same integers in every
    //! build. Each digest below was recorded once and must hold on all
    //! profiles; the accuracy of each engine is established separately by the
    //! mpmath gates of the profile that computes with it.
    use super::atan_tier_n_plus_1::*;
    use super::exp_tier_n_plus_1::*;
    use super::ln_tier_n_plus_1::*;
    use super::sin_cos_tier_n_plus_1::*;
    use super::sqrt_tier_n_plus_1::*;
    use crate::fixed_point::{I1024, I256, I512};

    const N: usize = 24;

    /// Deterministic inputs: `hi` is a signed value below 16 in magnitude
    /// with 56 fractional bits, `lo` fills the bits below it.
    fn inputs() -> Vec<(i64, u64)> {
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            s ^ (s >> 29)
        };
        (0..N).map(|_| (((next() >> 3) as i64) - (1i64 << 60), next())).collect()
    }

    trait Fmt: Copy {
        fn make(hi: i64, lo: u64) -> Self;
        fn abs_nonzero(self) -> Self;
        fn feed(self, h: &mut u64);
    }
    fn fnv(h: &mut u64, w: u64) {
        for b in w.to_le_bytes() {
            *h = (*h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    impl Fmt for i128 {
        fn make(hi: i64, lo: u64) -> Self { ((hi as i128) << 8) | (lo >> 56) as i128 }
        fn abs_nonzero(self) -> Self { if self <= 0 { 1 - self } else { self } }
        fn feed(self, h: &mut u64) { fnv(h, self as u64); fnv(h, (self >> 64) as u64); }
    }
    macro_rules! wide_fmt {
        ($t:ident, $shift:expr) => {
            impl Fmt for $t {
                fn make(hi: i64, lo: u64) -> Self {
                    ($t::from_i128(hi as i128) << $shift) + $t::from_i128((lo >> 1) as i128)
                }
                fn abs_nonzero(self) -> Self {
                    if self <= $t::zero() { $t::from_i128(1) - self } else { self }
                }
                fn feed(self, h: &mut u64) { for w in self.words { fnv(h, w); } }
            }
        };
    }
    wide_fmt!(I256, 72);
    wide_fmt!(I512, 200);
    wide_fmt!(I1024, 456);

    fn digest1<T: Fmt>(positive: bool, f: impl Fn(T) -> T) -> u64 {
        let mut h = 0xCBF2_9CE4_8422_2325u64;
        for (hi, lo) in inputs() {
            let x = T::make(hi, lo);
            f(if positive { x.abs_nonzero() } else { x }).feed(&mut h);
        }
        h
    }
    fn digest_pair<T: Fmt>(f: impl Fn(T) -> (T, T)) -> u64 {
        let mut h = 0xCBF2_9CE4_8422_2325u64;
        for (hi, lo) in inputs() {
            let (a, b) = f(T::make(hi, lo));
            a.feed(&mut h);
            b.feed(&mut h);
        }
        h
    }
    fn digest2<T: Fmt>(f: impl Fn(T, T) -> T) -> u64 {
        let mut h = 0xCBF2_9CE4_8422_2325u64;
        let v = inputs();
        for i in 0..N {
            let (y, x) = (v[i], v[(i + 7) % N]);
            f(T::make(y.0, y.1), T::make(x.0, x.1)).feed(&mut h);
        }
        h
    }

    fn all() -> Vec<(&'static str, u64)> {
        vec![
            ("exp_q64_64", digest1::<i128>(false, exp_q64_64_native)),
            ("exp_q128_128", digest1::<I256>(false, exp_q128_128_native)),
            ("exp_q256_256", digest1::<I512>(false, exp_q256_256_native)),
            ("exp_q512_512", digest1::<I1024>(false, exp_q512_512_native)),
            ("ln_q64_64", digest1::<i128>(true, ln_q64_64_native)),
            ("ln_q128_128", digest1::<I256>(true, ln_q128_128_native)),
            ("ln_q256_256", digest1::<I512>(true, ln_q256_256_native)),
            ("ln_q512_512", digest1::<I1024>(true, ln_q512_512_native)),
            ("sqrt_q64_64", digest1::<i128>(true, sqrt_q64_64_native)),
            ("sqrt_q128_128", digest1::<I256>(true, sqrt_q128_128_native)),
            ("sqrt_q256_256", digest1::<I512>(true, sqrt_q256_256_native)),
            ("sqrt_q512_512", digest1::<I1024>(true, sqrt_q512_512_native)),
            ("sincos_q64_64", digest_pair::<i128>(sincos_q64_64_native)),
            ("sincos_q128_128", digest_pair::<I256>(sincos_q128_128_native)),
            ("sincos_q256_256", digest_pair::<I512>(sincos_q256_256_native)),
            ("sincos_q512_512", digest_pair::<I1024>(sincos_q512_512_native)),
            ("atan_q64_64", digest1::<i128>(false, atan_q64_64_native)),
            ("atan_q128_128", digest1::<I256>(false, atan_q128_128_native)),
            ("atan_q256_256", digest1::<I512>(false, atan_q256_256_native)),
            ("atan_q512_512", digest1::<I1024>(false, atan_q512_512_native)),
            ("atan_q128_128_compute", digest1::<I256>(false, atan_q128_128_compute)),
            ("atan_q256_256_compute", digest1::<I512>(false, atan_q256_256_compute)),
            ("atan2_q64_64", digest2::<i128>(atan2_q64_64_native)),
            ("atan2_q128_128", digest2::<I256>(atan2_q128_128_native)),
            ("atan2_q256_256", digest2::<I512>(atan2_q256_256_native)),
            ("atan2_q512_512", digest2::<I1024>(atan2_q512_512_native)),
            ("atan2_q128_128_compute", digest2::<I256>(atan2_q128_128_compute)),
            ("atan2_q256_256_compute", digest2::<I512>(atan2_q256_256_compute)),
        ]
    }

    const EXPECTED: &[(&str, u64)] = &[
        ("exp_q64_64", 0x69A6F886355E457B),
        ("exp_q128_128", 0xCDFE13FF9074C858),
        ("exp_q256_256", 0x14AF8CB2AE9F805F),
        ("exp_q512_512", 0x1AD95FEC73C76C72),
        ("ln_q64_64", 0xB8F7832C0117771B),
        ("ln_q128_128", 0x2DCA1D208B1C7C79),
        ("ln_q256_256", 0xBA05DE429E849712),
        ("ln_q512_512", 0x173965474DE46D30),
        ("sqrt_q64_64", 0x86F85AACCF5786D6),
        ("sqrt_q128_128", 0xF6CB95184DEEE373),
        ("sqrt_q256_256", 0xDA0283F57964943E),
        ("sqrt_q512_512", 0xE3C4BE210FF10DEB),
        ("sincos_q64_64", 0x612993E21B6A43B5),
        ("sincos_q128_128", 0x762877FC28547539),
        ("sincos_q256_256", 0xE5BD8A2681FA9E0A),
        ("sincos_q512_512", 0x292872BD72763885),
        ("atan_q64_64", 0xC1E0A84B8345D2F9),
        ("atan_q128_128", 0x5C58E0A5CADA6049),
        ("atan_q256_256", 0x614CBB54C2F597BD),
        ("atan_q512_512", 0x51D6839FDD7372B1),
        ("atan_q128_128_compute", 0x0514252DF596858A),
        ("atan_q256_256_compute", 0x06594EDCA60B4B65),
        ("atan2_q64_64", 0xC34556B2EB03F90B),
        ("atan2_q128_128", 0x27216FAB6E2EAFAB),
        ("atan2_q256_256", 0x59FA9D4620DF5BAA),
        ("atan2_q512_512", 0x20B329BBEFA12959),
        ("atan2_q128_128_compute", 0xB9D938EADE60EC5A),
        ("atan2_q256_256_compute", 0x51A2B2D53AE83152),
    ];

    #[test]
    fn engines_return_the_same_integers_on_every_profile() {
        let got = all();
        assert_eq!(got.len(), EXPECTED.len());
        for ((name, d), (ename, ed)) in got.iter().zip(EXPECTED) {
            assert_eq!(name, ename);
            assert_eq!(d, ed, "{} differs from its recorded digest", name);
        }
    }
}
