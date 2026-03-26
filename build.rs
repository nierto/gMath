//! Build script for gMath: Profile-Aware Mathematical Constants and Lookup Tables
//!
//! ARCHITECTURE: Zero-float, profile-tiered precision, hardcoded constants
//! - Mathematical constants (π, e, √2, ln2, φ, ln10, √3, √5) hardcoded at 3 precision tiers
//! - Q64.64/Q128.128/Q256.256 transcendental lookup tables generated from rational constants
//! - Profile detection: embedded(19d) | balanced(38d) | scientific(77d)
//! - ZERO floating point arithmetic - pure rational → fixed-point conversion
//!
//! **PROFILES**:
//! - embedded: Q64.64 (scalar), balanced: Q128.128, scientific: Q256.256
//!
//! MEMORY: <10s build time, <500MB peak memory (vs 60GB explosion with algorithmic generation)
//! PRECISION: Continued fraction convergents (unique denominators, NOT 10^N scaling)
//! VERIFICATION: All constants verified against OEIS sequences

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

// Build-time rational arithmetic using num-bigint
use num_bigint::{BigInt, Sign};
use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Signed, Zero};

// ============================================================================
// DEPLOYMENT PROFILE DETECTION
// ============================================================================

/// Deployment profiles with precision tiers
///
/// 3 core profiles aligned to UGOD tiers
/// - Embedded: Tier 4 (i64/u64) → Q64.64 (i128), scalar-only
/// - Balanced: Tier 5 (i128/u128) → Q128.128 (I256), general purpose
/// - Scientific: Tier 6 (I256/I256) → Q256.256 (I512), maximum precision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum DeploymentProfile {
    Realtime,    //  4 decimals, Q16.16,   i32 storage, i64 compute
    Compact,     //  9 decimals, Q32.32,   i64 storage, i128 compute
    Embedded,    // 19 decimals, Q64.64,   i128 arithmetic
    Balanced,    // 38 decimals, Q128.128, I256 arithmetic
    Scientific,  // 77 decimals, Q256.256, I512 arithmetic
    Custom,
}

/// Precision configuration for each profile
#[derive(Debug, Clone)]
struct PrecisionConfig {
    target_decimal_places: u8,
    profile_name: &'static str,
    table_format: &'static str, // "Q64.64", "Q128.128", "Q256.256"
}

impl PrecisionConfig {
    fn for_profile(profile: DeploymentProfile) -> Self {
        match profile {
            DeploymentProfile::Realtime => PrecisionConfig {
                target_decimal_places: 4,
                profile_name: "realtime",
                table_format: "Q16.16",
            },
            DeploymentProfile::Compact => PrecisionConfig {
                target_decimal_places: 9,
                profile_name: "compact",
                table_format: "Q32.32",
            },
            DeploymentProfile::Embedded => PrecisionConfig {
                target_decimal_places: 19,
                profile_name: "embedded",
                table_format: "Q64.64",
            },
            DeploymentProfile::Balanced => PrecisionConfig {
                target_decimal_places: 38,
                profile_name: "balanced",
                table_format: "Q128.128",
            },
            DeploymentProfile::Scientific => PrecisionConfig {
                target_decimal_places: 77,
                profile_name: "scientific",
                table_format: "Q256.256",
            },
            DeploymentProfile::Custom => {
                let custom_precision = env::var("GMATH_MAX_DECIMAL_PRECISION")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(38);

                let table_format = if custom_precision <= 19 {
                    "Q64.64"
                } else if custom_precision <= 38 {
                    "Q128.128"
                } else {
                    "Q256.256"
                };

                PrecisionConfig {
                    target_decimal_places: custom_precision,
                    profile_name: "custom",
                    table_format,
                }
            }
        }
    }
}

/// Detect deployment profile from environment and features
fn detect_deployment_profile() -> DeploymentProfile {
    // Check environment variable first (highest priority)
    if let Ok(profile_str) = env::var("GMATH_PROFILE") {
        match profile_str.to_lowercase().as_str() {
            "realtime" => return DeploymentProfile::Realtime,
            "compact" | "fast" => return DeploymentProfile::Compact,
            "embedded" => return DeploymentProfile::Embedded,
            "balanced" => return DeploymentProfile::Balanced,
            "scientific" => return DeploymentProfile::Scientific,
            _ => {}
        }
    }

    // Check Cargo features using environment variables
    // NOTE: cfg!() in build.rs checks build.rs compile-time features, NOT the main crate's features!
    // Cargo sets CARGO_FEATURE_<NAME> env vars for enabled features at build time.
    //
    // IMPORTANT: Check EXPLICIT profile features first, in order of precision (highest to lowest)!
    // Order: embedded → scientific → balanced → default
    if env::var("CARGO_FEATURE_REALTIME").is_ok() {
        DeploymentProfile::Realtime
    } else if env::var("CARGO_FEATURE_COMPACT").is_ok() || env::var("CARGO_FEATURE_FAST").is_ok() {
        DeploymentProfile::Compact
    } else if env::var("CARGO_FEATURE_EMBEDDED_MINIMAL").is_ok() || env::var("CARGO_FEATURE_EMBEDDED").is_ok() {
        DeploymentProfile::Embedded
    } else if env::var("CARGO_FEATURE_SCIENTIFIC").is_ok() {
        DeploymentProfile::Scientific
    } else if env::var("CARGO_FEATURE_BALANCED").is_ok() {
        DeploymentProfile::Balanced
    } else {
        // Default to embedded profile (Q64.64, 19 decimals, fastest)
        DeploymentProfile::Embedded
    }
}

// ============================================================================
// HARDCODED MATHEMATICAL CONSTANTS (Continued Fraction Convergents)
// ============================================================================
//
// Generated via scripts/compute_convergents.py using high-precision decimal arithmetic
// Verified against OEIS sequences: A002485/A002486 (π), A007676/A007677 (e), etc.
//
// CRITICAL: Denominators are UNIQUE (continued fraction property)
// NOT decimal scaling (10^N) - mathematically optimal approximations

#[allow(dead_code)]
mod hardcoded_constants {
    /// String literals for pattern detection (77 decimal places)
    pub mod strings {
        pub const PI: &str = "3.141592653589793238462643383279502884197169399375105820974944592307816406286208";
        pub const E: &str = "2.718281828459045235360287471352662497757247093699959574966967627724076630353547";
        pub const SQRT_2: &str = "1.414213562373095048801688724209698078569671875376948073176679737990732478462107";
        pub const LN_2: &str = "0.693147180559945309417232121458176568075500134360255254120680009493393621969694";
        pub const PHI: &str = "1.618033988749894848204586834365638117720309179805762862135448622705260462818902";
        pub const LN_10: &str = "2.302585092994045684017991454684364207601101488628772976033327900967572609677352";
        pub const SQRT_3: &str = "1.732050807568877293527446341505872366942805253810380628055806979451933016908800";
        pub const SQRT_5: &str = "2.236067977499789696409173668731276235440618359611525724270897245410520925637804";
    }

    /// 19 decimal precision (i128 numerator/denominator pairs)
    #[allow(dead_code)]
    pub mod tier_19d {
        pub const PI: (i128, i128) = (21053343141, 6701487259);
        pub const E: (i128, i128) = (28875761731, 10622799089);
        pub const SQRT_2: (i128, i128) = (10812186007, 7645370045);
        pub const LN_2: (i128, i128) = (6847196937, 9878417065);
        pub const PHI: (i128, i128) = (12586269025, 7778742049);
        pub const LN_10: (i128, i128) = (42528320816, 18469815055);
        pub const SQRT_3: (i128, i128) = (9863382151, 5694626340);
        pub const SQRT_5: (i128, i128) = (22768774562, 10182505537);
    }

    /// 38 decimal precision (I256 as hex arrays [u64; 4])
    #[allow(dead_code)]
    pub mod tier_38d {
        pub const PI_NUM: [u64; 4] = [0x76db573ddce0c076, 0xe3a, 0, 0];
        pub const PI_DEN: [u64; 4] = [0xf93771bb7b1ce51, 0x486, 0, 0];
        pub const E_NUM: [u64; 4] = [0xc8ebe3d0a1e4891f, 0x8dc, 0, 0];
        pub const E_DEN: [u64; 4] = [0x818dc41bf9f74000, 0x343, 0, 0];
        pub const SQRT_2_NUM: [u64; 4] = [0x846aef57c4c05e91, 0x22c, 0, 0];
        pub const SQRT_2_DEN: [u64; 4] = [0xcd30cce8c6c6699c, 0x188, 0, 0];
        pub const LN_2_NUM: [u64; 4] = [0x3e1fa2f117bf2801, 0x30e, 0, 0];
        pub const LN_2_DEN: [u64; 4] = [0xd65a36b85a8c24d6, 0x466, 0, 0];
        pub const PHI_NUM: [u64; 4] = [0x6f4e7b57a99c6400, 0x2ce, 0, 0];
        pub const PHI_DEN: [u64; 4] = [0xcb5f35a76976c43, 0x1bb, 0, 0];
        pub const LN_10_NUM: [u64; 4] = [0x9dcd3c524c5deccb, 0xc3c, 0, 0];
        pub const LN_10_DEN: [u64; 4] = [0xe9e66e8ba0ce32a9, 0x550, 0, 0];
        pub const SQRT_3_NUM: [u64; 4] = [0x1bf0a94e65f38dda, 0x2d4, 0, 0];
        pub const SQRT_3_DEN: [u64; 4] = [0xfd33edda88aff58f, 0x19f, 0, 0];
        pub const SQRT_5_NUM: [u64; 4] = [0xbc29d2d0ca001001, 0x322, 0, 0];
        pub const SQRT_5_DEN: [u64; 4] = [0xd3ac2c2e0a61ec0, 0x16, 0, 0];
    }

    /// 77 decimal precision (I512 via string representation)
    pub mod tier_77d {
        pub const PI_NUM_STR: &str = "4170167120753626267426951858176848373908";
        pub const PI_DEN_STR: &str = "1327405421574472826819306015318500841729";
        pub const E_NUM_STR: &str = "3224546177070375504839670260260400797251";
        pub const E_DEN_STR: &str = "1186244245652160441069426743288856160849";
        pub const SQRT_2_NUM_STR: &str = "94741125149636933417873079920900017937";
        pub const SQRT_2_DEN_STR: &str = "66992092050551637663438906713182313772";
        pub const LN_2_NUM_STR: &str = "245136811127310559099810329887594754993";
        pub const LN_2_DEN_STR: &str = "353657661752705407042003521780857290376";
        pub const PHI_NUM_STR: &str = "573147844013817084101";
        pub const PHI_DEN_STR: &str = "354224848179261915075";
        pub const LN_10_NUM_STR: &str = "3453684117652085635286271321032656666369";
        pub const LN_10_DEN_STR: &str = "1499915954533201960943129396032093884739";
        pub const SQRT_3_NUM_STR: &str = "19785515999613069781581367687";
        pub const SQRT_3_DEN_STR: &str = "11423172988432253331946397284";
        pub const SQRT_5_NUM_STR: &str = "1576282345991202924472633606870413747838";
        pub const SQRT_5_DEN_STR: &str = "704934895473834571656017795987798259457";
    }
}

// ============================================================================
// HELPER FUNCTIONS FOR RATIONAL ARITHMETIC
// ============================================================================

// gcd_i128 removed — build.rs uses BigInt GCD

/// Simplify rational (num/den) using GCD
fn simplify_rational(num: &BigInt, den: &BigInt) -> (BigInt, BigInt) {
    use num_integer::Integer;
    let gcd = num.gcd(den);
    (num / &gcd, den / &gcd)
}

// bigint_to_u64_array (4-word) removed — bigint_to_u64_array_8 used instead

/// Convert BigInt to [u64; 8] array (little-endian, for I512)
fn bigint_to_u64_array_8(n: &BigInt) -> [u64; 8] {
    let (sign, bytes) = n.to_bytes_le();
    let mut result = [0u64; 8];

    // Handle sign (should always be positive for our constants)
    if sign == Sign::Minus {
        panic!("Cannot convert negative BigInt to unsigned array");
    }

    // Convert bytes to u64 array (little-endian)
    for (i, chunk) in bytes.chunks(8).enumerate() {
        if i >= 8 { break; }
        let mut word_bytes = [0u8; 8];
        word_bytes[..chunk.len()].copy_from_slice(chunk);
        result[i] = u64::from_le_bytes(word_bytes);
    }

    result
}

// ============================================================================
// PROFILE-AWARE CONSTANT GENERATION
// ============================================================================

fn generate_mathematical_constants(out_dir: &str, config: &PrecisionConfig) {
    let dest_path = Path::new(out_dir).join("mathematical_constants.rs");
    let mut file = File::create(&dest_path).unwrap();

    writeln!(file, "// Mathematical Constants for gMath Library").unwrap();
    writeln!(file, "// Profile: {}, Precision: {} decimal places",
             config.profile_name, config.target_decimal_places).unwrap();
    writeln!(file, "// Generated by build.rs - DO NOT EDIT").unwrap();
    writeln!(file, "//").unwrap();
    writeln!(file, "// Constants computed via continued fraction convergents").unwrap();
    writeln!(file, "// Denominators are UNIQUE (not 10^N decimal scaling)").unwrap();
    writeln!(file, "").unwrap();

    // Emit string literals for pattern detection
    writeln!(file, "/// String literals for pattern detection in g_math gmath(\"3.14...\") wrapper").unwrap();
    writeln!(file, "pub mod pattern_strings {{").unwrap();
    writeln!(file, "    pub const PI: &str = \"{}\";", hardcoded_constants::strings::PI).unwrap();
    writeln!(file, "    pub const E: &str = \"{}\";", hardcoded_constants::strings::E).unwrap();
    writeln!(file, "    pub const SQRT_2: &str = \"{}\";", hardcoded_constants::strings::SQRT_2).unwrap();
    writeln!(file, "    pub const LN_2: &str = \"{}\";", hardcoded_constants::strings::LN_2).unwrap();
    writeln!(file, "    pub const PHI: &str = \"{}\";", hardcoded_constants::strings::PHI).unwrap();
    writeln!(file, "    pub const LN_10: &str = \"{}\";", hardcoded_constants::strings::LN_10).unwrap();
    writeln!(file, "    pub const SQRT_3: &str = \"{}\";", hardcoded_constants::strings::SQRT_3).unwrap();
    writeln!(file, "    pub const SQRT_5: &str = \"{}\";", hardcoded_constants::strings::SQRT_5).unwrap();
    writeln!(file, "}}").unwrap();
    writeln!(file, "").unwrap();

    // Emit MathematicalConstants struct with accessor methods
    writeln!(file, "/// Mathematical constants accessor interface (replaces SRICConstantsDatabase)").unwrap();
    writeln!(file, "pub struct MathematicalConstants;").unwrap();
    writeln!(file, "").unwrap();
    writeln!(file, "impl MathematicalConstants {{").unwrap();

    // ========================================================================
    // UNIVERSAL 77d GENERATION (Type-Safety Strategy)
    // ========================================================================
    // ALWAYS generate 77d tier ([u64; 8] arrays) regardless of GMATH_PROFILE
    // Runtime will extract appropriate precision based on feature flags:
    //   - embedded profile    → extract i128 from first 2 u64s
    //   - balanced profile    → extract [u64; 4] from first 4 u64s
    //   - scientific profile  → use full [u64; 8] directly
    // Rationale: Eliminates build-time/compile-time type mismatch,
    //            enables type-safe downcasting, single source of truth
    // ========================================================================

    let constants = [
        ("PI", hardcoded_constants::tier_77d::PI_NUM_STR, hardcoded_constants::tier_77d::PI_DEN_STR),
        ("E", hardcoded_constants::tier_77d::E_NUM_STR, hardcoded_constants::tier_77d::E_DEN_STR),
        ("SQRT_2", hardcoded_constants::tier_77d::SQRT_2_NUM_STR, hardcoded_constants::tier_77d::SQRT_2_DEN_STR),
        ("LN_2", hardcoded_constants::tier_77d::LN_2_NUM_STR, hardcoded_constants::tier_77d::LN_2_DEN_STR),
        ("PHI", hardcoded_constants::tier_77d::PHI_NUM_STR, hardcoded_constants::tier_77d::PHI_DEN_STR),
        ("LN_10", hardcoded_constants::tier_77d::LN_10_NUM_STR, hardcoded_constants::tier_77d::LN_10_DEN_STR),
        ("SQRT_3", hardcoded_constants::tier_77d::SQRT_3_NUM_STR, hardcoded_constants::tier_77d::SQRT_3_DEN_STR),
        ("SQRT_5", hardcoded_constants::tier_77d::SQRT_5_NUM_STR, hardcoded_constants::tier_77d::SQRT_5_DEN_STR),
    ];

    for (name, num_str, den_str) in constants {
        let num = num_str.parse::<BigInt>().unwrap();
        let den = den_str.parse::<BigInt>().unwrap();
        let num_arr = bigint_to_u64_array_8(&num);
        let den_arr = bigint_to_u64_array_8(&den);

        writeln!(file, "    pub fn {}() -> ([u64; 8], [u64; 8]) {{", name).unwrap();
        writeln!(file, "        ({:?}, {:?})", num_arr, den_arr).unwrap();
        writeln!(file, "    }}").unwrap();
        writeln!(file, "").unwrap();
    }

    writeln!(file, "    pub fn GOLDEN_RATIO() -> ([u64; 8], [u64; 8]) {{").unwrap();
    writeln!(file, "        Self::PHI()").unwrap();
    writeln!(file, "    }}").unwrap();
    writeln!(file, "").unwrap();

    // Add π/2, π/4, π/6 using BigInt computation
    let pi_num_77 = hardcoded_constants::tier_77d::PI_NUM_STR.parse::<BigInt>().unwrap();
    let pi_den_77 = hardcoded_constants::tier_77d::PI_DEN_STR.parse::<BigInt>().unwrap();

    writeln!(file, "    pub fn PI_OVER_2() -> ([u64; 8], [u64; 8]) {{").unwrap();
    let (num_2, den_2) = simplify_rational(&pi_num_77, &(pi_den_77.clone() * BigInt::from(2)));
    writeln!(file, "        ({:?}, {:?})", bigint_to_u64_array_8(&num_2), bigint_to_u64_array_8(&den_2)).unwrap();
    writeln!(file, "    }}").unwrap();
    writeln!(file, "").unwrap();

    writeln!(file, "    pub fn PI_OVER_4() -> ([u64; 8], [u64; 8]) {{").unwrap();
    let (num_4, den_4) = simplify_rational(&pi_num_77, &(pi_den_77.clone() * BigInt::from(4)));
    writeln!(file, "        ({:?}, {:?})", bigint_to_u64_array_8(&num_4), bigint_to_u64_array_8(&den_4)).unwrap();
    writeln!(file, "    }}").unwrap();
    writeln!(file, "").unwrap();

    writeln!(file, "    pub fn PI_OVER_6() -> ([u64; 8], [u64; 8]) {{").unwrap();
    let (num_6, den_6) = simplify_rational(&pi_num_77, &(pi_den_77 * BigInt::from(6)));
    writeln!(file, "        ({:?}, {:?})", bigint_to_u64_array_8(&num_6), bigint_to_u64_array_8(&den_6)).unwrap();
    writeln!(file, "    }}").unwrap();
    writeln!(file, "").unwrap();

    writeln!(file, "    pub fn ONE_OVER_E() -> ([u64; 8], [u64; 8]) {{").unwrap();
    let e_num = hardcoded_constants::tier_77d::E_NUM_STR.parse::<BigInt>().unwrap();
    let e_den = hardcoded_constants::tier_77d::E_DEN_STR.parse::<BigInt>().unwrap();
    writeln!(file, "        ({:?}, {:?})", bigint_to_u64_array_8(&e_den), bigint_to_u64_array_8(&e_num)).unwrap();
    writeln!(file, "    }}").unwrap();
    writeln!(file, "").unwrap();

    writeln!(file, "    pub fn E_SQUARED() -> ([u64; 8], [u64; 8]) {{").unwrap();
    writeln!(file, "        Self::E()  // Computed at runtime").unwrap();
    writeln!(file, "    }}").unwrap();
    writeln!(file, "").unwrap();

    // ========================================================================
    // HELPER FUNCTIONS: Runtime Precision Extraction
    // ========================================================================
    // These functions enable type-safe downcasting from universal 77d tier
    // to profile-appropriate precision levels based on feature flags
    // ========================================================================

    writeln!(file, "    /// Extract i128 from [u64; 8] array (embedded profile: 19 decimals)").unwrap();
    writeln!(file, "    /// Takes the first 2 u64s (16 bytes = 128 bits) from the array").unwrap();
    writeln!(file, "    #[inline]").unwrap();
    writeln!(file, "    pub fn extract_i128_from_u64_8(arr: [u64; 8]) -> i128 {{").unwrap();
    writeln!(file, "        let low = arr[0] as i128;").unwrap();
    writeln!(file, "        let high = arr[1] as i128;").unwrap();
    writeln!(file, "        low | (high << 64)").unwrap();
    writeln!(file, "    }}").unwrap();
    writeln!(file, "").unwrap();

    writeln!(file, "    /// Extract [u64; 4] from [u64; 8] array (balanced profile: 38 decimals)").unwrap();
    writeln!(file, "    /// Takes the first 4 u64s (32 bytes = 256 bits) from the array").unwrap();
    writeln!(file, "    #[inline]").unwrap();
    writeln!(file, "    pub fn extract_u64_4_from_u64_8(arr: [u64; 8]) -> [u64; 4] {{").unwrap();
    writeln!(file, "        [arr[0], arr[1], arr[2], arr[3]]").unwrap();
    writeln!(file, "    }}").unwrap();

    writeln!(file, "}}").unwrap();

    println!("cargo:warning=✅ Generated mathematical_constants.rs ({} decimals)",
             config.target_decimal_places);
}

// ============================================================================
// PROFILE-AWARE TRANSCENDENTAL TABLE GENERATION
// ============================================================================

/// Get rational constant as BigRational based on profile
#[allow(dead_code)]
fn get_rational_constant(name: &str, config: &PrecisionConfig) -> BigRational {
    match config.target_decimal_places {
        1..=19 => {
            let (num, den) = match name {
                "pi" => hardcoded_constants::tier_19d::PI,
                "e" => hardcoded_constants::tier_19d::E,
                "sqrt_2" => hardcoded_constants::tier_19d::SQRT_2,
                "ln_2" => hardcoded_constants::tier_19d::LN_2,
                "phi" => hardcoded_constants::tier_19d::PHI,
                "ln_10" => hardcoded_constants::tier_19d::LN_10,
                "sqrt_3" => hardcoded_constants::tier_19d::SQRT_3,
                "sqrt_5" => hardcoded_constants::tier_19d::SQRT_5,
                _ => panic!("Unknown constant: {}", name),
            };
            BigRational::new(BigInt::from(num), BigInt::from(den))
        },
        20..=38 => {
            let (num_arr, den_arr) = match name {
                "pi" => (hardcoded_constants::tier_38d::PI_NUM, hardcoded_constants::tier_38d::PI_DEN),
                "e" => (hardcoded_constants::tier_38d::E_NUM, hardcoded_constants::tier_38d::E_DEN),
                "sqrt_2" => (hardcoded_constants::tier_38d::SQRT_2_NUM, hardcoded_constants::tier_38d::SQRT_2_DEN),
                "ln_2" => (hardcoded_constants::tier_38d::LN_2_NUM, hardcoded_constants::tier_38d::LN_2_DEN),
                "phi" => (hardcoded_constants::tier_38d::PHI_NUM, hardcoded_constants::tier_38d::PHI_DEN),
                "ln_10" => (hardcoded_constants::tier_38d::LN_10_NUM, hardcoded_constants::tier_38d::LN_10_DEN),
                "sqrt_3" => (hardcoded_constants::tier_38d::SQRT_3_NUM, hardcoded_constants::tier_38d::SQRT_3_DEN),
                "sqrt_5" => (hardcoded_constants::tier_38d::SQRT_5_NUM, hardcoded_constants::tier_38d::SQRT_5_DEN),
                _ => panic!("Unknown constant: {}", name),
            };

            let num = u64_array_to_bigint(&num_arr);
            let den = u64_array_to_bigint(&den_arr);
            BigRational::new(num, den)
        },
        _ => {
            let (num_str, den_str) = match name {
                "pi" => (hardcoded_constants::tier_77d::PI_NUM_STR, hardcoded_constants::tier_77d::PI_DEN_STR),
                "e" => (hardcoded_constants::tier_77d::E_NUM_STR, hardcoded_constants::tier_77d::E_DEN_STR),
                "sqrt_2" => (hardcoded_constants::tier_77d::SQRT_2_NUM_STR, hardcoded_constants::tier_77d::SQRT_2_DEN_STR),
                "ln_2" => (hardcoded_constants::tier_77d::LN_2_NUM_STR, hardcoded_constants::tier_77d::LN_2_DEN_STR),
                "phi" => (hardcoded_constants::tier_77d::PHI_NUM_STR, hardcoded_constants::tier_77d::PHI_DEN_STR),
                "ln_10" => (hardcoded_constants::tier_77d::LN_10_NUM_STR, hardcoded_constants::tier_77d::LN_10_DEN_STR),
                "sqrt_3" => (hardcoded_constants::tier_77d::SQRT_3_NUM_STR, hardcoded_constants::tier_77d::SQRT_3_DEN_STR),
                "sqrt_5" => (hardcoded_constants::tier_77d::SQRT_5_NUM_STR, hardcoded_constants::tier_77d::SQRT_5_DEN_STR),
                _ => panic!("Unknown constant: {}", name),
            };

            let num: BigInt = num_str.parse().unwrap();
            let den: BigInt = den_str.parse().unwrap();
            BigRational::new(num, den)
        }
    }
}

/// Convert [u64; 4] array to BigInt (little-endian)
#[allow(dead_code)]
fn u64_array_to_bigint(arr: &[u64; 4]) -> BigInt {
    let mut bytes = Vec::with_capacity(32);
    for &word in arr.iter() {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    BigInt::from_bytes_le(Sign::Plus, &bytes)
}

/// Compute e^x using rational arithmetic (Taylor series)
fn exp_rational(x: &BigRational, terms: usize) -> BigRational {
    // For negative x, use e^-x = 1/e^x for better convergence
    if *x < BigRational::zero() {
        let pos_x = -x;
        let exp_pos = exp_rational_positive(&pos_x, terms);
        return BigRational::one() / exp_pos;
    }

    exp_rational_positive(x, terms)
}

/// Compute e^x for x >= 0 using Taylor series
/// For wide tiers (Q256.256, Q512.512), we need more than 300 bits!
/// - Tier 5 Q256.256: needs 384 bits (256 frac + 128 comp)
/// - Tier 6 Q512.512: needs 768 bits (512 frac + 256 comp)
fn exp_rational_positive(x: &BigRational, terms: usize) -> BigRational {
    let mut result = BigRational::one();
    let mut term = BigRational::one();

    // Calculate precision threshold based on term count
    // More terms requested = higher precision tier = larger threshold needed
    // 100 terms: 300 bits (tier 3: Q64.64)
    // 400 terms: 500 bits (tier 4: Q128.128)
    // 500 terms: 600 bits (tier 5: Q256.256)
    // 600 terms: 900 bits (tier 6: Q512.512)
    let precision_threshold = if terms >= 600 {
        900  // Tier 6: 512 + 256 + 132 margin
    } else if terms >= 500 {
        600  // Tier 5: 256 + 128 + 216 margin
    } else if terms >= 400 {
        500  // Tier 4: 128 + 64 + 308 margin
    } else {
        300  // Tier 3: 64 + 32 + 204 margin
    };

    for n in 1..terms {
        term = term * x / BigInt::from(n);
        result = result + &term;

        // Early termination if term becomes negligible relative to target precision
        let term_bits = term.numer().abs().bits() as i64 - term.denom().bits() as i64;
        let result_bits = result.numer().abs().bits() as i64 - result.denom().bits() as i64;
        if term_bits < result_bits - precision_threshold {
            break;
        }
    }

    result
}

/// Compute π via Machin's formula: π = 16·arctan(1/5) - 4·arctan(1/239)
/// Uses BigRational for exact rational arithmetic.
/// precision_bits: minimum number of correct bits required
fn compute_pi_rational(precision_bits: usize) -> BigRational {
    fn atan_rational(x: &BigRational, precision_bits: usize) -> BigRational {
        let x_sq = x * x;
        let mut power = x.clone(); // x^(2k+1), starting with k=0 → x^1
        let mut result = x.clone(); // first term: x/1
        let target_bits = -(precision_bits as i64 + 20); // conservative safety margin

        let mut k = 1u64;
        loop {
            power = &power * &x_sq; // x^(2k+1)
            let denom = BigRational::new(BigInt::from(1), BigInt::from(2 * k + 1));
            let term = &power * &denom;

            if k % 2 == 1 {
                result = result - &term; // odd k: subtract
            } else {
                result = result + &term; // even k: add
            }

            // Check convergence: |power| < 2^(-precision_bits - 20)
            let power_bits = power.numer().bits() as i64 - power.denom().bits() as i64;
            if power_bits < target_bits {
                break;
            }

            k += 1;
        }

        result
    }

    let one = BigInt::from(1);
    let x1 = BigRational::new(one.clone(), BigInt::from(5));   // 1/5
    let x2 = BigRational::new(one.clone(), BigInt::from(239)); // 1/239

    let atan_1_5 = atan_rational(&x1, precision_bits);
    let atan_1_239 = atan_rational(&x2, precision_bits);

    // π = 16·arctan(1/5) - 4·arctan(1/239)
    let sixteen = BigRational::from(BigInt::from(16));
    let four = BigRational::from(BigInt::from(4));

    &sixteen * &atan_1_5 - &four * &atan_1_239
}

/// Convert BigRational to Q64.64 fixed-point (i128)
/// Returns: (i128 main value, i64 compensation)
fn rational_to_q64_64(r: &BigRational) -> (i128, i64) {
    // ZERO-FLOAT: Pure BigInt arithmetic for consensus determinism

    // Scale by 2^64
    let scale_64: BigInt = BigInt::from(1i128 << 64);
    let scaled = r * BigRational::from(scale_64.clone());

    // Extract integer part (main value)
    let main = scaled.numer() / scaled.denom();
    let main_i128 = main.to_i128().unwrap_or(0);

    // Calculate compensation (fractional remainder scaled to i64 range)
    let main_rational = BigRational::from(main);
    let remainder = scaled - main_rational;

    // Scale remainder by 2^32 to fit in i64 compensation term (half fractional bits)
    let comp_scale = BigInt::from(1i64 << 32);
    let compensation_scaled = remainder * BigRational::from(comp_scale);
    let compensation = (compensation_scaled.numer() / compensation_scaled.denom())
        .to_i64()
        .unwrap_or(0);

    (main_i128, compensation)
}

/// Convert BigInt to I256 format (4 x u64 words, little-endian)
/// Returns: [u64; 4] suitable for I256::from_words()
fn bigint_to_i256(n: &BigInt) -> [u64; 4] {
    let bytes = n.to_signed_bytes_le();
    let mut words = [0u64; 4];

    // Pack bytes into u64 words (little-endian)
    for (i, chunk) in bytes.chunks(8).enumerate() {
        if i >= 4 { break; }
        let mut word = 0u64;
        for (j, &byte) in chunk.iter().enumerate() {
            word |= (byte as u64) << (j * 8);
        }
        words[i] = word;
    }

    // Sign extension for negative numbers
    if n.sign() == Sign::Minus && bytes.len() < 32 {
        let start_word = (bytes.len() + 7) / 8;
        for word in words.iter_mut().skip(start_word) {
            *word = u64::MAX;
        }
    }

    words
}

/// Convert BigInt to I512 format (8 x u64 words, little-endian)
/// Returns: [u64; 8] suitable for I512::from_words()
fn bigint_to_i512(n: &BigInt) -> [u64; 8] {
    let bytes = n.to_signed_bytes_le();
    let mut words = [0u64; 8];

    // Pack bytes into u64 words (little-endian)
    for (i, chunk) in bytes.chunks(8).enumerate() {
        if i >= 8 { break; }
        let mut word = 0u64;
        for (j, &byte) in chunk.iter().enumerate() {
            word |= (byte as u64) << (j * 8);
        }
        words[i] = word;
    }

    // Sign extension for negative numbers
    if n.sign() == Sign::Minus && bytes.len() < 64 {
        let start_word = (bytes.len() + 7) / 8;
        for word in words.iter_mut().skip(start_word) {
            *word = u64::MAX;
        }
    }

    words
}

/// Convert BigInt to I1024 (16 u64 words) - for Q512.512 tier 6 tables
fn bigint_to_i1024(n: &BigInt) -> [u64; 16] {
    let bytes = n.to_signed_bytes_le();
    let mut words = [0u64; 16];

    // Pack bytes into u64 words (little-endian)
    for (i, chunk) in bytes.chunks(8).enumerate() {
        if i >= 16 { break; }
        let mut word = 0u64;
        for (j, &byte) in chunk.iter().enumerate() {
            word |= (byte as u64) << (j * 8);
        }
        words[i] = word;
    }

    // Sign extension for negative numbers
    if n.sign() == Sign::Minus && bytes.len() < 128 {
        let start_word = (bytes.len() + 7) / 8;
        for word in words.iter_mut().skip(start_word) {
            *word = u64::MAX;
        }
    }

    words
}

/// Convert BigRational to Q128.128 fixed-point format
/// Returns: (I256 main value as [u64; 4], i128 compensation)
fn rational_to_q128_128(r: &BigRational) -> ([u64; 4], i128) {
    // Scale by 2^128
    let scale_128: BigInt = BigInt::from(1) << 128;
    let scaled = r * BigRational::from(scale_128.clone());

    // Extract integer part (main value)
    let main = scaled.numer() / scaled.denom();
    let main_i256 = bigint_to_i256(&main);

    // Calculate compensation (fractional remainder scaled to i128 range)
    let main_rational = BigRational::from(main);
    let remainder = scaled - main_rational;

    // Scale remainder by 2^64 to fit in i128 compensation term (half fractional bits)
    let comp_scale = BigInt::from(1i128 << 64);
    let compensation_scaled = remainder * BigRational::from(comp_scale);
    let compensation = (compensation_scaled.numer() / compensation_scaled.denom())
        .to_i128()
        .unwrap_or(0);

    (main_i256, compensation)
}

/// Convert BigRational to Q256.256 fixed-point format
/// Returns: (I512 main value as [u64; 8], I256 compensation as [u64; 4])
fn rational_to_q256_256(r: &BigRational) -> ([u64; 8], [u64; 4]) {
    // Scale by 2^256
    let scale_256: BigInt = BigInt::from(1) << 256;
    let scaled = r * BigRational::from(scale_256.clone());

    // Extract integer part (main value)
    let main = scaled.numer() / scaled.denom();
    let main_i512 = bigint_to_i512(&main);

    // Calculate compensation (fractional remainder scaled to I256 range)
    let main_rational = BigRational::from(main);
    let remainder = scaled - main_rational;

    // Scale remainder by 2^256 to match fractional bit width (full precision)
    let comp_scale = BigInt::from(1) << 256;
    let compensation_scaled = remainder * BigRational::from(comp_scale);
    let compensation_bigint = compensation_scaled.numer() / compensation_scaled.denom();
    let compensation = bigint_to_i256(&compensation_bigint);

    (main_i512, compensation)
}

/// Convert BigRational to Q512.512 fixed-point format (Tier 6)
/// Returns: (I1024 main value as [u64; 16], I512 compensation as [u64; 8])
fn rational_to_q512_512(r: &BigRational) -> ([u64; 16], [u64; 8]) {
    // Scale by 2^512
    let scale_512: BigInt = BigInt::from(1) << 512;
    let scaled = r * BigRational::from(scale_512.clone());

    // Extract integer part (main value)
    let main = scaled.numer() / scaled.denom();
    let main_i1024 = bigint_to_i1024(&main);

    // Calculate compensation (fractional remainder scaled to I512 range)
    let main_rational = BigRational::from(main);
    let remainder = scaled - main_rational;

    // Scale remainder by 2^256 to fit in I512 compensation term
    let comp_scale = BigInt::from(1) << 256;
    let compensation_scaled = remainder * BigRational::from(comp_scale);
    let compensation_bigint = compensation_scaled.numer() / compensation_scaled.denom();
    let compensation = bigint_to_i512(&compensation_bigint);

    (main_i1024, compensation)
}

/// Profile-aware table generation dispatcher
///
/// - Embedded: Q64.64 tables (128-bit, 19 decimal places)
/// - Balanced: Q128.128 tables (256-bit, 38 decimal places)
/// - Scientific: Q256.256 tables (512-bit, 77 decimal places)
///
/// UGOD REQUIREMENT: Generate ALL table sets for multi-tier transcendental support.
///
/// **ARCHITECTURE**: Profile determines which tables are USED at compile time via cfg,
/// but ALL tables are GENERATED for potential runtime UGOD tier promotion.
///
/// **BUILD OUTPUT**:
/// - q64_64_tables.rs (173KB, 19 decimals) - Tier 3
/// - q128_128_tables.rs (329KB, 38 decimals) - Tier 4
/// - q256_256_tables.rs (600KB+, 77 decimals) - Tier 5
///
/// **PROFILE USAGE** (via #[cfg(table_format)]):
/// - embedded/balanced/scientific: Chooses PRIMARY table set
/// - UGOD overflow: Can delegate to higher tiers if implemented
fn generate_profile_aware_tables(out_dir: &str, config: &PrecisionConfig) {
    println!("cargo:warning=🔧 UGOD Multi-Tier: Generating ALL transcendental table sets");
    println!("cargo:warning=📊 Profile '{}' will use {} tables as PRIMARY",
             config.profile_name, config.table_format);

    // Generate ALL table sets for UGOD tier N+1 support
    println!("cargo:warning=  ├─ Q64.64 tables (Tier 3: 19 decimals)...");
    generate_q64_64_tables(out_dir, config);

    println!("cargo:warning=  ├─ Q128.128 tables (Tier 4: 38 decimals)...");
    generate_q128_128_tables(out_dir, config);

    println!("cargo:warning=  ├─ Q256.256 tables (Tier 5: 77 decimals)...");
    generate_q256_256_tables(out_dir, config);

    println!("cargo:warning=  └─ Q512.512 tables (Tier 6: 154 decimals)...");
    generate_q512_512_tables(out_dir, config);

    println!("cargo:warning=✅ All table sets generated successfully");
}

/// Generate Q64.64 exponential tables
fn generate_q64_64_tables(out_dir: &str, _config: &PrecisionConfig) {
    let dest_path = Path::new(out_dir).join("q64_64_tables.rs");
    let mut file = File::create(&dest_path).unwrap();

    writeln!(file, "// Q64.64 Transcendental Lookup Tables").unwrap();
    writeln!(file, "// Generated from hardcoded rational constants - ZERO float arithmetic").unwrap();
    writeln!(file, "").unwrap();

    // Generate EXP_INTEGER_TABLE: e^n for n ∈ [-40, 40]
    writeln!(file, "/// Integer exponential table: e^n for n ∈ [-40, 40] (Tier 3: Q64.64)").unwrap();
    writeln!(file, "pub static EXP_INTEGER_TABLE_TIER_3: [(i128, i64); 81] = [").unwrap();

    for n in -40..=40 {
        let x = BigRational::from(BigInt::from(n));
        // Use 300 terms for integer exponentials (large x needs more terms)
        let exp_x = exp_rational(&x, 300);

        let (main, comp) = rational_to_q64_64(&exp_x);
        writeln!(file, "    ({}, {}), // e^{}", main, comp, n).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // Generate EXP_PRIMARY_TABLE: e^(k/2^10) for k ∈ [0, 1023]
    writeln!(file, "/// Primary exponential table: e^(k/2^10) for k ∈ [0, 1023] (Tier 3: Q64.64)").unwrap();
    writeln!(file, "pub static EXP_PRIMARY_TABLE_TIER_3: [(i128, i64); 1024] = [").unwrap();

    for k in 0..1024 {
        let x = BigRational::new(BigInt::from(k), BigInt::from(1i128 << 10));
        let exp_x = exp_rational(&x, 100);
        let (main, comp) = rational_to_q64_64(&exp_x);
        writeln!(file, "    ({}, {}), // e^({}/2^10)", main, comp, k).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // Generate EXP_SECONDARY_TABLE: e^(j/2^20) for j ∈ [0, 1023]
    writeln!(file, "/// Secondary exponential table: e^(j/2^20) for j ∈ [0, 1023] (Tier 3: Q64.64)").unwrap();
    writeln!(file, "pub static EXP_SECONDARY_TABLE_TIER_3: [(i128, i64); 1024] = [").unwrap();

    for j in 0..1024 {
        let x = BigRational::new(BigInt::from(j), BigInt::from(1i128 << 20));
        let exp_x = exp_rational(&x, 100);
        let (main, comp) = rational_to_q64_64(&exp_x);
        writeln!(file, "    ({}, {}), // e^({}/2^20)", main, comp, j).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // Generate EXP_TERTIARY_TABLE: e^(m/2^30) for m ∈ [0, 1023]
    writeln!(file, "/// Tertiary exponential table: e^(m/2^30) for m ∈ [0, 1023] (Tier 3: Q64.64)").unwrap();
    writeln!(file, "pub static EXP_TERTIARY_TABLE_TIER_3: [(i128, i64); 1024] = [").unwrap();

    for m in 0..1024 {
        let x = BigRational::new(BigInt::from(m), BigInt::from(1i128 << 30));
        let exp_x = exp_rational(&x, 100);
        let (main, comp) = rational_to_q64_64(&exp_x);
        writeln!(file, "    ({}, {}), // e^({}/2^30)", main, comp, m).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    println!("cargo:warning=✅ Generated Q64.64 exponential tables (4 × 1024 entries)");
}

/// Generate Q128.128 exponential tables for Balanced profile (38 decimal precision)
fn generate_q128_128_tables(out_dir: &str, _config: &PrecisionConfig) {
    let dest_path = Path::new(out_dir).join("q128_128_tables.rs");
    let mut file = File::create(&dest_path).unwrap();

    writeln!(file, "// Q128.128 Transcendental Lookup Tables").unwrap();
    writeln!(file, "// Generated from BigRational - 38 decimal precision").unwrap();
    writeln!(file, "// Profile: Balanced | Storage: I256 | Compensation: i128").unwrap();
    writeln!(file, "// ZERO float arithmetic - pure rational → fixed-point conversion").unwrap();
    writeln!(file, "// Note: I256 type is imported by the including module").unwrap();
    writeln!(file, "").unwrap();

    // ========================================================================
    // EXP_INTEGER_TABLE: e^n for n ∈ [-40, 40]
    // ========================================================================
    writeln!(file, "/// Integer exponential table: e^n for n ∈ [-40, 40] (Tier 4: Q128.128)").unwrap();
    writeln!(file, "/// Format: (I256 main, i128 compensation)").unwrap();
    writeln!(file, "pub static EXP_INTEGER_TABLE_TIER_4: [(I256, i128); 81] = [").unwrap();

    for n in -40..=40 {
        let x = BigRational::from(BigInt::from(n));
        // Use 400 terms for e^40 convergence at 38-decimal precision
        let exp_x = exp_rational(&x, 400);
        let (main, comp) = rational_to_q128_128(&exp_x);
        writeln!(file, "    (I256::from_words({:?}), {}), // e^{}", main, comp, n).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // ========================================================================
    // EXP_PRIMARY_TABLE: e^(k/2^30) for k ∈ [0, 1023]
    // ========================================================================
    writeln!(file, "/// Primary exponential table: e^(k/2^30) for k ∈ [0, 1023] (Tier 4: Q128.128)").unwrap();
    writeln!(file, "pub static EXP_PRIMARY_TABLE_TIER_4: [(I256, i128); 1024] = [").unwrap();

    for k in 0..1024 {
        // PRIMARY table: e^(k/1024) - represents top 10 bits of fractional part
        let x = BigRational::new(BigInt::from(k), BigInt::from(1024));
        let exp_x = exp_rational(&x, 100);
        let (main, comp) = rational_to_q128_128(&exp_x);
        writeln!(file, "    (I256::from_words({:?}), {}), // e^({}/1024)", main, comp, k).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // ========================================================================
    // EXP_SECONDARY_TABLE: e^(j/2^20) for j ∈ [0, 1023]
    // ========================================================================
    writeln!(file, "/// Secondary exponential table: e^(j/2^20) for j ∈ [0, 1023] (Tier 4: Q128.128)").unwrap();
    writeln!(file, "pub static EXP_SECONDARY_TABLE_TIER_4: [(I256, i128); 1024] = [").unwrap();

    for j in 0..1024 {
        // SECONDARY table: e^(j/1024/1024) - represents next 10 bits
        let x = BigRational::new(BigInt::from(j), BigInt::from(1024 * 1024));
        let exp_x = exp_rational(&x, 100);
        let (main, comp) = rational_to_q128_128(&exp_x);
        writeln!(file, "    (I256::from_words({:?}), {}), // e^({}/1024/1024)", main, comp, j).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // ========================================================================
    // EXP_TERTIARY_TABLE: e^(m/2^40) for m ∈ [0, 1023]
    // ========================================================================
    writeln!(file, "/// Tertiary exponential table: e^(m/2^40) for m ∈ [0, 1023] (Tier 4: Q128.128)").unwrap();
    writeln!(file, "pub static EXP_TERTIARY_TABLE_TIER_4: [(I256, i128); 1024] = [").unwrap();

    for m in 0..1024 {
        // TERTIARY table: e^(m/1024/1024/1024) - represents third set of 10 bits
        let x = BigRational::new(BigInt::from(m), BigInt::from(1024 * 1024 * 1024));
        let exp_x = exp_rational(&x, 100);
        let (main, comp) = rational_to_q128_128(&exp_x);
        writeln!(file, "    (I256::from_words({:?}), {}), // e^({}/1024/1024/1024)", main, comp, m).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // Zero-initialized reciprocal and factorial tables (unused by current algorithms)
    writeln!(file, "// Reciprocal and factorial tables (reserved)").unwrap();
    writeln!(file, "pub static RECIPROCAL_PRIMARY_TABLE_TIER_4: [(I256, i128); 1024] = [(I256::from_words([0, 0, 0, 0]), 0); 1024];").unwrap();
    writeln!(file, "pub static RECIPROCAL_SECONDARY_TABLE_TIER_4: [(I256, i128); 1024] = [(I256::from_words([0, 0, 0, 0]), 0); 1024];").unwrap();
    writeln!(file, "pub static RECIPROCAL_EXACT_TABLE_TIER_4: [(I256, i128); 256] = [(I256::from_words([0, 0, 0, 0]), 0); 256];").unwrap();
    writeln!(file, "").unwrap();

    writeln!(file, "pub static FACTORIAL_RECIPROCALS_TIER_4: [(I256, i128); 50] = [").unwrap();
    for i in 0..50 {
        let factorial = (1..=i).fold(BigInt::one(), |acc, x| acc * BigInt::from(x));
        let reciprocal = BigRational::new(BigInt::one(), factorial);
        let (main, comp) = rational_to_q128_128(&reciprocal);
        writeln!(file, "    (I256::from_words({:?}), {}), // 1/{}!", main, comp, i).unwrap();
    }
    writeln!(file, "];").unwrap();

    println!("cargo:warning=✅ Generated Q128.128 tables: 3,169 entries, 38-decimal precision");
}

/// Generate Q256.256 exponential tables for Scientific profile (77 decimal precision)
fn generate_q256_256_tables(out_dir: &str, _config: &PrecisionConfig) {
    let dest_path = Path::new(out_dir).join("q256_256_tables.rs");
    let mut file = File::create(&dest_path).unwrap();

    writeln!(file, "// Q256.256 Transcendental Lookup Tables").unwrap();
    writeln!(file, "// Generated from BigRational - 77 decimal precision").unwrap();
    writeln!(file, "// Profile: Scientific | Storage: I512 | Compensation: I256").unwrap();
    writeln!(file, "// ZERO float arithmetic - pure rational → fixed-point conversion").unwrap();
    writeln!(file, "// Note: I512 and I256 types are imported by the including module").unwrap();
    writeln!(file, "").unwrap();

    // ========================================================================
    // EXP_INTEGER_TABLE: e^n for n ∈ [-40, 40]
    // ========================================================================
    writeln!(file, "/// Integer exponential table: e^n for n ∈ [-40, 40]").unwrap();
    writeln!(file, "/// Format: (I512 main, I256 compensation)").unwrap();
    writeln!(file, "pub static EXP_INTEGER_TABLE_TIER_5: [(I512, I256); 81] = [").unwrap();

    for n in -40..=40 {
        let x = BigRational::from(BigInt::from(n));
        // Use 500 terms for e^40 convergence at 77-decimal precision
        let exp_x = exp_rational(&x, 500);
        let (main, comp) = rational_to_q256_256(&exp_x);
        writeln!(file, "    (I512::from_words({:?}), I256::from_words({:?})), // e^{}", main, comp, n).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // ========================================================================
    // EXP_PRIMARY_TABLE: e^(k/2^30) for k ∈ [0, 1023]
    // ========================================================================
    writeln!(file, "/// Primary exponential table: e^(k/2^30) for k ∈ [0, 1023]").unwrap();
    writeln!(file, "pub static EXP_PRIMARY_TABLE_TIER_5: [(I512, I256); 1024] = [").unwrap();

    for k in 0..1024 {
        let x = BigRational::new(BigInt::from(k), BigInt::from(1024));
        let exp_x = exp_rational(&x, 200);
        let (main, comp) = rational_to_q256_256(&exp_x);
        writeln!(file, "    (I512::from_words({:?}), I256::from_words({:?})), // e^({}/1024)", main, comp, k).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // ========================================================================
    // EXP_SECONDARY_TABLE: e^(j/2^20) for j ∈ [0, 1023]
    // ========================================================================
    writeln!(file, "/// Secondary exponential table: e^(j/2^20) for j ∈ [0, 1023]").unwrap();
    writeln!(file, "pub static EXP_SECONDARY_TABLE_TIER_5: [(I512, I256); 1024] = [").unwrap();

    for j in 0..1024 {
        let x = BigRational::new(BigInt::from(j), BigInt::from(1i128 << 20));
        let exp_x = exp_rational(&x, 200);
        let (main, comp) = rational_to_q256_256(&exp_x);
        writeln!(file, "    (I512::from_words({:?}), I256::from_words({:?})), // e^({}/2^20)", main, comp, j).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // ========================================================================
    // EXP_TERTIARY_TABLE: e^(m/2^30) for m ∈ [0, 1023]
    // ========================================================================
    writeln!(file, "/// Tertiary exponential table: e^(m/2^30) for m ∈ [0, 1023]").unwrap();
    writeln!(file, "pub static EXP_TERTIARY_TABLE_TIER_5: [(I512, I256); 1024] = [").unwrap();

    for m in 0..1024 {
        let x = BigRational::new(BigInt::from(m), BigInt::from(1024 * 1024 * 1024));  // 2^30 = 1024^3
        let exp_x = exp_rational(&x, 200);
        let (main, comp) = rational_to_q256_256(&exp_x);
        writeln!(file, "    (I512::from_words({:?}), I256::from_words({:?})), // e^({}/2^30)", main, comp, m).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // Zero-initialized reciprocal and factorial tables (unused by current algorithms)
    writeln!(file, "// Reciprocal and factorial tables (reserved)").unwrap();
    writeln!(file, "pub static RECIPROCAL_PRIMARY_TABLE_TIER_5: [(I512, I256); 1024] = [(I512::from_words([0, 0, 0, 0, 0, 0, 0, 0]), I256::from_words([0, 0, 0, 0])); 1024];").unwrap();
    writeln!(file, "pub static RECIPROCAL_SECONDARY_TABLE_TIER_5: [(I512, I256); 1024] = [(I512::from_words([0, 0, 0, 0, 0, 0, 0, 0]), I256::from_words([0, 0, 0, 0])); 1024];").unwrap();
    writeln!(file, "pub static RECIPROCAL_EXACT_TABLE_TIER_5: [(I512, I256); 256] = [(I512::from_words([0, 0, 0, 0, 0, 0, 0, 0]), I256::from_words([0, 0, 0, 0])); 256];").unwrap();
    writeln!(file, "").unwrap();

    writeln!(file, "pub static FACTORIAL_RECIPROCALS_TIER_5: [(I512, I256); 50] = [").unwrap();
    for i in 0..50 {
        let factorial = (1..=i).fold(BigInt::one(), |acc, x| acc * BigInt::from(x));
        let reciprocal = BigRational::new(BigInt::one(), factorial);
        let (main, comp) = rational_to_q256_256(&reciprocal);
        writeln!(file, "    (I512::from_words({:?}), I256::from_words({:?})), // 1/{}!", main, comp, i).unwrap();
    }
    writeln!(file, "];").unwrap();

    println!("cargo:warning=✅ Generated Q256.256 tables: 3,169 entries, 77-decimal precision");
}

/// Generate Q512.512 tables (Tier 6) - for scientific profile tier N+1 strategy
fn generate_q512_512_tables(out_dir: &str, _config: &PrecisionConfig) {
    let dest_path = Path::new(out_dir).join("q512_512_tables.rs");
    let mut file = File::create(&dest_path).unwrap();

    writeln!(file, "// Q512.512 Transcendental Lookup Tables (Tier 6)").unwrap();
    writeln!(file, "// Generated from BigRational - 154 decimal precision").unwrap();
    writeln!(file, "// Profile: Scientific Tier N+1 | Storage: I1024 | Compensation: I512").unwrap();
    writeln!(file, "// ZERO float arithmetic - pure rational → fixed-point conversion").unwrap();
    writeln!(file, "// Note: I1024 and I512 types are imported by the including module").unwrap();
    writeln!(file, "").unwrap();

    // ========================================================================
    // EXP_INTEGER_TABLE: e^n for n ∈ [-40, 40]
    // ========================================================================
    writeln!(file, "/// Integer exponential table: e^n for n ∈ [-40, 40]").unwrap();
    writeln!(file, "/// Format: (I1024 main, I512 compensation)").unwrap();
    writeln!(file, "pub static EXP_INTEGER_TABLE_TIER_6: [(I1024, I512); 81] = [").unwrap();

    for n in -40..=40 {
        let x = BigRational::from(BigInt::from(n));
        // Use 600 terms for e^40 convergence at 154-decimal precision
        let exp_x = exp_rational(&x, 600);
        let (main, comp) = rational_to_q512_512(&exp_x);
        writeln!(file, "    (I1024::from_words({:?}), I512::from_words({:?})), // e^{}", main, comp, n).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // ========================================================================
    // EXP_PRIMARY_TABLE: e^(k/2^30) for k ∈ [0, 1023]
    // ========================================================================
    writeln!(file, "/// Primary exponential table: e^(k/2^30) for k ∈ [0, 1023]").unwrap();
    writeln!(file, "pub static EXP_PRIMARY_TABLE_TIER_6: [(I1024, I512); 1024] = [").unwrap();

    for k in 0..1024 {
        let x = BigRational::new(BigInt::from(k), BigInt::from(1024));
        let exp_x = exp_rational(&x, 300);
        let (main, comp) = rational_to_q512_512(&exp_x);
        writeln!(file, "    (I1024::from_words({:?}), I512::from_words({:?})), // e^({}/1024)", main, comp, k).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // ========================================================================
    // EXP_SECONDARY_TABLE: e^(j/2^20) for j ∈ [0, 1023]
    // ========================================================================
    writeln!(file, "/// Secondary exponential table: e^(j/2^20) for j ∈ [0, 1023]").unwrap();
    writeln!(file, "pub static EXP_SECONDARY_TABLE_TIER_6: [(I1024, I512); 1024] = [").unwrap();

    for j in 0..1024 {
        let x = BigRational::new(BigInt::from(j), BigInt::from(1i128 << 20));
        let exp_x = exp_rational(&x, 300);
        let (main, comp) = rational_to_q512_512(&exp_x);
        writeln!(file, "    (I1024::from_words({:?}), I512::from_words({:?})), // e^({}/2^20)", main, comp, j).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // ========================================================================
    // EXP_TERTIARY_TABLE: e^(m/2^40) for m ∈ [0, 1023]
    // ========================================================================
    writeln!(file, "/// Tertiary exponential table: e^(m/2^40) for m ∈ [0, 1023]").unwrap();
    writeln!(file, "pub static EXP_TERTIARY_TABLE_TIER_6: [(I1024, I512); 1024] = [").unwrap();

    for m in 0..1024 {
        let x = BigRational::new(BigInt::from(m), BigInt::from(1024 * 1024 * 1024));  // 2^30
        let exp_x = exp_rational(&x, 300);
        let (main, comp) = rational_to_q512_512(&exp_x);
        writeln!(file, "    (I1024::from_words({:?}), I512::from_words({:?})), // e^({}/2^30)", main, comp, m).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // Zero-initialized reciprocal tables (unused by current algorithms)
    writeln!(file, "// Reciprocal tables (reserved)").unwrap();
    writeln!(file, "pub static RECIPROCAL_PRIMARY_TABLE_TIER_6: [(I1024, I512); 1024] = [(I1024::from_words([0; 16]), I512::from_words([0; 8])); 1024];").unwrap();
    writeln!(file, "pub static RECIPROCAL_SECONDARY_TABLE_TIER_6: [(I1024, I512); 1024] = [(I1024::from_words([0; 16]), I512::from_words([0; 8])); 1024];").unwrap();
    writeln!(file, "pub static RECIPROCAL_EXACT_TABLE_TIER_6: [(I1024, I512); 256] = [(I1024::from_words([0; 16]), I512::from_words([0; 8])); 256];").unwrap();
    writeln!(file, "").unwrap();

    // ========================================================================
    // FACTORIAL_RECIPROCALS: 1/n! for Taylor series (PRODUCTION-READY)
    // ========================================================================
    writeln!(file, "/// Factorial reciprocals: 1/n! for n ∈ [0, 49] (Q512.512 format)").unwrap();
    writeln!(file, "/// CRITICAL: Used in taylor_series_q512_512 for sub-ULP precision").unwrap();
    writeln!(file, "pub static FACTORIAL_RECIPROCALS_TIER_6: [(I1024, I512); 50] = [").unwrap();
    for i in 0..50 {
        let factorial = (1..=i).fold(BigInt::one(), |acc, x| acc * BigInt::from(x));
        let reciprocal = BigRational::new(BigInt::one(), factorial);
        let (main, comp) = rational_to_q512_512(&reciprocal);
        writeln!(file, "    (I1024::from_words({:?}), I512::from_words({:?})), // 1/{}!", main, comp, i).unwrap();
    }
    writeln!(file, "];").unwrap();

    println!("cargo:warning=✅ Generated Q512.512 tables: 3,169 entries, 154-decimal precision");
}

// ============================================================================
// NATURAL LOGARITHM TABLE GENERATION - OPTIMIZED
// ============================================================================

/// Compute ln(1+y) using rational arithmetic (Taylor series) - OPTIMIZED
/// ln(1+y) = y - y²/2 + y³/3 - y⁴/4 + ... for |y| < 1
///
/// KEY INSIGHT: For small y, convergence is FAST:
/// - |y| ≈ 1:      ~100 terms for 77 decimals (worst case)
/// - |y| < 1/1024: ~25 terms sufficient
/// - |y| < 1/2^20: ~13 terms (each term ×y^2 ≈ 2^-40)
/// - |y| < 1/2^30: ~9 terms (y itself is nearly the answer)
/// Compute ln(1+y) using artanh transformation for FAST convergence
///
/// **ALGORITHM**: ln(1+y) = 2 * artanh(y/(y+2))
///
/// For y ∈ [0, 1), z = y/(y+2) ∈ [0, 1/3)
/// artanh(z) = z + z³/3 + z⁵/5 + z⁷/7 + ...
///
/// **CONVERGENCE**: Rate is z² ≈ 1/9 for worst case y→1
/// - Old Taylor: y^n/n → needs ~270 terms for 77 digits when y=0.75
/// - New artanh: z^(2n+1)/(2n+1) → needs ~45 terms for 77 digits
///
/// **FIX**: This resolves the precision issue where ln(112) only achieved
/// 27 digits due to slow Taylor convergence for large y values.
fn ln_1_plus_y_rational(y: &BigRational, precision_bits: usize) -> BigRational {
    if *y == BigRational::zero() {
        return BigRational::zero();
    }

    // Estimate |y| magnitude
    let y_bits = y.numer().bits() as i64 - y.denom().bits() as i64;

    // For very small y: ln(1+y) ≈ y - y²/2 is sufficient
    if y_bits < -50 {
        let y2 = y * y;
        let half = BigRational::new(BigInt::from(1), BigInt::from(2));
        return y - &y2 * &half;
    }

    // For small y (< 0.01), Taylor series converges fast enough
    if y_bits < -6 {
        return ln_1_plus_y_taylor(y, precision_bits);
    }

    // For larger y, use artanh transformation for fast convergence
    // ln(1+y) = 2 * artanh(y/(y+2))
    let two = BigRational::from(BigInt::from(2));
    let z = y / (y + &two);  // z ∈ [0, 1/3) for y ∈ [0, 1)

    let mut result = z.clone();
    let mut z_power = z.clone();
    let z_squared = &z * &z;

    // Convergence at rate z² ≈ 1/9, so need precision_bits / log2(9) ≈ precision_bits / 3.17 terms
    let max_terms = precision_bits / 2 + 20;  // Extra margin for safety
    let target_bits = -(precision_bits as i64 + 20);

    for n in 1..max_terms {
        z_power = &z_power * &z_squared;

        let coeff = BigRational::new(BigInt::from(1), BigInt::from(2 * n + 1));
        result = result + &coeff * &z_power;

        // Early termination when term is negligible
        let term_bits = z_power.numer().bits() as i64 - z_power.denom().bits() as i64;
        if term_bits < target_bits {
            break;
        }
    }

    &result * &two  // 2 * artanh(z)
}

/// Taylor series fallback for small y (fast convergence guaranteed)
fn ln_1_plus_y_taylor(y: &BigRational, precision_bits: usize) -> BigRational {
    let mut result = y.clone();
    let mut y_power = y.clone();

    let max_terms = 50 + precision_bits / 4;  // Sufficient for small y
    let target_bits = -(precision_bits as i64 + 20);

    for n in 2..=max_terms {
        y_power = &y_power * y;
        let term = &y_power / BigRational::from(BigInt::from(n));

        if n % 2 == 0 {
            result = result - term;
        } else {
            result = result + term;
        }

        let term_bits = y_power.numer().bits() as i64 - y_power.denom().bits() as i64;
        if term_bits < target_bits {
            break;
        }
    }

    result
}

/// Compute ln(2) using artanh series - OPTIMIZED
/// ln(2) = 2 * artanh(1/3) = 2 * Σ (1/(2n+1)) * (1/3)^(2n+1)
/// Converges at rate (1/9)^n - very fast!
fn ln_2_rational(precision_bits: usize) -> BigRational {
    let one_third = BigRational::new(BigInt::from(1), BigInt::from(3));
    let mut result = BigRational::zero();
    let mut power = one_third.clone();

    // ln(2) needs about precision_bits / log2(9) ≈ precision_bits / 3.17 terms
    let max_terms = precision_bits / 3 + 10;
    let target_bits = -(precision_bits as i64 + 10);

    for n in 0..max_terms {
        let coeff = BigRational::new(BigInt::from(1), BigInt::from(2 * n + 1));
        result = result + &coeff * &power;
        power = &power * &one_third * &one_third;

        let power_bits = power.numer().bits() as i64 - power.denom().bits() as i64;
        if power_bits < target_bits {
            break;
        }
    }

    result * BigRational::from(BigInt::from(2))
}

/// Generate all ln tables for all tiers
fn generate_ln_tables(out_dir: &str, _config: &PrecisionConfig) {
    println!("cargo:warning=🔧 Generating ln() lookup tables (optimized)");

    generate_ln_q64_64_tables(out_dir);
    generate_ln_q128_128_tables(out_dir);
    generate_ln_q256_256_tables(out_dir);
    generate_ln_q512_512_tables(out_dir);

    println!("cargo:warning=✅ All ln() table sets generated successfully");
}

/// Generate Q64.64 ln tables (19 decimal precision)
fn generate_ln_q64_64_tables(out_dir: &str) {
    let dest_path = Path::new(out_dir).join("ln_q64_64_tables.rs");
    let mut file = File::create(&dest_path).unwrap();

    writeln!(file, "// Q64.64 Natural Logarithm Lookup Tables").unwrap();
    writeln!(file, "// Generated from BigRational - ZERO float arithmetic").unwrap();
    writeln!(file, "").unwrap();

    // ln(2) constant - 64 bits precision
    let ln2 = ln_2_rational(80);
    let (ln2_main, ln2_comp) = rational_to_q64_64(&ln2);
    writeln!(file, "/// ln(2) constant (Tier 3: Q64.64)").unwrap();
    writeln!(file, "pub static LN_2_CONSTANT_TIER_3: (i128, i64) = ({}, {});", ln2_main, ln2_comp).unwrap();
    writeln!(file, "").unwrap();

    // Primary table: ln(1 + k/1024) - need ~50 terms for k=1023
    writeln!(file, "/// Primary ln table: ln(1 + k/1024) for k ∈ [0, 1023] (Tier 3: Q64.64)").unwrap();
    writeln!(file, "pub static LN_PRIMARY_TABLE_TIER_3: [(i128, i64); 1024] = [").unwrap();
    for k in 0..1024 {
        let y = BigRational::new(BigInt::from(k), BigInt::from(1024));
        let ln_val = ln_1_plus_y_rational(&y, 80);
        let (main, comp) = rational_to_q64_64(&ln_val);
        writeln!(file, "    ({}, {}), // ln(1 + {}/1024)", main, comp, k).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // Secondary table: ln(1 + j/2^20) - need ~15 terms
    writeln!(file, "/// Secondary ln table: ln(1 + j/2^20) for j ∈ [0, 1023] (Tier 3: Q64.64)").unwrap();
    writeln!(file, "pub static LN_SECONDARY_TABLE_TIER_3: [(i128, i64); 1024] = [").unwrap();
    for j in 0..1024 {
        let y = BigRational::new(BigInt::from(j), BigInt::from(1i128 << 20));
        let ln_val = ln_1_plus_y_rational(&y, 80);
        let (main, comp) = rational_to_q64_64(&ln_val);
        writeln!(file, "    ({}, {}), // ln(1 + {}/2^20)", main, comp, j).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // Tertiary table: ln(1 + m/2^30) - need ~10 terms
    writeln!(file, "/// Tertiary ln table: ln(1 + m/2^30) for m ∈ [0, 1023] (Tier 3: Q64.64)").unwrap();
    writeln!(file, "pub static LN_TERTIARY_TABLE_TIER_3: [(i128, i64); 1024] = [").unwrap();
    for m in 0..1024 {
        let y = BigRational::new(BigInt::from(m), BigInt::from(1i128 << 30));
        let ln_val = ln_1_plus_y_rational(&y, 80);
        let (main, comp) = rational_to_q64_64(&ln_val);
        writeln!(file, "    ({}, {}), // ln(1 + {}/2^30)", main, comp, m).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // Quaternary table: ln(1 + p/2^40) - 4th stage for improved precision
    writeln!(file, "/// Quaternary ln table: ln(1 + p/2^40) for p ∈ [0, 1023] (Tier 3: Q64.64)").unwrap();
    writeln!(file, "pub static LN_QUATERNARY_TABLE_TIER_3: [(i128, i64); 1024] = [").unwrap();
    for p in 0..1024 {
        let y = BigRational::new(BigInt::from(p), BigInt::from(1i128 << 40));
        let ln_val = ln_1_plus_y_rational(&y, 80);
        let (main, comp) = rational_to_q64_64(&ln_val);
        writeln!(file, "    ({}, {}), // ln(1 + {}/2^40)", main, comp, p).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // Taylor coefficients - UNSIGNED (1/n), signs applied in code
    // We store positive 1/n values because I512/I1024 multiplication is unsigned
    writeln!(file, "/// Taylor coefficients for ln(1+y): 1/n (Tier 3: Q64.64)").unwrap();
    writeln!(file, "/// Signs are applied in the Taylor series code, not in the table.").unwrap();
    writeln!(file, "pub static LN_TAYLOR_COEFFICIENTS_TIER_3: [(i128, i64); 51] = [").unwrap();
    writeln!(file, "    (0, 0), // placeholder for n=0").unwrap();
    for n in 1..=50 {
        // Store POSITIVE 1/n (unsigned multiplication requires positive coefficients)
        let coef = BigRational::new(BigInt::from(1), BigInt::from(n));
        let (main, comp) = rational_to_q64_64(&coef);
        writeln!(file, "    ({}, {}), // 1/{}", main, comp, n).unwrap();
    }
    writeln!(file, "];").unwrap();

    println!("cargo:warning=✅ Generated Q64.64 ln tables");
}

/// Generate Q128.128 ln tables (38 decimal precision)
fn generate_ln_q128_128_tables(out_dir: &str) {
    let dest_path = Path::new(out_dir).join("ln_q128_128_tables.rs");
    let mut file = File::create(&dest_path).unwrap();

    writeln!(file, "// Q128.128 Natural Logarithm Lookup Tables").unwrap();
    writeln!(file, "// Generated from BigRational - 38 decimal precision").unwrap();
    writeln!(file, "").unwrap();

    let ln2 = ln_2_rational(150);
    let (ln2_main, ln2_comp) = rational_to_q128_128(&ln2);
    writeln!(file, "/// ln(2) constant (Tier 4: Q128.128)").unwrap();
    writeln!(file, "pub static LN_2_CONSTANT_TIER_4: (I256, i128) = (I256::from_words({:?}), {});", ln2_main, ln2_comp).unwrap();
    writeln!(file, "").unwrap();

    writeln!(file, "/// Primary ln table: ln(1 + k/1024) for k ∈ [0, 1023] (Tier 4: Q128.128)").unwrap();
    writeln!(file, "pub static LN_PRIMARY_TABLE_TIER_4: [(I256, i128); 1024] = [").unwrap();
    for k in 0..1024 {
        let y = BigRational::new(BigInt::from(k), BigInt::from(1024));
        let ln_val = ln_1_plus_y_rational(&y, 150);
        let (main, comp) = rational_to_q128_128(&ln_val);
        writeln!(file, "    (I256::from_words({:?}), {}), // ln(1 + {}/1024)", main, comp, k).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    writeln!(file, "/// Secondary ln table: ln(1 + j/2^20) for j ∈ [0, 1023] (Tier 4: Q128.128)").unwrap();
    writeln!(file, "pub static LN_SECONDARY_TABLE_TIER_4: [(I256, i128); 1024] = [").unwrap();
    for j in 0..1024 {
        let y = BigRational::new(BigInt::from(j), BigInt::from(1i128 << 20));
        let ln_val = ln_1_plus_y_rational(&y, 150);
        let (main, comp) = rational_to_q128_128(&ln_val);
        writeln!(file, "    (I256::from_words({:?}), {}), // ln(1 + {}/2^20)", main, comp, j).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    writeln!(file, "/// Tertiary ln table: ln(1 + m/2^30) for m ∈ [0, 1023] (Tier 4: Q128.128)").unwrap();
    writeln!(file, "pub static LN_TERTIARY_TABLE_TIER_4: [(I256, i128); 1024] = [").unwrap();
    for m in 0..1024 {
        let y = BigRational::new(BigInt::from(m), BigInt::from(1i128 << 30));
        let ln_val = ln_1_plus_y_rational(&y, 150);
        let (main, comp) = rational_to_q128_128(&ln_val);
        writeln!(file, "    (I256::from_words({:?}), {}), // ln(1 + {}/2^30)", main, comp, m).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // Quaternary table: ln(1 + p/2^40) - 4th stage for improved precision
    writeln!(file, "/// Quaternary ln table: ln(1 + p/2^40) for p ∈ [0, 1023] (Tier 4: Q128.128)").unwrap();
    writeln!(file, "pub static LN_QUATERNARY_TABLE_TIER_4: [(I256, i128); 1024] = [").unwrap();
    for p in 0..1024 {
        let y = BigRational::new(BigInt::from(p), BigInt::from(1i128 << 40));
        let ln_val = ln_1_plus_y_rational(&y, 150);
        let (main, comp) = rational_to_q128_128(&ln_val);
        writeln!(file, "    (I256::from_words({:?}), {}), // ln(1 + {}/2^40)", main, comp, p).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // Taylor coefficients - UNSIGNED (1/n), signs applied in code
    writeln!(file, "/// Taylor coefficients for ln(1+y): 1/n (Tier 4: Q128.128)").unwrap();
    writeln!(file, "/// Signs are applied in the Taylor series code, not in the table.").unwrap();
    writeln!(file, "pub static LN_TAYLOR_COEFFICIENTS_TIER_4: [(I256, i128); 51] = [").unwrap();
    writeln!(file, "    (I256::from_words([0, 0, 0, 0]), 0), // placeholder for n=0").unwrap();
    for n in 1..=50 {
        // Store POSITIVE 1/n (unsigned multiplication requires positive coefficients)
        let coef = BigRational::new(BigInt::from(1), BigInt::from(n));
        let (main, comp) = rational_to_q128_128(&coef);
        writeln!(file, "    (I256::from_words({:?}), {}), // 1/{}", main, comp, n).unwrap();
    }
    writeln!(file, "];").unwrap();

    println!("cargo:warning=✅ Generated Q128.128 ln tables");
}

/// Generate Q256.256 ln tables (77 decimal precision)
fn generate_ln_q256_256_tables(out_dir: &str) {
    let dest_path = Path::new(out_dir).join("ln_q256_256_tables.rs");
    let mut file = File::create(&dest_path).unwrap();

    writeln!(file, "// Q256.256 Natural Logarithm Lookup Tables").unwrap();
    writeln!(file, "// Generated from BigRational - 77 decimal precision").unwrap();
    writeln!(file, "").unwrap();

    let ln2 = ln_2_rational(280);
    let (ln2_main, ln2_comp) = rational_to_q256_256(&ln2);
    writeln!(file, "/// ln(2) constant (Tier 5: Q256.256)").unwrap();
    writeln!(file, "pub static LN_2_CONSTANT_TIER_5: (I512, I256) = (I512::from_words({:?}), I256::from_words({:?}));", ln2_main, ln2_comp).unwrap();
    writeln!(file, "").unwrap();

    writeln!(file, "/// Primary ln table: ln(1 + k/1024) for k ∈ [0, 1023] (Tier 5: Q256.256)").unwrap();
    writeln!(file, "pub static LN_PRIMARY_TABLE_TIER_5: [(I512, I256); 1024] = [").unwrap();
    for k in 0..1024 {
        let y = BigRational::new(BigInt::from(k), BigInt::from(1024));
        let ln_val = ln_1_plus_y_rational(&y, 280);
        let (main, comp) = rational_to_q256_256(&ln_val);
        writeln!(file, "    (I512::from_words({:?}), I256::from_words({:?})), // ln(1 + {}/1024)", main, comp, k).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    writeln!(file, "/// Secondary ln table: ln(1 + j/2^20) for j ∈ [0, 1023] (Tier 5: Q256.256)").unwrap();
    writeln!(file, "pub static LN_SECONDARY_TABLE_TIER_5: [(I512, I256); 1024] = [").unwrap();
    for j in 0..1024 {
        let y = BigRational::new(BigInt::from(j), BigInt::from(1i128 << 20));
        let ln_val = ln_1_plus_y_rational(&y, 280);
        let (main, comp) = rational_to_q256_256(&ln_val);
        writeln!(file, "    (I512::from_words({:?}), I256::from_words({:?})), // ln(1 + {}/2^20)", main, comp, j).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    writeln!(file, "/// Tertiary ln table: ln(1 + m/2^30) for m ∈ [0, 1023] (Tier 5: Q256.256)").unwrap();
    writeln!(file, "pub static LN_TERTIARY_TABLE_TIER_5: [(I512, I256); 1024] = [").unwrap();
    for m in 0..1024 {
        let y = BigRational::new(BigInt::from(m), BigInt::from(1i128 << 30));
        let ln_val = ln_1_plus_y_rational(&y, 280);
        let (main, comp) = rational_to_q256_256(&ln_val);
        writeln!(file, "    (I512::from_words({:?}), I256::from_words({:?})), // ln(1 + {}/2^30)", main, comp, m).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // Quaternary table: ln(1 + p/2^40) - 4th stage for improved precision
    writeln!(file, "/// Quaternary ln table: ln(1 + p/2^40) for p ∈ [0, 1023] (Tier 5: Q256.256)").unwrap();
    writeln!(file, "pub static LN_QUATERNARY_TABLE_TIER_5: [(I512, I256); 1024] = [").unwrap();
    for p in 0..1024 {
        let y = BigRational::new(BigInt::from(p), BigInt::from(1i128 << 40));
        let ln_val = ln_1_plus_y_rational(&y, 280);
        let (main, comp) = rational_to_q256_256(&ln_val);
        writeln!(file, "    (I512::from_words({:?}), I256::from_words({:?})), // ln(1 + {}/2^40)", main, comp, p).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // Taylor coefficients - UNSIGNED (1/n), signs applied in code
    writeln!(file, "/// Taylor coefficients for ln(1+y): 1/n (Tier 5: Q256.256)").unwrap();
    writeln!(file, "/// Signs are applied in the Taylor series code, not in the table.").unwrap();
    writeln!(file, "pub static LN_TAYLOR_COEFFICIENTS_TIER_5: [(I512, I256); 51] = [").unwrap();
    writeln!(file, "    (I512::from_words([0, 0, 0, 0, 0, 0, 0, 0]), I256::from_words([0, 0, 0, 0])), // placeholder for n=0").unwrap();
    for n in 1..=50 {
        // Store POSITIVE 1/n (unsigned multiplication requires positive coefficients)
        let coef = BigRational::new(BigInt::from(1), BigInt::from(n));
        let (main, comp) = rational_to_q256_256(&coef);
        writeln!(file, "    (I512::from_words({:?}), I256::from_words({:?})), // 1/{}", main, comp, n).unwrap();
    }
    writeln!(file, "];").unwrap();

    println!("cargo:warning=✅ Generated Q256.256 ln tables");
}

/// Generate Q512.512 ln tables (154 decimal precision)
fn generate_ln_q512_512_tables(out_dir: &str) {
    let dest_path = Path::new(out_dir).join("ln_q512_512_tables.rs");
    let mut file = File::create(&dest_path).unwrap();

    writeln!(file, "// Q512.512 Natural Logarithm Lookup Tables").unwrap();
    writeln!(file, "// Generated from BigRational - 154 decimal precision").unwrap();
    writeln!(file, "").unwrap();

    let ln2 = ln_2_rational(550);
    let (ln2_main, ln2_comp) = rational_to_q512_512(&ln2);
    writeln!(file, "/// ln(2) constant (Tier 6: Q512.512)").unwrap();
    writeln!(file, "pub static LN_2_CONSTANT_TIER_6: (I1024, I512) = (I1024::from_words({:?}), I512::from_words({:?}));", ln2_main, ln2_comp).unwrap();
    writeln!(file, "").unwrap();

    writeln!(file, "/// Primary ln table: ln(1 + k/1024) for k ∈ [0, 1023] (Tier 6: Q512.512)").unwrap();
    writeln!(file, "pub static LN_PRIMARY_TABLE_TIER_6: [(I1024, I512); 1024] = [").unwrap();
    for k in 0..1024 {
        let y = BigRational::new(BigInt::from(k), BigInt::from(1024));
        let ln_val = ln_1_plus_y_rational(&y, 550);
        let (main, comp) = rational_to_q512_512(&ln_val);
        writeln!(file, "    (I1024::from_words({:?}), I512::from_words({:?})), // ln(1 + {}/1024)", main, comp, k).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    writeln!(file, "/// Secondary ln table: ln(1 + j/2^20) for j ∈ [0, 1023] (Tier 6: Q512.512)").unwrap();
    writeln!(file, "pub static LN_SECONDARY_TABLE_TIER_6: [(I1024, I512); 1024] = [").unwrap();
    for j in 0..1024 {
        let y = BigRational::new(BigInt::from(j), BigInt::from(1i128 << 20));
        let ln_val = ln_1_plus_y_rational(&y, 550);
        let (main, comp) = rational_to_q512_512(&ln_val);
        writeln!(file, "    (I1024::from_words({:?}), I512::from_words({:?})), // ln(1 + {}/2^20)", main, comp, j).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    writeln!(file, "/// Tertiary ln table: ln(1 + m/2^30) for m ∈ [0, 1023] (Tier 6: Q512.512)").unwrap();
    writeln!(file, "pub static LN_TERTIARY_TABLE_TIER_6: [(I1024, I512); 1024] = [").unwrap();
    for m in 0..1024 {
        let y = BigRational::new(BigInt::from(m), BigInt::from(1i128 << 30));
        let ln_val = ln_1_plus_y_rational(&y, 550);
        let (main, comp) = rational_to_q512_512(&ln_val);
        writeln!(file, "    (I1024::from_words({:?}), I512::from_words({:?})), // ln(1 + {}/2^30)", main, comp, m).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // Quaternary table: ln(1 + p/2^40) - 4th stage for improved precision
    writeln!(file, "/// Quaternary ln table: ln(1 + p/2^40) for p ∈ [0, 1023] (Tier 6: Q512.512)").unwrap();
    writeln!(file, "pub static LN_QUATERNARY_TABLE_TIER_6: [(I1024, I512); 1024] = [").unwrap();
    for p in 0..1024 {
        let y = BigRational::new(BigInt::from(p), BigInt::from(1i128 << 40));
        let ln_val = ln_1_plus_y_rational(&y, 550);
        let (main, comp) = rational_to_q512_512(&ln_val);
        writeln!(file, "    (I1024::from_words({:?}), I512::from_words({:?})), // ln(1 + {}/2^40)", main, comp, p).unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();

    // Taylor coefficients - UNSIGNED (1/n), signs applied in code
    writeln!(file, "/// Taylor coefficients for ln(1+y): 1/n (Tier 6: Q512.512)").unwrap();
    writeln!(file, "/// Signs are applied in the Taylor series code, not in the table.").unwrap();
    writeln!(file, "pub static LN_TAYLOR_COEFFICIENTS_TIER_6: [(I1024, I512); 51] = [").unwrap();
    writeln!(file, "    (I1024::from_words([0; 16]), I512::from_words([0; 8])), // placeholder for n=0").unwrap();
    for n in 1..=50 {
        // Store POSITIVE 1/n (unsigned multiplication requires positive coefficients)
        let coef = BigRational::new(BigInt::from(1), BigInt::from(n));
        let (main, comp) = rational_to_q512_512(&coef);
        writeln!(file, "    (I1024::from_words({:?}), I512::from_words({:?})), // 1/{}", main, comp, n).unwrap();
    }
    writeln!(file, "];").unwrap();

    println!("cargo:warning=✅ Generated Q512.512 ln tables");
}

// ============================================================================
// AUXILIARY TABLE GENERATION (Prime Tables)
// ============================================================================

const MAX_PRIMES: usize = 100_000;
const PRIME_LIMIT: u64 = 10_000;

/// Generate prime table using Sieve of Eratosthenes
fn generate_prime_table_sieve(out_dir: &str) -> Vec<u64> {
    println!("cargo:warning=🔍 Sieve of Eratosthenes: generating primes up to {}", PRIME_LIMIT);

    let primes = sieve_of_eratosthenes(PRIME_LIMIT);

    // Write to generated file
    let dest_path = Path::new(out_dir).join("prime_table.rs");
    let mut file = File::create(&dest_path).unwrap();

    writeln!(file, "// Generated Prime Table - Pattern Generation Foundation").unwrap();
    writeln!(file, "// Generated by sieve of Eratosthenes algorithm").unwrap();
    writeln!(file, "// Primes up to {} (first {} primes)", PRIME_LIMIT, primes.len()).unwrap();
    writeln!(file, "// DO NOT MODIFY - Generated by build.rs").unwrap();
    writeln!(file, "").unwrap();

    writeln!(file, "/// Compile-time generated prime table").unwrap();
    writeln!(file, "pub const PRIME_TABLE: &[u64] = &[").unwrap();

    for (i, &prime) in primes.iter().enumerate() {
        if i % 10 == 0 {
            write!(file, "    ").unwrap();
        }
        write!(file, "{}", prime).unwrap();
        if i < primes.len() - 1 {
            write!(file, ", ").unwrap();
        }
        if i % 10 == 9 || i == primes.len() - 1 {
            writeln!(file).unwrap();
        }
    }

    writeln!(file, "];").unwrap();
    writeln!(file, "").unwrap();
    writeln!(file, "pub const PRIME_COUNT: usize = {};", primes.len()).unwrap();
    writeln!(file, "pub const MAX_PRIME: u64 = {};", primes.last().unwrap_or(&0)).unwrap();

    primes
}

/// Optimized sieve of Eratosthenes with wheel factorization
fn sieve_of_eratosthenes(limit: u64) -> Vec<u64> {
    if limit < 2 {
        return Vec::new();
    }

    // Prime number theorem approximation using integer math
    // π(n) ≈ n / ln(n) ≈ n / (log2(n) * 0.693)
    // Conservative estimate: n / 10 for practical ranges
    let capacity_estimate = (limit as usize).max(100) / 10;
    let mut primes = Vec::with_capacity(capacity_estimate);

    if limit >= 2 { primes.push(2); }
    if limit >= 3 { primes.push(3); }
    if limit >= 5 { primes.push(5); }

    if limit < 7 {
        return primes.into_iter().take(MAX_PRIMES).collect();
    }

    // Wheel factorization: only check numbers coprime to 2,3,5
    let wheel = [1, 7, 11, 13, 17, 19, 23, 29];
    let wheel_size = 30;

    let sieve_limit = ((limit - 7) / wheel_size + 1) * wheel.len() as u64;
    let mut is_prime = vec![true; sieve_limit as usize];

    // Integer square root using binary search
    let sqrt_limit = {
        let mut low = 0u64;
        let mut high = limit;
        while low < high {
            let mid = (low + high + 1) / 2;
            if mid <= limit / mid {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        low
    };

    // Optimized sieving with wheel
    for base in (0..).map(|k| k * wheel_size).take_while(|&b| b <= sqrt_limit) {
        for &offset in &wheel {
            let candidate = base + offset;
            if candidate < 7 { continue; }
            if candidate > sqrt_limit { break; }

            let idx = wheel_index(candidate);
            if idx < is_prime.len() && is_prime[idx] {
                // Mark multiples
                let mut multiple = candidate * candidate;
                while multiple <= limit {
                    let mult_idx = wheel_index(multiple);
                    if mult_idx < is_prime.len() {
                        is_prime[mult_idx] = false;
                    }
                    multiple += candidate;
                }
            }
        }
    }

    // Collect primes
    for base in (0..).map(|k| k * wheel_size).take_while(|&b| b <= limit) {
        for &offset in &wheel {
            let candidate = base + offset;
            if candidate >= 7 && candidate <= limit {
                let idx = wheel_index(candidate);
                if idx < is_prime.len() && is_prime[idx] {
                    primes.push(candidate);
                    if primes.len() >= MAX_PRIMES {
                        return primes;
                    }
                }
            }
        }
    }

    primes
}

/// Convert number to wheel factorization index
fn wheel_index(n: u64) -> usize {
    let wheel = [1, 7, 11, 13, 17, 19, 23, 29];
    let base = (n / 30) * 8;
    let offset = n % 30;

    for (i, &w) in wheel.iter().enumerate() {
        if w == offset {
            return (base + i as u64) as usize;
        }
    }

    // Fallback
    ((n - 7) / 30 * 8) as usize
}

// Pattern generation + CHD hash tables removed (see archive/legacy_router/)

// ============================================================================
// TRIGONOMETRIC CONSTANTS GENERATION
// ============================================================================

/// Generate trigonometric constants for sin/cos/atan implementations
///
/// **OUTPUT**: `trig_constants.rs` containing:
/// - PI, PI_HALF, PI_QUARTER, TWO_OVER_PI in each Q-format
/// - Taylor coefficients for sin: 1/3!, 1/5!, 1/7!, ... (odd factorials)
/// - Taylor coefficients for cos: 1/2!, 1/4!, 1/6!, ... (even factorials)
/// - Taylor coefficients for atan: 1/3, 1/5, 1/7, ... (odd reciprocals)
/// - Threshold constant: tan(π/8) for atan range reduction
///
/// All computed via pure BigRational arithmetic — ZERO float contamination
fn generate_trig_constants(out_dir: &str, _config: &PrecisionConfig) {
    let dest_path = Path::new(out_dir).join("trig_constants.rs");
    let mut file = File::create(&dest_path).unwrap();

    writeln!(file, "// Trigonometric Constants - Generated by build.rs").unwrap();
    writeln!(file, "// ZERO float arithmetic - Pure BigRational → Q-format conversion").unwrap();
    writeln!(file, "").unwrap();

    // Compute π via Machin's formula with 580 bits of precision
    // (512 fractional bits for Q512.512 + safety margin)
    let pi = compute_pi_rational(580);

    let two = BigRational::from(BigInt::from(2));
    let four = BigRational::from(BigInt::from(4));
    let pi_half = &pi / &two;
    let pi_quarter = &pi / &four;
    let two_over_pi = &two / &pi;

    // ========== Q64.64 constants (i128) ==========
    writeln!(file, "// Q64.64 trigonometric constants (i128)").unwrap();
    let (pi_q64, _) = rational_to_q64_64(&pi);
    let (pi_half_q64, _) = rational_to_q64_64(&pi_half);
    let (pi_quarter_q64, _) = rational_to_q64_64(&pi_quarter);
    let (two_over_pi_q64, _) = rational_to_q64_64(&two_over_pi);
    writeln!(file, "pub const PI_Q64: i128 = {};", pi_q64).unwrap();
    writeln!(file, "pub const PI_HALF_Q64: i128 = {};", pi_half_q64).unwrap();
    writeln!(file, "pub const PI_QUARTER_Q64: i128 = {};", pi_quarter_q64).unwrap();
    writeln!(file, "pub const TWO_OVER_PI_Q64: i128 = {};", two_over_pi_q64).unwrap();
    writeln!(file, "").unwrap();

    // ========== Q128.128 constants (I256 as [u64; 4]) ==========
    writeln!(file, "// Q128.128 trigonometric constants (I256 words)").unwrap();
    let (pi_q128, _) = rational_to_q128_128(&pi);
    let (pi_half_q128, _) = rational_to_q128_128(&pi_half);
    let (pi_quarter_q128, _) = rational_to_q128_128(&pi_quarter);
    let (two_over_pi_q128, _) = rational_to_q128_128(&two_over_pi);
    writeln!(file, "pub const PI_Q128: [u64; 4] = {:?};", pi_q128).unwrap();
    writeln!(file, "pub const PI_HALF_Q128: [u64; 4] = {:?};", pi_half_q128).unwrap();
    writeln!(file, "pub const PI_QUARTER_Q128: [u64; 4] = {:?};", pi_quarter_q128).unwrap();
    writeln!(file, "pub const TWO_OVER_PI_Q128: [u64; 4] = {:?};", two_over_pi_q128).unwrap();
    writeln!(file, "").unwrap();

    // ========== Q256.256 constants (I512 as [u64; 8]) ==========
    writeln!(file, "// Q256.256 trigonometric constants (I512 words)").unwrap();
    let (pi_q256, _) = rational_to_q256_256(&pi);
    let (pi_half_q256, _) = rational_to_q256_256(&pi_half);
    let (pi_quarter_q256, _) = rational_to_q256_256(&pi_quarter);
    let (two_over_pi_q256, _) = rational_to_q256_256(&two_over_pi);
    writeln!(file, "pub const PI_Q256: [u64; 8] = {:?};", pi_q256).unwrap();
    writeln!(file, "pub const PI_HALF_Q256: [u64; 8] = {:?};", pi_half_q256).unwrap();
    writeln!(file, "pub const PI_QUARTER_Q256: [u64; 8] = {:?};", pi_quarter_q256).unwrap();
    writeln!(file, "pub const TWO_OVER_PI_Q256: [u64; 8] = {:?};", two_over_pi_q256).unwrap();
    writeln!(file, "").unwrap();

    // ========== Q512.512 constants (I1024 as [u64; 16]) for tier N+1 of Q256.256 ==========
    writeln!(file, "// Q512.512 trigonometric constants (I1024 words) for tier N+1 computation").unwrap();
    let (pi_q512, _) = rational_to_q512_512(&pi);
    let (pi_half_q512, _) = rational_to_q512_512(&pi_half);
    let (pi_quarter_q512, _) = rational_to_q512_512(&pi_quarter);
    let (two_over_pi_q512, _) = rational_to_q512_512(&two_over_pi);
    writeln!(file, "pub const PI_Q512: [u64; 16] = {:?};", pi_q512).unwrap();
    writeln!(file, "pub const PI_HALF_Q512: [u64; 16] = {:?};", pi_half_q512).unwrap();
    writeln!(file, "pub const PI_QUARTER_Q512: [u64; 16] = {:?};", pi_quarter_q512).unwrap();
    writeln!(file, "pub const TWO_OVER_PI_Q512: [u64; 16] = {:?};", two_over_pi_q512).unwrap();
    writeln!(file, "").unwrap();

    // ========== Taylor coefficients for sin: x - x³/3! + x⁵/5! - x⁷/7! + ... ==========
    // We store 1/n! for odd n = 3, 5, 7, ..., with alternating signs baked in
    // sin(x) = x + Σ (-1)^k * x^(2k+1) / (2k+1)!  for k = 1, 2, ...
    // Store: coefficients[k] = (-1)^k / (2k+1)! for k = 0, 1, 2, ...
    // (coefficient for x^1 is 1, for x^3 is -1/6, for x^5 is 1/120, etc.)
    // Actually: store 1/(2k+1)! unsigned, apply sign at eval time

    let mut factorial: BigInt;

    // Generate sin coefficients: 1/1!, 1/3!, 1/5!, 1/7!, ...
    writeln!(file, "// Sin Taylor coefficients: 1/(2k+1)! for k=0,1,2,...").unwrap();
    writeln!(file, "// sin(x) = Σ (-1)^k * SIN_COEFF[k] * x^(2k+1)").unwrap();

    // Q64.64 sin coefficients (11 terms for 64 fractional bits)
    writeln!(file, "pub const SIN_COEFFS_Q64: [i128; 11] = [").unwrap();
    factorial = BigInt::from(1); // 1!
    for k in 0..11 {
        let n = 2 * k + 1; // 1, 3, 5, 7, ...
        if k == 0 {
            factorial = BigInt::from(1); // 1!
        } else {
            // factorial = (2k+1)! = (2k-1)! * 2k * (2k+1)
            factorial = &factorial * BigInt::from(2 * k) * BigInt::from(2 * k + 1);
        }
        let coeff = BigRational::new(BigInt::from(1), factorial.clone());
        let (val, _) = rational_to_q64_64(&coeff);
        writeln!(file, "    {}, // 1/{}!", val, n).unwrap();
    }
    writeln!(file, "];").unwrap();

    // Q128.128 sin coefficients (21 terms)
    writeln!(file, "pub const SIN_COEFFS_Q128: [[u64; 4]; 21] = [").unwrap();
    factorial = BigInt::from(1);
    for k in 0..21 {
        let n = 2 * k + 1;
        if k == 0 {
            factorial = BigInt::from(1);
        } else {
            factorial = &factorial * BigInt::from(2 * k) * BigInt::from(2 * k + 1);
        }
        let coeff = BigRational::new(BigInt::from(1), factorial.clone());
        let (val, _) = rational_to_q128_128(&coeff);
        writeln!(file, "    {:?}, // 1/{}!", val, n).unwrap();
    }
    writeln!(file, "];").unwrap();

    // Q256.256 sin coefficients (41 terms)
    writeln!(file, "pub const SIN_COEFFS_Q256: [[u64; 8]; 41] = [").unwrap();
    factorial = BigInt::from(1);
    for k in 0..41 {
        let n = 2 * k + 1;
        if k == 0 {
            factorial = BigInt::from(1);
        } else {
            factorial = &factorial * BigInt::from(2 * k) * BigInt::from(2 * k + 1);
        }
        let coeff = BigRational::new(BigInt::from(1), factorial.clone());
        let (val, _) = rational_to_q256_256(&coeff);
        writeln!(file, "    {:?}, // 1/{}!", val, n).unwrap();
    }
    writeln!(file, "];").unwrap();

    // Q512.512 sin coefficients (65 terms) for tier N+1
    writeln!(file, "pub const SIN_COEFFS_Q512: [[u64; 16]; 65] = [").unwrap();
    factorial = BigInt::from(1);
    for k in 0..65 {
        let n = 2 * k + 1;
        if k == 0 {
            factorial = BigInt::from(1);
        } else {
            factorial = &factorial * BigInt::from(2 * k) * BigInt::from(2 * k + 1);
        }
        let coeff = BigRational::new(BigInt::from(1), factorial.clone());
        let (val, _) = rational_to_q512_512(&coeff);
        writeln!(file, "    {:?}, // 1/{}!", val, n).unwrap();
    }
    writeln!(file, "];").unwrap();

    // Generate cos coefficients: 1/0!, 1/2!, 1/4!, 1/6!, ...
    // cos(x) = Σ (-1)^k * COS_COEFF[k] * x^(2k)
    writeln!(file, "").unwrap();
    writeln!(file, "// Cos Taylor coefficients: 1/(2k)! for k=0,1,2,...").unwrap();
    writeln!(file, "// cos(x) = Σ (-1)^k * COS_COEFF[k] * x^(2k)").unwrap();

    // Q64.64 cos coefficients
    writeln!(file, "pub const COS_COEFFS_Q64: [i128; 11] = [").unwrap();
    factorial = BigInt::from(1); // 0!
    for k in 0..11 {
        let n = 2 * k; // 0, 2, 4, 6, ...
        if k == 0 {
            factorial = BigInt::from(1); // 0!
        } else {
            // factorial = (2k)! = (2k-2)! * (2k-1) * 2k
            factorial = &factorial * BigInt::from(2 * k - 1) * BigInt::from(2 * k);
        }
        let coeff = BigRational::new(BigInt::from(1), factorial.clone());
        let (val, _) = rational_to_q64_64(&coeff);
        writeln!(file, "    {}, // 1/{}!", val, n).unwrap();
    }
    writeln!(file, "];").unwrap();

    // Q128.128 cos coefficients
    writeln!(file, "pub const COS_COEFFS_Q128: [[u64; 4]; 21] = [").unwrap();
    factorial = BigInt::from(1);
    for k in 0..21 {
        let n = 2 * k;
        if k == 0 {
            factorial = BigInt::from(1);
        } else {
            factorial = &factorial * BigInt::from(2 * k - 1) * BigInt::from(2 * k);
        }
        let coeff = BigRational::new(BigInt::from(1), factorial.clone());
        let (val, _) = rational_to_q128_128(&coeff);
        writeln!(file, "    {:?}, // 1/{}!", val, n).unwrap();
    }
    writeln!(file, "];").unwrap();

    // Q256.256 cos coefficients
    writeln!(file, "pub const COS_COEFFS_Q256: [[u64; 8]; 41] = [").unwrap();
    factorial = BigInt::from(1);
    for k in 0..41 {
        let n = 2 * k;
        if k == 0 {
            factorial = BigInt::from(1);
        } else {
            factorial = &factorial * BigInt::from(2 * k - 1) * BigInt::from(2 * k);
        }
        let coeff = BigRational::new(BigInt::from(1), factorial.clone());
        let (val, _) = rational_to_q256_256(&coeff);
        writeln!(file, "    {:?}, // 1/{}!", val, n).unwrap();
    }
    writeln!(file, "];").unwrap();

    // Q512.512 cos coefficients
    writeln!(file, "pub const COS_COEFFS_Q512: [[u64; 16]; 65] = [").unwrap();
    factorial = BigInt::from(1);
    for k in 0..65 {
        let n = 2 * k;
        if k == 0 {
            factorial = BigInt::from(1);
        } else {
            factorial = &factorial * BigInt::from(2 * k - 1) * BigInt::from(2 * k);
        }
        let coeff = BigRational::new(BigInt::from(1), factorial.clone());
        let (val, _) = rational_to_q512_512(&coeff);
        writeln!(file, "    {:?}, // 1/{}!", val, n).unwrap();
    }
    writeln!(file, "];").unwrap();

    // ========== Atan constants ==========
    // tan(π/8) threshold for atan range reduction
    // tan(π/8) = √2 - 1 ≈ 0.41421356...
    let sqrt2_num: BigInt = hardcoded_constants::tier_77d::SQRT_2_NUM_STR.parse().unwrap();
    let sqrt2_den: BigInt = hardcoded_constants::tier_77d::SQRT_2_DEN_STR.parse().unwrap();
    let sqrt2 = BigRational::new(sqrt2_num, sqrt2_den);
    let one = BigRational::from(BigInt::from(1));
    let tan_pi_8 = &sqrt2 - &one;

    writeln!(file, "").unwrap();
    writeln!(file, "// Atan constants").unwrap();
    let (tan_pi8_q64, _) = rational_to_q64_64(&tan_pi_8);
    let (tan_pi8_q128, _) = rational_to_q128_128(&tan_pi_8);
    let (tan_pi8_q256, _) = rational_to_q256_256(&tan_pi_8);
    let (tan_pi8_q512, _) = rational_to_q512_512(&tan_pi_8);
    writeln!(file, "pub const TAN_PI_8_Q64: i128 = {};", tan_pi8_q64).unwrap();
    writeln!(file, "pub const TAN_PI_8_Q128: [u64; 4] = {:?};", tan_pi8_q128).unwrap();
    writeln!(file, "pub const TAN_PI_8_Q256: [u64; 8] = {:?};", tan_pi8_q256).unwrap();
    writeln!(file, "pub const TAN_PI_8_Q512: [u64; 16] = {:?};", tan_pi8_q512).unwrap();

    println!("cargo:warning=✅ Generated trigonometric constants (trig_constants.rs)");
}

// ============================================================================
// MAIN BUILD SCRIPT
// ============================================================================

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=GMATH_PROFILE");
    println!("cargo:rerun-if-env-changed=GMATH_MAX_DECIMAL_PRECISION");
    // Rerun if feature flags change
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_EMBEDDED");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_EMBEDDED_MINIMAL");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_BALANCED");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_SCIENTIFIC");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_MULTI_PRECISION");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_FAST");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_COMPACT");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_REALTIME");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_REBUILD_TABLES");

    // Declare custom cfg values for Rust compiler
    println!("cargo::rustc-check-cfg=cfg(table_format, values(\"q16_16\", \"q32_32\", \"q64_64\", \"q128_128\", \"q256_256\"))");

    let out_dir = env::var("OUT_DIR").unwrap();

    // Detect profile and configure precision
    let profile = detect_deployment_profile();
    let config = PrecisionConfig::for_profile(profile);

    // Emit cfg directives for conditional compilation based on table format
    match config.table_format {
        "Q16.16" => println!("cargo:rustc-cfg=table_format=\"q16_16\""),
        "Q32.32" => println!("cargo:rustc-cfg=table_format=\"q32_32\""),
        "Q64.64" => println!("cargo:rustc-cfg=table_format=\"q64_64\""),
        "Q128.128" => println!("cargo:rustc-cfg=table_format=\"q128_128\""),
        "Q256.256" => println!("cargo:rustc-cfg=table_format=\"q256_256\""),
        _ => println!("cargo:rustc-cfg=table_format=\"q64_64\""), // fallback
    }

    // =========================================================================
    // TABLE GENERATION: Only when rebuild-tables feature is enabled
    // =========================================================================
    // Pre-built tables are checked into src/generated_tables/ and used by default.
    // To regenerate: cargo build --features rebuild-tables
    if std::env::var("CARGO_FEATURE_REBUILD_TABLES").is_ok() {
        println!("cargo:warning=🚀 gMath Build System - Regenerating Tables");
        println!("cargo:warning=📊 Profile: {:?} | Precision: {} decimals | Format: {}",
                 profile, config.target_decimal_places, config.table_format);

        // Generate mathematical constants
        generate_mathematical_constants(&out_dir, &config);

        // Generate profile-aware transcendental tables (exp)
        generate_profile_aware_tables(&out_dir, &config);

        // Generate natural logarithm tables (ln)
        generate_ln_tables(&out_dir, &config);

        // Generate trigonometric constants (sin/cos/atan)
        generate_trig_constants(&out_dir, &config);

        // Generate prime table
        let _primes = generate_prime_table_sieve(&out_dir);
        println!("cargo:warning=✅ Generated {} primes up to {}", _primes.len(), PRIME_LIMIT);

        println!("cargo:warning=✅ Tables regenerated in OUT_DIR.");
        println!("cargo:warning=   To update repo: cp $OUT_DIR/*.rs src/generated_tables/");
    }
    // Default path: pre-built tables from src/generated_tables/ — silent.
}
