//! The published crate's format, read from the variables it reads itself,
//! so the gate can select the code paths that exist in that build.
use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=GMATH_PROFILE");
    let profile = match env::var("GMATH_PROFILE").unwrap_or_default().to_lowercase().as_str() {
        "realtime" => "q16_16",
        "compact" | "fast" => "q32_32",
        "" | "embedded" => "q64_64",
        "balanced" => "q128_128",
        "scientific" => "q256_256",
        other => panic!("unknown GMATH_PROFILE '{other}'"),
    };
    println!("cargo:rustc-cfg=published_profile=\"{profile}\"");
}
