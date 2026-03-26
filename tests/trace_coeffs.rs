#[allow(dead_code)]
mod coeffs {
    // Include trig constants directly
    #[cfg(feature = "rebuild-tables")]
    include!(concat!(env!("OUT_DIR"), "/trig_constants.rs"));
    #[cfg(not(feature = "rebuild-tables"))]
    include!("../src/generated_tables/trig_constants.rs");

    #[test]
    fn print_sin_coeffs() {
        println!("SIN_COEFFS_Q64[0] = {}", SIN_COEFFS_Q64[0]);
        println!("SIN_COEFFS_Q64[1] = {}", SIN_COEFFS_Q64[1]);
        println!("SIN_COEFFS_Q64[2] = {}", SIN_COEFFS_Q64[2]);
        println!("COS_COEFFS_Q64[0] = {}", COS_COEFFS_Q64[0]);
        println!("TWO_OVER_PI_Q64 = {}", TWO_OVER_PI_Q64);
        println!("PI_HALF_Q64 = {}", PI_HALF_Q64);
    }
}
