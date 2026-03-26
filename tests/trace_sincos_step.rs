#[allow(dead_code)]
mod step {
    use g_math::fixed_point::i256::I256;

    // Include trig constants
    #[cfg(not(feature = "rebuild-tables"))]
    include!("../src/generated_tables/trig_constants.rs");
    #[cfg(feature = "rebuild-tables")]
    include!(concat!(env!("OUT_DIR"), "/trig_constants.rs"));

    #[test]
    fn step_by_step_sincos() {
        let x: i128 = 3_i128 << 64; // 3.0 in Q64.64
        
        // Step 1: k = round(x * 2/pi)
        let two_over_pi = TWO_OVER_PI_Q64;
        let product = {
            let x_wide = I256::from_i128(x);
            let twopi_wide = I256::from_i128(two_over_pi);
            let prod = x_wide * twopi_wide;
            (prod >> 64u32).as_i128()
        };
        println!("x * (2/pi) in Q64.64 = {}", product);
        
        let k = (product + (1_i128 << 63)) >> 64;
        println!("k = {}", k);
        
        // Step 2: r = x - k * pi/2
        let pi_half = PI_HALF_Q64;
        let r = x - k * pi_half;
        println!("r = x - k*pi/2 = {}", r);
        println!("r as f64 = {}", r as f64 / (1u128 << 64) as f64);
        
        // Step 3: r^2
        let r_sq = {
            let r_wide = I256::from_i128(r);
            ((r_wide * r_wide) >> 64u32).as_i128()
        };
        println!("r^2 = {}", r_sq);
        
        // Step 4: Taylor sin(r)
        let num_terms = 11;
        let mut result: i128 = 0;
        for i in (1..num_terms).rev() {
            let coeff = SIN_COEFFS_Q64[i];
            let term = {
                let r_sq_wide = I256::from_i128(r_sq);
                let result_wide = I256::from_i128(result);
                ((r_sq_wide * result_wide) >> 64u32).as_i128()
            };
            result = coeff - term;
        }
        // Final: sin(r) = r * result / one
        let sin_r = {
            let r_wide = I256::from_i128(r);
            let result_wide = I256::from_i128(result);
            ((r_wide * result_wide) >> 64u32).as_i128()
        };
        println!("sin(r) = {}", sin_r);
        
        // Step 5: Quadrant correction (k=2 means sin(x) = -sin(r), cos(x) = -cos(r))
        let quadrant = ((k % 4) + 4) % 4;
        println!("quadrant = {}", quadrant);
        
        let final_sin = match quadrant {
            0 => sin_r,
            1 => { println!("cos_r needed"); 0 }, // placeholder
            2 => -sin_r,
            3 => { println!("-cos_r needed"); 0 }, // placeholder
            _ => unreachable!(),
        };
        println!("final sin(3) = {}", final_sin);
        println!("expected     = 2603255189406976154");
        println!("ULP = {}", (final_sin - 2603255189406976154_i128).abs());
    }
}
