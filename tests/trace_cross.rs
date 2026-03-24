mod cross {
    use g_math::fixed_point::i256::I256;

    #[test]
    fn test_basic_i256_multiply() {
        // This MUST produce the same result on every profile
        let a = I256::from_i128(3_i128 << 64);
        let b = I256::from_i128(11743562013128004905_i128); // TWO_OVER_PI_Q64
        let product = a * b;
        let shifted = (product >> 64u32).as_i128();
        println!("3.0 * (2/pi) in Q64.64 = {}", shifted);
        // This should be ~1.909... * 2^64 = 35208275930186131046
        // k = round(shifted / 2^64) = round(1.909...) = 2
        let k = (shifted + (1_i128 << 63)) >> 64;
        println!("k = {}", k);
    }
}
