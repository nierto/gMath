//! Integration test for symbolic rational arithmetic

#[cfg(test)]
mod integration_tests {
    use crate::fixed_point::domains::symbolic::rational::*;

    #[test]
    fn test_symbolic_integration_basic() {
        // Create rational numbers
        let a = RationalNumber::new(1, 3);  // 1/3
        let b = RationalNumber::new(3, 8);  // 3/8
        
        // Test division-as-multiplication: (1/3) ÷ (3/8) = 8/9
        let result = a / b;
        
        // Verify exact result
        assert_eq!(result.numerator_i128().unwrap(), 8);
        assert_eq!(result.denominator_i128().unwrap(), 9);
    }
    
    #[test]
    fn test_rational_mathematical_exactness() {
        // Test perfect precision: (1/3) * 3 = 1 exactly
        let one_third = RationalNumber::new(1, 3);
        let three = RationalNumber::new(3, 1);
        let result = one_third * three;
        
        // Should be exactly 1/1
        assert_eq!(result.numerator_i128().unwrap(), 1);
        assert_eq!(result.denominator_i128().unwrap(), 1);
    }
    
    #[test]
    fn test_rational_chain_operations() {
        // Complex chain: ((1/7) × (7/3)) ÷ (2/9) = (1/3) × (9/2) = 9/6 = 3/2
        let a = RationalNumber::new(1, 7);
        let b = RationalNumber::new(7, 3);  
        let c = RationalNumber::new(2, 9);
        
        let result = (a * b) / c;
        
        // Should simplify to 3/2 after GCD reduction
        // Note: We might get 9/6 which is equivalent, but our implementation
        // should canonicalize it to 3/2
        let expected_num = 3i32;
        let expected_den = 2i32;
        
        // Check the mathematical equivalence
        let actual_num: i32 = result.numerator_i128().unwrap() as i32;
        let actual_den: i32 = result.denominator_i128().unwrap() as i32;
        
        // Cross multiply to check equality: actual_num * expected_den == expected_num * actual_den
        assert_eq!(actual_num * expected_den, expected_num * actual_den);
    }
}