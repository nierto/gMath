//! Domain Implementations — 4 precision domains for FASC+UGOD pipeline
//!
//! Each domain provides domain-native arithmetic and type representations:
//! - `binary_fixed`: Q64.64/Q128.128/Q256.256 binary fixed-point (tier N+1 transcendentals)
//! - `decimal_fixed`: Exact decimal arithmetic (financial)
//! - `balanced_ternary`: Balanced ternary UGOD tier arithmetic
//! - `symbolic`: Exact rational arithmetic (RationalNumber)

// Binary Fixed-Point Domain
pub mod binary_fixed;
pub use binary_fixed::{
    I256, I512, I1024,
    multiply_binary_i128,
    mul_i128_to_i256,
    transcendental,
};

// Decimal Fixed-Point Domain
pub mod decimal_fixed;
pub use decimal_fixed::DecimalFixed;

// Balanced Ternary Domain
pub mod balanced_ternary;

// Symbolic Domain (rational arithmetic)
pub mod symbolic;
pub use symbolic::{
    RationalNumber, RationalStorage,
    SymbolicConstants,
};

// Re-export domain-specific modules for internal use
pub use binary_fixed::{
    i256, i512,
};
