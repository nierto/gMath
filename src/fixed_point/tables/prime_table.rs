//! Prime Table — Build-Time Generated Reference Database
//!
//! Contains all primes up to 10,000 generated via Sieve of Eratosthenes
//! at compile time by `build.rs`. Zero external dependencies.
//!
//! # Usage
//!
//! ```rust
//! use g_math::fixed_point::tables::prime_table::{PRIME_TABLE, PRIME_COUNT, MAX_PRIME};
//!
//! // Check if 97 is prime
//! assert!(PRIME_TABLE.binary_search(&97).is_ok());
//!
//! // Iterate first 10 primes
//! for &p in &PRIME_TABLE[..10] {
//!     println!("{}", p);
//! }
//! ```

#[cfg(feature = "rebuild-tables")]
include!(concat!(env!("OUT_DIR"), "/prime_table.rs"));
#[cfg(not(feature = "rebuild-tables"))]
include!("../../generated_tables/prime_table.rs");

/// Check if a number exists in the precomputed prime table.
///
/// Uses binary search — O(log n) where n = PRIME_COUNT (1,145 primes).
/// Only valid for values up to MAX_PRIME (9,973).
#[inline]
pub fn is_prime(n: u64) -> bool {
    PRIME_TABLE.binary_search(&n).is_ok()
}

/// Return the nth prime (0-indexed). Returns None if index >= PRIME_COUNT.
#[inline]
pub fn nth_prime(index: usize) -> Option<u64> {
    PRIME_TABLE.get(index).copied()
}

/// Count of primes up to and including `limit`.
/// Returns 0 if limit < 2.
pub fn prime_count_up_to(limit: u64) -> usize {
    match PRIME_TABLE.binary_search(&limit) {
        Ok(idx) => idx + 1,
        Err(idx) => idx,
    }
}
