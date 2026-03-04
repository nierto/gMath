//! Core Types Module
//!
//! Shared type definitions used across all precision domains.

pub mod precision;
pub mod errors;
pub mod domain_metadata;

// Re-export commonly used types
pub use precision::PrecisionTier;
pub use errors::OverflowDetected;
pub use domain_metadata::{DomainType, ExactnessType, ShadowMetadata};
