//! Mode Routing System (`compute_mode:output_mode`)
//!
//! Controls which domain computes and which domain formats the result.
//! Default is `Auto:Auto` — existing parse_literal() routing is untouched.
//!
//! **Examples**:
//! - `"auto:auto"` — default behavior, no override
//! - `"binary:ternary"` — parse all inputs as Binary, convert result to Ternary
//! - `"decimal:symbolic"` — parse all inputs as Decimal, convert result to Symbolic

use core::cell::Cell;

/// Which domain computes the expression
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeMode {
    /// Existing parse_literal() routing — DEFAULT
    Auto,
    /// All inputs → Binary domain
    Binary,
    /// All inputs → Decimal domain
    Decimal,
    /// All inputs → Symbolic domain
    Symbolic,
    /// All inputs → Ternary domain
    Ternary,
}

/// Which domain formats the result
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputMode {
    /// Return as-is (whatever domain computed) — DEFAULT
    Auto,
    /// Convert result to Binary before returning
    Binary,
    /// Convert result to Decimal before returning
    Decimal,
    /// Convert result to Symbolic (rational) before returning
    Symbolic,
    /// Convert result to Ternary before returning
    Ternary,
}

/// Combined compute + output mode
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GmathMode {
    pub compute: ComputeMode,
    pub output: OutputMode,
}

impl Default for GmathMode {
    fn default() -> Self {
        GmathMode {
            compute: ComputeMode::Auto,
            output: OutputMode::Auto,
        }
    }
}

impl GmathMode {
    /// Parse "binary:ternary", "auto:auto", etc.
    pub fn from_str(s: &str) -> Result<Self, &'static str> {
        let mut parts = s.split(':');
        let compute_str = parts.next().ok_or("expected format 'compute:output'")?;
        let output_str = parts.next().ok_or("expected format 'compute:output'")?;
        if parts.next().is_some() {
            return Err("expected format 'compute:output'");
        }
        Ok(GmathMode {
            compute: match compute_str {
                "auto" => ComputeMode::Auto,
                "binary" => ComputeMode::Binary,
                "decimal" => ComputeMode::Decimal,
                "symbolic" => ComputeMode::Symbolic,
                "ternary" => ComputeMode::Ternary,
                _ => return Err("unknown compute mode"),
            },
            output: match output_str {
                "auto" => OutputMode::Auto,
                "binary" => OutputMode::Binary,
                "decimal" => OutputMode::Decimal,
                "symbolic" => OutputMode::Symbolic,
                "ternary" => OutputMode::Ternary,
                _ => return Err("unknown output mode"),
            },
        })
    }
}

// Thread-local mode state (mirrors EVALUATOR pattern)
thread_local! {
    static GMATH_MODE: Cell<GmathMode> = Cell::new(GmathMode::default());
}

/// Set the compute:output mode for the current thread
pub fn set_mode(mode: GmathMode) {
    GMATH_MODE.with(|m| m.set(mode));
}

/// Get the current compute:output mode for the current thread
pub fn get_mode() -> GmathMode {
    GMATH_MODE.with(|m| m.get())
}

/// Reset to default Auto:Auto mode
pub fn reset_mode() {
    GMATH_MODE.with(|m| m.set(GmathMode::default()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_mode() {
        let mode = GmathMode::default();
        assert_eq!(mode.compute, ComputeMode::Auto);
        assert_eq!(mode.output, OutputMode::Auto);
    }

    #[test]
    fn test_parse_auto_auto() {
        let mode = GmathMode::from_str("auto:auto").unwrap();
        assert_eq!(mode.compute, ComputeMode::Auto);
        assert_eq!(mode.output, OutputMode::Auto);
    }

    #[test]
    fn test_parse_binary_ternary() {
        let mode = GmathMode::from_str("binary:ternary").unwrap();
        assert_eq!(mode.compute, ComputeMode::Binary);
        assert_eq!(mode.output, OutputMode::Ternary);
    }

    #[test]
    fn test_parse_all_modes() {
        for compute in &["auto", "binary", "decimal", "symbolic", "ternary"] {
            for output in &["auto", "binary", "decimal", "symbolic", "ternary"] {
                let s = format!("{}:{}", compute, output);
                assert!(GmathMode::from_str(&s).is_ok(), "failed to parse: {}", s);
            }
        }
    }

    #[test]
    fn test_parse_invalid() {
        assert!(GmathMode::from_str("binary").is_err());
        assert!(GmathMode::from_str("binary:ternary:extra").is_err());
        assert!(GmathMode::from_str("unknown:auto").is_err());
        assert!(GmathMode::from_str("auto:unknown").is_err());
    }

    #[test]
    fn test_thread_local_set_get_reset() {
        reset_mode();
        assert_eq!(get_mode(), GmathMode::default());

        let mode = GmathMode::from_str("binary:symbolic").unwrap();
        set_mode(mode);
        assert_eq!(get_mode().compute, ComputeMode::Binary);
        assert_eq!(get_mode().output, OutputMode::Symbolic);

        reset_mode();
        assert_eq!(get_mode(), GmathMode::default());
    }
}
