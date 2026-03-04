//! Cross-domain conversion without floating-point contamination

use crate::fixed_point::domains::symbolic::rational::RationalNumber;

/// Parse error for string conversion
#[derive(Debug, Clone)]
pub enum ParseError {
    InvalidFormat(String),
    InvalidNumerator,
    InvalidDenominator,
    UnrecognizedInput(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            ParseError::InvalidNumerator => write!(f, "Invalid numerator"),
            ParseError::InvalidDenominator => write!(f, "Invalid denominator"),
            ParseError::UnrecognizedInput(input) => write!(f, "Unrecognized input: {}", input),
        }
    }
}

impl std::error::Error for ParseError {}

/// Direct conversion implementations for common types
impl RationalNumber {
    /// Create from integer
    pub fn from_integer(value: i128) -> Self {
        Self::new(value, 1)
    }

    /// Create from decimal string without floating point
    pub fn from_decimal_str(s: &str) -> Result<Self, ParseError> {
        // Split on decimal point
        let parts: Vec<&str> = s.split('.').collect();

        match parts.len() {
            1 => {
                // Integer only
                if let Ok(num) = parts[0].parse::<i128>() {
                    Ok(Self::from_integer(num))
                } else {
                    Err(ParseError::InvalidNumerator)
                }
            }
            2 => {
                // Has decimal part
                let integer_part = parts[0].parse::<i128>()
                    .map_err(|_| ParseError::InvalidNumerator)?;
                let decimal_digits = parts[1].len() as u8;
                let decimal_part = parts[1].parse::<i128>()
                    .map_err(|_| ParseError::InvalidNumerator)?;

                // Construct as (integer * 10^decimals + decimal) / 10^decimals
                let scale = 10i128.pow(decimal_digits as u32);
                let numerator = integer_part * scale + decimal_part;

                Ok(Self::new(numerator, scale as u128))
            }
            _ => Err(ParseError::InvalidFormat("Too many decimal points".to_string()))
        }
    }

    /// Create from fraction string (e.g., "22/7")
    pub fn from_fraction_str(s: &str) -> Result<Self, ParseError> {
        let parts: Vec<&str> = s.split('/').collect();

        if parts.len() != 2 {
            return Err(ParseError::InvalidFormat("Expected format: numerator/denominator".to_string()));
        }

        let num = parts[0].parse::<i128>()
            .map_err(|_| ParseError::InvalidNumerator)?;
        let den = parts[1].parse::<u128>()
            .map_err(|_| ParseError::InvalidDenominator)?;

        if den == 0 {
            return Err(ParseError::InvalidFormat("Division by zero".to_string()));
        }

        Ok(Self::new(num, den))
    }

    /// Try to convert to i128 if exactly representable
    pub fn to_i128(&self) -> Option<i128> {
        if self.is_integer() {
            self.numerator_i128()
        } else {
            None
        }
    }

    /// High-precision π using precomputed rational approximation
    pub fn pi_high_precision() -> Self {
        // 355/113 is accurate to 6 decimal places
        Self::new(355, 113)
    }

    /// High-precision e using precomputed rational approximation
    pub fn e_high_precision() -> Self {
        // 2718/1000 is a reasonable approximation
        Self::new(2718, 1000)
    }
}
