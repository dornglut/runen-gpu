use super::super::contract_diagnostics::{GpuProgramContractCause, GpuProgramContractError};
use core::fmt;
use core::str::FromStr;

const MAX_SPECIALIZATION_KEY_BYTES: usize = 256;

/// Validated source-level WGSL override identifier.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuSpecializationKey(String);

impl GpuSpecializationKey {
    pub fn new(value: impl Into<String>) -> Result<Self, GpuProgramContractError> {
        let value = value.into();
        let mut chars = value.chars();
        let valid_start = chars
            .next()
            .is_some_and(|character| character == '_' || character.is_ascii_alphabetic());
        let valid_rest =
            chars.all(|character| character == '_' || character.is_ascii_alphanumeric());
        if !valid_start || !valid_rest || value.len() > MAX_SPECIALIZATION_KEY_BYTES {
            return Err(GpuProgramContractError::invalid(
                "construct GPU specialization key",
                if value.is_empty() {
                    "<empty>"
                } else {
                    value.as_str()
                },
                GpuProgramContractCause::InvalidSpecializationKey,
                "provide a bounded WGSL-compatible identifier beginning with a letter or underscore",
            ));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl AsRef<str> for GpuSpecializationKey {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for GpuSpecializationKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl FromStr for GpuSpecializationKey {
    type Err = GpuProgramContractError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}
