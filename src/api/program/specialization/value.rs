use super::super::contract_diagnostics::{GpuProgramContractCause, GpuProgramContractError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuSpecializationValueType {
    Bool,
    U32,
    I32,
    F32,
}

/// Finite canonical F32 specialization value.
///
/// Negative zero is stored as positive zero. NaN and infinity are rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuSpecializationF32(u32);

impl GpuSpecializationF32 {
    pub fn try_new(value: f32) -> Result<Self, GpuProgramContractError> {
        if !value.is_finite() {
            return Err(GpuProgramContractError::invalid(
                "construct GPU F32 specialization value",
                format!("value={value:?}"),
                GpuProgramContractCause::InvalidSpecializationValue,
                "provide a finite F32 value",
            ));
        }
        let normalized = if value == 0.0 { 0.0 } else { value };
        Ok(Self(normalized.to_bits()))
    }

    pub const fn canonical_bits(self) -> u32 {
        self.0
    }

    pub fn get(self) -> f32 {
        f32::from_bits(self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuSpecializationValue {
    Bool(bool),
    U32(u32),
    I32(i32),
    F32(GpuSpecializationF32),
}

impl GpuSpecializationValue {
    pub const fn value_type(self) -> GpuSpecializationValueType {
        match self {
            Self::Bool(_) => GpuSpecializationValueType::Bool,
            Self::U32(_) => GpuSpecializationValueType::U32,
            Self::I32(_) => GpuSpecializationValueType::I32,
            Self::F32(_) => GpuSpecializationValueType::F32,
        }
    }
}

impl From<bool> for GpuSpecializationValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<u32> for GpuSpecializationValue {
    fn from(value: u32) -> Self {
        Self::U32(value)
    }
}

impl From<i32> for GpuSpecializationValue {
    fn from(value: i32) -> Self {
        Self::I32(value)
    }
}

impl From<GpuSpecializationF32> for GpuSpecializationValue {
    fn from(value: GpuSpecializationF32) -> Self {
        Self::F32(value)
    }
}
