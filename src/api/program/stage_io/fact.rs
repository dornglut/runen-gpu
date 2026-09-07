use super::super::contract_diagnostics::{GpuProgramContractCause, GpuProgramContractError};
use core::num::NonZeroU8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuShaderIoScalarClass {
    Float,
    Sint,
    Uint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuShaderIoValueType {
    scalar_class: GpuShaderIoScalarClass,
    vector_width: NonZeroU8,
}

impl GpuShaderIoValueType {
    pub fn try_new(
        scalar_class: GpuShaderIoScalarClass,
        vector_width: u8,
    ) -> Result<Self, GpuProgramContractError> {
        let requested_width = vector_width;
        let vector_width = NonZeroU8::new(vector_width).filter(|width| width.get() <= 4);
        let Some(vector_width) = vector_width else {
            return Err(GpuProgramContractError::invalid(
                "construct GPU shader-stage IO value type",
                format!("vector_width={requested_width}"),
                GpuProgramContractCause::StageIoSignatureInvalid,
                "use one scalar component or a vector width from two through four",
            ));
        };
        Ok(Self {
            scalar_class,
            vector_width,
        })
    }

    pub const fn scalar_class(self) -> GpuShaderIoScalarClass {
        self.scalar_class
    }

    pub const fn vector_width(self) -> NonZeroU8 {
        self.vector_width
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuShaderIoLocation {
    location: u32,
    value_type: GpuShaderIoValueType,
}

impl GpuShaderIoLocation {
    pub const fn new(location: u32, value_type: GpuShaderIoValueType) -> Self {
        Self {
            location,
            value_type,
        }
    }

    pub const fn location(self) -> u32 {
        self.location
    }

    pub const fn value_type(self) -> GpuShaderIoValueType {
        self.value_type
    }
}
