use super::super::super::contract_diagnostics::{GpuProgramContractCause, GpuProgramContractError};

/// Typed shader stage used by entry points and binding visibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuShaderStage {
    Compute,
    Vertex,
    Fragment,
}

/// Non-empty normalized set of shader stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuShaderStages(u8);

impl GpuShaderStages {
    const COMPUTE_BIT: u8 = 1 << 0;
    const VERTEX_BIT: u8 = 1 << 1;
    const FRAGMENT_BIT: u8 = 1 << 2;

    pub fn new(
        stages: impl IntoIterator<Item = GpuShaderStage>,
    ) -> Result<Self, GpuProgramContractError> {
        let mut bits = 0u8;
        for stage in stages {
            bits |= match stage {
                GpuShaderStage::Compute => Self::COMPUTE_BIT,
                GpuShaderStage::Vertex => Self::VERTEX_BIT,
                GpuShaderStage::Fragment => Self::FRAGMENT_BIT,
            };
        }
        if bits == 0 {
            return Err(GpuProgramContractError::invalid(
                "construct GPU shader stage visibility",
                "<empty>",
                GpuProgramContractCause::EmptyStageVisibility,
                "declare at least one typed shader stage",
            ));
        }
        Ok(Self(bits))
    }

    pub const fn one(stage: GpuShaderStage) -> Self {
        Self(match stage {
            GpuShaderStage::Compute => Self::COMPUTE_BIT,
            GpuShaderStage::Vertex => Self::VERTEX_BIT,
            GpuShaderStage::Fragment => Self::FRAGMENT_BIT,
        })
    }

    pub const fn contains(self, stage: GpuShaderStage) -> bool {
        let bit = match stage {
            GpuShaderStage::Compute => Self::COMPUTE_BIT,
            GpuShaderStage::Vertex => Self::VERTEX_BIT,
            GpuShaderStage::Fragment => Self::FRAGMENT_BIT,
        };
        self.0 & bit != 0
    }

    pub fn iter(self) -> impl Iterator<Item = GpuShaderStage> {
        [
            GpuShaderStage::Compute,
            GpuShaderStage::Vertex,
            GpuShaderStage::Fragment,
        ]
        .into_iter()
        .filter(move |stage| self.contains(*stage))
    }
}
