use super::contract_diagnostics::GpuProgramContractError;
use super::specialization::{GpuSpecializationSchema, GpuSpecializationValueSet};
use crate::GpuCapabilityRequirements;

mod compute;
mod render;
mod render_state;
mod requirements;

pub use compute::*;
pub use render::*;
pub use render_state::*;

/// Optional caller-controlled pipeline semantics.
///
/// The pipeline layout is never caller-controlled: compute and render descriptors derive it from
/// the admitted program interface. `Default` represents the ordinary case with no specialization
/// contract and no capability requirements beyond those derived by the pipeline itself.
#[derive(Debug, Clone, Default)]
pub struct GpuPipelineConfiguration {
    specialization: Option<GpuSpecializationValueSet>,
    additional_requirements: Option<GpuCapabilityRequirements>,
}

impl GpuPipelineConfiguration {
    pub fn new(
        specialization: Option<GpuSpecializationValueSet>,
        additional_requirements: Option<GpuCapabilityRequirements>,
    ) -> Self {
        Self {
            specialization,
            additional_requirements,
        }
    }

    fn resolve(
        self,
    ) -> Result<(GpuSpecializationValueSet, GpuCapabilityRequirements), GpuProgramContractError>
    {
        let specialization = match self.specialization {
            Some(specialization) => specialization,
            None => {
                let schema = GpuSpecializationSchema::new([])?;
                GpuSpecializationValueSet::new(schema, [])?
            }
        };
        Ok((
            specialization,
            self.additional_requirements.unwrap_or_default(),
        ))
    }
}
