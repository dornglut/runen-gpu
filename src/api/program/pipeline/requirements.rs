use super::super::contract_diagnostics::{GpuProgramContractCause, GpuProgramContractError};
use crate::{GpuCapabilityRequirement, GpuCapabilityRequirements};

pub(super) fn insert_pipeline_requirement(
    operation: &'static str,
    label: impl Into<String>,
    requirements: &mut GpuCapabilityRequirements,
    requirement: GpuCapabilityRequirement,
) -> Result<(), GpuProgramContractError> {
    requirements.insert(requirement).map_err(|error| {
        GpuProgramContractError::invalid(
            operation,
            format!("{}: {error}", label.into()),
            GpuProgramContractCause::PipelineDescriptorInvalid,
            "remove conflicting capability requirements from the pipeline inputs",
        )
    })
}
