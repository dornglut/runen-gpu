use super::super::contract_diagnostics::{GpuProgramContractCause, GpuProgramContractError};
use super::super::requirement_identity::hash_capability_requirements;
use super::super::{
    GpuEntryPointName, GpuPipelineLayoutDescriptor, GpuProgramDescriptor, GpuShaderStage,
    GpuSpecializationValueSet,
};
use super::GpuPipelineConfiguration;
use super::requirements::insert_pipeline_requirement;
use crate::{
    GpuCapabilityFeature, GpuCapabilityRequirement, GpuCapabilityRequirements,
    GpuRuntimeBindingSet, GpuRuntimeBindingValue,
};
use core::hash::Hash;
use std::sync::Arc;

#[derive(Debug)]
struct GpuComputePipelineDescriptorInner {
    program: GpuProgramDescriptor,
    entry_point: GpuEntryPointName,
    layout: GpuPipelineLayoutDescriptor,
    specialization: GpuSpecializationValueSet,
    requirements: GpuCapabilityRequirements,
}

/// Complete backend-neutral compute pipeline contract.
#[derive(Debug, Clone)]
pub struct GpuComputePipelineDescriptor(Arc<GpuComputePipelineDescriptorInner>);

impl GpuComputePipelineDescriptor {
    pub fn new(
        program: GpuProgramDescriptor,
        entry_point: GpuEntryPointName,
        configuration: GpuPipelineConfiguration,
    ) -> Result<Self, GpuProgramContractError> {
        if program
            .entry_point(GpuShaderStage::Compute, &entry_point)
            .is_none()
        {
            return Err(GpuProgramContractError::invalid(
                "construct GPU compute pipeline descriptor",
                entry_point.to_string(),
                GpuProgramContractCause::PipelineDescriptorInvalid,
                "select a compute entry point declared by the admitted program",
            ));
        }

        let layout = GpuPipelineLayoutDescriptor::from_interface(program.interface())?;
        let (specialization, mut requirements) = configuration.resolve()?;
        let operation = "construct GPU compute pipeline descriptor";
        insert_pipeline_requirement(
            operation,
            entry_point.to_string(),
            &mut requirements,
            GpuCapabilityRequirement::Required(GpuCapabilityFeature::Compute),
        )?;
        for requirement in program.requirements().iter() {
            insert_pipeline_requirement(
                operation,
                entry_point.to_string(),
                &mut requirements,
                requirement,
            )?;
        }
        for requirement in specialization.requirements().iter() {
            insert_pipeline_requirement(
                operation,
                entry_point.to_string(),
                &mut requirements,
                requirement,
            )?;
        }

        Ok(Self(Arc::new(GpuComputePipelineDescriptorInner {
            program,
            entry_point,
            layout,
            specialization,
            requirements,
        })))
    }

    pub fn program(&self) -> &GpuProgramDescriptor {
        &self.0.program
    }

    pub fn entry_point(&self) -> &GpuEntryPointName {
        &self.0.entry_point
    }

    pub fn layout(&self) -> &GpuPipelineLayoutDescriptor {
        &self.0.layout
    }

    /// Validates runtime binding values against this pipeline's exact derived layout.
    ///
    /// Callers state binding locations and resources; they do not need to clone and
    /// restate the layout already owned by the pipeline. Compatibility, effective
    /// access derivation, and writable-alias checks remain owned by
    /// [`GpuRuntimeBindingSet::new`].
    pub fn runtime_bindings(
        &self,
        values: impl IntoIterator<Item = GpuRuntimeBindingValue>,
    ) -> Result<GpuRuntimeBindingSet, GpuProgramContractError> {
        GpuRuntimeBindingSet::new(self.layout().clone(), values)
    }

    pub fn specialization(&self) -> &GpuSpecializationValueSet {
        &self.0.specialization
    }

    pub fn requirements(&self) -> &GpuCapabilityRequirements {
        &self.0.requirements
    }

    pub fn is_same_record(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl PartialEq for GpuComputePipelineDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.0.program == other.0.program
            && self.0.entry_point == other.0.entry_point
            && self.0.layout == other.0.layout
            && self.0.specialization == other.0.specialization
            && self.0.requirements == other.0.requirements
    }
}

impl Eq for GpuComputePipelineDescriptor {}

impl Hash for GpuComputePipelineDescriptor {
    fn hash<State: core::hash::Hasher>(&self, state: &mut State) {
        self.0.program.hash(state);
        self.0.entry_point.hash(state);
        self.0.layout.hash(state);
        self.0.specialization.hash(state);
        hash_capability_requirements(&self.0.requirements, state);
    }
}
