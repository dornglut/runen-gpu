use super::super::contract_diagnostics::{GpuProgramContractCause, GpuProgramContractError};
use super::super::requirement_identity::hash_capability_requirements;
use super::super::{
    GpuEntryPointName, GpuExpectedFragmentOutputSignature, GpuExpectedVertexInputSignature,
    GpuPipelineLayoutDescriptor, GpuProgramDescriptor, GpuShaderStage, GpuSpecializationValueSet,
    compare_fragment_output_signatures, compare_vertex_input_signatures,
};
use super::GpuPipelineConfiguration;
use super::render_state::GpuRenderPipelineStateDescriptor;
use super::requirements::insert_pipeline_requirement;
use crate::{
    GpuCapabilityFeature, GpuCapabilityRequirement, GpuCapabilityRequirements,
    GpuRuntimeBindingSet, GpuRuntimeBindingValue,
};
use core::hash::Hash;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuRenderEntryPoints {
    vertex: GpuEntryPointName,
    fragment: Option<GpuEntryPointName>,
}

impl GpuRenderEntryPoints {
    pub fn new(vertex: GpuEntryPointName, fragment: Option<GpuEntryPointName>) -> Self {
        Self { vertex, fragment }
    }

    pub fn vertex(&self) -> &GpuEntryPointName {
        &self.vertex
    }

    pub fn fragment(&self) -> Option<&GpuEntryPointName> {
        self.fragment.as_ref()
    }

    fn diagnostic_label(&self) -> String {
        match self.fragment() {
            Some(fragment) => format!("vertex={}, fragment={fragment}", self.vertex()),
            None => format!("vertex={}, fragment=<none>", self.vertex()),
        }
    }
}

#[derive(Debug)]
struct GpuRenderPipelineDescriptorInner {
    program: GpuProgramDescriptor,
    entry_points: GpuRenderEntryPoints,
    state: GpuRenderPipelineStateDescriptor,
    layout: GpuPipelineLayoutDescriptor,
    specialization: GpuSpecializationValueSet,
    requirements: GpuCapabilityRequirements,
}

/// Complete backend-neutral render pipeline contract.
#[derive(Debug, Clone)]
pub struct GpuRenderPipelineDescriptor(Arc<GpuRenderPipelineDescriptorInner>);

impl GpuRenderPipelineDescriptor {
    pub fn new(
        program: GpuProgramDescriptor,
        entry_points: GpuRenderEntryPoints,
        state: GpuRenderPipelineStateDescriptor,
        configuration: GpuPipelineConfiguration,
    ) -> Result<Self, GpuProgramContractError> {
        let operation = "construct GPU render pipeline descriptor";
        let label = entry_points.diagnostic_label();

        if program
            .entry_point(GpuShaderStage::Vertex, entry_points.vertex())
            .is_none()
        {
            return Err(GpuProgramContractError::invalid(
                operation,
                label,
                GpuProgramContractCause::PipelineDescriptorInvalid,
                "select a vertex entry point declared by the admitted program",
            ));
        }

        if entry_points.fragment().is_some() != state.has_fragment_stage() {
            return Err(GpuProgramContractError::invalid(
                operation,
                entry_points.diagnostic_label(),
                GpuProgramContractCause::PipelineDescriptorInvalid,
                "select a fragment entry point exactly when fragment-output state is present",
            ));
        }

        if let Some(fragment) = entry_points.fragment() {
            if program
                .entry_point(GpuShaderStage::Fragment, fragment)
                .is_none()
            {
                return Err(GpuProgramContractError::invalid(
                    operation,
                    entry_points.diagnostic_label(),
                    GpuProgramContractCause::PipelineDescriptorInvalid,
                    "select a fragment entry point declared by the admitted program",
                ));
            }
        }

        validate_stage_io(&program, &entry_points, &state)?;

        let layout = GpuPipelineLayoutDescriptor::from_interface(program.interface())?;
        let (specialization, mut requirements) = configuration.resolve()?;
        insert_pipeline_requirement(
            operation,
            entry_points.diagnostic_label(),
            &mut requirements,
            GpuCapabilityRequirement::Required(GpuCapabilityFeature::RenderPipeline),
        )?;
        if state.depth_stencil().is_some() {
            insert_pipeline_requirement(
                operation,
                entry_points.diagnostic_label(),
                &mut requirements,
                GpuCapabilityRequirement::Required(GpuCapabilityFeature::DepthAttachment),
            )?;
        }
        for requirement in program.requirements().iter() {
            insert_pipeline_requirement(
                operation,
                entry_points.diagnostic_label(),
                &mut requirements,
                requirement,
            )?;
        }
        for requirement in specialization.requirements().iter() {
            insert_pipeline_requirement(
                operation,
                entry_points.diagnostic_label(),
                &mut requirements,
                requirement,
            )?;
        }

        Ok(Self(Arc::new(GpuRenderPipelineDescriptorInner {
            program,
            entry_points,
            state,
            layout,
            specialization,
            requirements,
        })))
    }

    pub fn program(&self) -> &GpuProgramDescriptor {
        &self.0.program
    }

    pub fn entry_points(&self) -> &GpuRenderEntryPoints {
        &self.0.entry_points
    }

    pub fn state(&self) -> &GpuRenderPipelineStateDescriptor {
        &self.0.state
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

    pub fn expected_vertex_input_signature(
        &self,
    ) -> Result<GpuExpectedVertexInputSignature, GpuProgramContractError> {
        self.0
            .state
            .vertex_input()
            .expected_signature(self.0.entry_points.vertex().clone())
    }

    pub fn expected_fragment_output_signature(
        &self,
    ) -> Result<Option<GpuExpectedFragmentOutputSignature>, GpuProgramContractError> {
        match (
            self.0.entry_points.fragment(),
            self.0.state.fragment_output(),
        ) {
            (Some(entry_point), Some(output)) => {
                output.expected_signature(entry_point.clone()).map(Some)
            }
            (None, None) => Ok(None),
            _ => unreachable!("render descriptor construction preserves fragment-stage parity"),
        }
    }

    pub fn is_same_record(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

fn validate_stage_io(
    program: &GpuProgramDescriptor,
    entry_points: &GpuRenderEntryPoints,
    state: &GpuRenderPipelineStateDescriptor,
) -> Result<(), GpuProgramContractError> {
    let operation = "construct GPU render pipeline descriptor";
    let expected_vertex = state
        .vertex_input()
        .expected_signature(entry_points.vertex().clone())?;
    let observed_vertex = program
        .observed_vertex_input_signature(entry_points.vertex())
        .ok_or_else(|| {
            GpuProgramContractError::invalid(
                operation,
                entry_points.diagnostic_label(),
                GpuProgramContractCause::PipelineStageIoMismatch,
                "use the compiler-observed vertex entry point admitted with this program",
            )
        })?;
    compare_vertex_input_signatures(&expected_vertex, observed_vertex)?;

    match (entry_points.fragment(), state.fragment_output()) {
        (Some(fragment), Some(output)) => {
            let expected_fragment = output.expected_signature(fragment.clone())?;
            let observed_fragment = program
                .observed_fragment_output_signature(fragment)
                .ok_or_else(|| {
                    GpuProgramContractError::invalid(
                        operation,
                        entry_points.diagnostic_label(),
                        GpuProgramContractCause::PipelineStageIoMismatch,
                        "use the compiler-observed fragment entry point admitted with this program",
                    )
                })?;
            compare_fragment_output_signatures(&expected_fragment, observed_fragment)
        }
        (None, None) => Ok(()),
        _ => unreachable!("render descriptor construction checks fragment-stage parity first"),
    }
}

impl PartialEq for GpuRenderPipelineDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.0.program == other.0.program
            && self.0.entry_points == other.0.entry_points
            && self.0.state == other.0.state
            && self.0.layout == other.0.layout
            && self.0.specialization == other.0.specialization
            && self.0.requirements == other.0.requirements
    }
}

impl Eq for GpuRenderPipelineDescriptor {}

impl Hash for GpuRenderPipelineDescriptor {
    fn hash<State: core::hash::Hasher>(&self, state: &mut State) {
        self.0.program.hash(state);
        self.0.entry_points.hash(state);
        self.0.state.hash(state);
        self.0.layout.hash(state);
        self.0.specialization.hash(state);
        hash_capability_requirements(&self.0.requirements, state);
    }
}
