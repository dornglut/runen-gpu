use super::analysis::analyze_program;
use super::contract_diagnostics::{GpuProgramContractCause, GpuProgramContractError};
use super::entry_point::{GpuEntryPointDescriptor, GpuEntryPointName};
use super::interface::{
    GpuBindingClass, GpuBindingLayoutRefinement, GpuProgramInterfaceDescriptor, GpuShaderStage,
};
use super::requirement_identity::hash_capability_requirements;
use super::source::GpuAdmittedProgramSource;
use super::stage_io::{GpuObservedFragmentOutputSignature, GpuObservedVertexInputSignature};
use crate::{GpuCapabilityFeature, GpuCapabilityRequirement, GpuCapabilityRequirements};
use core::hash::Hash;
use std::sync::Arc;

#[derive(Debug)]
struct GpuProgramDescriptorInner {
    source: GpuAdmittedProgramSource,
    interface: GpuProgramInterfaceDescriptor,
    entry_points: Vec<GpuEntryPointDescriptor>,
    vertex_inputs: Vec<GpuObservedVertexInputSignature>,
    fragment_outputs: Vec<GpuObservedFragmentOutputSignature>,
    requirements: GpuCapabilityRequirements,
}

/// Immutable admitted program contract retained by later realization records.
///
/// Canonical WGSL is parsed and validated during construction. Shader-defined entry-point,
/// resource-interface, and stage-IO facts are compiler-derived; callers provide only selected
/// entry-point names and sparse host/layout refinements.
///
/// The record cannot be fabricated without source admission and validation:
///
/// ```compile_fail
/// use runen_gpu::GpuProgramDescriptor;
///
/// let _fabricated = GpuProgramDescriptor(());
/// ```
#[derive(Debug, Clone)]
pub struct GpuProgramDescriptor(Arc<GpuProgramDescriptorInner>);

impl GpuProgramDescriptor {
    pub fn new(
        source: GpuAdmittedProgramSource,
        selected_entry_points: impl IntoIterator<Item = GpuEntryPointName>,
        refinements: impl IntoIterator<Item = GpuBindingLayoutRefinement>,
    ) -> Result<Self, GpuProgramContractError> {
        let mut analysis = analyze_program(&source, selected_entry_points, refinements)?;
        analysis.entry_points.sort_by(|left, right| {
            left.stage()
                .cmp(&right.stage())
                .then_with(|| left.name().cmp(right.name()))
        });

        let mut requirements = GpuCapabilityRequirements::new();
        for binding in analysis.interface.bindings() {
            if binding.kind().class() == GpuBindingClass::StorageTexture {
                insert_interface_requirement(
                    &mut requirements,
                    &source,
                    GpuCapabilityFeature::StorageTexture,
                )?;
            }
            if binding.array_count().is_some() {
                for feature in fixed_array_capabilities(binding.kind().class()) {
                    insert_interface_requirement(&mut requirements, &source, *feature)?;
                }
            }
        }

        Ok(Self(Arc::new(GpuProgramDescriptorInner {
            source,
            interface: analysis.interface,
            entry_points: analysis.entry_points,
            vertex_inputs: analysis.vertex_inputs,
            fragment_outputs: analysis.fragment_outputs,
            requirements,
        })))
    }

    pub fn source(&self) -> &GpuAdmittedProgramSource {
        &self.0.source
    }

    pub fn interface(&self) -> &GpuProgramInterfaceDescriptor {
        &self.0.interface
    }

    pub fn entry_points(&self) -> impl ExactSizeIterator<Item = &GpuEntryPointDescriptor> {
        self.0.entry_points.iter()
    }

    pub fn entry_point(
        &self,
        stage: GpuShaderStage,
        name: &GpuEntryPointName,
    ) -> Option<&GpuEntryPointDescriptor> {
        self.0
            .entry_points
            .binary_search_by(|entry_point| {
                entry_point
                    .stage()
                    .cmp(&stage)
                    .then_with(|| entry_point.name().cmp(name))
            })
            .ok()
            .map(|index| &self.0.entry_points[index])
    }

    pub(crate) fn observed_vertex_input_signature(
        &self,
        name: &GpuEntryPointName,
    ) -> Option<&GpuObservedVertexInputSignature> {
        self.0
            .vertex_inputs
            .binary_search_by(|signature| signature.entry_point().cmp(name))
            .ok()
            .map(|index| &self.0.vertex_inputs[index])
    }

    pub(crate) fn observed_fragment_output_signature(
        &self,
        name: &GpuEntryPointName,
    ) -> Option<&GpuObservedFragmentOutputSignature> {
        self.0
            .fragment_outputs
            .binary_search_by(|signature| signature.entry_point().cmp(name))
            .ok()
            .map(|index| &self.0.fragment_outputs[index])
    }

    pub fn requirements(&self) -> &GpuCapabilityRequirements {
        &self.0.requirements
    }

    pub fn is_same_record(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

fn fixed_array_capabilities(class: GpuBindingClass) -> &'static [GpuCapabilityFeature] {
    match class {
        GpuBindingClass::UniformBuffer => &[
            GpuCapabilityFeature::BufferBindingArray,
            GpuCapabilityFeature::UniformBufferBindingArray,
        ],
        GpuBindingClass::StorageBuffer => &[
            GpuCapabilityFeature::BufferBindingArray,
            GpuCapabilityFeature::StorageResourceBindingArray,
        ],
        GpuBindingClass::SampledTexture | GpuBindingClass::Sampler => {
            &[GpuCapabilityFeature::TextureBindingArray]
        }
        GpuBindingClass::StorageTexture => &[
            GpuCapabilityFeature::TextureBindingArray,
            GpuCapabilityFeature::StorageResourceBindingArray,
        ],
    }
}

fn insert_interface_requirement(
    requirements: &mut GpuCapabilityRequirements,
    source: &GpuAdmittedProgramSource,
    feature: GpuCapabilityFeature,
) -> Result<(), GpuProgramContractError> {
    requirements
        .insert(GpuCapabilityRequirement::Required(feature))
        .map_err(|error| {
            GpuProgramContractError::invalid(
                "construct admitted GPU program",
                format!("{}: {error}", source.identity().diagnostic_label()),
                GpuProgramContractCause::ProgramInterfaceMismatch,
                "remove conflicting capability requirements implied by the program interface",
            )
        })
}

impl PartialEq for GpuProgramDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.0.source == other.0.source
            && self.0.interface == other.0.interface
            && self.0.entry_points == other.0.entry_points
            && self.0.vertex_inputs == other.0.vertex_inputs
            && self.0.fragment_outputs == other.0.fragment_outputs
            && self.0.requirements == other.0.requirements
    }
}

impl Eq for GpuProgramDescriptor {}

impl Hash for GpuProgramDescriptor {
    fn hash<State: core::hash::Hasher>(&self, state: &mut State) {
        self.0.source.hash(state);
        self.0.interface.hash(state);
        self.0.entry_points.hash(state);
        self.0.vertex_inputs.hash(state);
        self.0.fragment_outputs.hash(state);
        hash_capability_requirements(&self.0.requirements, state);
    }
}
