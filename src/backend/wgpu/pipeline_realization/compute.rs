use super::G4C3_WGPU_PIPELINE_COMPATIBILITY_REVISION;
use super::diagnostics::PipelineCacheFamily;
use super::publication::{ensure_available, scoped_create};
use super::records::ComputePipelineRealizationRecord;
use super::registry::{self, InFlightOutcome, Reservation};
use crate::{
    GpuAlignmentFacts, GpuAlignmentKind, GpuCapabilityAdmission, GpuCapabilityFeature,
    GpuComputePipelineDescriptor, GpuContext, GpuContextAffinity, GpuLimits,
    GpuPipelineLayoutDescriptor, GpuPipelineRealizationError, GpuPipelineRealizationErrorCategory,
    GpuProgramBindingRealizationError, GpuProgramBindingRealizationErrorCategory,
    GpuRealizedComputePipeline, GpuRealizedPipelineLayout, GpuRealizedProgram, GpuShaderStage,
    GpuSpecializationValue,
};
use std::sync::Arc;
use wgpu::{ComputePipelineDescriptor, PipelineCompilationOptions};

pub(super) type ComputeRecord = ComputePipelineRealizationRecord;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct ComputePipelineRequestKey {
    affinity: GpuContextAffinity,
    descriptor: GpuComputePipelineDescriptor,
    enabled_features: Vec<GpuCapabilityFeature>,
    device_limits: GpuLimits,
    device_alignments: GpuAlignmentFacts,
    workload_limits: GpuLimits,
    workload_alignment_maximums: Vec<(GpuAlignmentKind, u64)>,
    wgpu_pipeline_compatibility_revision: u32,
}

impl ComputePipelineRequestKey {
    fn new(context: &GpuContext, descriptor: &GpuComputePipelineDescriptor) -> Self {
        let device = context.device_facts();
        Self {
            affinity: context.affinity(),
            descriptor: descriptor.clone(),
            enabled_features: device.enabled_features().collect(),
            device_limits: device.device_limits().values(),
            device_alignments: device.device_limits().alignments(),
            workload_limits: device.workload_budget().limits(),
            workload_alignment_maximums: device.workload_budget().alignment_maximums().collect(),
            wgpu_pipeline_compatibility_revision: G4C3_WGPU_PIPELINE_COMPATIBILITY_REVISION,
        }
    }

    fn matches_record(&self, record: &ComputeRecord) -> bool {
        record.affinity() == self.affinity
            && record.descriptor() == &self.descriptor
            && record.program.affinity() == self.affinity
            && record.program.descriptor() == self.descriptor.program()
            && record.layout.affinity() == self.affinity
            && record.layout.descriptor() == self.descriptor.layout()
    }
}

impl GpuContext {
    /// Realizes one complete accepted G4B compute-pipeline request against exact G4C2
    /// program and pipeline-layout realizations.
    pub async fn realize_compute_pipeline(
        &self,
        descriptor: &GpuComputePipelineDescriptor,
        program: &GpuRealizedProgram,
        layout: &GpuRealizedPipelineLayout,
    ) -> Result<GpuRealizedComputePipeline, GpuPipelineRealizationError> {
        let request = compute_request_name(descriptor);
        validate_compute_descriptor(descriptor, request.clone())?;
        validate_dependency_affinity(self.affinity(), program.affinity(), request.clone())?;
        validate_dependency_affinity(self.affinity(), layout.affinity(), request.clone())?;
        if descriptor.program() != program.descriptor()
            || descriptor.layout() != layout.descriptor()
        {
            return Err(GpuPipelineRealizationError::new(
                GpuPipelineRealizationErrorCategory::ProgramInterfaceMismatch,
                request,
                "the compute descriptor does not name the exact realized program and pipeline layout",
            ));
        }

        let canonical_program = self
            .realize_program(descriptor.program())
            .await
            .map_err(|error| map_dependency_error(compute_request_name(descriptor), error))?;
        if !canonical_program.is_same_record(program) {
            return Err(GpuPipelineRealizationError::new(
                GpuPipelineRealizationErrorCategory::UnknownProgramOrLayoutRealization,
                compute_request_name(descriptor),
                "the supplied program handle is not the authoritative G4C2 record for this request",
            ));
        }
        let canonical_layout = self
            .realize_pipeline_layout(descriptor.layout())
            .await
            .map_err(|error| map_dependency_error(compute_request_name(descriptor), error))?;
        if !canonical_layout.is_same_record(layout) {
            return Err(GpuPipelineRealizationError::new(
                GpuPipelineRealizationErrorCategory::UnknownProgramOrLayoutRealization,
                compute_request_name(descriptor),
                "the supplied pipeline-layout handle is not the authoritative G4C2 record for this request",
            ));
        }

        GpuCapabilityAdmission::evaluate(
            compute_request_name(descriptor),
            descriptor.requirements(),
            self.adapter_facts().supported(),
            self.device_facts().enabled_features(),
        )
        .map_err(|error| {
            GpuPipelineRealizationError::new(
                GpuPipelineRealizationErrorCategory::RequirementNotAdmitted,
                compute_request_name(descriptor),
                error.to_string(),
            )
        })?;

        loop {
            let request = compute_request_name(descriptor);
            ensure_available(&self.backend.pipeline_realization, request.clone())?;
            let key = ComputePipelineRequestKey::new(self, descriptor);
            let (reservation, observation) = registry::reserve(
                &self.backend.pipeline_realization.compute,
                self.backend.pipeline_realization.max_records,
                key,
                request.clone(),
                ComputePipelineRequestKey::matches_record,
            )?;
            self.backend.pipeline_realization.observe_cache(
                PipelineCacheFamily::Compute,
                observation,
                &request,
            );
            match reservation {
                Reservation::Ready(record) => {
                    return Ok(GpuRealizedComputePipeline::from_record(record));
                }
                Reservation::Waiter(attempt) => match attempt.wait().await {
                    InFlightOutcome::Complete(outcome) => {
                        return outcome.map(GpuRealizedComputePipeline::from_record);
                    }
                    InFlightOutcome::Abandoned => continue,
                    InFlightOutcome::Pending => {
                        unreachable!("wait never returns a pending pipeline attempt")
                    }
                },
                Reservation::Owner(owner) => {
                    let outcome = self
                        .realize_compute_pipeline_owner(descriptor, program, layout)
                        .await;
                    return owner
                        .finish(outcome)
                        .map(GpuRealizedComputePipeline::from_record);
                }
            }
        }
    }

    async fn realize_compute_pipeline_owner(
        &self,
        descriptor: &GpuComputePipelineDescriptor,
        program: &GpuRealizedProgram,
        layout: &GpuRealizedPipelineLayout,
    ) -> Result<Arc<ComputeRecord>, GpuPipelineRealizationError> {
        let constants = wgpu_specialization_constants(descriptor);
        let request = compute_request_name(descriptor);
        let object = scoped_create(
            &self.backend.device,
            &self.backend.pipeline_realization,
            request,
            || {
                self.backend
                    .device
                    .create_compute_pipeline(&ComputePipelineDescriptor {
                        label: Some("runengpu-compute-pipeline"),
                        layout: Some(layout.record.wgpu_object()),
                        module: program.record.wgpu_object(),
                        entry_point: Some(descriptor.entry_point().as_str()),
                        compilation_options: PipelineCompilationOptions {
                            constants: constants.as_slice(),
                            ..PipelineCompilationOptions::default()
                        },
                        cache: None,
                    })
            },
        )
        .await?;
        Ok(Arc::new(ComputeRecord {
            affinity: self.affinity(),
            descriptor: descriptor.clone(),
            object,
            program: Arc::clone(&program.record),
            layout: Arc::clone(&layout.record),
        }))
    }
}

fn validate_compute_descriptor(
    descriptor: &GpuComputePipelineDescriptor,
    request: String,
) -> Result<(), GpuPipelineRealizationError> {
    if descriptor
        .program()
        .entry_point(GpuShaderStage::Compute, descriptor.entry_point())
        .is_none()
    {
        return Err(GpuPipelineRealizationError::new(
            GpuPipelineRealizationErrorCategory::EntryPointStageMismatch,
            request,
            "the compute entry point is absent from the descriptor-owned program",
        ));
    }
    let expected_layout = GpuPipelineLayoutDescriptor::from_interface(
        descriptor.program().interface(),
    )
    .map_err(|error| {
        GpuPipelineRealizationError::new(
            GpuPipelineRealizationErrorCategory::PipelineDescriptorInvalid,
            request.clone(),
            error.to_string(),
        )
    })?;
    if descriptor.layout() != &expected_layout {
        return Err(GpuPipelineRealizationError::new(
            GpuPipelineRealizationErrorCategory::ProgramInterfaceMismatch,
            request,
            "the compute pipeline layout does not match the program's explicit resource interface",
        ));
    }
    descriptor
        .specialization()
        .validate_override_support(true)
        .map_err(|error| {
            GpuPipelineRealizationError::new(
                GpuPipelineRealizationErrorCategory::PipelineDescriptorInvalid,
                compute_request_name(descriptor),
                error.to_string(),
            )
        })
}

fn validate_dependency_affinity(
    expected: GpuContextAffinity,
    observed: GpuContextAffinity,
    request: String,
) -> Result<(), GpuPipelineRealizationError> {
    if observed.context() != expected.context() {
        return Err(GpuPipelineRealizationError::affinity(
            GpuPipelineRealizationErrorCategory::ForeignContext,
            request,
            expected,
            observed,
        ));
    }
    if observed.generation() != expected.generation() {
        return Err(GpuPipelineRealizationError::affinity(
            GpuPipelineRealizationErrorCategory::StaleDeviceGeneration,
            request,
            expected,
            observed,
        ));
    }
    Ok(())
}

fn map_dependency_error(
    request: String,
    error: GpuProgramBindingRealizationError,
) -> GpuPipelineRealizationError {
    let category = match error.category() {
        GpuProgramBindingRealizationErrorCategory::ForeignContext => {
            GpuPipelineRealizationErrorCategory::ForeignContext
        }
        GpuProgramBindingRealizationErrorCategory::StaleDeviceGeneration => {
            GpuPipelineRealizationErrorCategory::StaleDeviceGeneration
        }
        GpuProgramBindingRealizationErrorCategory::BackendResourceExhaustion => {
            GpuPipelineRealizationErrorCategory::BackendResourceExhaustion
        }
        GpuProgramBindingRealizationErrorCategory::ContextOrDeviceUnavailableOrLost => {
            GpuPipelineRealizationErrorCategory::ContextOrDeviceUnavailableOrLost
        }
        _ => GpuPipelineRealizationErrorCategory::UnknownProgramOrLayoutRealization,
    };
    GpuPipelineRealizationError::new(category, request, error.to_string())
}

fn compute_request_name(descriptor: &GpuComputePipelineDescriptor) -> String {
    format!("compute pipeline entry={}", descriptor.entry_point())
}

fn wgpu_specialization_constants(descriptor: &GpuComputePipelineDescriptor) -> Vec<(&str, f64)> {
    descriptor
        .specialization()
        .entries()
        .map(|entry| {
            let value = match entry.value() {
                GpuSpecializationValue::Bool(value) => {
                    if value {
                        1.0
                    } else {
                        0.0
                    }
                }
                GpuSpecializationValue::U32(value) => f64::from(value),
                GpuSpecializationValue::I32(value) => f64::from(value),
                GpuSpecializationValue::F32(value) => f64::from(value.get()),
            };
            (entry.key().as_str(), value)
        })
        .collect()
}
