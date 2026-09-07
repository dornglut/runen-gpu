//! Context/device-generation-bound G4C3 pipeline realization.

mod compute;
mod diagnostics;
mod publication;
mod records;
mod registry;
mod render;
mod render_lowering;
mod render_mapping;
mod render_validation;

pub(crate) use records::{ComputePipelineRealizationRecord, RenderPipelineRealizationRecord};

use super::program_binding_realization::ProgramBindingRealizationState;
use super::{WgpuDeviceHealth, WgpuErrorAttributionGate};
use crate::{
    GpuContextAffinity, GpuPipelineRealizationError, GpuPipelineRealizationErrorCategory,
    GpuProgramBindingRealizationError, GpuProgramBindingRealizationErrorCategory,
    GpuRealizedComputePipeline, GpuRealizedRenderPipeline,
};
use compute::{ComputePipelineRequestKey, ComputeRecord};
use diagnostics::PipelineCacheDiagnosticRegistry;
use registry::SingleFlightRegistry;
use render::RenderRecord;
use render_validation::RenderPipelineRequestKey;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

const DEFAULT_MAX_PIPELINE_REALIZATION_RECORDS: NonZeroUsize =
    NonZeroUsize::new(16_384).expect("the default pipeline realization-record bound is nonzero");
const G4C3_WGPU_PIPELINE_COMPATIBILITY_REVISION: u32 = 2;

/// The sole private G4C3 owner for one admitted WGPU device generation.
pub(crate) struct PipelineRealizationState {
    affinity: GpuContextAffinity,
    max_records: NonZeroUsize,
    compute: Arc<Mutex<SingleFlightRegistry<ComputePipelineRequestKey, ComputeRecord>>>,
    render: Arc<Mutex<SingleFlightRegistry<RenderPipelineRequestKey, RenderRecord>>>,
    cache_diagnostics: Mutex<PipelineCacheDiagnosticRegistry>,
    health: Arc<WgpuDeviceHealth>,
    error_attribution_gate: Arc<WgpuErrorAttributionGate>,
}

impl PipelineRealizationState {
    pub(crate) fn new(
        affinity: GpuContextAffinity,
        health: Arc<WgpuDeviceHealth>,
        error_attribution_gate: Arc<WgpuErrorAttributionGate>,
    ) -> Self {
        Self {
            affinity,
            max_records: DEFAULT_MAX_PIPELINE_REALIZATION_RECORDS,
            compute: Arc::new(Mutex::new(SingleFlightRegistry::default())),
            render: Arc::new(Mutex::new(SingleFlightRegistry::default())),
            cache_diagnostics: Mutex::new(PipelineCacheDiagnosticRegistry::default()),
            health,
            error_attribution_gate,
        }
    }

    /// Validates one already-issued opaque compute-pipeline handle for the lexical execution
    /// bridge and lends only its private WGPU object to the supplied call. Pipeline lookup-cache
    /// residency is deliberately not consulted: the retained G4C3 record is execution authority.
    pub(crate) fn with_execution_compute_pipeline<R>(
        &self,
        pipeline: &GpuRealizedComputePipeline,
        program_binding_state: &ProgramBindingRealizationState,
        operation: impl FnOnce(&wgpu::ComputePipeline) -> R,
    ) -> Result<R, GpuPipelineRealizationError> {
        let request = "current render compute-pipeline execution";
        let record = &pipeline.record;
        self.validate_execution_affinity(request, record.affinity())?;
        publication::ensure_available(self, request)?;
        self.validate_compute_execution_record(request, record, program_binding_state)?;
        Ok(operation(&record.object))
    }

    /// Validates a lexical set of render pipelines before one render pass that may switch between
    /// them. This is required by UI encoding and still exposes no reusable raw-pipeline authority.
    pub(crate) fn with_execution_render_pipelines<R>(
        &self,
        pipelines: &[&GpuRealizedRenderPipeline],
        program_binding_state: &ProgramBindingRealizationState,
        operation: impl FnOnce(&[&wgpu::RenderPipeline]) -> R,
    ) -> Result<R, GpuPipelineRealizationError> {
        let request = "current render render-pipeline-set execution";
        publication::ensure_available(self, request)?;
        let mut objects = Vec::with_capacity(pipelines.len());
        for pipeline in pipelines {
            let record = &pipeline.record;
            self.validate_execution_affinity(request, record.affinity())?;
            self.validate_render_execution_record(request, record, program_binding_state)?;
            objects.push(&record.object);
        }
        Ok(operation(&objects))
    }

    fn validate_compute_execution_record(
        &self,
        request: &'static str,
        record: &Arc<ComputePipelineRealizationRecord>,
        program_binding_state: &ProgramBindingRealizationState,
    ) -> Result<(), GpuPipelineRealizationError> {
        if record.program.affinity() != record.affinity()
            || record.layout.affinity() != record.affinity()
            || record.program.descriptor() != record.descriptor().program()
            || record.layout.descriptor() != record.descriptor().layout()
        {
            return Err(execution_authority_violation(
                request,
                "the retained compute-pipeline record no longer agrees with its exact G4C2 dependencies",
            ));
        }
        program_binding_state
            .validate_execution_program(&record.program)
            .map_err(|error| map_execution_dependency_error(request, error))?;
        program_binding_state
            .validate_execution_pipeline_layout(&record.layout)
            .map_err(|error| map_execution_dependency_error(request, error))?;
        Ok(())
    }

    fn validate_render_execution_record(
        &self,
        request: &'static str,
        record: &Arc<RenderPipelineRealizationRecord>,
        program_binding_state: &ProgramBindingRealizationState,
    ) -> Result<(), GpuPipelineRealizationError> {
        if record.program.affinity() != record.affinity()
            || record.layout.affinity() != record.affinity()
            || record.program.descriptor() != record.descriptor().program()
            || record.layout.descriptor() != record.descriptor().layout()
        {
            return Err(execution_authority_violation(
                request,
                "the retained render-pipeline record no longer agrees with its exact G4C2 dependencies",
            ));
        }
        program_binding_state
            .validate_execution_program(&record.program)
            .map_err(|error| map_execution_dependency_error(request, error))?;
        program_binding_state
            .validate_execution_pipeline_layout(&record.layout)
            .map_err(|error| map_execution_dependency_error(request, error))?;
        Ok(())
    }

    fn validate_execution_affinity(
        &self,
        request: &'static str,
        observed: GpuContextAffinity,
    ) -> Result<(), GpuPipelineRealizationError> {
        if observed.context() != self.affinity.context() {
            return Err(GpuPipelineRealizationError::affinity(
                GpuPipelineRealizationErrorCategory::ForeignContext,
                request,
                self.affinity,
                observed,
            ));
        }
        if observed.generation() != self.affinity.generation() {
            return Err(GpuPipelineRealizationError::affinity(
                GpuPipelineRealizationErrorCategory::StaleDeviceGeneration,
                request,
                self.affinity,
                observed,
            ));
        }
        Ok(())
    }
}

fn map_execution_dependency_error(
    request: &'static str,
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
        _ => GpuPipelineRealizationErrorCategory::ExecutionAuthorityViolation,
    };
    GpuPipelineRealizationError::new(category, request, error.to_string())
}

fn execution_authority_violation(
    request: &'static str,
    detail: &'static str,
) -> GpuPipelineRealizationError {
    GpuPipelineRealizationError::new(
        GpuPipelineRealizationErrorCategory::ExecutionAuthorityViolation,
        request,
        detail,
    )
}

impl core::fmt::Debug for PipelineRealizationState {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("PipelineRealizationState")
            .field("affinity", &self.affinity)
            .field(
                "compute_records",
                &self
                    .compute
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .total_len(),
            )
            .field(
                "render_records",
                &self
                    .render
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .total_len(),
            )
            .field(
                "cache_diagnostics",
                &*self
                    .cache_diagnostics
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            )
            .field("max_records_per_family", &self.max_records)
            .finish_non_exhaustive()
    }
}
