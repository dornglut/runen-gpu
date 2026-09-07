use super::diagnostics::PipelineCacheFamily;
use super::publication::{ensure_available, scoped_create};
use super::records::RenderPipelineRealizationRecord;
use super::registry::{self, InFlightOutcome, Reservation};
use super::render_lowering::{LoweredRenderPipeline, lower_render_pipeline};
use super::render_validation::{
    RenderPipelineRequestKey, map_dependency_error, render_request_name,
    validate_admitted_format_roles, validate_dependency_affinity, validate_render_descriptor,
    wgpu_specialization_constants,
};
use crate::{
    GpuCapabilityAdmission, GpuContext, GpuPipelineRealizationError,
    GpuPipelineRealizationErrorCategory, GpuRealizedPipelineLayout, GpuRealizedProgram,
    GpuRealizedRenderPipeline, GpuRenderPipelineDescriptor,
};
use std::sync::Arc;
use wgpu::{FragmentState, PipelineCompilationOptions, RenderPipelineDescriptor, VertexState};

pub(super) type RenderRecord = RenderPipelineRealizationRecord;

impl GpuContext {
    /// Realizes one already-validated render-pipeline request against exact private program and
    /// pipeline-layout realizations plus current device capability/limit facts.
    pub async fn realize_render_pipeline(
        &self,
        descriptor: &GpuRenderPipelineDescriptor,
        program: &GpuRealizedProgram,
        layout: &GpuRealizedPipelineLayout,
    ) -> Result<GpuRealizedRenderPipeline, GpuPipelineRealizationError> {
        let request = render_request_name(descriptor);
        validate_render_descriptor(descriptor, request.clone())?;
        validate_dependency_affinity(self.affinity(), program.affinity(), request.clone())?;
        validate_dependency_affinity(self.affinity(), layout.affinity(), request.clone())?;
        if descriptor.program() != program.descriptor()
            || descriptor.layout() != layout.descriptor()
        {
            return Err(GpuPipelineRealizationError::new(
                GpuPipelineRealizationErrorCategory::ProgramInterfaceMismatch,
                request,
                "the render descriptor does not name the exact realized program and pipeline layout",
            ));
        }

        let canonical_program = self
            .realize_program(descriptor.program())
            .await
            .map_err(|error| map_dependency_error(render_request_name(descriptor), error))?;
        if !canonical_program.is_same_record(program) {
            return Err(GpuPipelineRealizationError::new(
                GpuPipelineRealizationErrorCategory::UnknownProgramOrLayoutRealization,
                render_request_name(descriptor),
                "the supplied program handle is not the authoritative G4C2 record for this request",
            ));
        }
        let canonical_layout = self
            .realize_pipeline_layout(descriptor.layout())
            .await
            .map_err(|error| map_dependency_error(render_request_name(descriptor), error))?;
        if !canonical_layout.is_same_record(layout) {
            return Err(GpuPipelineRealizationError::new(
                GpuPipelineRealizationErrorCategory::UnknownProgramOrLayoutRealization,
                render_request_name(descriptor),
                "the supplied pipeline-layout handle is not the authoritative G4C2 record for this request",
            ));
        }

        validate_admitted_format_roles(self, descriptor, &request)?;
        GpuCapabilityAdmission::evaluate(
            render_request_name(descriptor),
            descriptor.requirements(),
            self.adapter_facts().supported(),
            self.device_facts().enabled_features(),
        )
        .map_err(|error| {
            GpuPipelineRealizationError::new(
                GpuPipelineRealizationErrorCategory::RequirementNotAdmitted,
                render_request_name(descriptor),
                error.to_string(),
            )
        })?;

        // Complete private lowering validates current device limits and attachment sample support
        // before the WGPU creation call. Shader-stage IO was already validated by the logical
        // render-pipeline descriptor against the admitted program.
        let lowered = lower_render_pipeline(self, descriptor, &request)?;

        loop {
            let request = render_request_name(descriptor);
            ensure_available(&self.backend.pipeline_realization, request.clone())?;
            let key = RenderPipelineRequestKey::new(self, descriptor);
            let (reservation, observation) = registry::reserve(
                &self.backend.pipeline_realization.render,
                self.backend.pipeline_realization.max_records,
                key,
                request.clone(),
                RenderPipelineRequestKey::matches_record,
            )?;
            self.backend.pipeline_realization.observe_cache(
                PipelineCacheFamily::Render,
                observation,
                &request,
            );
            match reservation {
                Reservation::Ready(record) => {
                    return Ok(GpuRealizedRenderPipeline::from_record(record));
                }
                Reservation::Waiter(attempt) => match attempt.wait().await {
                    InFlightOutcome::Complete(outcome) => {
                        return outcome.map(GpuRealizedRenderPipeline::from_record);
                    }
                    InFlightOutcome::Abandoned => continue,
                    InFlightOutcome::Pending => {
                        unreachable!("wait never returns a pending pipeline attempt")
                    }
                },
                Reservation::Owner(owner) => {
                    let outcome = self
                        .realize_render_pipeline_owner(descriptor, program, layout, &lowered)
                        .await;
                    return owner
                        .finish(outcome)
                        .map(GpuRealizedRenderPipeline::from_record);
                }
            }
        }
    }

    async fn realize_render_pipeline_owner(
        &self,
        descriptor: &GpuRenderPipelineDescriptor,
        program: &GpuRealizedProgram,
        layout: &GpuRealizedPipelineLayout,
        lowered: &LoweredRenderPipeline,
    ) -> Result<Arc<RenderRecord>, GpuPipelineRealizationError> {
        let constants = wgpu_specialization_constants(descriptor);
        let vertex_buffers = lowered.vertex_buffer_layouts();
        let request = render_request_name(descriptor);
        let object = scoped_create(
            &self.backend.device,
            &self.backend.pipeline_realization,
            request,
            || {
                let fragment =
                    descriptor
                        .entry_points()
                        .fragment()
                        .map(|entry_point| FragmentState {
                            module: program.record.wgpu_object(),
                            entry_point: Some(entry_point.as_str()),
                            compilation_options: PipelineCompilationOptions {
                                constants: constants.as_slice(),
                                ..PipelineCompilationOptions::default()
                            },
                            targets: lowered.color_targets.as_slice(),
                        });
                self.backend
                    .device
                    .create_render_pipeline(&RenderPipelineDescriptor {
                        label: Some("runengpu-render-pipeline"),
                        layout: Some(layout.record.wgpu_object()),
                        vertex: VertexState {
                            module: program.record.wgpu_object(),
                            entry_point: Some(descriptor.entry_points().vertex().as_str()),
                            compilation_options: PipelineCompilationOptions {
                                constants: constants.as_slice(),
                                ..PipelineCompilationOptions::default()
                            },
                            buffers: vertex_buffers.as_slice(),
                        },
                        primitive: lowered.primitive,
                        depth_stencil: lowered.depth_stencil.clone(),
                        multisample: lowered.multisample,
                        fragment,
                        multiview_mask: None,
                        cache: None,
                    })
            },
        )
        .await?;
        Ok(Arc::new(RenderRecord {
            affinity: self.affinity(),
            descriptor: descriptor.clone(),
            object,
            program: Arc::clone(&program.record),
            layout: Arc::clone(&layout.record),
        }))
    }
}
