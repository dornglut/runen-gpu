use crate::{GpuComputePipelineDescriptor, GpuContextAffinity, GpuRenderPipelineDescriptor};
use std::sync::Arc;
use wgpu::{ComputePipeline, RenderPipeline};

use super::super::program_binding_realization::{
    PipelineLayoutRealizationRecord, ProgramRealizationRecord,
};

/// One accepted context/device-generation-bound compute pipeline and its exact G4C2 dependencies.
pub(crate) struct ComputePipelineRealizationRecord {
    pub(super) affinity: GpuContextAffinity,
    pub(super) descriptor: GpuComputePipelineDescriptor,
    #[allow(
        dead_code,
        reason = "G4C3 retains the private WGPU pipeline object for the serialized renderer cutover"
    )]
    pub(super) object: ComputePipeline,
    #[allow(
        dead_code,
        reason = "a live G4C3 pipeline must retain its exact G4C2 program dependency"
    )]
    pub(super) program: Arc<ProgramRealizationRecord>,
    #[allow(
        dead_code,
        reason = "a live G4C3 pipeline must retain its exact G4C2 pipeline-layout dependency"
    )]
    pub(super) layout: Arc<PipelineLayoutRealizationRecord>,
}

impl ComputePipelineRealizationRecord {
    pub(crate) const fn affinity(&self) -> GpuContextAffinity {
        self.affinity
    }

    pub(crate) fn descriptor(&self) -> &GpuComputePipelineDescriptor {
        &self.descriptor
    }
}

/// One accepted context/device-generation-bound render pipeline and its exact G4C2 dependencies.
pub(crate) struct RenderPipelineRealizationRecord {
    pub(super) affinity: GpuContextAffinity,
    pub(super) descriptor: GpuRenderPipelineDescriptor,
    #[allow(
        dead_code,
        reason = "G4C3 retains the private WGPU pipeline object for the serialized renderer cutover"
    )]
    pub(super) object: RenderPipeline,
    #[allow(
        dead_code,
        reason = "a live G4C3 pipeline must retain its exact G4C2 program dependency"
    )]
    pub(super) program: Arc<ProgramRealizationRecord>,
    #[allow(
        dead_code,
        reason = "a live G4C3 pipeline must retain its exact G4C2 pipeline-layout dependency"
    )]
    pub(super) layout: Arc<PipelineLayoutRealizationRecord>,
}

impl RenderPipelineRealizationRecord {
    pub(crate) const fn affinity(&self) -> GpuContextAffinity {
        self.affinity
    }

    pub(crate) fn descriptor(&self) -> &GpuRenderPipelineDescriptor {
        &self.descriptor
    }
}
