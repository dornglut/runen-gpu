use super::{
    PipelineRealizationState, ProgramBindingRealizationState, ResourceRealizationState,
    WgpuDeviceHealth, WgpuErrorAttributionGate, WgpuExecutionState, WgpuSurfaceState,
};
use std::sync::Arc;
use wgpu::{Adapter, Device, Instance, Queue};

/// The sole private owner of WGPU context objects.
#[derive(Debug)]
pub(crate) struct WgpuContextState {
    pub(super) instance: Instance,
    pub(super) adapter: Adapter,
    pub(super) device: Arc<Device>,
    pub(super) queue: Arc<Queue>,
    /// One shared terminal device-fault truth for every current same-device operation.
    pub(super) health: Arc<WgpuDeviceHealth>,
    /// One non-reentrant error-scope serialization authority for this device generation.
    pub(super) error_attribution_gate: Arc<WgpuErrorAttributionGate>,
    pub(super) resource_realization: ResourceRealizationState,
    pub(super) program_binding_realization: ProgramBindingRealizationState,
    pub(super) pipeline_realization: PipelineRealizationState,
    pub(super) execution: Arc<WgpuExecutionState>,
    pub(super) surfaces: WgpuSurfaceState,
}
