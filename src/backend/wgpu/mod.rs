//! Private WGPU containment for G4A context admission.
//!
//! The submodules own state, normalized adapter mapping, device requests, execution, surfaces, and
//! bounded migration bridges. This root only composes those owners.

mod adapter_mapping;
mod device_request;
mod execution;
#[cfg(test)]
mod execution_tests;
mod health;
#[cfg(test)]
mod initial_content_tests;
mod pipeline_realization;
#[cfg(test)]
mod pressure_diagnostics_tests;
mod program_binding_realization;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod reconstruction_tests;
mod resource_realization;
mod state;
mod surface;
mod timestamp;

pub(crate) use device_request::request_headless;
pub(crate) use execution::WgpuExecutionState;
pub(crate) use health::{WgpuDeviceHealth, WgpuErrorAttributionGate};
pub(crate) use pipeline_realization::{
    ComputePipelineRealizationRecord, PipelineRealizationState, RenderPipelineRealizationRecord,
};
pub(crate) use program_binding_realization::{
    BindGroupLayoutRealizationRecord, BindGroupRealizationRecord, PipelineLayoutRealizationRecord,
    ProgramBindingRealizationState, ProgramRealizationRecord,
};
pub(crate) use resource_realization::{
    BufferRealizationRecord, QuerySetRealizationRecord, ResourceRealizationState,
    SamplerRealizationRecord, TextureRealizationRecord, TextureViewRealizationRecord,
};
pub(crate) use state::WgpuContextState;
pub(crate) use surface::WgpuSurfaceState;
