//! Backend-neutral GPU execution contracts with a private WGPU realization.
//!
//! RunenGPU owns reusable GPU capabilities, resources, programs, work graphs,
//! submission, transfers, readback, surfaces, completion, and diagnostics.
//! Renderer image formation, application lifecycle, window policy, shader-file
//! policy, and product artifact policy remain outside this crate.

pub mod api;
mod backend;

pub(crate) use api::GpuPreparedInitialContent;
pub use api::*;
