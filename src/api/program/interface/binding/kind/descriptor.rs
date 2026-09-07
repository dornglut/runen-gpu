use super::{
    GpuSamplerClass, GpuStorageBufferAccess, GpuStorageTextureAccess, GpuTextureSampleClass,
    GpuTextureViewDimension,
};
use crate::GpuTextureFormat;
use core::num::NonZeroU64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuBindingClass {
    UniformBuffer,
    StorageBuffer,
    SampledTexture,
    StorageTexture,
    Sampler,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum GpuBindingKindInner {
    UniformBuffer {
        dynamic_offset: bool,
        minimum_size: Option<NonZeroU64>,
    },
    StorageBuffer {
        access: GpuStorageBufferAccess,
        dynamic_offset: bool,
        minimum_size: Option<NonZeroU64>,
    },
    SampledTexture {
        sample_class: GpuTextureSampleClass,
        view_dimension: GpuTextureViewDimension,
        multisampled: bool,
    },
    StorageTexture {
        access: GpuStorageTextureAccess,
        format: GpuTextureFormat,
        view_dimension: GpuTextureViewDimension,
    },
    Sampler {
        class: GpuSamplerClass,
    },
}

/// Structurally validated shader-visible binding kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuBindingKind(pub(super) GpuBindingKindInner);
