use super::{
    GpuBindingClass, GpuBindingKind, GpuBindingKindInner, GpuSamplerClass, GpuStorageBufferAccess,
    GpuStorageTextureAccess, GpuTextureSampleClass, GpuTextureViewDimension,
};
use crate::GpuTextureFormat;
use core::num::NonZeroU64;

impl GpuBindingKind {
    pub const fn class(self) -> GpuBindingClass {
        match self.0 {
            GpuBindingKindInner::UniformBuffer { .. } => GpuBindingClass::UniformBuffer,
            GpuBindingKindInner::StorageBuffer { .. } => GpuBindingClass::StorageBuffer,
            GpuBindingKindInner::SampledTexture { .. } => GpuBindingClass::SampledTexture,
            GpuBindingKindInner::StorageTexture { .. } => GpuBindingClass::StorageTexture,
            GpuBindingKindInner::Sampler { .. } => GpuBindingClass::Sampler,
        }
    }

    pub const fn minimum_buffer_size(self) -> Option<NonZeroU64> {
        match self.0 {
            GpuBindingKindInner::UniformBuffer { minimum_size, .. }
            | GpuBindingKindInner::StorageBuffer { minimum_size, .. } => minimum_size,
            _ => None,
        }
    }

    pub const fn uses_dynamic_offset(self) -> bool {
        match self.0 {
            GpuBindingKindInner::UniformBuffer { dynamic_offset, .. }
            | GpuBindingKindInner::StorageBuffer { dynamic_offset, .. } => dynamic_offset,
            _ => false,
        }
    }

    pub const fn storage_buffer_access(self) -> Option<GpuStorageBufferAccess> {
        match self.0 {
            GpuBindingKindInner::StorageBuffer { access, .. } => Some(access),
            _ => None,
        }
    }

    pub const fn texture_view_dimension(self) -> Option<GpuTextureViewDimension> {
        match self.0 {
            GpuBindingKindInner::SampledTexture { view_dimension, .. }
            | GpuBindingKindInner::StorageTexture { view_dimension, .. } => Some(view_dimension),
            _ => None,
        }
    }

    pub const fn texture_sample_class(self) -> Option<GpuTextureSampleClass> {
        match self.0 {
            GpuBindingKindInner::SampledTexture { sample_class, .. } => Some(sample_class),
            _ => None,
        }
    }

    pub const fn is_multisampled_texture(self) -> bool {
        matches!(
            self.0,
            GpuBindingKindInner::SampledTexture {
                multisampled: true,
                ..
            }
        )
    }

    pub const fn storage_texture_access(self) -> Option<GpuStorageTextureAccess> {
        match self.0 {
            GpuBindingKindInner::StorageTexture { access, .. } => Some(access),
            _ => None,
        }
    }

    pub const fn storage_texture_format(self) -> Option<GpuTextureFormat> {
        match self.0 {
            GpuBindingKindInner::StorageTexture { format, .. } => Some(format),
            _ => None,
        }
    }

    pub const fn sampler_class(self) -> Option<GpuSamplerClass> {
        match self.0 {
            GpuBindingKindInner::Sampler { class } => Some(class),
            _ => None,
        }
    }
}
