use super::{
    GpuBindingKind, GpuBindingKindInner, GpuSamplerClass, GpuStorageBufferAccess,
    GpuStorageTextureAccess, GpuTextureSampleClass, GpuTextureViewDimension,
};
use crate::{GpuProgramContractCause, GpuProgramContractError, GpuTextureFormat};
use core::num::NonZeroU64;

impl GpuBindingKind {
    pub fn uniform_buffer(dynamic_offset: bool, minimum_size: Option<NonZeroU64>) -> Self {
        Self(GpuBindingKindInner::UniformBuffer {
            dynamic_offset,
            minimum_size,
        })
    }

    pub fn storage_buffer(
        access: GpuStorageBufferAccess,
        dynamic_offset: bool,
        minimum_size: Option<NonZeroU64>,
    ) -> Self {
        Self(GpuBindingKindInner::StorageBuffer {
            access,
            dynamic_offset,
            minimum_size,
        })
    }

    pub fn sampled_texture(
        sample_class: GpuTextureSampleClass,
        view_dimension: GpuTextureViewDimension,
        multisampled: bool,
    ) -> Result<Self, GpuProgramContractError> {
        if multisampled && view_dimension != GpuTextureViewDimension::D2 {
            return Err(GpuProgramContractError::invalid(
                "construct GPU sampled-texture binding",
                format!("view_dimension={view_dimension:?}"),
                GpuProgramContractCause::BindingDeclarationInvalid,
                "use D2 for multisampled sampled textures",
            ));
        }
        Ok(Self(GpuBindingKindInner::SampledTexture {
            sample_class,
            view_dimension,
            multisampled,
        }))
    }

    pub fn storage_texture(
        access: GpuStorageTextureAccess,
        format: GpuTextureFormat,
        view_dimension: GpuTextureViewDimension,
    ) -> Result<Self, GpuProgramContractError> {
        if format.is_depth()
            || matches!(
                view_dimension,
                GpuTextureViewDimension::Cube | GpuTextureViewDimension::CubeArray
            )
        {
            return Err(GpuProgramContractError::invalid(
                "construct GPU storage-texture binding",
                format!("format={format:?}, view_dimension={view_dimension:?}"),
                GpuProgramContractCause::BindingDeclarationInvalid,
                "use a non-depth format and D1, D2, D2Array, or D3 view dimension",
            ));
        }
        Ok(Self(GpuBindingKindInner::StorageTexture {
            access,
            format,
            view_dimension,
        }))
    }

    pub fn sampler(class: GpuSamplerClass) -> Self {
        Self(GpuBindingKindInner::Sampler { class })
    }
}
