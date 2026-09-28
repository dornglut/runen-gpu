use super::interface::GpuBindingClass;
use crate::GpuCapabilityFeature;

/// Whole canonical-module features required merely for a fixed binding-array declaration to exist.
///
/// This is intentionally narrower than selected-layout requirements for uniform-buffer arrays:
/// pinned WGPU/Naga needs BUFFER_BINDING_ARRAY to compile the declaration, while
/// UNIFORM_BUFFER_BINDING_ARRAYS represents the stronger selected-layout/indexing capability.
pub(super) fn fixed_array_compilation_capabilities(
    class: GpuBindingClass,
) -> &'static [GpuCapabilityFeature] {
    match class {
        GpuBindingClass::UniformBuffer => &[GpuCapabilityFeature::BufferBindingArray],
        GpuBindingClass::StorageBuffer => &[
            GpuCapabilityFeature::BufferBindingArray,
            GpuCapabilityFeature::StorageResourceBindingArray,
        ],
        GpuBindingClass::SampledTexture | GpuBindingClass::Sampler => {
            &[GpuCapabilityFeature::TextureBindingArray]
        }
        GpuBindingClass::StorageTexture => &[
            GpuCapabilityFeature::TextureBindingArray,
            GpuCapabilityFeature::StorageResourceBindingArray,
        ],
    }
}

/// Features required when a fixed binding array participates in the selected public layout.
pub(crate) fn fixed_array_layout_capabilities(
    class: GpuBindingClass,
) -> &'static [GpuCapabilityFeature] {
    match class {
        GpuBindingClass::UniformBuffer => &[
            GpuCapabilityFeature::BufferBindingArray,
            GpuCapabilityFeature::UniformBufferBindingArray,
        ],
        class => fixed_array_compilation_capabilities(class),
    }
}

/// Normalized capability required when a fixed binding array is indexed dynamically
/// with a Naga-proven non-uniform index.
pub(super) fn fixed_array_non_uniform_indexing_capability(
    class: GpuBindingClass,
) -> Result<GpuCapabilityFeature, &'static str> {
    match class {
        GpuBindingClass::UniformBuffer => {
            Err("dynamically non-uniform uniform-buffer binding-array indexing is not normalized")
        }
        GpuBindingClass::StorageBuffer => {
            Ok(GpuCapabilityFeature::StorageBufferBindingArrayNonUniformIndexing)
        }
        GpuBindingClass::SampledTexture | GpuBindingClass::Sampler => {
            Ok(GpuCapabilityFeature::TextureBindingArrayNonUniformIndexing)
        }
        GpuBindingClass::StorageTexture => {
            Ok(GpuCapabilityFeature::StorageTextureBindingArrayNonUniformIndexing)
        }
    }
}
