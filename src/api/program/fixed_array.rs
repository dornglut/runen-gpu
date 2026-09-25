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
pub(super) fn fixed_array_layout_capabilities(
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
