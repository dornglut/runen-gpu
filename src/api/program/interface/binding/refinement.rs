use super::{GpuBindingKey, GpuSamplerClass, GpuShaderStages, GpuTextureSampleClass};
use core::num::NonZeroU64;

/// Sparse host/layout policy for one compiler-derived program binding.
///
/// The canonical WGSL remains authoritative for binding identity, resource class,
/// shader access, fixed array cardinality, and observed stage use. This record carries
/// only policy the shader cannot encode itself.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuBindingLayoutRefinement {
    key: GpuBindingKey,
    dynamic_offset: bool,
    host_minimum_size: Option<NonZeroU64>,
    texture_sample_class: Option<GpuTextureSampleClass>,
    sampler_class: Option<GpuSamplerClass>,
    visibility: Option<GpuShaderStages>,
}

impl GpuBindingLayoutRefinement {
    pub const fn new(key: GpuBindingKey) -> Self {
        Self {
            key,
            dynamic_offset: false,
            host_minimum_size: None,
            texture_sample_class: None,
            sampler_class: None,
            visibility: None,
        }
    }

    pub const fn with_dynamic_offset(mut self, enabled: bool) -> Self {
        self.dynamic_offset = enabled;
        self
    }

    pub const fn with_host_minimum_size(mut self, minimum_size: NonZeroU64) -> Self {
        self.host_minimum_size = Some(minimum_size);
        self
    }

    pub const fn with_texture_sample_class(mut self, class: GpuTextureSampleClass) -> Self {
        self.texture_sample_class = Some(class);
        self
    }

    pub const fn with_sampler_class(mut self, class: GpuSamplerClass) -> Self {
        self.sampler_class = Some(class);
        self
    }

    pub const fn with_visibility(mut self, visibility: GpuShaderStages) -> Self {
        self.visibility = Some(visibility);
        self
    }

    pub const fn key(&self) -> GpuBindingKey {
        self.key
    }

    pub const fn dynamic_offset(&self) -> bool {
        self.dynamic_offset
    }

    pub const fn host_minimum_size(&self) -> Option<NonZeroU64> {
        self.host_minimum_size
    }

    pub const fn texture_sample_class(&self) -> Option<GpuTextureSampleClass> {
        self.texture_sample_class
    }

    pub const fn sampler_class(&self) -> Option<GpuSamplerClass> {
        self.sampler_class
    }

    pub const fn visibility(&self) -> Option<GpuShaderStages> {
        self.visibility
    }
}
