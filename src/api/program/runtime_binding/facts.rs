use crate::{GpuTextureFormat, GpuTextureFormatCapabilities};
use core::num::NonZeroU64;
use std::collections::BTreeMap;

/// Internal normalized device facts for contextual runtime-binding validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GpuRuntimeBindingDeviceFacts {
    uniform_buffer_offset_alignment: Option<NonZeroU64>,
    storage_buffer_offset_alignment: Option<NonZeroU64>,
    max_bind_groups: u32,
    max_dynamic_uniform_buffers_per_pipeline_layout: u32,
    max_dynamic_storage_buffers_per_pipeline_layout: u32,
    format_capabilities: BTreeMap<GpuTextureFormat, GpuTextureFormatCapabilities>,
}

impl GpuRuntimeBindingDeviceFacts {
    pub(crate) fn new(
        uniform_buffer_offset_alignment: Option<NonZeroU64>,
        storage_buffer_offset_alignment: Option<NonZeroU64>,
        max_bind_groups: u32,
        max_dynamic_uniform_buffers_per_pipeline_layout: u32,
        max_dynamic_storage_buffers_per_pipeline_layout: u32,
        format_capabilities: impl IntoIterator<Item = (GpuTextureFormat, GpuTextureFormatCapabilities)>,
    ) -> Self {
        Self {
            uniform_buffer_offset_alignment,
            storage_buffer_offset_alignment,
            max_bind_groups,
            max_dynamic_uniform_buffers_per_pipeline_layout,
            max_dynamic_storage_buffers_per_pipeline_layout,
            format_capabilities: format_capabilities.into_iter().collect(),
        }
    }

    pub(crate) const fn uniform_buffer_offset_alignment(&self) -> Option<NonZeroU64> {
        self.uniform_buffer_offset_alignment
    }

    pub(crate) const fn storage_buffer_offset_alignment(&self) -> Option<NonZeroU64> {
        self.storage_buffer_offset_alignment
    }

    pub(crate) const fn max_bind_groups(&self) -> u32 {
        self.max_bind_groups
    }

    pub(crate) const fn max_dynamic_uniform_buffers_per_pipeline_layout(&self) -> u32 {
        self.max_dynamic_uniform_buffers_per_pipeline_layout
    }

    pub(crate) const fn max_dynamic_storage_buffers_per_pipeline_layout(&self) -> u32 {
        self.max_dynamic_storage_buffers_per_pipeline_layout
    }

    pub(crate) fn format_capabilities(
        &self,
        format: GpuTextureFormat,
    ) -> Option<GpuTextureFormatCapabilities> {
        self.format_capabilities.get(&format).copied()
    }
}
