use super::device_request::profile_limits;
use crate::{
    GpuAdapterClass, GpuAdapterFacts, GpuAdapterLimits, GpuAlignmentFacts, GpuBackendFamily,
    GpuCapabilities, GpuCapabilityFeature, GpuDeviceRequestProfile, GpuFallbackStatus, GpuLimits,
    GpuSoftwareStatus, GpuTextureFormat, GpuTextureFormatCapabilities,
};
use wgpu::{
    Adapter, Backend, DeviceType, DownlevelCapabilities, DownlevelFlags, Features, TextureFormat,
    TextureFormatFeatureFlags, TextureUsages,
};

pub(super) fn adapter_facts(
    adapter: &Adapter,
    surface_compatible: bool,
    fallback: GpuFallbackStatus,
) -> GpuAdapterFacts {
    let info = adapter.get_info();
    let downlevel = adapter.get_downlevel_capabilities();
    let native_limits = adapter.limits();
    let formats = known_formats().into_iter().map(|(normalized, native)| {
        (
            normalized,
            format_capabilities(native, adapter.get_texture_format_features(native)),
        )
    });
    let profile = select_device_request_profile(info.backend, &downlevel);
    let supported = normalized_features(
        adapter.features(),
        downlevel.flags,
        downlevel.is_webgpu_compliant(),
        surface_compatible,
    );
    let supports_storage_texture = known_formats().iter().any(|(_, format)| {
        adapter
            .get_texture_format_features(*format)
            .allowed_usages
            .contains(TextureUsages::STORAGE_BINDING)
    });
    let supports_depth_attachment = adapter
        .get_texture_format_features(TextureFormat::Depth32Float)
        .allowed_usages
        .contains(TextureUsages::RENDER_ATTACHMENT);
    let mut supported = supported;
    if downlevel.is_webgpu_compliant() && supports_storage_texture {
        supported.push(GpuCapabilityFeature::StorageTexture);
    }
    if downlevel.is_webgpu_compliant() && supports_depth_attachment {
        supported.push(GpuCapabilityFeature::DepthAttachment);
    }
    let adapter_limits = GpuLimits::from_validated_adapter_facts(
        native_limits.max_uniform_buffer_binding_size,
        native_limits.max_storage_buffer_binding_size,
        native_limits.max_color_attachments,
        native_limits.max_vertex_buffers,
        native_limits.max_bindings_per_bind_group,
        native_limits.max_texture_dimension_2d,
        native_limits.max_bind_groups,
        native_limits.max_bind_groups_plus_vertex_buffers,
        native_limits.max_dynamic_uniform_buffers_per_pipeline_layout,
        native_limits.max_dynamic_storage_buffers_per_pipeline_layout,
        native_limits.max_compute_workgroups_per_dimension,
    );
    GpuAdapterFacts::new(
        map_backend(info.backend),
        map_class(info.device_type),
        map_software(info.device_type),
        fallback,
        GpuCapabilities::from_normalized_facts(supported, adapter_limits, formats),
        GpuAdapterLimits::new(adapter_limits),
        GpuAlignmentFacts {
            uniform_dynamic_offset: Some(u64::from(
                native_limits.min_uniform_buffer_offset_alignment,
            )),
            storage_dynamic_offset: Some(u64::from(
                native_limits.min_storage_buffer_offset_alignment,
            )),
            copy_buffer_offset: Some(wgpu::COPY_BUFFER_ALIGNMENT),
            bytes_per_row: Some(u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)),
            query_resolve_destination: Some(wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT),
        },
    )
    .with_device_profile(
        profile,
        profile_limits(profile).check_limits(&native_limits),
    )
    .with_diagnostics(
        info.name,
        info.vendor,
        info.device,
        info.driver,
        info.driver_info,
    )
}

/// Maps only downlevel capabilities WGPU explicitly proves. Unknown flag bits suppress
/// broad capability claims rather than being guessed into the portable baseline.
pub(super) fn normalized_features(
    features: Features,
    flags: DownlevelFlags,
    webgpu_compliant: bool,
    surface_compatible: bool,
) -> Vec<GpuCapabilityFeature> {
    let unknown_flags = flags.bits() & !DownlevelFlags::all().bits() != 0;
    let baseline = webgpu_compliant && !unknown_flags;
    let mut supported = Vec::new();
    if baseline {
        supported.push(GpuCapabilityFeature::RenderPipeline);
        supported.push(GpuCapabilityFeature::Copy);
    }
    if !unknown_flags && flags.contains(DownlevelFlags::COMPUTE_SHADERS) {
        supported.push(GpuCapabilityFeature::Compute);
    }
    if !unknown_flags
        && flags.contains(DownlevelFlags::COMPUTE_SHADERS)
        && flags.contains(DownlevelFlags::INDIRECT_EXECUTION)
    {
        supported.push(GpuCapabilityFeature::IndirectExecution);
    }
    if features.contains(Features::TIMESTAMP_QUERY) {
        supported.push(GpuCapabilityFeature::TimestampQuery);
    }
    if features.contains(Features::TEXTURE_BINDING_ARRAY) {
        supported.push(GpuCapabilityFeature::TextureBindingArray);
    }
    if features.contains(Features::BUFFER_BINDING_ARRAY) {
        supported.push(GpuCapabilityFeature::BufferBindingArray);
    }
    if features.contains(Features::STORAGE_RESOURCE_BINDING_ARRAY) {
        supported.push(GpuCapabilityFeature::StorageResourceBindingArray);
    }
    if surface_compatible {
        supported.push(GpuCapabilityFeature::Presentation);
    }
    supported
}

pub(super) fn select_device_request_profile(
    backend: Backend,
    downlevel: &DownlevelCapabilities,
) -> GpuDeviceRequestProfile {
    if backend == Backend::BrowserWebGpu {
        GpuDeviceRequestProfile::BrowserWebGpu
    } else if !downlevel.is_webgpu_compliant() && backend == Backend::Gl {
        GpuDeviceRequestProfile::DownlevelWebGl2
    } else if !downlevel.is_webgpu_compliant() {
        GpuDeviceRequestProfile::Downlevel
    } else {
        GpuDeviceRequestProfile::ModernPortable
    }
}

pub(super) fn known_formats() -> [(GpuTextureFormat, TextureFormat); 7] {
    [
        (GpuTextureFormat::R8Unorm, TextureFormat::R8Unorm),
        (GpuTextureFormat::Rgba8Unorm, TextureFormat::Rgba8Unorm),
        (
            GpuTextureFormat::Rgba8UnormSrgb,
            TextureFormat::Rgba8UnormSrgb,
        ),
        (GpuTextureFormat::Bgra8Unorm, TextureFormat::Bgra8Unorm),
        (
            GpuTextureFormat::Bgra8UnormSrgb,
            TextureFormat::Bgra8UnormSrgb,
        ),
        (GpuTextureFormat::R32Uint, TextureFormat::R32Uint),
        (GpuTextureFormat::Depth32Float, TextureFormat::Depth32Float),
    ]
}

pub(super) fn format_capabilities(
    format: TextureFormat,
    features: wgpu::TextureFormatFeatures,
) -> GpuTextureFormatCapabilities {
    GpuTextureFormatCapabilities {
        sampled: features
            .allowed_usages
            .contains(TextureUsages::TEXTURE_BINDING),
        filterable: features
            .flags
            .contains(TextureFormatFeatureFlags::FILTERABLE),
        storage_read: features
            .flags
            .contains(TextureFormatFeatureFlags::STORAGE_READ_ONLY)
            || features
                .flags
                .contains(TextureFormatFeatureFlags::STORAGE_READ_WRITE),
        storage_write: features
            .flags
            .contains(TextureFormatFeatureFlags::STORAGE_WRITE_ONLY)
            || features
                .flags
                .contains(TextureFormatFeatureFlags::STORAGE_READ_WRITE),
        color_attachment: !format.is_depth_stencil_format()
            && features
                .allowed_usages
                .contains(TextureUsages::RENDER_ATTACHMENT),
        depth_stencil: format.is_depth_stencil_format()
            && features
                .allowed_usages
                .contains(TextureUsages::RENDER_ATTACHMENT),
        copy_source: features.allowed_usages.contains(TextureUsages::COPY_SRC),
        copy_destination: features.allowed_usages.contains(TextureUsages::COPY_DST),
        block_dimensions: Some(format.block_dimensions()),
        block_copy_size: format.block_copy_size(None),
    }
}

pub(super) const fn map_backend(backend: Backend) -> GpuBackendFamily {
    match backend {
        Backend::Vulkan => GpuBackendFamily::Vulkan,
        Backend::Metal => GpuBackendFamily::Metal,
        Backend::Dx12 => GpuBackendFamily::Direct3D12,
        Backend::Gl => GpuBackendFamily::OpenGl,
        Backend::BrowserWebGpu => GpuBackendFamily::BrowserWebGpu,
        Backend::Noop => GpuBackendFamily::UnknownBackend,
    }
}

pub(super) const fn map_class(class: DeviceType) -> GpuAdapterClass {
    match class {
        DeviceType::DiscreteGpu => GpuAdapterClass::Discrete,
        DeviceType::IntegratedGpu => GpuAdapterClass::Integrated,
        DeviceType::VirtualGpu => GpuAdapterClass::Virtual,
        DeviceType::Cpu => GpuAdapterClass::Cpu,
        DeviceType::Other => GpuAdapterClass::Other,
    }
}

pub(super) const fn map_software(class: DeviceType) -> GpuSoftwareStatus {
    match class {
        DeviceType::Cpu => GpuSoftwareStatus::Software,
        _ => GpuSoftwareStatus::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GpuCapabilityAdmission, GpuCapabilityAdmissionCause, GpuCapabilityAdmissionError,
        GpuCapabilityRequirement, GpuCapabilityRequirements, GpuLimits,
    };

    fn test_limits() -> GpuLimits {
        GpuLimits::new(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1).unwrap()
    }

    #[test]
    fn downlevel_mapping_claims_only_explicitly_proven_operations() {
        let full = normalized_features(
            Features::TIMESTAMP_QUERY,
            DownlevelFlags::all(),
            true,
            false,
        );
        assert!(full.contains(&GpuCapabilityFeature::Compute));
        assert!(full.contains(&GpuCapabilityFeature::IndirectExecution));
        assert!(full.contains(&GpuCapabilityFeature::RenderPipeline));
        assert!(full.contains(&GpuCapabilityFeature::Copy));
        assert!(!full.contains(&GpuCapabilityFeature::Presentation));

        let missing_compute =
            normalized_features(Features::empty(), DownlevelFlags::empty(), false, false);
        assert!(!missing_compute.contains(&GpuCapabilityFeature::Compute));
        assert!(!missing_compute.contains(&GpuCapabilityFeature::IndirectExecution));
        assert!(!missing_compute.contains(&GpuCapabilityFeature::RenderPipeline));
        assert!(!missing_compute.contains(&GpuCapabilityFeature::Copy));

        let unknown = normalized_features(
            Features::empty(),
            DownlevelFlags::from_bits_retain(DownlevelFlags::all().bits() | (1 << 31)),
            true,
            true,
        );
        assert!(!unknown.contains(&GpuCapabilityFeature::Compute));
        assert!(!unknown.contains(&GpuCapabilityFeature::IndirectExecution));
        assert!(!unknown.contains(&GpuCapabilityFeature::RenderPipeline));
        assert!(!unknown.contains(&GpuCapabilityFeature::Copy));
        assert!(unknown.contains(&GpuCapabilityFeature::Presentation));
    }

    #[test]
    fn native_binding_array_features_preserve_the_accepted_normalized_profile() {
        let normalized = normalized_features(
            Features::TEXTURE_BINDING_ARRAY
                | Features::BUFFER_BINDING_ARRAY
                | Features::STORAGE_RESOURCE_BINDING_ARRAY
                | Features::UNIFORM_BUFFER_BINDING_ARRAYS,
            DownlevelFlags::empty(),
            false,
            false,
        );

        assert!(normalized.contains(&GpuCapabilityFeature::TextureBindingArray));
        assert!(normalized.contains(&GpuCapabilityFeature::BufferBindingArray));
        assert!(normalized.contains(&GpuCapabilityFeature::StorageResourceBindingArray));
        assert!(
            !normalized.contains(&GpuCapabilityFeature::UniformBufferBindingArray),
            "the backend refresh must not expand the normalized profile without RunenGPU authority"
        );
    }

    #[test]
    fn refreshed_backend_rejects_unadmitted_uniform_buffer_array_capability() {
        let capabilities = GpuCapabilities::from_normalized_facts(
            normalized_features(
                Features::UNIFORM_BUFFER_BINDING_ARRAYS,
                DownlevelFlags::empty(),
                false,
                false,
            ),
            test_limits(),
            [],
        );
        let mut requirements = GpuCapabilityRequirements::new();
        requirements
            .insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::UniformBufferBindingArray,
            ))
            .unwrap();

        assert!(matches!(
            GpuCapabilityAdmission::evaluate(
                "uniform binding array",
                &requirements,
                &capabilities,
                []
            ),
            Err(GpuCapabilityAdmissionError::Rejected {
                cause: GpuCapabilityAdmissionCause::RequiredUnavailable,
                ..
            })
        ));
    }

    #[test]
    fn pinned_backend_and_adapter_mappings_remain_exhaustive() {
        assert_eq!(map_backend(Backend::Vulkan), GpuBackendFamily::Vulkan);
        assert_eq!(map_backend(Backend::Metal), GpuBackendFamily::Metal);
        assert_eq!(map_backend(Backend::Dx12), GpuBackendFamily::Direct3D12);
        assert_eq!(map_backend(Backend::Gl), GpuBackendFamily::OpenGl);
        assert_eq!(
            map_backend(Backend::BrowserWebGpu),
            GpuBackendFamily::BrowserWebGpu
        );
        assert_eq!(map_backend(Backend::Noop), GpuBackendFamily::UnknownBackend);
        assert_eq!(map_class(DeviceType::Cpu), GpuAdapterClass::Cpu);
        assert_eq!(map_software(DeviceType::Cpu), GpuSoftwareStatus::Software);
    }

    #[test]
    fn texture_mapping_preserves_normalized_roles_and_block_facts() {
        let features = wgpu::TextureFormatFeatures {
            allowed_usages: TextureUsages::TEXTURE_BINDING
                | TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST,
            flags: TextureFormatFeatureFlags::FILTERABLE,
        };
        let color = format_capabilities(TextureFormat::Rgba8Unorm, features);
        assert!(color.color_attachment);
        assert!(!color.depth_stencil);
        assert_eq!(color.block_dimensions, Some((1, 1)));
        let depth = format_capabilities(TextureFormat::Depth32Float, features);
        assert!(depth.depth_stencil);
        assert!(!depth.color_attachment);
    }
}
