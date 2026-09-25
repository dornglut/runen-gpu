use super::device_request::profile_limits;
use crate::api::texture_format;
use crate::{
    GpuAdapterClass, GpuAdapterFacts, GpuAdapterLimits, GpuAlignmentFacts, GpuBackendFamily,
    GpuCapabilities, GpuCapabilityFeature, GpuDeviceRequestProfile, GpuFallbackStatus, GpuLimits,
    GpuSoftwareStatus, GpuTextureAspect, GpuTextureFormat, GpuTextureFormatCapabilities,
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
    let formats = texture_formats().into_iter().map(|(normalized, native)| {
        (
            normalized,
            format_capabilities(normalized, adapter.get_texture_format_features(native)),
        )
    });
    let profile = select_device_request_profile(info.backend, &downlevel);
    let supported = normalized_features(
        info.backend,
        adapter.features(),
        downlevel.flags,
        downlevel.is_webgpu_compliant(),
        surface_compatible,
    );
    let supports_storage_texture = texture_formats().iter().any(|(_, format)| {
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
    let adapter_limits = normalized_limits(&native_limits);
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

fn normalized_limits(native: &wgpu::Limits) -> GpuLimits {
    GpuLimits::from_validated_adapter_facts(
        native.max_uniform_buffer_binding_size,
        native.max_storage_buffer_binding_size,
        native.max_color_attachments,
        native.max_vertex_buffers,
        native.max_bindings_per_bind_group,
        native.max_texture_dimension_2d,
        native.max_bind_groups,
        native.max_bind_groups_plus_vertex_buffers,
        native.max_dynamic_uniform_buffers_per_pipeline_layout,
        native.max_dynamic_storage_buffers_per_pipeline_layout,
        native.max_compute_workgroups_per_dimension,
        native.max_buffer_size,
        native.max_texture_dimension_1d,
        native.max_texture_dimension_3d,
        native.max_texture_array_layers,
        native.max_vertex_attributes,
        native.max_vertex_buffer_array_stride,
    )
}

/// Maps only downlevel capabilities WGPU explicitly proves. Unknown flag bits suppress
/// broad capability claims rather than being guessed into the portable baseline.
pub(super) fn normalized_features(
    backend: Backend,
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
    // The current WGPU Metal timestamp surface is not sufficient proof of RunenGPU's ordered
    // marker semantics. Keep the normalized capability conservative until Apple-native
    // marker + resolve/readback evidence proves the backend path truthful.
    if backend != Backend::Metal && features.contains(Features::TIMESTAMP_QUERY) {
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

/// Closed G7A presentation vocabulary retained for the surface owner.
/// Ordinary texture-format growth is enumerated by `texture_formats()` and cannot expand this set.
pub(super) fn known_formats() -> Vec<(GpuTextureFormat, TextureFormat)> {
    texture_formats()
        .into_iter()
        .filter(|(format, _)| is_g7a_presentation_format(*format))
        .collect()
}

const fn is_g7a_presentation_format(format: GpuTextureFormat) -> bool {
    matches!(
        format,
        GpuTextureFormat::R8Unorm
            | GpuTextureFormat::Rgba8Unorm
            | GpuTextureFormat::Rgba8UnormSrgb
            | GpuTextureFormat::Bgra8Unorm
            | GpuTextureFormat::Bgra8UnormSrgb
            | GpuTextureFormat::R32Uint
            | GpuTextureFormat::R32Sint
            | GpuTextureFormat::R32Float
            | GpuTextureFormat::Rg32Uint
            | GpuTextureFormat::Rg32Sint
            | GpuTextureFormat::Rg32Float
            | GpuTextureFormat::Rgba32Uint
            | GpuTextureFormat::Rgba32Sint
            | GpuTextureFormat::Rgba32Float
            | GpuTextureFormat::Depth32Float
    )
}

fn texture_formats() -> [(GpuTextureFormat, TextureFormat); 38] {
    [
        (GpuTextureFormat::R8Unorm, TextureFormat::R8Unorm),
        (GpuTextureFormat::R8Snorm, TextureFormat::R8Snorm),
        (GpuTextureFormat::R8Uint, TextureFormat::R8Uint),
        (GpuTextureFormat::R8Sint, TextureFormat::R8Sint),
        (GpuTextureFormat::Rg8Unorm, TextureFormat::Rg8Unorm),
        (GpuTextureFormat::Rg8Snorm, TextureFormat::Rg8Snorm),
        (GpuTextureFormat::Rg8Uint, TextureFormat::Rg8Uint),
        (GpuTextureFormat::Rg8Sint, TextureFormat::Rg8Sint),
        (GpuTextureFormat::R16Uint, TextureFormat::R16Uint),
        (GpuTextureFormat::R16Sint, TextureFormat::R16Sint),
        (GpuTextureFormat::R16Float, TextureFormat::R16Float),
        (GpuTextureFormat::Rg16Uint, TextureFormat::Rg16Uint),
        (GpuTextureFormat::Rg16Sint, TextureFormat::Rg16Sint),
        (GpuTextureFormat::Rg16Float, TextureFormat::Rg16Float),
        (GpuTextureFormat::Rgba8Unorm, TextureFormat::Rgba8Unorm),
        (
            GpuTextureFormat::Rgba8UnormSrgb,
            TextureFormat::Rgba8UnormSrgb,
        ),
        (GpuTextureFormat::Rgba8Snorm, TextureFormat::Rgba8Snorm),
        (GpuTextureFormat::Rgba8Uint, TextureFormat::Rgba8Uint),
        (GpuTextureFormat::Rgba8Sint, TextureFormat::Rgba8Sint),
        (GpuTextureFormat::Bgra8Unorm, TextureFormat::Bgra8Unorm),
        (
            GpuTextureFormat::Bgra8UnormSrgb,
            TextureFormat::Bgra8UnormSrgb,
        ),
        (GpuTextureFormat::R32Uint, TextureFormat::R32Uint),
        (GpuTextureFormat::R32Sint, TextureFormat::R32Sint),
        (GpuTextureFormat::R32Float, TextureFormat::R32Float),
        (GpuTextureFormat::Rg32Uint, TextureFormat::Rg32Uint),
        (GpuTextureFormat::Rg32Sint, TextureFormat::Rg32Sint),
        (GpuTextureFormat::Rg32Float, TextureFormat::Rg32Float),
        (GpuTextureFormat::Rgba32Uint, TextureFormat::Rgba32Uint),
        (GpuTextureFormat::Rgba32Sint, TextureFormat::Rgba32Sint),
        (GpuTextureFormat::Rgba32Float, TextureFormat::Rgba32Float),
        (GpuTextureFormat::Rgba16Uint, TextureFormat::Rgba16Uint),
        (GpuTextureFormat::Rgba16Sint, TextureFormat::Rgba16Sint),
        (GpuTextureFormat::Rgba16Float, TextureFormat::Rgba16Float),
        (GpuTextureFormat::Stencil8, TextureFormat::Stencil8),
        (GpuTextureFormat::Depth16Unorm, TextureFormat::Depth16Unorm),
        (GpuTextureFormat::Depth24Plus, TextureFormat::Depth24Plus),
        (
            GpuTextureFormat::Depth24PlusStencil8,
            TextureFormat::Depth24PlusStencil8,
        ),
        (GpuTextureFormat::Depth32Float, TextureFormat::Depth32Float),
    ]
}

pub(super) fn format_capabilities(
    format: GpuTextureFormat,
    features: wgpu::TextureFormatFeatures,
) -> GpuTextureFormatCapabilities {
    let render_attachment = features
        .allowed_usages
        .contains(TextureUsages::RENDER_ATTACHMENT);
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
        color_attachment: render_attachment
            && texture_format::supports_aspect(format, GpuTextureAspect::Color),
        depth_stencil: render_attachment
            && (texture_format::supports_aspect(format, GpuTextureAspect::DepthOnly)
                || texture_format::supports_aspect(format, GpuTextureAspect::StencilOnly)),
        copy_source: features.allowed_usages.contains(TextureUsages::COPY_SRC),
        copy_destination: features.allowed_usages.contains(TextureUsages::COPY_DST),
        block_dimensions: None,
        block_copy_size: None,
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
        GpuLimits::new(
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            256 * 1024 * 1024,
            8192,
            2048,
            256,
            16,
            2048,
        )
        .unwrap()
    }

    #[test]
    fn adapter_limit_mapping_preserves_public_descriptor_limits() {
        let mut native = wgpu::Limits::defaults();
        native.max_buffer_size = 123_456_789;
        native.max_texture_dimension_1d = 4096;
        native.max_texture_dimension_3d = 1024;
        native.max_texture_array_layers = 128;
        native.max_vertex_attributes = 12;
        native.max_vertex_buffer_array_stride = 1024;
        let limits = normalized_limits(&native);
        assert_eq!(limits.max_buffer_size(), 123_456_789);
        assert_eq!(limits.max_texture_dimension_1d(), 4096);
        assert_eq!(limits.max_texture_dimension_3d(), 1024);
        assert_eq!(limits.max_texture_array_layers(), 128);
        assert_eq!(limits.max_vertex_attributes(), 12);
        assert_eq!(limits.max_vertex_buffer_array_stride(), 1024);
    }

    #[test]
    fn downlevel_mapping_claims_only_explicitly_proven_operations() {
        let full = normalized_features(
            Backend::Vulkan,
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

        let missing_compute = normalized_features(
            Backend::Vulkan,
            Features::empty(),
            DownlevelFlags::empty(),
            false,
            false,
        );
        assert!(!missing_compute.contains(&GpuCapabilityFeature::Compute));
        assert!(!missing_compute.contains(&GpuCapabilityFeature::IndirectExecution));
        assert!(!missing_compute.contains(&GpuCapabilityFeature::RenderPipeline));
        assert!(!missing_compute.contains(&GpuCapabilityFeature::Copy));

        let unknown = normalized_features(
            Backend::Vulkan,
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
    fn metal_does_not_claim_timestamp_query_from_the_advertised_backend_bit() {
        let metal = normalized_features(
            Backend::Metal,
            Features::TIMESTAMP_QUERY,
            DownlevelFlags::all(),
            true,
            false,
        );
        assert!(!metal.contains(&GpuCapabilityFeature::TimestampQuery));

        let vulkan = normalized_features(
            Backend::Vulkan,
            Features::TIMESTAMP_QUERY,
            DownlevelFlags::all(),
            true,
            false,
        );
        assert!(vulkan.contains(&GpuCapabilityFeature::TimestampQuery));
    }

    #[test]
    fn native_binding_array_features_preserve_the_accepted_normalized_profile() {
        let normalized = normalized_features(
            Backend::Vulkan,
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
                Backend::Vulkan,
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
    fn presentation_format_census_is_closed_against_texture_growth() {
        assert_eq!(known_formats().len(), 15);
        assert!(
            known_formats().contains(&(GpuTextureFormat::Rgba32Float, TextureFormat::Rgba32Float))
        );
        for pair in [
            (GpuTextureFormat::Rgba8Snorm, TextureFormat::Rgba8Snorm),
            (GpuTextureFormat::Rgba8Uint, TextureFormat::Rgba8Uint),
            (GpuTextureFormat::Rgba8Sint, TextureFormat::Rgba8Sint),
            (GpuTextureFormat::Rgba16Uint, TextureFormat::Rgba16Uint),
            (GpuTextureFormat::Rgba16Sint, TextureFormat::Rgba16Sint),
            (GpuTextureFormat::Rgba16Float, TextureFormat::Rgba16Float),
        ] {
            assert!(!known_formats().contains(&pair));
            assert!(texture_formats().contains(&pair));
        }
    }

    #[test]
    fn baseline_32bit_format_census_is_complete() {
        for pair in [
            (GpuTextureFormat::R32Uint, TextureFormat::R32Uint),
            (GpuTextureFormat::R32Sint, TextureFormat::R32Sint),
            (GpuTextureFormat::R32Float, TextureFormat::R32Float),
            (GpuTextureFormat::Rg32Uint, TextureFormat::Rg32Uint),
            (GpuTextureFormat::Rg32Sint, TextureFormat::Rg32Sint),
            (GpuTextureFormat::Rg32Float, TextureFormat::Rg32Float),
            (GpuTextureFormat::Rgba32Uint, TextureFormat::Rgba32Uint),
            (GpuTextureFormat::Rgba32Sint, TextureFormat::Rgba32Sint),
            (GpuTextureFormat::Rgba32Float, TextureFormat::Rgba32Float),
        ] {
            assert!(texture_formats().contains(&pair));
        }
    }

    #[test]
    fn rgba8_core_format_census_and_optional_roles_follow_backend_facts() {
        assert_eq!(texture_formats().len(), 38);
        for (format, native) in [
            (GpuTextureFormat::Rgba8Snorm, TextureFormat::Rgba8Snorm),
            (GpuTextureFormat::Rgba8Uint, TextureFormat::Rgba8Uint),
            (GpuTextureFormat::Rgba8Sint, TextureFormat::Rgba8Sint),
        ] {
            assert!(texture_formats().contains(&(format, native)));
            let absent = format_capabilities(
                format,
                wgpu::TextureFormatFeatures {
                    allowed_usages: TextureUsages::COPY_SRC,
                    flags: TextureFormatFeatureFlags::empty(),
                },
            );
            assert!(absent.copy_source);
            assert!(!absent.copy_destination);
            assert!(!absent.sampled);
            assert!(!absent.filterable);
            assert!(!absent.storage_read);
            assert!(!absent.storage_write);
            assert!(!absent.color_attachment);
            assert!(!absent.depth_stencil);
            let observed = format_capabilities(
                format,
                wgpu::TextureFormatFeatures {
                    allowed_usages: TextureUsages::TEXTURE_BINDING
                        | TextureUsages::STORAGE_BINDING
                        | TextureUsages::RENDER_ATTACHMENT
                        | TextureUsages::COPY_DST,
                    flags: TextureFormatFeatureFlags::FILTERABLE
                        | TextureFormatFeatureFlags::STORAGE_READ_WRITE,
                },
            );
            assert!(observed.sampled);
            assert!(observed.filterable);
            assert!(observed.storage_read);
            assert!(observed.storage_write);
            assert!(observed.color_attachment);
            assert!(observed.copy_destination);
            assert!(!observed.copy_source);
        }
    }

    #[test]
    fn rgba16_format_census_and_optional_roles_follow_backend_facts() {
        assert_eq!(texture_formats().len(), 38);
        for (format, native) in [
            (GpuTextureFormat::Rgba16Uint, TextureFormat::Rgba16Uint),
            (GpuTextureFormat::Rgba16Sint, TextureFormat::Rgba16Sint),
            (GpuTextureFormat::Rgba16Float, TextureFormat::Rgba16Float),
        ] {
            assert!(texture_formats().contains(&(format, native)));
            let absent = format_capabilities(
                format,
                wgpu::TextureFormatFeatures {
                    allowed_usages: TextureUsages::COPY_SRC,
                    flags: TextureFormatFeatureFlags::empty(),
                },
            );
            assert!(absent.copy_source);
            assert!(!absent.copy_destination);
            assert!(!absent.sampled);
            assert!(!absent.filterable);
            assert!(!absent.storage_read);
            assert!(!absent.storage_write);
            assert!(!absent.color_attachment);
            assert!(!absent.depth_stencil);
            let observed = format_capabilities(
                format,
                wgpu::TextureFormatFeatures {
                    allowed_usages: TextureUsages::TEXTURE_BINDING
                        | TextureUsages::STORAGE_BINDING
                        | TextureUsages::RENDER_ATTACHMENT
                        | TextureUsages::COPY_DST,
                    flags: TextureFormatFeatureFlags::FILTERABLE
                        | TextureFormatFeatureFlags::STORAGE_READ_WRITE,
                },
            );
            assert!(observed.sampled);
            assert!(observed.filterable);
            assert!(observed.storage_read);
            assert!(observed.storage_write);
            assert!(observed.color_attachment);
            assert!(observed.copy_destination);
            assert!(!observed.copy_source);
        }
    }

    #[test]
    fn r32float_mapping_preserves_backend_reported_roles() {
        assert!(texture_formats().contains(&(GpuTextureFormat::R32Float, TextureFormat::R32Float)));

        let sampled_copy_source = format_capabilities(
            GpuTextureFormat::R32Float,
            wgpu::TextureFormatFeatures {
                allowed_usages: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
                flags: TextureFormatFeatureFlags::empty(),
            },
        );
        assert!(sampled_copy_source.sampled);
        assert!(sampled_copy_source.copy_source);
        assert!(!sampled_copy_source.copy_destination);
        assert!(!sampled_copy_source.filterable);
        assert!(!sampled_copy_source.storage_read);
        assert!(!sampled_copy_source.storage_write);
        assert!(!sampled_copy_source.depth_stencil);
        assert_eq!(sampled_copy_source.block_dimensions, None);
        assert_eq!(sampled_copy_source.block_copy_size, None);

        let richer = format_capabilities(
            GpuTextureFormat::R32Float,
            wgpu::TextureFormatFeatures {
                allowed_usages: TextureUsages::TEXTURE_BINDING
                    | TextureUsages::STORAGE_BINDING
                    | TextureUsages::COPY_DST,
                flags: TextureFormatFeatureFlags::FILTERABLE
                    | TextureFormatFeatureFlags::STORAGE_READ_WRITE,
            },
        );
        assert!(richer.sampled);
        assert!(richer.filterable);
        assert!(richer.storage_read);
        assert!(richer.storage_write);
        assert!(richer.copy_destination);
        assert!(!richer.copy_source);
    }

    #[test]
    fn new_32bit_structure_does_not_manufacture_backend_roles() {
        for format in [GpuTextureFormat::Rg32Float, GpuTextureFormat::Rgba32Float] {
            let facts = format_capabilities(
                format,
                wgpu::TextureFormatFeatures {
                    allowed_usages: TextureUsages::COPY_SRC,
                    flags: TextureFormatFeatureFlags::empty(),
                },
            );
            assert!(facts.copy_source);
            assert!(!facts.copy_destination);
            assert!(!facts.sampled);
            assert!(!facts.filterable);
            assert!(!facts.storage_read);
            assert!(!facts.storage_write);
            assert!(!facts.color_attachment);
            assert!(!facts.depth_stencil);
            assert_eq!(facts.block_dimensions, None);
            assert_eq!(facts.block_copy_size, None);
        }
    }

    #[test]
    fn texture_mapping_preserves_normalized_roles_without_owning_structure() {
        let features = wgpu::TextureFormatFeatures {
            allowed_usages: TextureUsages::TEXTURE_BINDING
                | TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST,
            flags: TextureFormatFeatureFlags::FILTERABLE,
        };
        let color = format_capabilities(GpuTextureFormat::Rgba8Unorm, features);
        assert!(color.color_attachment);
        assert!(!color.depth_stencil);
        assert_eq!(color.block_dimensions, None);
        assert_eq!(color.block_copy_size, None);
        let depth = format_capabilities(GpuTextureFormat::Depth32Float, features);
        assert!(depth.depth_stencil);
        assert!(!depth.color_attachment);
        assert_eq!(depth.block_dimensions, None);
        assert_eq!(depth.block_copy_size, None);
    }
    #[test]
    fn stencil8_maps_exactly_and_preserves_observed_roles() {
        let format = GpuTextureFormat::Stencil8;
        assert!(texture_formats().contains(&(format, TextureFormat::Stencil8)));
        assert!(!is_g7a_presentation_format(format));
        assert_eq!(format.copy_block_size(GpuTextureAspect::All), Some(1));
        let facts = format_capabilities(
            format,
            wgpu::TextureFormatFeatures {
                allowed_usages: TextureUsages::TEXTURE_BINDING
                    | TextureUsages::RENDER_ATTACHMENT
                    | TextureUsages::COPY_SRC
                    | TextureUsages::COPY_DST,
                flags: TextureFormatFeatureFlags::empty(),
            },
        );
        assert!(facts.sampled);
        assert!(facts.depth_stencil);
        assert!(facts.copy_source);
        assert!(facts.copy_destination);
        assert!(!facts.color_attachment);
        assert!(!facts.storage_read);
        assert!(!facts.storage_write);
    }

    #[test]
    fn baseline_depth_formats_map_exactly_and_preserve_observed_roles() {
        for (format, native, expected_copy_size) in [
            (
                GpuTextureFormat::Depth16Unorm,
                TextureFormat::Depth16Unorm,
                Some(2),
            ),
            (
                GpuTextureFormat::Depth24Plus,
                TextureFormat::Depth24Plus,
                None,
            ),
        ] {
            assert!(texture_formats().contains(&(format, native)));
            assert!(!is_g7a_presentation_format(format));
            assert_eq!(
                format.copy_block_size(GpuTextureAspect::All),
                expected_copy_size
            );

            let facts = format_capabilities(
                format,
                wgpu::TextureFormatFeatures {
                    allowed_usages: TextureUsages::TEXTURE_BINDING
                        | TextureUsages::RENDER_ATTACHMENT
                        | TextureUsages::COPY_SRC
                        | TextureUsages::COPY_DST,
                    flags: TextureFormatFeatureFlags::FILTERABLE,
                },
            );
            assert!(facts.sampled);
            assert!(facts.filterable);
            assert!(facts.depth_stencil);
            assert!(facts.copy_source);
            assert!(facts.copy_destination);
            assert!(!facts.color_attachment);
            assert!(!facts.storage_read);
            assert!(!facts.storage_write);
            assert_eq!(facts.block_dimensions, None);
            assert_eq!(facts.block_copy_size, None);
        }
    }
}

#[cfg(test)]
mod r1_r_rg8_mapping_tests {
    use super::*;

    #[test]
    fn thirty_eight_unique_private_mappings_preserve_closed_presentation() {
        let mappings = texture_formats();
        assert_eq!(mappings.len(), 38);
        let mut normalized = Vec::new();
        let mut native = Vec::new();
        for (format, wgpu_format) in mappings {
            assert!(!normalized.contains(&format));
            assert!(!native.contains(&wgpu_format));
            normalized.push(format);
            native.push(wgpu_format);
        }
        assert_eq!(known_formats().len(), 15);
        for (format, wgpu_format) in [
            (GpuTextureFormat::R8Snorm, TextureFormat::R8Snorm),
            (GpuTextureFormat::R8Uint, TextureFormat::R8Uint),
            (GpuTextureFormat::R8Sint, TextureFormat::R8Sint),
            (GpuTextureFormat::Rg8Unorm, TextureFormat::Rg8Unorm),
            (GpuTextureFormat::Rg8Snorm, TextureFormat::Rg8Snorm),
            (GpuTextureFormat::Rg8Uint, TextureFormat::Rg8Uint),
            (GpuTextureFormat::Rg8Sint, TextureFormat::Rg8Sint),
            (GpuTextureFormat::R16Uint, TextureFormat::R16Uint),
            (GpuTextureFormat::R16Sint, TextureFormat::R16Sint),
            (GpuTextureFormat::R16Float, TextureFormat::R16Float),
            (GpuTextureFormat::Rg16Uint, TextureFormat::Rg16Uint),
            (GpuTextureFormat::Rg16Sint, TextureFormat::Rg16Sint),
            (GpuTextureFormat::Rg16Float, TextureFormat::Rg16Float),
        ] {
            assert!(!is_g7a_presentation_format(format));
            assert_eq!(
                texture_formats()
                    .into_iter()
                    .find(|(value, _)| *value == format),
                Some((format, wgpu_format))
            );
        }
    }
}
