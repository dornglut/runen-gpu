use super::{TextureRealizationRecord, WgpuContextState};
use crate::{
    GpuAddressMode, GpuBufferDescriptor, GpuBufferUsage, GpuCapabilityFeature, GpuCompareFunction,
    GpuContext, GpuFilterMode, GpuFormatRole, GpuMemoryIntent, GpuQueryKind, GpuQuerySetDescriptor,
    GpuResourceCommon, GpuResourceOwnership, GpuResourceRealizationError,
    GpuResourceRealizationErrorCategory, GpuSamplerDescriptor, GpuTextureAspect,
    GpuTextureDescriptor, GpuTextureDimension, GpuTextureFormat, GpuTextureUsage,
    GpuTextureViewDescriptor, GpuWorkResourceId,
};
use wgpu::{
    AddressMode, BufferUsages, CompareFunction, DownlevelFlags, Extent3d, Features, FilterMode,
    MipmapFilterMode, QueryType, TextureAspect, TextureDimension, TextureFormat,
    TextureFormatFeatureFlags, TextureFormatFeatures, TextureUsages, TextureViewDimension,
};

pub(super) struct LoweredTexture {
    pub(super) size: Extent3d,
    pub(super) dimension: TextureDimension,
    pub(super) format: TextureFormat,
    pub(super) usage: TextureUsages,
    pub(super) paired_view_format: Option<TextureFormat>,
    pub(super) permits_format_reinterpretation: bool,
}

pub(super) fn lower_buffer(
    context: &GpuContext,
    identity: GpuWorkResourceId,
    descriptor: &GpuBufferDescriptor,
) -> Result<BufferUsages, GpuResourceRealizationError> {
    validate_resource_ownership(identity, descriptor.common())?;

    let usages = descriptor.usages();
    if usages.contains(GpuBufferUsage::CopySource)
        || usages.contains(GpuBufferUsage::CopyDestination)
    {
        require_feature(context, identity, GpuCapabilityFeature::Copy)?;
    }
    if usages.contains(GpuBufferUsage::Indirect) {
        require_feature(context, identity, GpuCapabilityFeature::IndirectExecution)?;
        if !context
            .backend
            .adapter
            .get_downlevel_capabilities()
            .flags
            .contains(DownlevelFlags::INDIRECT_EXECUTION)
        {
            return Err(incompatible(
                identity,
                "the admitted device lacks indirect-execution support for this buffer usage",
            ));
        }
    }
    if usages.contains(GpuBufferUsage::QueryResolve) {
        require_feature(context, identity, GpuCapabilityFeature::TimestampQuery)?;
    }
    if descriptor.size_bytes() > context.backend.device.limits().max_buffer_size {
        return Err(incompatible(
            identity,
            "buffer size exceeds the created device's maximum buffer size",
        ));
    }

    let mut native = usages.iter().fold(BufferUsages::empty(), |native, usage| {
        native | map_buffer_usage(usage)
    });
    match descriptor.common().memory_intent() {
        GpuMemoryIntent::Device => {}
        GpuMemoryIntent::Upload => native |= BufferUsages::MAP_WRITE,
        GpuMemoryIntent::Readback => native |= BufferUsages::MAP_READ,
    }

    if !context
        .backend
        .device
        .features()
        .contains(Features::MAPPABLE_PRIMARY_BUFFERS)
    {
        let write_mismatch = native.contains(BufferUsages::MAP_WRITE)
            && !(BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC).contains(native);
        let read_mismatch = native.contains(BufferUsages::MAP_READ)
            && !(BufferUsages::MAP_READ | BufferUsages::COPY_DST).contains(native);
        if write_mismatch || read_mismatch {
            return Err(incompatible(
                identity,
                "upload/readback memory intent cannot combine primary mapping with these usages",
            ));
        }
    }

    let restricted_index_combination = native.contains(BufferUsages::INDEX)
        && native.intersects(
            BufferUsages::VERTEX
                | BufferUsages::UNIFORM
                | BufferUsages::INDIRECT
                | BufferUsages::STORAGE,
        );
    if restricted_index_combination
        && !context
            .backend
            .adapter
            .get_downlevel_capabilities()
            .flags
            .contains(DownlevelFlags::UNRESTRICTED_INDEX_BUFFER)
    {
        return Err(incompatible(
            identity,
            "the admitted device restricts index buffers from the requested combined usages",
        ));
    }

    Ok(native)
}

pub(super) fn lower_texture(
    context: &GpuContext,
    identity: GpuWorkResourceId,
    descriptor: &GpuTextureDescriptor,
) -> Result<LoweredTexture, GpuResourceRealizationError> {
    validate_resource_ownership(identity, descriptor.common())?;

    let native_format = map_texture_format(descriptor.format());
    let native_usage = descriptor
        .usages()
        .iter()
        .fold(TextureUsages::empty(), |native, usage| {
            native | map_texture_usage(usage)
        });
    validate_texture_roles(
        context,
        identity,
        descriptor.format(),
        descriptor.usages().iter(),
    )?;

    let extent = descriptor.extent();
    let limits = context.backend.device.limits();
    let within_limits = match descriptor.dimension() {
        GpuTextureDimension::D1 => extent.width() <= limits.max_texture_dimension_1d,
        GpuTextureDimension::D2 => {
            extent.width() <= limits.max_texture_dimension_2d
                && extent.height() <= limits.max_texture_dimension_2d
                && extent.depth_or_layers() <= limits.max_texture_array_layers
        }
        GpuTextureDimension::D3 => {
            extent.width() <= limits.max_texture_dimension_3d
                && extent.height() <= limits.max_texture_dimension_3d
                && extent.depth_or_layers() <= limits.max_texture_dimension_3d
        }
    };
    if !within_limits {
        return Err(incompatible(
            identity,
            "texture extent exceeds the created device's dimension or array-layer limits",
        ));
    }
    if descriptor.dimension() != GpuTextureDimension::D2 && descriptor.format().is_depth() {
        return Err(incompatible(
            identity,
            "the private backend supports normalized depth textures only in two dimensions",
        ));
    }
    if descriptor.dimension() == GpuTextureDimension::D1
        && native_usage.contains(TextureUsages::RENDER_ATTACHMENT)
    {
        return Err(incompatible(
            identity,
            "one-dimensional textures cannot carry render-attachment usage",
        ));
    }
    if descriptor.sample_count() > 1
        && (!native_usage.contains(TextureUsages::RENDER_ATTACHMENT)
            || extent.depth_or_layers() != 1)
    {
        return Err(incompatible(
            identity,
            "multisampled textures require render-attachment usage and one array layer",
        ));
    }

    let format_features = device_format_features(&context.backend, native_format);
    if !format_features.allowed_usages.contains(native_usage) {
        return Err(incompatible(
            identity,
            "texture usages are not available from the created device for this format",
        ));
    }
    validate_storage_access_flags(identity, descriptor, format_features.flags)?;
    if !format_features
        .flags
        .sample_count_supported(descriptor.sample_count())
    {
        return Err(incompatible(
            identity,
            "texture sample count is not supported for this format on the created device",
        ));
    }

    let downlevel = context.backend.adapter.get_downlevel_capabilities();
    let permits_format_reinterpretation = downlevel.flags.contains(DownlevelFlags::VIEW_FORMATS);
    Ok(LoweredTexture {
        size: Extent3d {
            width: extent.width(),
            height: extent.height(),
            depth_or_array_layers: extent.depth_or_layers(),
        },
        dimension: map_texture_dimension(descriptor.dimension()),
        format: native_format,
        usage: native_usage,
        paired_view_format: permits_format_reinterpretation
            .then(|| paired_view_format(native_format))
            .flatten(),
        permits_format_reinterpretation,
    })
}

pub(super) fn validate_texture_view(
    context: &GpuContext,
    identity: GpuWorkResourceId,
    descriptor: &GpuTextureViewDescriptor,
    parent: &TextureRealizationRecord,
) -> Result<(), GpuResourceRealizationError> {
    validate_resource_ownership(identity, descriptor.common())?;
    let declared_parent = descriptor.texture();
    if declared_parent.diagnostic_identity() != parent.logical_identity {
        return Err(GpuResourceRealizationError::new(
            GpuResourceRealizationErrorCategory::UnknownLogicalResource,
            Some(identity),
            "the view descriptor names a different logical parent texture",
        ));
    }
    if declared_parent.descriptor() != parent.descriptor() {
        return Err(GpuResourceRealizationError::new(
            GpuResourceRealizationErrorCategory::DescriptorChangedForIdentity,
            Some(declared_parent.diagnostic_identity()),
            "the view retained changed semantics for its logical parent texture",
        ));
    }

    let effective_format = descriptor.format().unwrap_or(parent.descriptor().format());
    if effective_format != parent.descriptor().format() {
        if !parent.permits_format_reinterpretation {
            return Err(incompatible(
                identity,
                "the created device cannot reinterpret this parent texture's view format",
            ));
        }
        validate_texture_roles(
            context,
            identity,
            effective_format,
            parent.descriptor().usages().iter(),
        )?;
        let native_usage = parent
            .descriptor()
            .usages()
            .iter()
            .fold(TextureUsages::empty(), |native, usage| {
                native | map_texture_usage(usage)
            });
        if !device_format_features(&context.backend, map_texture_format(effective_format))
            .allowed_usages
            .contains(native_usage)
        {
            return Err(incompatible(
                identity,
                "the view format cannot carry the parent texture's admitted usages",
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_sampler(
    identity: GpuWorkResourceId,
    descriptor: &GpuSamplerDescriptor,
) -> Result<(), GpuResourceRealizationError> {
    validate_resource_ownership(identity, descriptor.common())?;
    if descriptor.lod_range().0 < 0.0 {
        return Err(incompatible(
            identity,
            "sampler minimum LOD must be nonnegative for the private backend",
        ));
    }
    Ok(())
}

pub(super) fn validate_query_set(
    context: &GpuContext,
    identity: GpuWorkResourceId,
    descriptor: &GpuQuerySetDescriptor,
) -> Result<(), GpuResourceRealizationError> {
    validate_resource_ownership(identity, descriptor.common())?;
    match descriptor.kind() {
        GpuQueryKind::Timestamp => {
            require_feature(context, identity, GpuCapabilityFeature::TimestampQuery)?;
        }
    }
    if descriptor.count() > wgpu::QUERY_SET_MAX_QUERIES {
        return Err(incompatible(
            identity,
            "query count exceeds the pinned backend's maximum query-set size",
        ));
    }
    Ok(())
}

pub(super) const fn map_texture_format(format: GpuTextureFormat) -> TextureFormat {
    match format {
        GpuTextureFormat::R8Unorm => TextureFormat::R8Unorm,
        GpuTextureFormat::Rgba8Unorm => TextureFormat::Rgba8Unorm,
        GpuTextureFormat::Rgba8UnormSrgb => TextureFormat::Rgba8UnormSrgb,
        GpuTextureFormat::Bgra8Unorm => TextureFormat::Bgra8Unorm,
        GpuTextureFormat::Bgra8UnormSrgb => TextureFormat::Bgra8UnormSrgb,
        GpuTextureFormat::R32Uint => TextureFormat::R32Uint,
        GpuTextureFormat::Depth32Float => TextureFormat::Depth32Float,
    }
}

pub(super) const fn map_texture_dimension(dimension: GpuTextureDimension) -> TextureDimension {
    match dimension {
        GpuTextureDimension::D1 => TextureDimension::D1,
        GpuTextureDimension::D2 => TextureDimension::D2,
        GpuTextureDimension::D3 => TextureDimension::D3,
    }
}

pub(super) const fn map_texture_view_dimension(
    descriptor: &GpuTextureViewDescriptor,
) -> TextureViewDimension {
    match descriptor.dimension() {
        GpuTextureDimension::D1 => TextureViewDimension::D1,
        GpuTextureDimension::D2 if descriptor.subresources().array_layer_count() > 1 => {
            TextureViewDimension::D2Array
        }
        GpuTextureDimension::D2 => TextureViewDimension::D2,
        GpuTextureDimension::D3 => TextureViewDimension::D3,
    }
}

pub(super) const fn map_texture_aspect(aspect: GpuTextureAspect) -> TextureAspect {
    match aspect {
        GpuTextureAspect::All | GpuTextureAspect::Color => TextureAspect::All,
        GpuTextureAspect::DepthOnly => TextureAspect::DepthOnly,
        GpuTextureAspect::StencilOnly => TextureAspect::StencilOnly,
    }
}

pub(super) const fn map_address_mode(mode: GpuAddressMode) -> AddressMode {
    match mode {
        GpuAddressMode::ClampToEdge => AddressMode::ClampToEdge,
        GpuAddressMode::Repeat => AddressMode::Repeat,
        GpuAddressMode::MirrorRepeat => AddressMode::MirrorRepeat,
    }
}

pub(super) const fn map_filter_mode(mode: GpuFilterMode) -> FilterMode {
    match mode {
        GpuFilterMode::Nearest => FilterMode::Nearest,
        GpuFilterMode::Linear => FilterMode::Linear,
    }
}

pub(super) const fn map_mipmap_filter_mode(mode: GpuFilterMode) -> MipmapFilterMode {
    match mode {
        GpuFilterMode::Nearest => MipmapFilterMode::Nearest,
        GpuFilterMode::Linear => MipmapFilterMode::Linear,
    }
}

pub(super) const fn map_compare_function(function: GpuCompareFunction) -> CompareFunction {
    match function {
        GpuCompareFunction::Never => CompareFunction::Never,
        GpuCompareFunction::Less => CompareFunction::Less,
        GpuCompareFunction::Equal => CompareFunction::Equal,
        GpuCompareFunction::LessEqual => CompareFunction::LessEqual,
        GpuCompareFunction::Greater => CompareFunction::Greater,
        GpuCompareFunction::NotEqual => CompareFunction::NotEqual,
        GpuCompareFunction::GreaterEqual => CompareFunction::GreaterEqual,
        GpuCompareFunction::Always => CompareFunction::Always,
    }
}

pub(super) const fn map_query_kind(kind: GpuQueryKind) -> QueryType {
    match kind {
        GpuQueryKind::Timestamp => QueryType::Timestamp,
    }
}

pub(super) fn validate_resource_ownership(
    identity: GpuWorkResourceId,
    common: &GpuResourceCommon,
) -> Result<(), GpuResourceRealizationError> {
    match common.ownership() {
        GpuResourceOwnership::Owned => Ok(()),
        GpuResourceOwnership::Imported => Err(GpuResourceRealizationError::new(
            GpuResourceRealizationErrorCategory::ImportSourceUnavailable,
            Some(identity),
            "no accepted concrete import-source contract is available in G4C1",
        )),
        GpuResourceOwnership::SurfaceAcquired => Err(GpuResourceRealizationError::new(
            GpuResourceRealizationErrorCategory::RequirementNotAdmitted,
            Some(identity),
            "surface-acquired resource realization remains owned by G7",
        )),
    }
}

fn require_feature(
    context: &GpuContext,
    identity: GpuWorkResourceId,
    feature: GpuCapabilityFeature,
) -> Result<(), GpuResourceRealizationError> {
    if context.device_facts().is_enabled(feature) {
        Ok(())
    } else {
        Err(GpuResourceRealizationError::new(
            GpuResourceRealizationErrorCategory::RequirementNotAdmitted,
            Some(identity),
            format!("required normalized capability was not admitted: {feature:?}"),
        ))
    }
}

fn validate_texture_roles(
    context: &GpuContext,
    identity: GpuWorkResourceId,
    format: GpuTextureFormat,
    usages: impl Iterator<Item = GpuTextureUsage>,
) -> Result<(), GpuResourceRealizationError> {
    let admitted_roles = context
        .device_facts()
        .admission_contract()
        .format_roles()
        .collect::<std::collections::BTreeSet<_>>();
    for usage in usages {
        let role = match usage {
            GpuTextureUsage::Sampled => GpuFormatRole::Sampled,
            GpuTextureUsage::StorageRead => GpuFormatRole::StorageRead,
            GpuTextureUsage::StorageWrite => GpuFormatRole::StorageWrite,
            GpuTextureUsage::ColorAttachment => GpuFormatRole::ColorAttachment,
            GpuTextureUsage::DepthStencilAttachment => GpuFormatRole::DepthStencil,
            GpuTextureUsage::CopySource => GpuFormatRole::CopySource,
            GpuTextureUsage::CopyDestination => GpuFormatRole::CopyDestination,
        };
        if !admitted_roles.contains(&(format, role)) {
            return Err(incompatible(
                identity,
                "texture format role was not admitted by the context request",
            ));
        }
        match usage {
            GpuTextureUsage::StorageRead | GpuTextureUsage::StorageWrite => {
                require_feature(context, identity, GpuCapabilityFeature::StorageTexture)?;
            }
            GpuTextureUsage::ColorAttachment => {
                require_feature(context, identity, GpuCapabilityFeature::RenderPipeline)?;
            }
            GpuTextureUsage::DepthStencilAttachment => {
                require_feature(context, identity, GpuCapabilityFeature::DepthAttachment)?;
            }
            GpuTextureUsage::CopySource | GpuTextureUsage::CopyDestination => {
                require_feature(context, identity, GpuCapabilityFeature::Copy)?;
            }
            GpuTextureUsage::Sampled => {}
        }
    }
    Ok(())
}

fn validate_storage_access_flags(
    identity: GpuWorkResourceId,
    descriptor: &GpuTextureDescriptor,
    flags: TextureFormatFeatureFlags,
) -> Result<(), GpuResourceRealizationError> {
    let read = descriptor.usages().contains(GpuTextureUsage::StorageRead);
    let write = descriptor.usages().contains(GpuTextureUsage::StorageWrite);
    let supports_read = flags.intersects(
        TextureFormatFeatureFlags::STORAGE_READ_ONLY
            | TextureFormatFeatureFlags::STORAGE_READ_WRITE,
    );
    let supports_write = flags.intersects(
        TextureFormatFeatureFlags::STORAGE_WRITE_ONLY
            | TextureFormatFeatureFlags::STORAGE_READ_WRITE,
    );
    if (read && !supports_read) || (write && !supports_write) {
        Err(incompatible(
            identity,
            "texture format lacks the requested storage access on the created device",
        ))
    } else {
        Ok(())
    }
}

fn device_format_features(
    state: &WgpuContextState,
    format: TextureFormat,
) -> TextureFormatFeatures {
    let device_features = state.device.features();
    let downlevel = state.adapter.get_downlevel_capabilities();
    if device_features.contains(Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
        || !downlevel
            .flags
            .contains(DownlevelFlags::WEBGPU_TEXTURE_FORMAT_SUPPORT)
    {
        state.adapter.get_texture_format_features(format)
    } else {
        format.guaranteed_format_features(device_features)
    }
}

const fn map_buffer_usage(usage: GpuBufferUsage) -> BufferUsages {
    match usage {
        GpuBufferUsage::Uniform => BufferUsages::UNIFORM,
        GpuBufferUsage::Storage => BufferUsages::STORAGE,
        GpuBufferUsage::Vertex => BufferUsages::VERTEX,
        GpuBufferUsage::Index => BufferUsages::INDEX,
        GpuBufferUsage::Indirect => BufferUsages::INDIRECT,
        GpuBufferUsage::CopySource => BufferUsages::COPY_SRC,
        GpuBufferUsage::CopyDestination => BufferUsages::COPY_DST,
        GpuBufferUsage::QueryResolve => BufferUsages::QUERY_RESOLVE,
    }
}

const fn map_texture_usage(usage: GpuTextureUsage) -> TextureUsages {
    match usage {
        GpuTextureUsage::Sampled => TextureUsages::TEXTURE_BINDING,
        GpuTextureUsage::StorageRead | GpuTextureUsage::StorageWrite => {
            TextureUsages::STORAGE_BINDING
        }
        GpuTextureUsage::ColorAttachment | GpuTextureUsage::DepthStencilAttachment => {
            TextureUsages::RENDER_ATTACHMENT
        }
        GpuTextureUsage::CopySource => TextureUsages::COPY_SRC,
        GpuTextureUsage::CopyDestination => TextureUsages::COPY_DST,
    }
}

const fn paired_view_format(format: TextureFormat) -> Option<TextureFormat> {
    match format {
        TextureFormat::Rgba8Unorm => Some(TextureFormat::Rgba8UnormSrgb),
        TextureFormat::Rgba8UnormSrgb => Some(TextureFormat::Rgba8Unorm),
        TextureFormat::Bgra8Unorm => Some(TextureFormat::Bgra8UnormSrgb),
        TextureFormat::Bgra8UnormSrgb => Some(TextureFormat::Bgra8Unorm),
        TextureFormat::R8Unorm | TextureFormat::R32Uint | TextureFormat::Depth32Float => None,
        _ => None,
    }
}

fn incompatible(identity: GpuWorkResourceId, detail: &'static str) -> GpuResourceRealizationError {
    GpuResourceRealizationError::new(
        GpuResourceRealizationErrorCategory::FormatOrAlignmentNotAdmitted,
        Some(identity),
        detail,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalized_mappings_cover_every_current_resource_enum() {
        assert_eq!(
            map_buffer_usage(GpuBufferUsage::Uniform),
            BufferUsages::UNIFORM
        );
        assert_eq!(
            map_texture_usage(GpuTextureUsage::StorageRead),
            TextureUsages::STORAGE_BINDING
        );
        assert_eq!(
            map_texture_format(GpuTextureFormat::Depth32Float),
            TextureFormat::Depth32Float
        );
        assert_eq!(
            map_address_mode(GpuAddressMode::MirrorRepeat),
            AddressMode::MirrorRepeat
        );
        assert_eq!(
            map_compare_function(GpuCompareFunction::LessEqual),
            CompareFunction::LessEqual
        );
        assert!(paired_view_format(TextureFormat::R32Uint).is_none());
    }
}
