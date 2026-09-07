use super::{
    GpuAccessCause, GpuAccessError, GpuBufferHandle, GpuBufferUsage, GpuQuerySetHandle,
    GpuResourceAccessIntent, GpuSamplerHandle, GpuTextureAspect, GpuTextureDimension,
    GpuTextureHandle, GpuTextureSubresourceRange, GpuTextureUsage, GpuTextureViewHandle,
    GpuWorkResourceId,
};

/// A descriptor-bounded byte range.
///
/// Raw construction is intentionally unavailable; callers must use the
/// checked constructors.
///
/// ```compile_fail
/// use runen_gpu::GpuBufferRange;
///
/// let _unchecked = GpuBufferRange { offset: 0, size: 4 };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuBufferRange {
    offset: u64,
    size: u64,
}

impl GpuBufferRange {
    pub fn new(handle: &GpuBufferHandle, offset: u64, size: u64) -> Result<Self, GpuAccessError> {
        if size == 0 {
            return Err(buffer_range_error(
                handle,
                GpuAccessCause::ZeroRange,
                "provide a nonzero byte range",
            ));
        }
        let end = offset.checked_add(size).ok_or_else(|| {
            buffer_range_error(
                handle,
                GpuAccessCause::ArithmeticOverflow,
                "reduce the byte offset or size",
            )
        })?;
        if end > handle.descriptor().size_bytes() {
            return Err(buffer_range_error(
                handle,
                GpuAccessCause::OutOfBounds,
                "keep the byte range inside the buffer descriptor",
            ));
        }
        Ok(Self { offset, size })
    }

    pub fn whole(handle: &GpuBufferHandle) -> Result<Self, GpuAccessError> {
        Self::new(handle, 0, handle.descriptor().size_bytes())
    }

    pub const fn offset(self) -> u64 {
        self.offset
    }

    pub const fn size(self) -> u64 {
        self.size
    }

    pub const fn end(self) -> u64 {
        // Construction proves this addition cannot overflow.
        self.offset + self.size
    }

    pub const fn overlaps(self, other: Self) -> bool {
        self.offset < other.end() && other.offset < self.end()
    }

    pub const fn contains(self, other: Self) -> bool {
        self.offset <= other.offset && self.end() >= other.end()
    }
}

fn buffer_range_error(
    handle: &GpuBufferHandle,
    cause: GpuAccessCause,
    correction: &'static str,
) -> GpuAccessError {
    GpuAccessError::invalid(
        "construct GPU buffer range",
        handle.descriptor().common().label().as_str(),
        Some(handle.diagnostic_identity()),
        cause,
        correction,
    )
}

/// A descriptor-bounded query interval.
///
/// ```compile_fail
/// use runen_gpu::GpuQueryRange;
///
/// let _unchecked = GpuQueryRange { first: 0, count: 2 };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuQueryRange {
    first: u32,
    count: u32,
}

impl GpuQueryRange {
    pub fn new(handle: &GpuQuerySetHandle, first: u32, count: u32) -> Result<Self, GpuAccessError> {
        if count == 0 {
            return Err(query_range_error(
                handle,
                GpuAccessCause::ZeroRange,
                "provide a nonzero query count",
            ));
        }
        let end = first.checked_add(count).ok_or_else(|| {
            query_range_error(
                handle,
                GpuAccessCause::ArithmeticOverflow,
                "reduce the first query or count",
            )
        })?;
        if end > handle.descriptor().count() {
            return Err(query_range_error(
                handle,
                GpuAccessCause::OutOfBounds,
                "keep the query range inside the query-set descriptor",
            ));
        }
        Ok(Self { first, count })
    }

    pub fn whole(handle: &GpuQuerySetHandle) -> Result<Self, GpuAccessError> {
        Self::new(handle, 0, handle.descriptor().count())
    }

    pub const fn first(self) -> u32 {
        self.first
    }

    pub const fn count(self) -> u32 {
        self.count
    }

    pub const fn end(self) -> u32 {
        self.first + self.count
    }

    pub const fn overlaps(self, other: Self) -> bool {
        self.first < other.end() && other.first < self.end()
    }

    pub const fn contains(self, other: Self) -> bool {
        self.first <= other.first && self.end() >= other.end()
    }
}

fn query_range_error(
    handle: &GpuQuerySetHandle,
    cause: GpuAccessCause,
    correction: &'static str,
) -> GpuAccessError {
    GpuAccessError::invalid(
        "construct GPU query range",
        handle.descriptor().common().label().as_str(),
        Some(handle.diagnostic_identity()),
        cause,
        correction,
    )
}

impl GpuTextureSubresourceRange {
    pub fn whole(texture: &GpuTextureHandle) -> Result<Self, GpuAccessError> {
        let descriptor = texture.descriptor();
        let layer_count = match descriptor.dimension() {
            GpuTextureDimension::D2 => descriptor.extent().depth_or_layers(),
            GpuTextureDimension::D1 | GpuTextureDimension::D3 => 1,
        };
        Self::new(
            descriptor.common().label(),
            0,
            descriptor.mip_level_count(),
            0,
            layer_count,
            if descriptor.format().is_depth() {
                GpuTextureAspect::DepthOnly
            } else {
                GpuTextureAspect::Color
            },
        )
        .map_err(|_| {
            texture_access_error(
                texture,
                GpuAccessCause::OutOfBounds,
                "use the checked texture descriptor subresources",
            )
        })
    }

    pub fn checked_for(texture: &GpuTextureHandle, range: Self) -> Result<Self, GpuAccessError> {
        validate_texture_range(texture, range)?;
        Ok(range)
    }

    pub const fn mip_end(self) -> u32 {
        self.base_mip_level() + self.mip_level_count()
    }

    pub const fn layer_end(self) -> u32 {
        self.base_array_layer() + self.array_layer_count()
    }

    pub fn overlaps(self, other: Self, parent_aspect: GpuTextureAspect) -> bool {
        self.base_mip_level() < other.mip_end()
            && other.base_mip_level() < self.mip_end()
            && self.base_array_layer() < other.layer_end()
            && other.base_array_layer() < self.layer_end()
            && aspects_overlap(self.aspect(), other.aspect(), parent_aspect)
    }

    pub fn contains(self, other: Self, parent_aspect: GpuTextureAspect) -> bool {
        self.base_mip_level() <= other.base_mip_level()
            && self.mip_end() >= other.mip_end()
            && self.base_array_layer() <= other.base_array_layer()
            && self.layer_end() >= other.layer_end()
            && aspect_contains(self.aspect(), other.aspect(), parent_aspect)
    }
}

fn validate_texture_range(
    texture: &GpuTextureHandle,
    range: GpuTextureSubresourceRange,
) -> Result<(), GpuAccessError> {
    let descriptor = texture.descriptor();
    if descriptor.dimension() == GpuTextureDimension::D3
        && (range.base_array_layer() != 0 || range.array_layer_count() != 1)
    {
        return Err(texture_access_error(
            texture,
            GpuAccessCause::InvalidD3Interpretation,
            "address each D3 mip as one whole volume with logical array layer 0",
        ));
    }
    let layers = match descriptor.dimension() {
        GpuTextureDimension::D2 => descriptor.extent().depth_or_layers(),
        GpuTextureDimension::D1 | GpuTextureDimension::D3 => 1,
    };
    if range.mip_end() > descriptor.mip_level_count() || range.layer_end() > layers {
        return Err(texture_access_error(
            texture,
            GpuAccessCause::OutOfBounds,
            "keep mip and array-layer coverage inside the parent texture descriptor",
        ));
    }
    let aspect_valid = if descriptor.format().is_depth() {
        matches!(
            range.aspect(),
            GpuTextureAspect::All | GpuTextureAspect::DepthOnly
        )
    } else {
        matches!(
            range.aspect(),
            GpuTextureAspect::All | GpuTextureAspect::Color
        )
    };
    if !aspect_valid {
        return Err(texture_access_error(
            texture,
            GpuAccessCause::InvalidTextureAspect,
            "select an aspect represented by the parent texture format",
        ));
    }
    Ok(())
}

fn texture_access_error(
    texture: &GpuTextureHandle,
    cause: GpuAccessCause,
    correction: &'static str,
) -> GpuAccessError {
    GpuAccessError::invalid(
        "construct GPU texture access",
        texture.descriptor().common().label().as_str(),
        Some(texture.diagnostic_identity()),
        cause,
        correction,
    )
}

fn aspects_overlap(
    left: GpuTextureAspect,
    right: GpuTextureAspect,
    parent: GpuTextureAspect,
) -> bool {
    canonical_aspect(left, parent) == canonical_aspect(right, parent)
}

fn aspect_contains(
    outer: GpuTextureAspect,
    inner: GpuTextureAspect,
    parent: GpuTextureAspect,
) -> bool {
    canonical_aspect(outer, parent) == canonical_aspect(inner, parent)
}

fn canonical_aspect(value: GpuTextureAspect, parent: GpuTextureAspect) -> GpuTextureAspect {
    if value == GpuTextureAspect::All {
        parent
    } else {
        value
    }
}

fn intersect_texture_ranges(
    left: GpuTextureSubresourceRange,
    right: GpuTextureSubresourceRange,
    texture: &GpuTextureHandle,
) -> Result<GpuTextureSubresourceRange, GpuAccessError> {
    let mip_start = left.base_mip_level().max(right.base_mip_level());
    let mip_end = left.mip_end().min(right.mip_end());
    let layer_start = left.base_array_layer().max(right.base_array_layer());
    let layer_end = left.layer_end().min(right.layer_end());
    let parent_aspect = if texture.descriptor().format().is_depth() {
        GpuTextureAspect::DepthOnly
    } else {
        GpuTextureAspect::Color
    };
    let left_aspect = canonical_aspect(left.aspect(), parent_aspect);
    let right_aspect = canonical_aspect(right.aspect(), parent_aspect);
    if mip_start >= mip_end || layer_start >= layer_end || left_aspect != right_aspect {
        return Err(texture_access_error(
            texture,
            GpuAccessCause::InvalidViewIntersection,
            "request mip, layer, and aspect coverage inside the texture view",
        ));
    }
    GpuTextureSubresourceRange::new(
        texture.descriptor().common().label(),
        mip_start,
        mip_end - mip_start,
        layer_start,
        layer_end - layer_start,
        left_aspect,
    )
    .map_err(|_| {
        texture_access_error(
            texture,
            GpuAccessCause::InvalidViewIntersection,
            "request a nonempty checked texture-view intersection",
        )
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuBufferAccessKind {
    UniformRead,
    StorageRead,
    StorageWrite,
    StorageReadWrite,
    VertexRead,
    IndexRead,
    IndirectRead,
    CopySource,
    CopyDestination,
    QueryResolveDestination,
}

impl GpuBufferAccessKind {
    pub const fn reads(self) -> bool {
        matches!(
            self,
            Self::UniformRead
                | Self::StorageRead
                | Self::StorageReadWrite
                | Self::VertexRead
                | Self::IndexRead
                | Self::IndirectRead
                | Self::CopySource
        )
    }

    pub const fn writes(self) -> bool {
        matches!(
            self,
            Self::StorageWrite
                | Self::StorageReadWrite
                | Self::CopyDestination
                | Self::QueryResolveDestination
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuAttachmentLoadKind {
    Load,
    Clear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuAttachmentStore {
    Store,
    Discard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuDepthStencilAccess {
    ReadOnly,
    ReadWrite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuTextureAccessKind {
    SampledRead,
    StorageRead,
    StorageWrite,
    StorageReadWrite,
    CopySource,
    CopyDestination,
    ColorAttachment {
        load_kind: GpuAttachmentLoadKind,
        store: GpuAttachmentStore,
    },
    MultisampleResolveDestination,
    DepthStencilAttachment {
        access: GpuDepthStencilAccess,
        load_kind: GpuAttachmentLoadKind,
        store: GpuAttachmentStore,
    },
    Present,
}

impl GpuTextureAccessKind {
    pub const fn reads(self) -> bool {
        match self {
            Self::SampledRead | Self::StorageRead | Self::StorageReadWrite | Self::CopySource => {
                true
            }
            Self::ColorAttachment { load_kind, .. }
            | Self::DepthStencilAttachment { load_kind, .. } => {
                matches!(load_kind, GpuAttachmentLoadKind::Load)
            }
            Self::Present => true,
            Self::StorageWrite | Self::CopyDestination | Self::MultisampleResolveDestination => {
                false
            }
        }
    }

    pub const fn writes(self) -> bool {
        match self {
            Self::StorageWrite
            | Self::StorageReadWrite
            | Self::CopyDestination
            | Self::MultisampleResolveDestination => true,
            Self::ColorAttachment { .. } => true,
            Self::DepthStencilAttachment { access, .. } => {
                matches!(access, GpuDepthStencilAccess::ReadWrite)
            }
            Self::SampledRead | Self::StorageRead | Self::CopySource | Self::Present => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuQueryAccessKind {
    WriteTimestamp,
    ResolveSource,
}

impl GpuQueryAccessKind {
    pub const fn reads(self) -> bool {
        matches!(self, Self::ResolveSource)
    }

    pub const fn writes(self) -> bool {
        matches!(self, Self::WriteTimestamp)
    }
}

/// Typed buffer access; cross-kind construction is rejected by Rust's type
/// system before graph preparation.
///
/// ```compile_fail
/// use runen_gpu::{
///     GpuBufferAccess, GpuBufferAccessKind, GpuBufferRange, GpuTextureHandle,
/// };
///
/// fn cross_kind(texture: &GpuTextureHandle, range: GpuBufferRange) {
///     let _ = GpuBufferAccess::new(texture, range, GpuBufferAccessKind::StorageRead);
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuBufferAccess {
    buffer: GpuBufferHandle,
    range: GpuBufferRange,
    kind: GpuBufferAccessKind,
}

impl GpuBufferAccess {
    pub fn new(
        buffer: &GpuBufferHandle,
        range: GpuBufferRange,
        kind: GpuBufferAccessKind,
    ) -> Result<Self, GpuAccessError> {
        GpuBufferRange::new(buffer, range.offset(), range.size())?;
        let usage = match kind {
            GpuBufferAccessKind::UniformRead => GpuBufferUsage::Uniform,
            GpuBufferAccessKind::StorageRead
            | GpuBufferAccessKind::StorageWrite
            | GpuBufferAccessKind::StorageReadWrite => GpuBufferUsage::Storage,
            GpuBufferAccessKind::VertexRead => GpuBufferUsage::Vertex,
            GpuBufferAccessKind::IndexRead => GpuBufferUsage::Index,
            GpuBufferAccessKind::IndirectRead => GpuBufferUsage::Indirect,
            GpuBufferAccessKind::CopySource => GpuBufferUsage::CopySource,
            GpuBufferAccessKind::CopyDestination => GpuBufferUsage::CopyDestination,
            GpuBufferAccessKind::QueryResolveDestination => GpuBufferUsage::QueryResolve,
        };
        if !buffer.descriptor().usages().contains(usage) {
            return Err(GpuAccessError::invalid(
                "construct GPU buffer access",
                buffer.descriptor().common().label().as_str(),
                Some(buffer.diagnostic_identity()),
                GpuAccessCause::InvalidDescriptorUsage,
                "add the matching normalized usage to the buffer descriptor",
            ));
        }
        Ok(Self {
            buffer: buffer.clone(),
            range,
            kind,
        })
    }

    pub fn buffer(&self) -> &GpuBufferHandle {
        &self.buffer
    }

    pub const fn range(&self) -> GpuBufferRange {
        self.range
    }

    pub const fn kind(&self) -> GpuBufferAccessKind {
        self.kind
    }

    pub fn resource_identity(&self) -> GpuWorkResourceId {
        self.buffer.diagnostic_identity()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuTextureAccessResource {
    Texture(GpuTextureHandle),
    TextureView(GpuTextureViewHandle),
}

impl From<GpuTextureHandle> for GpuTextureAccessResource {
    fn from(value: GpuTextureHandle) -> Self {
        Self::Texture(value)
    }
}

impl From<GpuTextureViewHandle> for GpuTextureAccessResource {
    fn from(value: GpuTextureViewHandle) -> Self {
        Self::TextureView(value)
    }
}

impl GpuTextureAccessResource {
    pub fn parent_texture(&self) -> &GpuTextureHandle {
        match self {
            Self::Texture(texture) => texture,
            Self::TextureView(view) => view.descriptor().texture(),
        }
    }

    pub fn diagnostic_identity(&self) -> GpuWorkResourceId {
        match self {
            Self::Texture(texture) => texture.diagnostic_identity(),
            Self::TextureView(view) => view.diagnostic_identity(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuTextureAccess {
    resource: GpuTextureAccessResource,
    requested_subresources: GpuTextureSubresourceRange,
    normalized_texture: GpuTextureHandle,
    normalized_subresources: GpuTextureSubresourceRange,
    kind: GpuTextureAccessKind,
}

impl GpuTextureAccess {
    pub fn new(
        resource: GpuTextureAccessResource,
        requested_subresources: GpuTextureSubresourceRange,
        kind: GpuTextureAccessKind,
    ) -> Result<Self, GpuAccessError> {
        let parent = resource.parent_texture().clone();
        validate_texture_range(&parent, requested_subresources)?;
        let normalized_subresources = match &resource {
            GpuTextureAccessResource::Texture(_) => requested_subresources,
            GpuTextureAccessResource::TextureView(view) => {
                if view.descriptor().texture() != &parent {
                    return Err(texture_access_error(
                        &parent,
                        GpuAccessCause::ParentLeaseMismatch,
                        "use a texture view retaining the same checked parent lease",
                    ));
                }
                intersect_texture_ranges(
                    view.descriptor().subresources(),
                    requested_subresources,
                    &parent,
                )?
            }
        };
        validate_texture_usage(&parent, kind)?;
        Ok(Self {
            resource,
            requested_subresources,
            normalized_texture: parent,
            normalized_subresources,
            kind,
        })
    }

    pub fn resource(&self) -> &GpuTextureAccessResource {
        &self.resource
    }

    pub const fn requested_subresources(&self) -> GpuTextureSubresourceRange {
        self.requested_subresources
    }

    pub fn normalized_texture(&self) -> &GpuTextureHandle {
        &self.normalized_texture
    }

    pub const fn normalized_subresources(&self) -> GpuTextureSubresourceRange {
        self.normalized_subresources
    }

    pub const fn kind(&self) -> GpuTextureAccessKind {
        self.kind
    }

    pub fn resource_identity(&self) -> GpuWorkResourceId {
        self.normalized_texture.diagnostic_identity()
    }
}

fn validate_texture_usage(
    texture: &GpuTextureHandle,
    kind: GpuTextureAccessKind,
) -> Result<(), GpuAccessError> {
    let required = match kind {
        GpuTextureAccessKind::SampledRead => Some(GpuTextureUsage::Sampled),
        GpuTextureAccessKind::StorageRead => Some(GpuTextureUsage::StorageRead),
        GpuTextureAccessKind::StorageWrite | GpuTextureAccessKind::StorageReadWrite => {
            Some(GpuTextureUsage::StorageWrite)
        }
        GpuTextureAccessKind::CopySource => Some(GpuTextureUsage::CopySource),
        GpuTextureAccessKind::CopyDestination => Some(GpuTextureUsage::CopyDestination),
        GpuTextureAccessKind::ColorAttachment { .. }
        | GpuTextureAccessKind::MultisampleResolveDestination => {
            Some(GpuTextureUsage::ColorAttachment)
        }
        GpuTextureAccessKind::DepthStencilAttachment { .. } => {
            Some(GpuTextureUsage::DepthStencilAttachment)
        }
        GpuTextureAccessKind::Present => None,
    };
    let usages = texture.descriptor().usages();
    let valid = match kind {
        GpuTextureAccessKind::StorageReadWrite => {
            usages.contains(GpuTextureUsage::StorageRead)
                && usages.contains(GpuTextureUsage::StorageWrite)
        }
        _ => required.is_none_or(|usage| usages.contains(usage)),
    };
    if !valid {
        return Err(texture_access_error(
            texture,
            GpuAccessCause::InvalidDescriptorUsage,
            "add the matching normalized usage to the texture descriptor",
        ));
    }
    let depth = texture.descriptor().format().is_depth();
    if matches!(kind, GpuTextureAccessKind::DepthStencilAttachment { .. }) != depth
        && matches!(
            kind,
            GpuTextureAccessKind::ColorAttachment { .. }
                | GpuTextureAccessKind::MultisampleResolveDestination
                | GpuTextureAccessKind::DepthStencilAttachment { .. }
        )
    {
        return Err(texture_access_error(
            texture,
            GpuAccessCause::InvalidTextureAspect,
            "use color attachment roles with color formats and depth roles with depth formats",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuQueryAccess {
    query_set: GpuQuerySetHandle,
    range: GpuQueryRange,
    kind: GpuQueryAccessKind,
}

impl GpuQueryAccess {
    pub fn new(
        query_set: &GpuQuerySetHandle,
        range: GpuQueryRange,
        kind: GpuQueryAccessKind,
    ) -> Result<Self, GpuAccessError> {
        GpuQueryRange::new(query_set, range.first(), range.count())?;
        Ok(Self {
            query_set: query_set.clone(),
            range,
            kind,
        })
    }

    pub fn query_set(&self) -> &GpuQuerySetHandle {
        &self.query_set
    }

    pub const fn range(&self) -> GpuQueryRange {
        self.range
    }

    pub const fn kind(&self) -> GpuQueryAccessKind {
        self.kind
    }

    pub fn resource_identity(&self) -> GpuWorkResourceId {
        self.query_set.diagnostic_identity()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuSamplerUse {
    sampler: GpuSamplerHandle,
}

impl GpuSamplerUse {
    pub fn new(sampler: &GpuSamplerHandle) -> Self {
        Self {
            sampler: sampler.clone(),
        }
    }

    pub fn sampler(&self) -> &GpuSamplerHandle {
        &self.sampler
    }

    pub fn resource_identity(&self) -> GpuWorkResourceId {
        self.sampler.diagnostic_identity()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuResourceAccess {
    Buffer(GpuBufferAccess),
    Texture(GpuTextureAccess),
    Query(GpuQueryAccess),
    Sampler(GpuSamplerUse),
}

impl GpuResourceAccess {
    pub fn declared_resource_identity(&self) -> GpuWorkResourceId {
        match self {
            Self::Buffer(access) => access.buffer().diagnostic_identity(),
            Self::Texture(access) => access.resource().diagnostic_identity(),
            Self::Query(access) => access.query_set().diagnostic_identity(),
            Self::Sampler(access) => access.sampler().diagnostic_identity(),
        }
    }

    pub fn resource_identity(&self) -> GpuWorkResourceId {
        match self {
            Self::Buffer(access) => access.resource_identity(),
            Self::Texture(access) => access.resource_identity(),
            Self::Query(access) => access.resource_identity(),
            Self::Sampler(access) => access.resource_identity(),
        }
    }

    pub const fn reads(&self) -> bool {
        match self {
            Self::Buffer(access) => access.kind().reads(),
            Self::Texture(access) => access.kind().reads(),
            Self::Query(access) => access.kind().reads(),
            Self::Sampler(_) => true,
        }
    }

    pub const fn writes(&self) -> bool {
        match self {
            Self::Buffer(access) => access.kind().writes(),
            Self::Texture(access) => access.kind().writes(),
            Self::Query(access) => access.kind().writes(),
            Self::Sampler(_) => false,
        }
    }

    pub const fn intent(&self) -> GpuResourceAccessIntent {
        match (self.reads(), self.writes()) {
            (true, false) => GpuResourceAccessIntent::Read,
            (false, true) => GpuResourceAccessIntent::Write,
            (true, true) => GpuResourceAccessIntent::ReadWrite,
            (false, false) => GpuResourceAccessIntent::Read,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GpuBufferDescriptor, GpuBufferInitialization, GpuBufferUsages, GpuMemoryIntent,
        GpuQueryKind, GpuQuerySetDescriptor, GpuReconstruction, GpuResourceCommon,
        GpuResourceLabel, GpuResourceLifetime, GpuResourceProvenance, GpuTextureDescriptor,
        GpuTextureExtent, GpuTextureFormat, GpuTextureInitialization, GpuTextureUsages,
        GpuTextureViewDescriptor, GpuWorkResourceIdAllocator,
    };
    use std::{
        collections::hash_map::DefaultHasher,
        hash::{Hash, Hasher},
        num::NonZeroU64,
    };

    fn label(value: &str) -> GpuResourceLabel {
        GpuResourceLabel::new(value).unwrap()
    }

    fn common(value: &str) -> GpuResourceCommon {
        let label = label(value);
        GpuResourceCommon::owned(
            label.clone(),
            GpuResourceLifetime::Transient,
            GpuMemoryIntent::Device,
            GpuReconstruction::SourceBacked,
            GpuResourceProvenance::new(label, None, None),
        )
        .unwrap()
    }

    fn allocator() -> GpuWorkResourceIdAllocator {
        GpuWorkResourceIdAllocator::for_owner_scope(NonZeroU64::new(41).unwrap())
    }

    fn buffer(
        allocator: &mut GpuWorkResourceIdAllocator,
        name: &str,
        usages: impl IntoIterator<Item = GpuBufferUsage>,
    ) -> GpuBufferHandle {
        let label = label(name);
        allocator
            .allocate_buffer_handle(
                GpuBufferDescriptor::new(
                    common(name),
                    64,
                    GpuBufferUsages::new(&label, usages).unwrap(),
                    GpuBufferInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap()
    }

    fn semantic_hash(value: impl Hash) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn checked_buffer_and_query_ranges_reject_zero_overflow_and_bounds() {
        let mut allocator = allocator();
        let buffer = buffer(&mut allocator, "buffer", [GpuBufferUsage::Storage]);
        assert_eq!(GpuBufferRange::whole(&buffer).unwrap().size(), 64);
        assert!(GpuBufferRange::new(&buffer, 4, 8).is_ok());
        assert_eq!(
            GpuBufferRange::new(&buffer, 0, 0).unwrap_err().cause(),
            GpuAccessCause::ZeroRange
        );
        assert_eq!(
            GpuBufferRange::new(&buffer, u64::MAX, 2)
                .unwrap_err()
                .cause(),
            GpuAccessCause::ArithmeticOverflow
        );
        assert_eq!(
            GpuBufferRange::new(&buffer, 60, 8).unwrap_err().cause(),
            GpuAccessCause::OutOfBounds
        );

        let queries = allocator
            .allocate_query_set_handle(
                GpuQuerySetDescriptor::new(common("queries"), GpuQueryKind::Timestamp, 8).unwrap(),
            )
            .unwrap();
        assert_eq!(GpuQueryRange::whole(&queries).unwrap().count(), 8);
        assert!(GpuQueryRange::new(&queries, 2, 4).is_ok());
        assert_eq!(
            GpuQueryRange::new(&queries, 0, 0).unwrap_err().cause(),
            GpuAccessCause::ZeroRange
        );
        assert_eq!(
            GpuQueryRange::new(&queries, u32::MAX, 2)
                .unwrap_err()
                .cause(),
            GpuAccessCause::ArithmeticOverflow
        );
        assert_eq!(
            GpuQueryRange::new(&queries, 7, 2).unwrap_err().cause(),
            GpuAccessCause::OutOfBounds
        );
    }

    #[test]
    fn buffer_and_query_ranges_have_diagnostic_free_order_and_hashing() {
        let mut allocator = allocator();
        let first_buffer = buffer(&mut allocator, "first buffer", [GpuBufferUsage::Storage]);
        let second_buffer = buffer(&mut allocator, "second buffer", [GpuBufferUsage::Storage]);
        let first = GpuBufferRange::new(&first_buffer, 8, 16).unwrap();
        let same = GpuBufferRange::new(&second_buffer, 8, 16).unwrap();
        let later = GpuBufferRange::new(&first_buffer, 24, 8).unwrap();
        assert_eq!(first, same);
        assert_eq!(semantic_hash(first), semantic_hash(same));
        assert!(first < later);

        let first_queries = allocator
            .allocate_query_set_handle(
                GpuQuerySetDescriptor::new(common("first queries"), GpuQueryKind::Timestamp, 8)
                    .unwrap(),
            )
            .unwrap();
        let second_queries = allocator
            .allocate_query_set_handle(
                GpuQuerySetDescriptor::new(common("second queries"), GpuQueryKind::Timestamp, 8)
                    .unwrap(),
            )
            .unwrap();
        let first = GpuQueryRange::new(&first_queries, 2, 3).unwrap();
        let same = GpuQueryRange::new(&second_queries, 2, 3).unwrap();
        let later = GpuQueryRange::new(&first_queries, 5, 1).unwrap();
        assert_eq!(first, same);
        assert_eq!(semantic_hash(first), semantic_hash(same));
        assert!(first < later);
    }

    #[test]
    fn texture_subresource_order_and_hashing_ignore_diagnostic_labels() {
        let first = GpuTextureSubresourceRange::new(
            &label("first diagnostic"),
            0,
            1,
            1,
            2,
            GpuTextureAspect::Color,
        )
        .unwrap();
        let same = GpuTextureSubresourceRange::new(
            &label("second diagnostic"),
            0,
            1,
            1,
            2,
            GpuTextureAspect::Color,
        )
        .unwrap();
        let later = GpuTextureSubresourceRange::new(
            &label("third diagnostic"),
            1,
            1,
            0,
            1,
            GpuTextureAspect::Color,
        )
        .unwrap();
        assert_eq!(first, same);
        assert_eq!(semantic_hash(first), semantic_hash(same));
        assert!(first < later);
    }

    #[test]
    fn access_kinds_require_the_matching_descriptor_usage() {
        let mut allocator = allocator();
        let storage = buffer(&mut allocator, "storage", [GpuBufferUsage::Storage]);
        let range = GpuBufferRange::whole(&storage).unwrap();
        assert!(GpuBufferAccess::new(&storage, range, GpuBufferAccessKind::StorageRead).is_ok());
        assert!(
            GpuBufferAccess::new(
                &storage,
                range,
                GpuBufferAccessKind::QueryResolveDestination,
            )
            .is_err()
        );
        let resolve = buffer(&mut allocator, "resolve", [GpuBufferUsage::QueryResolve]);
        let range = GpuBufferRange::whole(&resolve).unwrap();
        assert!(
            GpuBufferAccess::new(
                &resolve,
                range,
                GpuBufferAccessKind::QueryResolveDestination,
            )
            .is_ok()
        );
    }

    #[test]
    fn texture_view_access_normalizes_to_parent_storage() {
        let mut allocator = allocator();
        let texture_label = label("texture");
        let texture = allocator
            .allocate_texture_handle(
                GpuTextureDescriptor::new(
                    common("texture"),
                    GpuTextureDimension::D2,
                    GpuTextureExtent::new(&texture_label, GpuTextureDimension::D2, 16, 16, 4)
                        .unwrap(),
                    3,
                    1,
                    GpuTextureFormat::Rgba8Unorm,
                    GpuTextureUsages::new(&texture_label, [GpuTextureUsage::Sampled]).unwrap(),
                    GpuTextureInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();
        let view_label = label("view");
        let view_range =
            GpuTextureSubresourceRange::new(&view_label, 1, 2, 1, 2, GpuTextureAspect::Color)
                .unwrap();
        let view = allocator
            .allocate_texture_view_handle(
                GpuTextureViewDescriptor::new(
                    common("view"),
                    &texture,
                    None,
                    GpuTextureDimension::D2,
                    view_range,
                )
                .unwrap(),
            )
            .unwrap();
        let requested =
            GpuTextureSubresourceRange::new(&view_label, 2, 1, 2, 1, GpuTextureAspect::All)
                .unwrap();
        let access = GpuTextureAccess::new(
            GpuTextureAccessResource::TextureView(view.clone()),
            requested,
            GpuTextureAccessKind::SampledRead,
        )
        .unwrap();
        assert_eq!(access.normalized_texture(), &texture);
        assert_eq!(access.normalized_subresources().base_mip_level(), 2);
        assert_eq!(access.normalized_subresources().base_array_layer(), 2);
        assert_eq!(
            access.normalized_subresources().aspect(),
            GpuTextureAspect::Color
        );

        let outside_view =
            GpuTextureSubresourceRange::new(&view_label, 0, 1, 0, 1, GpuTextureAspect::Color)
                .unwrap();
        assert_eq!(
            GpuTextureAccess::new(
                GpuTextureAccessResource::TextureView(view),
                outside_view,
                GpuTextureAccessKind::SampledRead,
            )
            .unwrap_err()
            .cause(),
            GpuAccessCause::InvalidViewIntersection
        );
    }

    #[test]
    fn texture_ranges_are_bounded_and_overlap_aware() {
        let mut allocator = allocator();
        let texture_label = label("array texture");
        let texture = allocator
            .allocate_texture_handle(
                GpuTextureDescriptor::new(
                    common("array texture"),
                    GpuTextureDimension::D2,
                    GpuTextureExtent::new(&texture_label, GpuTextureDimension::D2, 16, 16, 4)
                        .unwrap(),
                    3,
                    1,
                    GpuTextureFormat::Rgba8Unorm,
                    GpuTextureUsages::new(&texture_label, [GpuTextureUsage::Sampled]).unwrap(),
                    GpuTextureInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();
        let first =
            GpuTextureSubresourceRange::new(&texture_label, 0, 1, 0, 2, GpuTextureAspect::Color)
                .unwrap();
        let overlapping =
            GpuTextureSubresourceRange::new(&texture_label, 0, 1, 1, 2, GpuTextureAspect::All)
                .unwrap();
        let disjoint_mip =
            GpuTextureSubresourceRange::new(&texture_label, 1, 1, 0, 2, GpuTextureAspect::Color)
                .unwrap();
        assert!(GpuTextureSubresourceRange::checked_for(&texture, first).is_ok());
        assert!(first.overlaps(overlapping, GpuTextureAspect::Color));
        assert!(!first.overlaps(disjoint_mip, GpuTextureAspect::Color));

        let mip_out_of_bounds =
            GpuTextureSubresourceRange::new(&texture_label, 2, 2, 0, 1, GpuTextureAspect::Color)
                .unwrap();
        assert_eq!(
            GpuTextureSubresourceRange::checked_for(&texture, mip_out_of_bounds)
                .unwrap_err()
                .cause(),
            GpuAccessCause::OutOfBounds
        );
        let layer_out_of_bounds =
            GpuTextureSubresourceRange::new(&texture_label, 0, 1, 3, 2, GpuTextureAspect::Color)
                .unwrap();
        assert_eq!(
            GpuTextureSubresourceRange::checked_for(&texture, layer_out_of_bounds)
                .unwrap_err()
                .cause(),
            GpuAccessCause::OutOfBounds
        );
        let invalid_aspect = GpuTextureSubresourceRange::new(
            &texture_label,
            0,
            1,
            0,
            1,
            GpuTextureAspect::DepthOnly,
        )
        .unwrap();
        assert_eq!(
            GpuTextureSubresourceRange::checked_for(&texture, invalid_aspect)
                .unwrap_err()
                .cause(),
            GpuAccessCause::InvalidTextureAspect
        );
    }

    #[test]
    fn d3_mips_are_addressed_as_whole_volumes() {
        let mut allocator = allocator();
        let volume_label = label("volume texture");
        let volume = allocator
            .allocate_texture_handle(
                GpuTextureDescriptor::new(
                    common("volume texture"),
                    GpuTextureDimension::D3,
                    GpuTextureExtent::new(&volume_label, GpuTextureDimension::D3, 8, 8, 8).unwrap(),
                    4,
                    1,
                    GpuTextureFormat::Rgba8Unorm,
                    GpuTextureUsages::new(&volume_label, [GpuTextureUsage::Sampled]).unwrap(),
                    GpuTextureInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();
        let whole_volume = GpuTextureSubresourceRange::whole(&volume).unwrap();
        assert_eq!(whole_volume.mip_level_count(), 4);
        assert_eq!(whole_volume.base_array_layer(), 0);
        assert_eq!(whole_volume.array_layer_count(), 1);
        let sliced_volume =
            GpuTextureSubresourceRange::new(&volume_label, 0, 1, 1, 1, GpuTextureAspect::Color)
                .unwrap();
        assert_eq!(
            GpuTextureSubresourceRange::checked_for(&volume, sliced_volume)
                .unwrap_err()
                .cause(),
            GpuAccessCause::InvalidD3Interpretation
        );
    }
}
