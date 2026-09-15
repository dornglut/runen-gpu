use super::{
    GpuBufferAccess, GpuBufferAccessKind, GpuBufferRegion, GpuReadbackId, GpuResourceAccess,
    GpuTextureAccess, GpuTextureAccessKind, GpuTextureAccessResource, GpuTextureCopyRegion,
    GpuTextureDimension, GpuWorkOperationCause, GpuWorkOperationError, PreparedGpuData,
    TransferData,
};

/// Exact logical source/destination region for CPU/GPU transfer work.
///
/// Texture regions retain their exact origin and extent even though the accepted G3 hazard envelope
/// is conservatively normalized to texture subresources.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuTransferRegion {
    Buffer(GpuBufferRegion),
    Texture(GpuTextureCopyRegion),
}

impl GpuTransferRegion {
    pub fn logical_byte_len(&self) -> Result<u64, GpuWorkOperationError> {
        match self {
            Self::Buffer(region) => Ok(region.range().size()),
            Self::Texture(region) => region.tightly_packed_byte_len().ok_or_else(|| {
                transfer_error(
                    "derive GPU texture transfer byte length",
                    region.texture().diagnostic_identity(),
                    "reduce the logical texture transfer extent or choose a copyable normalized format aspect",
                )
            }),
        }
    }

    pub(crate) fn access(
        &self,
        buffer_kind: GpuBufferAccessKind,
        texture_kind: GpuTextureAccessKind,
        operation: &'static str,
    ) -> Result<GpuResourceAccess, GpuWorkOperationError> {
        match self {
            Self::Buffer(region) => {
                GpuBufferAccess::new(region.buffer(), region.range(), buffer_kind)
                    .map(GpuResourceAccess::Buffer)
                    .map_err(|source| {
                        GpuWorkOperationError::from_access(
                            operation,
                            region.buffer().descriptor().common().label().as_str(),
                            GpuWorkOperationCause::InvalidCopyRegion,
                            "use a checked transfer range with matching copy usage",
                            source,
                        )
                    })
            }
            Self::Texture(region) => GpuTextureAccess::new(
                GpuTextureAccessResource::Texture(region.texture().clone()),
                region.subresources(),
                texture_kind,
            )
            .map(GpuResourceAccess::Texture)
            .map_err(|source| {
                GpuWorkOperationError::from_access(
                    operation,
                    region.texture().descriptor().common().label().as_str(),
                    GpuWorkOperationCause::InvalidCopyRegion,
                    "use a checked transfer region with matching copy usage",
                    source,
                )
            }),
        }
    }

    /// Whether the exact logical region establishes initialization for every normalized texture
    /// subresource it touches. Buffers are byte-range exact and therefore always return true.
    pub(crate) fn establishes_initialization_effect(&self) -> bool {
        let Self::Texture(region) = self else {
            return true;
        };
        texture_region_completely_covers_selected_subresources(region)
    }
}

impl From<GpuBufferRegion> for GpuTransferRegion {
    fn from(value: GpuBufferRegion) -> Self {
        Self::Buffer(value)
    }
}

impl From<GpuTextureCopyRegion> for GpuTransferRegion {
    fn from(value: GpuTextureCopyRegion) -> Self {
        Self::Texture(value)
    }
}

/// Immutable logical CPU-to-GPU transfer work. Physical staging and queue strategy remain G5B.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuUploadOperation {
    destination: GpuTransferRegion,
    payload: PreparedGpuData<TransferData>,
    destination_access: GpuResourceAccess,
}

impl GpuUploadOperation {
    pub fn new(
        destination: GpuTransferRegion,
        payload: PreparedGpuData<TransferData>,
    ) -> Result<Self, GpuWorkOperationError> {
        let expected = destination.logical_byte_len()?;
        if payload.layout().byte_len() != expected {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU upload operation",
                format!(
                    "payload_bytes={}, destination_bytes={expected}",
                    payload.layout().byte_len()
                ),
                Some(destination_identity(&destination)),
                GpuWorkOperationCause::InvalidCopyLayout,
                "provide an immutable tightly packed transfer payload exactly covering the logical destination region",
            ));
        }
        let destination_access = destination.access(
            GpuBufferAccessKind::CopyDestination,
            GpuTextureAccessKind::CopyDestination,
            "derive GPU upload destination access",
        )?;
        Ok(Self {
            destination,
            payload,
            destination_access,
        })
    }

    pub fn destination(&self) -> &GpuTransferRegion {
        &self.destination
    }

    pub fn payload(&self) -> &PreparedGpuData<TransferData> {
        &self.payload
    }

    pub fn destination_access(&self) -> &GpuResourceAccess {
        &self.destination_access
    }

    pub(crate) fn establishes_initialization_effect(&self) -> bool {
        self.destination.establishes_initialization_effect()
    }
}

/// Logical GPU-to-CPU transfer request. Result storage/mapping/materialization remain G5B.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuReadbackOperation {
    source: GpuTransferRegion,
    id: GpuReadbackId,
    source_access: GpuResourceAccess,
}

impl GpuReadbackOperation {
    pub fn new(
        source: GpuTransferRegion,
        id: GpuReadbackId,
    ) -> Result<Self, GpuWorkOperationError> {
        let source_access = source.access(
            GpuBufferAccessKind::CopySource,
            GpuTextureAccessKind::CopySource,
            "derive GPU readback source access",
        )?;
        Ok(Self {
            source,
            id,
            source_access,
        })
    }

    pub fn source(&self) -> &GpuTransferRegion {
        &self.source
    }

    pub const fn id(&self) -> GpuReadbackId {
        self.id
    }

    pub fn source_access(&self) -> &GpuResourceAccess {
        &self.source_access
    }

    pub fn logical_byte_len(&self) -> Result<u64, GpuWorkOperationError> {
        self.source.logical_byte_len()
    }
}

fn destination_identity(region: &GpuTransferRegion) -> super::GpuWorkResourceId {
    match region {
        GpuTransferRegion::Buffer(region) => region.buffer().diagnostic_identity(),
        GpuTransferRegion::Texture(region) => region.texture().diagnostic_identity(),
    }
}

fn texture_region_completely_covers_selected_subresources(region: &GpuTextureCopyRegion) -> bool {
    let descriptor = region.texture().descriptor();
    let mip = region.mip_level();
    let width = (descriptor.extent().width() >> mip).max(1);
    let height = (descriptor.extent().height() >> mip).max(1);
    let origin = region.origin();
    let extent = region.extent();
    if origin.x() != 0 || origin.y() != 0 || extent.width() != width || extent.height() != height {
        return false;
    }
    match descriptor.dimension() {
        GpuTextureDimension::D1 => origin.z() == 0 && extent.depth_or_layers() == 1,
        GpuTextureDimension::D2 => true,
        GpuTextureDimension::D3 => {
            origin.z() == 0
                && extent.depth_or_layers() == (descriptor.extent().depth_or_layers() >> mip).max(1)
        }
    }
}

fn transfer_error(
    operation: &'static str,
    resource: super::GpuWorkResourceId,
    correction: &'static str,
) -> GpuWorkOperationError {
    GpuWorkOperationError::invalid(
        operation,
        "transfer region",
        Some(resource),
        GpuWorkOperationCause::InvalidCopyLayout,
        correction,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GpuBufferDescriptor, GpuBufferInitialization, GpuBufferRange, GpuBufferUsage,
        GpuBufferUsages, GpuCopyExtent, GpuMemoryIntent, GpuReconstruction, GpuResourceCommon,
        GpuResourceLabel, GpuResourceLifetime, GpuResourceProvenance, GpuTextureAspect,
        GpuTextureDescriptor, GpuTextureExtent, GpuTextureFormat, GpuTextureInitialization,
        GpuTextureOrigin, GpuTextureUsage, GpuTextureUsages, GpuWorkResourceIdAllocator,
    };
    use std::num::NonZeroU64;

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

    fn transfer_payload(name: &str, byte_len: usize) -> PreparedGpuData<TransferData> {
        let resource_label = label(name);
        PreparedGpuData::from_pod_transfer(
            name,
            &vec![0_u8; byte_len],
            GpuResourceProvenance::new(resource_label, None, None),
        )
        .unwrap()
    }

    fn texture_region(
        allocator: &mut GpuWorkResourceIdAllocator,
        name: &str,
        format: GpuTextureFormat,
        aspect: GpuTextureAspect,
    ) -> GpuTextureCopyRegion {
        let resource_label = label(name);
        let texture = allocator
            .allocate_texture_handle(
                GpuTextureDescriptor::new(
                    common(name),
                    GpuTextureDimension::D2,
                    GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, 8, 8, 1)
                        .unwrap(),
                    1,
                    1,
                    format,
                    GpuTextureUsages::new(
                        &resource_label,
                        [
                            GpuTextureUsage::CopyDestination,
                            GpuTextureUsage::CopySource,
                        ],
                    )
                    .unwrap(),
                    GpuTextureInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();
        GpuTextureCopyRegion::new(
            &texture,
            0,
            GpuTextureOrigin::new(0, 0, 0),
            aspect,
            GpuCopyExtent::new(8, 8, 1).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn buffer_upload_requires_exact_payload_and_derives_copy_destination_access() {
        let mut allocator =
            GpuWorkResourceIdAllocator::for_owner_scope(NonZeroU64::new(111).unwrap());
        let resource_label = label("upload buffer");
        let buffer = allocator
            .allocate_buffer_handle(
                GpuBufferDescriptor::new(
                    common("upload buffer"),
                    32,
                    GpuBufferUsages::new(
                        &resource_label,
                        [GpuBufferUsage::CopyDestination, GpuBufferUsage::CopySource],
                    )
                    .unwrap(),
                    GpuBufferInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();
        let region =
            GpuBufferRegion::new(&buffer, GpuBufferRange::new(&buffer, 8, 16).unwrap()).unwrap();

        let upload =
            GpuUploadOperation::new(region.clone().into(), transfer_payload("payload", 16))
                .unwrap();
        assert_eq!(upload.payload().layout().byte_len(), 16);
        assert!(upload.destination_access().writes());
        assert!(upload.establishes_initialization_effect());
        assert!(GpuUploadOperation::new(region.into(), transfer_payload("short", 15)).is_err());
    }

    #[test]
    fn texture_transfer_uses_tightly_packed_logical_bytes_and_partial_region_is_not_full_init() {
        let mut allocator =
            GpuWorkResourceIdAllocator::for_owner_scope(NonZeroU64::new(112).unwrap());
        let resource_label = label("texture");
        let texture = allocator
            .allocate_texture_handle(
                GpuTextureDescriptor::new(
                    common("texture"),
                    GpuTextureDimension::D2,
                    GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, 8, 8, 1)
                        .unwrap(),
                    1,
                    1,
                    GpuTextureFormat::Rgba8Unorm,
                    GpuTextureUsages::new(
                        &resource_label,
                        [
                            GpuTextureUsage::CopyDestination,
                            GpuTextureUsage::CopySource,
                        ],
                    )
                    .unwrap(),
                    GpuTextureInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();
        let region = GpuTextureCopyRegion::new(
            &texture,
            0,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::Color,
            GpuCopyExtent::new(4, 8, 1).unwrap(),
        )
        .unwrap();
        let upload = GpuUploadOperation::new(
            region.clone().into(),
            transfer_payload("texture payload", 4 * 8 * 4),
        )
        .unwrap();
        assert_eq!(upload.destination().logical_byte_len().unwrap(), 128);
        assert!(!upload.establishes_initialization_effect());

        let readback =
            GpuReadbackOperation::new(region.into(), GpuReadbackId::allocate().unwrap()).unwrap();
        assert_eq!(readback.logical_byte_len().unwrap(), 128);
        assert!(readback.source_access().reads());
        assert!(!readback.source_access().writes());
    }

    #[test]
    fn texture_transfer_byte_lengths_follow_normalized_format_structure() {
        let mut allocator =
            GpuWorkResourceIdAllocator::for_owner_scope(NonZeroU64::new(113).unwrap());
        for (name, format, aspect, expected) in [
            ("r8", GpuTextureFormat::R8Unorm, GpuTextureAspect::Color, 64),
            (
                "rgba8",
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureAspect::Color,
                256,
            ),
            (
                "r32float",
                GpuTextureFormat::R32Float,
                GpuTextureAspect::Color,
                256,
            ),
            (
                "depth32",
                GpuTextureFormat::Depth32Float,
                GpuTextureAspect::DepthOnly,
                256,
            ),
        ] {
            let region = texture_region(&mut allocator, name, format, aspect);
            assert_eq!(GpuTransferRegion::from(region.clone()).logical_byte_len().unwrap(), expected);
            let readback =
                GpuReadbackOperation::new(region.into(), GpuReadbackId::allocate().unwrap())
                    .unwrap();
            assert_eq!(readback.logical_byte_len().unwrap(), expected);
        }
    }
}
