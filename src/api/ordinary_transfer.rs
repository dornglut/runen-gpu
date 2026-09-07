use super::{
    GpuBufferHandle, GpuBufferRange, GpuBufferRegion, GpuCopyExtent, GpuDataPreparationError,
    GpuReadbackId, GpuReadbackIdAllocationError, GpuReadbackOperation, GpuResourceDescriptorError,
    GpuResourceLabel, GpuResourceProvenance, GpuTextureAspect, GpuTextureCopyRegion,
    GpuTextureHandle, GpuTextureOrigin, GpuTransferRegion, GpuUploadOperation,
    GpuWorkOperationCause, GpuWorkOperationError, PreparedGpuData, TransferData,
};
use core::fmt;

impl GpuBufferRegion {
    /// Constructs one transfer region covering the complete validated buffer.
    ///
    /// Subranges remain available through [`GpuBufferRegion::new`].
    pub fn whole(buffer: &GpuBufferHandle) -> Result<Self, GpuWorkOperationError> {
        let range = GpuBufferRange::whole(buffer).map_err(|source| {
            GpuWorkOperationError::from_access(
                "construct whole GPU buffer region",
                buffer.descriptor().common().label().as_str(),
                GpuWorkOperationCause::InvalidCopyRegion,
                "use a validated nonempty buffer descriptor",
                source,
            )
        })?;
        Self::new(buffer, range)
    }
}

impl GpuTextureCopyRegion {
    /// Constructs one transfer region covering the complete base mip from zero origin.
    ///
    /// The descriptor supplies the full base-mip extent, including all D2 array layers or
    /// D3 depth slices. The canonical texture-copy constructor still validates single-sample
    /// eligibility and normalizes the texture aspect from the format. Partial coverage and
    /// nonzero mip selection remain available through [`GpuTextureCopyRegion::new`].
    pub fn whole_base_mip(texture: &GpuTextureHandle) -> Result<Self, GpuWorkOperationError> {
        let extent = texture.descriptor().extent();
        Self::new(
            texture,
            0,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::All,
            GpuCopyExtent::new(extent.width(), extent.height(), extent.depth_or_layers())?,
        )
    }
}

/// Failure while preparing an ordinary Pod transfer payload.
///
/// Diagnostic-label validation and canonical data preparation remain separate authorities; this
/// error only preserves both outcomes for the convenience constructor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuOrdinaryTransferPreparationError {
    Label(GpuResourceDescriptorError),
    Data(GpuDataPreparationError),
}

impl fmt::Display for GpuOrdinaryTransferPreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Label(error) => error.fmt(formatter),
            Self::Data(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for GpuOrdinaryTransferPreparationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Label(error) => Some(error),
            Self::Data(error) => Some(error),
        }
    }
}

impl From<GpuResourceDescriptorError> for GpuOrdinaryTransferPreparationError {
    fn from(error: GpuResourceDescriptorError) -> Self {
        Self::Label(error)
    }
}

impl From<GpuDataPreparationError> for GpuOrdinaryTransferPreparationError {
    fn from(error: GpuDataPreparationError) -> Self {
        Self::Data(error)
    }
}

impl PreparedGpuData<TransferData> {
    /// Prepares ordinary Pod transfer data while deriving default diagnostic provenance.
    ///
    /// The caller supplies one diagnostic label and the typed payload. RunenGPU validates the
    /// label, then reuses it as the default provenance producer with no source generation or
    /// revision. Call [`PreparedGpuData::<TransferData>::from_pod_transfer`] when explicit
    /// provenance is a meaningful input.
    pub fn ordinary_pod_transfer<Source: bytemuck::Pod>(
        label: impl AsRef<str>,
        values: &[Source],
    ) -> Result<Self, GpuOrdinaryTransferPreparationError> {
        let label = GpuResourceLabel::new(label.as_ref())?;
        let provenance = GpuResourceProvenance::new(label.clone(), None, None);
        Self::from_pod_transfer(label.as_str(), values, provenance).map_err(Into::into)
    }
}

impl GpuUploadOperation {
    /// Constructs a canonical upload covering one complete validated buffer.
    ///
    /// Payload preparation remains explicit and reusable. Partial-buffer and texture uploads
    /// remain available through [`GpuUploadOperation::new`].
    pub fn whole_buffer(
        buffer: &GpuBufferHandle,
        payload: PreparedGpuData<TransferData>,
    ) -> Result<Self, GpuWorkOperationError> {
        Self::new(GpuBufferRegion::whole(buffer)?.into(), payload)
    }
}

/// Failure while constructing an ordinary readback request.
///
/// Correlation identity allocation and canonical operation validation remain separate authorities;
/// this error only preserves both outcomes for the convenience constructor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuReadbackRequestError {
    CorrelationId(GpuReadbackIdAllocationError),
    Operation(GpuWorkOperationError),
}

impl fmt::Display for GpuReadbackRequestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CorrelationId(error) => error.fmt(formatter),
            Self::Operation(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for GpuReadbackRequestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CorrelationId(error) => Some(error),
            Self::Operation(error) => Some(error),
        }
    }
}

impl From<GpuReadbackIdAllocationError> for GpuReadbackRequestError {
    fn from(error: GpuReadbackIdAllocationError) -> Self {
        Self::CorrelationId(error)
    }
}

impl From<GpuWorkOperationError> for GpuReadbackRequestError {
    fn from(error: GpuWorkOperationError) -> Self {
        Self::Operation(error)
    }
}

impl GpuReadbackOperation {
    /// Constructs an ordinary readback request and allocates its correlation identity internally.
    ///
    /// The caller chooses the source region and can retain [`GpuReadbackOperation::id`] before
    /// moving the operation into authored work. Use [`GpuReadbackOperation::new`] when an existing
    /// correlation identity is intentionally supplied.
    pub fn ordinary(source: GpuTransferRegion) -> Result<Self, GpuReadbackRequestError> {
        let id = GpuReadbackId::allocate()?;
        Self::new(source, id).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GpuBufferDescriptor, GpuBufferInitialization, GpuBufferUsage, GpuReconstruction,
        GpuResourceDescriptorCause, GpuResourceLifetime, GpuResourceScope, GpuTextureDescriptor,
        GpuTextureFormat, GpuTextureInitialization, GpuTextureUsage,
    };

    #[test]
    fn whole_buffer_region_preserves_complete_descriptor_range() {
        let mut resources = GpuResourceScope::new();
        let buffer = resources
            .buffer(
                GpuBufferDescriptor::ordinary_owned(
                    "whole transfer buffer",
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    96,
                    [GpuBufferUsage::CopySource],
                    GpuBufferInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();

        let region = GpuBufferRegion::whole(&buffer).unwrap();
        assert_eq!(region.buffer(), &buffer);
        assert_eq!(region.range().offset(), 0);
        assert_eq!(region.range().size(), 96);
    }

    #[test]
    fn whole_base_mip_region_preserves_full_ordinary_2d_extent_and_color_aspect() {
        let mut resources = GpuResourceScope::new();
        let texture = resources
            .texture(
                GpuTextureDescriptor::ordinary_owned_2d(
                    "whole transfer texture",
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    17,
                    11,
                    GpuTextureFormat::Rgba8Unorm,
                    [GpuTextureUsage::CopySource],
                    GpuTextureInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();

        let region = GpuTextureCopyRegion::whole_base_mip(&texture).unwrap();
        assert_eq!(region.texture(), &texture);
        assert_eq!(region.mip_level(), 0);
        assert_eq!(region.origin(), GpuTextureOrigin::new(0, 0, 0));
        assert_eq!(region.aspect(), GpuTextureAspect::Color);
        assert_eq!(region.extent(), GpuCopyExtent::new(17, 11, 1).unwrap());
        assert_eq!(region.subresources().base_mip_level(), 0);
        assert_eq!(region.subresources().mip_level_count(), 1);
        assert_eq!(region.subresources().base_array_layer(), 0);
        assert_eq!(region.subresources().array_layer_count(), 1);
    }

    #[test]
    fn ordinary_pod_transfer_derives_default_provenance_from_label() {
        let data = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
            "ordinary transfer payload",
            &[1_u32, 2_u32],
        )
        .unwrap();

        assert_eq!(
            data.provenance().producer().as_str(),
            "ordinary transfer payload"
        );
        assert_eq!(data.provenance().source_generation(), None);
        assert_eq!(data.provenance().source_revision(), None);
        assert_eq!(data.layout().element_count(), 2);
        assert_eq!(data.layout().byte_len(), 8);
    }

    #[test]
    fn ordinary_pod_transfer_preserves_label_validation() {
        let error =
            PreparedGpuData::<TransferData>::ordinary_pod_transfer("   ", &[1_u32]).unwrap_err();
        let GpuOrdinaryTransferPreparationError::Label(error) = error else {
            panic!("empty ordinary transfer labels must fail in the resource-label authority");
        };
        assert_eq!(error.cause(), GpuResourceDescriptorCause::EmptyLabel);
    }

    #[test]
    fn whole_buffer_upload_derives_region_and_preserves_canonical_validation() {
        let mut resources = GpuResourceScope::new();
        let buffer = resources
            .buffer(
                GpuBufferDescriptor::ordinary_owned(
                    "whole upload buffer",
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    8,
                    [GpuBufferUsage::CopyDestination],
                    GpuBufferInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();
        let payload = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
            "whole upload payload",
            &[1_u32, 2_u32],
        )
        .unwrap();

        let upload = GpuUploadOperation::whole_buffer(&buffer, payload).unwrap();
        let GpuTransferRegion::Buffer(region) = upload.destination() else {
            panic!("whole-buffer upload must retain a buffer transfer region");
        };
        assert_eq!(region.buffer(), &buffer);
        assert_eq!(region.range().offset(), 0);
        assert_eq!(region.range().size(), 8);

        let short = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
            "short whole upload payload",
            &[1_u32],
        )
        .unwrap();
        assert!(GpuUploadOperation::whole_buffer(&buffer, short).is_err());
    }

    #[test]
    fn ordinary_readback_allocates_correlation_identity_and_retains_source() {
        let mut resources = GpuResourceScope::new();
        let buffer = resources
            .buffer(
                GpuBufferDescriptor::ordinary_owned(
                    "ordinary readback buffer",
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    32,
                    [GpuBufferUsage::CopySource],
                    GpuBufferInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();
        let region = GpuBufferRegion::whole(&buffer).unwrap();

        let first = GpuReadbackOperation::ordinary(region.clone().into()).unwrap();
        let second = GpuReadbackOperation::ordinary(region.clone().into()).unwrap();

        assert_eq!(first.source(), &GpuTransferRegion::Buffer(region));
        assert_ne!(first.id(), second.id());
    }
}
