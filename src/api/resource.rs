use super::{
    GpuResourceDescriptorCause, GpuResourceDescriptorError, GpuResourceRef, GpuTextureFormat,
    GpuTextureHandle, PreparedGpuData, TransferData,
};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuResourceLabel(String);

impl GpuResourceLabel {
    pub fn new(value: impl Into<String>) -> Result<Self, GpuResourceDescriptorError> {
        let value = value.into();
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU resource label",
                "<empty>",
                GpuResourceDescriptorCause::EmptyLabel,
                "provide a non-empty diagnostic label",
            ));
        }
        Ok(Self(trimmed.to_string()))
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl AsRef<str> for GpuResourceLabel {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuResourceProvenance {
    producer: GpuResourceLabel,
    source_generation: Option<u64>,
    source_revision: Option<GpuResourceLabel>,
}

impl GpuResourceProvenance {
    pub fn new(
        producer: GpuResourceLabel,
        source_generation: Option<u64>,
        source_revision: Option<GpuResourceLabel>,
    ) -> Self {
        Self {
            producer,
            source_generation,
            source_revision,
        }
    }

    pub fn producer(&self) -> &GpuResourceLabel {
        &self.producer
    }
    pub const fn source_generation(&self) -> Option<u64> {
        self.source_generation
    }
    pub fn source_revision(&self) -> Option<&GpuResourceLabel> {
        self.source_revision.as_ref()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuResourceLifetime {
    Transient,
    Retained,
}

impl GpuResourceLifetime {
    pub const fn is_transient(self) -> bool {
        matches!(self, Self::Transient)
    }
    pub const fn is_retained(self) -> bool {
        matches!(self, Self::Retained)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuResourceOwnership {
    Owned,
    Imported,
    SurfaceAcquired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuReconstruction {
    SourceBacked,
    ExternallyReconstructed,
    NonReconstructable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuMemoryIntent {
    Device,
    Upload,
    Readback,
}

#[derive(Debug, Clone)]
pub struct GpuResourceCommon {
    label: GpuResourceLabel,
    lifetime: GpuResourceLifetime,
    ownership: GpuResourceOwnership,
    memory_intent: GpuMemoryIntent,
    reconstruction: GpuReconstruction,
    provenance: GpuResourceProvenance,
    retained_non_reconstructable_risk_accepted: bool,
}

impl PartialEq for GpuResourceCommon {
    fn eq(&self, other: &Self) -> bool {
        self.lifetime == other.lifetime
            && self.ownership == other.ownership
            && self.memory_intent == other.memory_intent
            && self.reconstruction == other.reconstruction
            && self.retained_non_reconstructable_risk_accepted
                == other.retained_non_reconstructable_risk_accepted
    }
}

impl Eq for GpuResourceCommon {}

impl GpuResourceCommon {
    pub fn owned(
        label: GpuResourceLabel,
        lifetime: GpuResourceLifetime,
        memory_intent: GpuMemoryIntent,
        reconstruction: GpuReconstruction,
        provenance: GpuResourceProvenance,
    ) -> Result<Self, GpuResourceDescriptorError> {
        if lifetime == GpuResourceLifetime::Retained
            && reconstruction == GpuReconstruction::NonReconstructable
        {
            return Err(GpuResourceDescriptorError::invalid(
                "construct owned GPU resource",
                label.as_str(),
                GpuResourceDescriptorCause::InvalidReconstruction,
                "use the explicit retained non-reconstructable constructor to accept device-loss risk",
            ));
        }
        Ok(Self {
            label,
            lifetime,
            ownership: GpuResourceOwnership::Owned,
            memory_intent,
            reconstruction,
            provenance,
            retained_non_reconstructable_risk_accepted: false,
        })
    }

    pub fn owned_retained_non_reconstructable(
        label: GpuResourceLabel,
        memory_intent: GpuMemoryIntent,
        provenance: GpuResourceProvenance,
    ) -> Self {
        Self {
            label,
            lifetime: GpuResourceLifetime::Retained,
            ownership: GpuResourceOwnership::Owned,
            memory_intent,
            reconstruction: GpuReconstruction::NonReconstructable,
            provenance,
            retained_non_reconstructable_risk_accepted: true,
        }
    }

    /// Describes externally owned resource state without adding owned bytes.
    ///
    /// ```
    /// use runen_gpu::*;
    /// let label = GpuResourceLabel::new("imported history")?;
    /// let provenance = GpuResourceProvenance::new(label.clone(), Some(7), None);
    /// let common = GpuResourceCommon::imported(
    ///     label, GpuResourceLifetime::Retained, provenance,
    /// );
    /// assert_eq!(common.ownership(), GpuResourceOwnership::Imported);
    /// # Ok::<(), GpuResourceDescriptorError>(())
    /// ```
    pub fn imported(
        label: GpuResourceLabel,
        lifetime: GpuResourceLifetime,
        provenance: GpuResourceProvenance,
    ) -> Self {
        Self {
            label,
            lifetime,
            ownership: GpuResourceOwnership::Imported,
            memory_intent: GpuMemoryIntent::Device,
            reconstruction: GpuReconstruction::ExternallyReconstructed,
            provenance,
            retained_non_reconstructable_risk_accepted: false,
        }
    }

    #[allow(
        dead_code,
        reason = "surface-acquired construction is reserved for the G7 adapter"
    )]
    pub(crate) fn surface_acquired(
        label: GpuResourceLabel,
        provenance: GpuResourceProvenance,
    ) -> Self {
        Self {
            label,
            lifetime: GpuResourceLifetime::Transient,
            ownership: GpuResourceOwnership::SurfaceAcquired,
            memory_intent: GpuMemoryIntent::Device,
            reconstruction: GpuReconstruction::ExternallyReconstructed,
            provenance,
            retained_non_reconstructable_risk_accepted: false,
        }
    }

    pub fn label(&self) -> &GpuResourceLabel {
        &self.label
    }
    pub const fn lifetime(&self) -> GpuResourceLifetime {
        self.lifetime
    }
    pub const fn ownership(&self) -> GpuResourceOwnership {
        self.ownership
    }
    pub const fn memory_intent(&self) -> GpuMemoryIntent {
        self.memory_intent
    }
    pub const fn reconstruction(&self) -> GpuReconstruction {
        self.reconstruction
    }
    pub fn provenance(&self) -> &GpuResourceProvenance {
        &self.provenance
    }
    pub const fn retained_non_reconstructable_risk_accepted(&self) -> bool {
        self.retained_non_reconstructable_risk_accepted
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuBufferUsage {
    Uniform,
    Storage,
    Vertex,
    Index,
    Indirect,
    CopySource,
    CopyDestination,
    QueryResolve,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuBufferUsages(BTreeSet<GpuBufferUsage>);

impl GpuBufferUsages {
    pub fn new(
        label: &GpuResourceLabel,
        usages: impl IntoIterator<Item = GpuBufferUsage>,
    ) -> Result<Self, GpuResourceDescriptorError> {
        let usages = usages.into_iter().collect::<BTreeSet<_>>();
        if usages.is_empty() {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU buffer usages",
                label.as_str(),
                GpuResourceDescriptorCause::EmptyUsage,
                "declare at least one normalized buffer usage",
            ));
        }
        Ok(Self(usages))
    }

    pub fn contains(&self, usage: GpuBufferUsage) -> bool {
        self.0.contains(&usage)
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = GpuBufferUsage> + '_ {
        self.0.iter().copied()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuTextureUsage {
    Sampled,
    StorageRead,
    StorageWrite,
    ColorAttachment,
    DepthStencilAttachment,
    CopySource,
    CopyDestination,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuTextureUsages(BTreeSet<GpuTextureUsage>);

impl GpuTextureUsages {
    pub fn new(
        label: &GpuResourceLabel,
        usages: impl IntoIterator<Item = GpuTextureUsage>,
    ) -> Result<Self, GpuResourceDescriptorError> {
        let usages = usages.into_iter().collect::<BTreeSet<_>>();
        if usages.is_empty() {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU texture usages",
                label.as_str(),
                GpuResourceDescriptorCause::EmptyUsage,
                "declare at least one normalized texture usage",
            ));
        }
        Ok(Self(usages))
    }

    pub fn contains(&self, usage: GpuTextureUsage) -> bool {
        self.0.contains(&usage)
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = GpuTextureUsage> + '_ {
        self.0.iter().copied()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuTextureDimension {
    D1,
    D2,
    D3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuTextureExtent {
    width: u32,
    height: u32,
    depth_or_layers: u32,
}

impl GpuTextureExtent {
    pub fn new(
        label: &GpuResourceLabel,
        dimension: GpuTextureDimension,
        width: u32,
        height: u32,
        depth_or_layers: u32,
    ) -> Result<Self, GpuResourceDescriptorError> {
        let valid = extent_is_compatible(dimension, width, height, depth_or_layers);
        if !valid {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU texture extent",
                label.as_str(),
                GpuResourceDescriptorCause::InvalidExtent,
                "provide nonzero dimensions compatible with the texture dimension",
            ));
        }
        Ok(Self {
            width,
            height,
            depth_or_layers,
        })
    }

    pub const fn width(self) -> u32 {
        self.width
    }
    pub const fn height(self) -> u32 {
        self.height
    }
    pub const fn depth_or_layers(self) -> u32 {
        self.depth_or_layers
    }
}

const fn extent_is_compatible(
    dimension: GpuTextureDimension,
    width: u32,
    height: u32,
    depth_or_layers: u32,
) -> bool {
    width > 0
        && height > 0
        && depth_or_layers > 0
        && match dimension {
            GpuTextureDimension::D1 => height == 1 && depth_or_layers == 1,
            GpuTextureDimension::D2 | GpuTextureDimension::D3 => true,
        }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuTextureAspect {
    All,
    Color,
    DepthOnly,
    StencilOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuTextureSubresourceRange {
    base_mip_level: u32,
    mip_level_count: u32,
    base_array_layer: u32,
    array_layer_count: u32,
    aspect: GpuTextureAspect,
}

impl GpuTextureSubresourceRange {
    pub fn new(
        label: &GpuResourceLabel,
        base_mip_level: u32,
        mip_level_count: u32,
        base_array_layer: u32,
        array_layer_count: u32,
        aspect: GpuTextureAspect,
    ) -> Result<Self, GpuResourceDescriptorError> {
        if mip_level_count == 0
            || array_layer_count == 0
            || base_mip_level.checked_add(mip_level_count).is_none()
            || base_array_layer.checked_add(array_layer_count).is_none()
        {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU texture subresource range",
                label.as_str(),
                GpuResourceDescriptorCause::SubresourceOutOfBounds,
                "provide nonzero mip and array-layer counts",
            ));
        }
        Ok(Self {
            base_mip_level,
            mip_level_count,
            base_array_layer,
            array_layer_count,
            aspect,
        })
    }

    pub const fn base_mip_level(self) -> u32 {
        self.base_mip_level
    }
    pub const fn mip_level_count(self) -> u32 {
        self.mip_level_count
    }
    pub const fn base_array_layer(self) -> u32 {
        self.base_array_layer
    }
    pub const fn array_layer_count(self) -> u32 {
        self.array_layer_count
    }
    pub const fn aspect(self) -> GpuTextureAspect {
        self.aspect
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuAddressMode {
    ClampToEdge,
    Repeat,
    MirrorRepeat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuFilterMode {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuCompareFunction {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuQueryKind {
    Timestamp,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuBufferInitialization {
    Uninitialized,
    Zeroed,
    Prepared(PreparedGpuData<TransferData>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuPreparedTextureData {
    data: PreparedGpuData<TransferData>,
    format: GpuTextureFormat,
    extent: GpuTextureExtent,
    bytes_per_row: u32,
    rows_per_image: u32,
}

impl GpuPreparedTextureData {
    pub fn new(
        label: &GpuResourceLabel,
        data: PreparedGpuData<TransferData>,
        format: GpuTextureFormat,
        extent: GpuTextureExtent,
        bytes_per_row: u32,
        rows_per_image: u32,
    ) -> Result<Self, GpuResourceDescriptorError> {
        let logical_row = extent
            .width()
            .checked_mul(format.bytes_per_texel())
            .ok_or_else(|| {
                GpuResourceDescriptorError::invalid(
                    "validate prepared GPU texture data",
                    label.as_str(),
                    GpuResourceDescriptorCause::ArithmeticOverflow,
                    "reduce the texture extent",
                )
            })?;
        let multiple_images = extent.depth_or_layers() > 1;
        if bytes_per_row < logical_row
            || bytes_per_row == 0
            || (multiple_images && rows_per_image < extent.height())
            || (!multiple_images && rows_per_image != 0 && rows_per_image < extent.height())
        {
            return Err(GpuResourceDescriptorError::invalid(
                "validate prepared GPU texture data",
                label.as_str(),
                GpuResourceDescriptorCause::InvalidRowLayout,
                "provide a row stride covering one logical row and enough rows per image",
            ));
        }
        let image_stride = if multiple_images {
            u64::from(bytes_per_row)
                .checked_mul(u64::from(rows_per_image))
                .ok_or_else(|| {
                    GpuResourceDescriptorError::invalid(
                        "validate prepared GPU texture data",
                        label.as_str(),
                        GpuResourceDescriptorCause::ArithmeticOverflow,
                        "reduce the texture row layout",
                    )
                })?
        } else {
            0
        };
        let preceding_images = u64::from(extent.depth_or_layers() - 1)
            .checked_mul(image_stride)
            .ok_or_else(|| {
                GpuResourceDescriptorError::invalid(
                    "validate prepared GPU texture data",
                    label.as_str(),
                    GpuResourceDescriptorCause::ArithmeticOverflow,
                    "reduce the texture layer count",
                )
            })?;
        let preceding_rows = u64::from(extent.height() - 1)
            .checked_mul(u64::from(bytes_per_row))
            .ok_or_else(|| {
                GpuResourceDescriptorError::invalid(
                    "validate prepared GPU texture data",
                    label.as_str(),
                    GpuResourceDescriptorCause::ArithmeticOverflow,
                    "reduce the texture height",
                )
            })?;
        let required = preceding_images
            .checked_add(preceding_rows)
            .and_then(|value| value.checked_add(u64::from(logical_row)))
            .ok_or_else(|| {
                GpuResourceDescriptorError::invalid(
                    "validate prepared GPU texture data",
                    label.as_str(),
                    GpuResourceDescriptorCause::ArithmeticOverflow,
                    "reduce the texture initialization size",
                )
            })?;
        if data.layout().byte_len() < required {
            return Err(GpuResourceDescriptorError::invalid(
                "validate prepared GPU texture data",
                label.as_str(),
                GpuResourceDescriptorCause::InsufficientTextureData,
                "provide bytes covering the complete checked row layout",
            ));
        }
        Ok(Self {
            data,
            format,
            extent,
            bytes_per_row,
            rows_per_image,
        })
    }

    pub fn data(&self) -> &PreparedGpuData<TransferData> {
        &self.data
    }
    pub const fn format(&self) -> GpuTextureFormat {
        self.format
    }
    pub const fn extent(&self) -> GpuTextureExtent {
        self.extent
    }
    pub const fn bytes_per_row(&self) -> u32 {
        self.bytes_per_row
    }
    pub const fn rows_per_image(&self) -> u32 {
        self.rows_per_image
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuTextureInitialization {
    Uninitialized,
    Zeroed,
    Prepared(GpuPreparedTextureData),
}

/// A checked backend-neutral logical buffer descriptor.
///
/// ```
/// use runen_gpu::*;
/// let label = GpuResourceLabel::new("particles")?;
/// let provenance = GpuResourceProvenance::new(label.clone(), None, None);
/// let common = GpuResourceCommon::owned(
///     label.clone(), GpuResourceLifetime::Retained, GpuMemoryIntent::Device,
///     GpuReconstruction::SourceBacked, provenance,
/// )?;
/// let usages = GpuBufferUsages::new(
///     &label, [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
/// )?;
/// let descriptor = GpuBufferDescriptor::new(
///     common, 64, usages, GpuBufferInitialization::Uninitialized,
/// )?;
/// assert_eq!(descriptor.size_bytes(), 64);
/// # Ok::<(), GpuResourceDescriptorError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuBufferDescriptor {
    common: GpuResourceCommon,
    size_bytes: u64,
    usages: GpuBufferUsages,
    initialization: GpuBufferInitialization,
}

impl GpuBufferDescriptor {
    pub fn new(
        common: GpuResourceCommon,
        size_bytes: u64,
        usages: GpuBufferUsages,
        initialization: GpuBufferInitialization,
    ) -> Result<Self, GpuResourceDescriptorError> {
        let label = common.label().as_str();
        if size_bytes == 0 {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU buffer descriptor",
                label,
                GpuResourceDescriptorCause::ZeroSize,
                "provide a nonzero buffer size",
            ));
        }
        if common.ownership() == GpuResourceOwnership::SurfaceAcquired {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU buffer descriptor",
                label,
                GpuResourceDescriptorCause::InvalidOwnership,
                "use surface-acquired ownership only for texture or texture-view descriptors",
            ));
        }
        if common.ownership() != GpuResourceOwnership::Owned
            && !matches!(initialization, GpuBufferInitialization::Uninitialized)
        {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU buffer descriptor",
                label,
                GpuResourceDescriptorCause::InvalidInitialization,
                "leave imported resources uninitialized by RunenGPU",
            ));
        }
        if matches!(initialization, GpuBufferInitialization::Prepared(_))
            && common.memory_intent() != GpuMemoryIntent::Device
        {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU buffer descriptor",
                label,
                GpuResourceDescriptorCause::InvalidMemoryIntent,
                "use Device memory intent with CopyDestination for prepared buffer initial content",
            ));
        }
        match common.memory_intent() {
            GpuMemoryIntent::Upload if !usages.contains(GpuBufferUsage::CopySource) => {
                return Err(GpuResourceDescriptorError::invalid(
                    "construct GPU buffer descriptor",
                    label,
                    GpuResourceDescriptorCause::InvalidMemoryIntent,
                    "add CopySource usage to an upload buffer",
                ));
            }
            GpuMemoryIntent::Readback if !usages.contains(GpuBufferUsage::CopyDestination) => {
                return Err(GpuResourceDescriptorError::invalid(
                    "construct GPU buffer descriptor",
                    label,
                    GpuResourceDescriptorCause::InvalidMemoryIntent,
                    "add CopyDestination usage to a readback buffer",
                ));
            }
            GpuMemoryIntent::Device
                if matches!(initialization, GpuBufferInitialization::Prepared(_))
                    && !usages.contains(GpuBufferUsage::CopyDestination) =>
            {
                return Err(GpuResourceDescriptorError::invalid(
                    "construct GPU buffer descriptor",
                    label,
                    GpuResourceDescriptorCause::InvalidInitialization,
                    "add CopyDestination usage for prepared device-buffer initialization",
                ));
            }
            _ => {}
        }
        if let GpuBufferInitialization::Prepared(data) = &initialization {
            if data.layout().byte_len() != size_bytes {
                return Err(GpuResourceDescriptorError::invalid(
                    "construct GPU buffer descriptor",
                    label,
                    GpuResourceDescriptorCause::InitializationLengthMismatch,
                    "make prepared byte length exactly match the buffer size",
                ));
            }
        }
        Ok(Self {
            common,
            size_bytes,
            usages,
            initialization,
        })
    }

    pub fn for_elements(
        common: GpuResourceCommon,
        element_count: u64,
        stride: u64,
        usages: GpuBufferUsages,
        initialization: GpuBufferInitialization,
    ) -> Result<Self, GpuResourceDescriptorError> {
        let label = common.label().as_str().to_string();
        let size_bytes = element_count.checked_mul(stride).ok_or_else(|| {
            GpuResourceDescriptorError::invalid(
                "construct GPU buffer descriptor",
                label,
                GpuResourceDescriptorCause::ArithmeticOverflow,
                "reduce element count or stride",
            )
        })?;
        Self::new(common, size_bytes, usages, initialization)
    }

    pub fn common(&self) -> &GpuResourceCommon {
        &self.common
    }
    pub const fn size_bytes(&self) -> u64 {
        self.size_bytes
    }
    pub fn usages(&self) -> &GpuBufferUsages {
        &self.usages
    }
    pub fn initialization(&self) -> &GpuBufferInitialization {
        &self.initialization
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuTextureDescriptor {
    common: GpuResourceCommon,
    dimension: GpuTextureDimension,
    extent: GpuTextureExtent,
    mip_level_count: u32,
    sample_count: u32,
    format: GpuTextureFormat,
    usages: GpuTextureUsages,
    initialization: GpuTextureInitialization,
}

impl GpuTextureDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        common: GpuResourceCommon,
        dimension: GpuTextureDimension,
        extent: GpuTextureExtent,
        mip_level_count: u32,
        sample_count: u32,
        format: GpuTextureFormat,
        usages: GpuTextureUsages,
        initialization: GpuTextureInitialization,
    ) -> Result<Self, GpuResourceDescriptorError> {
        let label = common.label().as_str();
        if common.memory_intent() != GpuMemoryIntent::Device {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU texture descriptor",
                label,
                GpuResourceDescriptorCause::InvalidMemoryIntent,
                "use Device memory intent for textures and explicit copy relationships for transfer",
            ));
        }
        if !extent_is_compatible(
            dimension,
            extent.width(),
            extent.height(),
            extent.depth_or_layers(),
        ) {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU texture descriptor",
                label,
                GpuResourceDescriptorCause::InvalidExtent,
                "use an extent constructed for the descriptor's texture dimension",
            ));
        }
        if common.ownership() != GpuResourceOwnership::Owned
            && !matches!(initialization, GpuTextureInitialization::Uninitialized)
        {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU texture descriptor",
                label,
                GpuResourceDescriptorCause::InvalidInitialization,
                "leave imported and surface-acquired textures uninitialized by RunenGPU",
            ));
        }
        let max_dimension = extent.width().max(extent.height()).max(match dimension {
            GpuTextureDimension::D3 => extent.depth_or_layers(),
            GpuTextureDimension::D1 | GpuTextureDimension::D2 => 1,
        });
        let max_mips = u32::BITS - max_dimension.leading_zeros();
        if mip_level_count == 0 || mip_level_count > max_mips {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU texture descriptor",
                label,
                GpuResourceDescriptorCause::InvalidMipCount,
                "choose a nonzero mip count bounded by the texture extent",
            ));
        }
        if !matches!(sample_count, 1 | 2 | 4 | 8 | 16)
            || (sample_count > 1
                && (mip_level_count != 1
                    || usages.contains(GpuTextureUsage::StorageRead)
                    || usages.contains(GpuTextureUsage::StorageWrite)))
        {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU texture descriptor",
                label,
                GpuResourceDescriptorCause::InvalidSampleCount,
                "use a supported power-of-two sample count and one non-storage mip for multisampling",
            ));
        }
        if matches!(initialization, GpuTextureInitialization::Prepared(_)) && sample_count != 1 {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU texture descriptor",
                label,
                GpuResourceDescriptorCause::InvalidSampleCount,
                "use sample_count 1 for prepared texture initial content so canonical texture upload can materialize it",
            ));
        }
        validate_texture_format_usages(label, format, &usages)?;
        if matches!(initialization, GpuTextureInitialization::Prepared(_))
            && !usages.contains(GpuTextureUsage::CopyDestination)
        {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU texture descriptor",
                label,
                GpuResourceDescriptorCause::InvalidInitialization,
                "add CopyDestination usage for prepared owned texture initialization",
            ));
        }
        if let GpuTextureInitialization::Prepared(data) = &initialization {
            if data.format() != format || data.extent() != extent {
                return Err(GpuResourceDescriptorError::invalid(
                    "construct GPU texture descriptor",
                    label,
                    GpuResourceDescriptorCause::InitializationLengthMismatch,
                    "bind prepared texture data with the descriptor's exact format and extent",
                ));
            }
        }
        Ok(Self {
            common,
            dimension,
            extent,
            mip_level_count,
            sample_count,
            format,
            usages,
            initialization,
        })
    }

    pub fn common(&self) -> &GpuResourceCommon {
        &self.common
    }
    pub const fn dimension(&self) -> GpuTextureDimension {
        self.dimension
    }
    pub const fn extent(&self) -> GpuTextureExtent {
        self.extent
    }
    pub const fn mip_level_count(&self) -> u32 {
        self.mip_level_count
    }
    pub const fn sample_count(&self) -> u32 {
        self.sample_count
    }
    pub const fn format(&self) -> GpuTextureFormat {
        self.format
    }
    pub fn usages(&self) -> &GpuTextureUsages {
        &self.usages
    }
    pub fn initialization(&self) -> &GpuTextureInitialization {
        &self.initialization
    }
}

fn validate_texture_format_usages(
    label: &str,
    format: GpuTextureFormat,
    usages: &GpuTextureUsages,
) -> Result<(), GpuResourceDescriptorError> {
    let depth_usage = usages.contains(GpuTextureUsage::DepthStencilAttachment);
    let color_usage = usages.contains(GpuTextureUsage::ColorAttachment);
    let storage_usage = usages.contains(GpuTextureUsage::StorageRead)
        || usages.contains(GpuTextureUsage::StorageWrite);
    if (format.is_depth() && (color_usage || storage_usage))
        || (!format.is_depth() && depth_usage)
        || (format.is_srgb() && storage_usage)
    {
        return Err(GpuResourceDescriptorError::invalid(
            "validate GPU texture format usages",
            label,
            GpuResourceDescriptorCause::InvalidFormatUsage,
            "choose usages compatible with the normalized texture format",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuTextureViewDescriptor {
    common: GpuResourceCommon,
    texture: GpuTextureHandle,
    format: Option<GpuTextureFormat>,
    dimension: GpuTextureDimension,
    subresources: GpuTextureSubresourceRange,
}

impl GpuTextureViewDescriptor {
    pub fn new(
        common: GpuResourceCommon,
        texture: &GpuTextureHandle,
        format: Option<GpuTextureFormat>,
        dimension: GpuTextureDimension,
        subresources: GpuTextureSubresourceRange,
    ) -> Result<Self, GpuResourceDescriptorError> {
        let label = common.label().as_str();
        let parent = texture.descriptor();
        if common.memory_intent() != GpuMemoryIntent::Device {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU texture-view descriptor",
                label,
                GpuResourceDescriptorCause::InvalidMemoryIntent,
                "use Device memory intent for texture views",
            ));
        }
        if common.ownership() != parent.common().ownership()
            || (parent.common().lifetime() == GpuResourceLifetime::Transient
                && common.lifetime() == GpuResourceLifetime::Retained)
        {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU texture-view descriptor",
                label,
                GpuResourceDescriptorCause::ParentLeaseMismatch,
                "keep view ownership and lifetime within the parent texture lease",
            ));
        }
        if dimension != parent.dimension() {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU texture-view descriptor",
                label,
                GpuResourceDescriptorCause::IncompatibleViewDimension,
                "use a view dimension compatible with the parent texture",
            ));
        }
        let mip_end = subresources
            .base_mip_level()
            .checked_add(subresources.mip_level_count());
        let layer_end = subresources
            .base_array_layer()
            .checked_add(subresources.array_layer_count());
        let parent_layers = match parent.dimension() {
            GpuTextureDimension::D2 => parent.extent().depth_or_layers(),
            GpuTextureDimension::D1 | GpuTextureDimension::D3 => 1,
        };
        if mip_end.is_none_or(|end| end > parent.mip_level_count())
            || layer_end.is_none_or(|end| end > parent_layers)
        {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU texture-view descriptor",
                label,
                GpuResourceDescriptorCause::SubresourceOutOfBounds,
                "keep mip and array-layer ranges inside the parent descriptor",
            ));
        }
        validate_aspect(label, parent.format(), subresources.aspect())?;
        if let Some(view_format) = format {
            if !formats_are_view_compatible(parent.format(), view_format) {
                return Err(GpuResourceDescriptorError::invalid(
                    "construct GPU texture-view descriptor",
                    label,
                    GpuResourceDescriptorCause::IncompatibleViewFormat,
                    "use the parent format or its normalized linear/sRGB pair",
                ));
            }
        }
        Ok(Self {
            common,
            texture: texture.clone(),
            format,
            dimension,
            subresources,
        })
    }

    pub fn common(&self) -> &GpuResourceCommon {
        &self.common
    }
    pub fn texture(&self) -> &GpuTextureHandle {
        &self.texture
    }
    pub const fn format(&self) -> Option<GpuTextureFormat> {
        self.format
    }
    pub const fn dimension(&self) -> GpuTextureDimension {
        self.dimension
    }
    pub const fn subresources(&self) -> GpuTextureSubresourceRange {
        self.subresources
    }
}

fn validate_aspect(
    label: &str,
    format: GpuTextureFormat,
    aspect: GpuTextureAspect,
) -> Result<(), GpuResourceDescriptorError> {
    let valid = if format.is_depth() {
        matches!(aspect, GpuTextureAspect::All | GpuTextureAspect::DepthOnly)
    } else {
        matches!(aspect, GpuTextureAspect::All | GpuTextureAspect::Color)
    };
    if !valid {
        return Err(GpuResourceDescriptorError::invalid(
            "validate GPU texture aspect",
            label,
            GpuResourceDescriptorCause::InvalidAspect,
            "select an aspect represented by the texture format",
        ));
    }
    Ok(())
}

fn formats_are_view_compatible(parent: GpuTextureFormat, view: GpuTextureFormat) -> bool {
    parent == view
        || matches!(
            (parent, view),
            (
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureFormat::Rgba8UnormSrgb
            ) | (
                GpuTextureFormat::Rgba8UnormSrgb,
                GpuTextureFormat::Rgba8Unorm
            ) | (
                GpuTextureFormat::Bgra8Unorm,
                GpuTextureFormat::Bgra8UnormSrgb
            ) | (
                GpuTextureFormat::Bgra8UnormSrgb,
                GpuTextureFormat::Bgra8Unorm
            )
        )
}

#[derive(Debug, Clone, PartialEq)]
pub struct GpuSamplerDescriptor {
    common: GpuResourceCommon,
    address_u: GpuAddressMode,
    address_v: GpuAddressMode,
    address_w: GpuAddressMode,
    mag_filter: GpuFilterMode,
    min_filter: GpuFilterMode,
    mipmap_filter: GpuFilterMode,
    lod_min: f32,
    lod_max: f32,
    compare: Option<GpuCompareFunction>,
}

// Construction rejects non-finite LOD values, so semantic equality is
// reflexive even though the stored representation uses `f32`.
impl Eq for GpuSamplerDescriptor {}

impl GpuSamplerDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        common: GpuResourceCommon,
        address_u: GpuAddressMode,
        address_v: GpuAddressMode,
        address_w: GpuAddressMode,
        mag_filter: GpuFilterMode,
        min_filter: GpuFilterMode,
        mipmap_filter: GpuFilterMode,
        lod_min: f32,
        lod_max: f32,
        compare: Option<GpuCompareFunction>,
    ) -> Result<Self, GpuResourceDescriptorError> {
        if common.memory_intent() != GpuMemoryIntent::Device {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU sampler descriptor",
                common.label().as_str(),
                GpuResourceDescriptorCause::InvalidMemoryIntent,
                "use Device memory intent for samplers",
            ));
        }
        if common.ownership() == GpuResourceOwnership::SurfaceAcquired {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU sampler descriptor",
                common.label().as_str(),
                GpuResourceDescriptorCause::InvalidOwnership,
                "use surface-acquired ownership only for texture or texture-view descriptors",
            ));
        }
        if !lod_min.is_finite() || !lod_max.is_finite() || lod_min > lod_max {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU sampler descriptor",
                common.label().as_str(),
                GpuResourceDescriptorCause::InvalidLodRange,
                "provide finite LOD bounds with minimum not greater than maximum",
            ));
        }
        Ok(Self {
            common,
            address_u,
            address_v,
            address_w,
            mag_filter,
            min_filter,
            mipmap_filter,
            lod_min,
            lod_max,
            compare,
        })
    }

    pub fn common(&self) -> &GpuResourceCommon {
        &self.common
    }
    pub const fn lod_range(&self) -> (f32, f32) {
        (self.lod_min, self.lod_max)
    }
    pub const fn address_modes(&self) -> (GpuAddressMode, GpuAddressMode, GpuAddressMode) {
        (self.address_u, self.address_v, self.address_w)
    }
    pub const fn filters(&self) -> (GpuFilterMode, GpuFilterMode, GpuFilterMode) {
        (self.mag_filter, self.min_filter, self.mipmap_filter)
    }
    pub const fn compare(&self) -> Option<GpuCompareFunction> {
        self.compare
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuQuerySetDescriptor {
    common: GpuResourceCommon,
    kind: GpuQueryKind,
    count: u32,
}

impl GpuQuerySetDescriptor {
    pub fn new(
        common: GpuResourceCommon,
        kind: GpuQueryKind,
        count: u32,
    ) -> Result<Self, GpuResourceDescriptorError> {
        if common.memory_intent() != GpuMemoryIntent::Device {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU query-set descriptor",
                common.label().as_str(),
                GpuResourceDescriptorCause::InvalidMemoryIntent,
                "use Device memory intent for query sets",
            ));
        }
        if common.ownership() == GpuResourceOwnership::SurfaceAcquired {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU query-set descriptor",
                common.label().as_str(),
                GpuResourceDescriptorCause::InvalidOwnership,
                "use surface-acquired ownership only for texture or texture-view descriptors",
            ));
        }
        if count == 0 {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU query-set descriptor",
                common.label().as_str(),
                GpuResourceDescriptorCause::InvalidQueryCount,
                "provide a nonzero query count",
            ));
        }
        Ok(Self {
            common,
            kind,
            count,
        })
    }

    pub fn common(&self) -> &GpuResourceCommon {
        &self.common
    }
    pub const fn kind(&self) -> GpuQueryKind {
        self.kind
    }
    pub const fn count(&self) -> u32 {
        self.count
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// A normalized descriptor cannot be replaced by framework or backend types.
///
/// ```compile_fail
/// use runen_gpu::GpuTextureFormat;
/// fn leak_backend(value: wgpu::TextureFormat) -> GpuTextureFormat { value }
/// ```
///
/// ```compile_fail
/// use runen_gpu::GpuResourceDescriptor;
/// fn leak_framework(value: ecs::World) -> GpuResourceDescriptor { value }
/// ```
pub enum GpuResourceDescriptor {
    Buffer(GpuBufferDescriptor),
    Texture(GpuTextureDescriptor),
    TextureView(GpuTextureViewDescriptor),
    Sampler(GpuSamplerDescriptor),
    QuerySet(GpuQuerySetDescriptor),
}

impl GpuResourceDescriptor {
    pub fn common(&self) -> &GpuResourceCommon {
        match self {
            Self::Buffer(value) => value.common(),
            Self::Texture(value) => value.common(),
            Self::TextureView(value) => value.common(),
            Self::Sampler(value) => value.common(),
            Self::QuerySet(value) => value.common(),
        }
    }
}

/// The already-existing neutral final access intent used only by export relationships.
/// G3 owns ranges, subresources, hazards, and work-time access validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuResourceAccessIntent {
    Read,
    Write,
    ReadWrite,
}

/// A consumer-owned semantic key used to connect fragment exports and imports.
///
/// Unlike labels, an export key participates in graph composition. It is still
/// process-local work authoring data and carries no persistence, replay, wire,
/// network, ABI, or cache stability promise.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuExportKey(String);

impl GpuExportKey {
    pub fn new(value: impl Into<String>) -> Result<Self, GpuResourceDescriptorError> {
        let value = value.into();
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(GpuResourceDescriptorError::invalid(
                "construct GPU export key",
                "<empty>",
                GpuResourceDescriptorCause::EmptyLabel,
                "provide a non-empty consumer-owned export key",
            ));
        }
        Ok(Self(trimmed.to_string()))
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

#[derive(Debug, Clone)]
pub struct GpuExportRelationship {
    resource: GpuResourceRef,
    export_key: GpuExportKey,
    required_final_access: GpuResourceAccessIntent,
    provenance: GpuResourceProvenance,
}

impl PartialEq for GpuExportRelationship {
    fn eq(&self, other: &Self) -> bool {
        self.resource == other.resource
            && self.export_key == other.export_key
            && self.required_final_access == other.required_final_access
    }
}

impl Eq for GpuExportRelationship {}

impl GpuExportRelationship {
    pub fn new(
        resource: GpuResourceRef,
        export_key: GpuExportKey,
        required_final_access: GpuResourceAccessIntent,
        provenance: GpuResourceProvenance,
    ) -> Self {
        Self {
            resource,
            export_key,
            required_final_access,
            provenance,
        }
    }

    pub fn resource(&self) -> &GpuResourceRef {
        &self.resource
    }
    pub fn export_key(&self) -> &GpuExportKey {
        &self.export_key
    }
    pub const fn required_final_access(&self) -> GpuResourceAccessIntent {
        self.required_final_access
    }
    pub fn provenance(&self) -> &GpuResourceProvenance {
        &self.provenance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuDataLayout, GpuWorkResourceIdAllocator};
    use std::num::NonZeroU64;

    fn label(value: &str) -> GpuResourceLabel {
        GpuResourceLabel::new(value).unwrap()
    }

    fn provenance(value: &str) -> GpuResourceProvenance {
        let label = label(value);
        GpuResourceProvenance::new(label, None, None)
    }

    fn common(value: &str) -> GpuResourceCommon {
        GpuResourceCommon::owned(
            label(value),
            GpuResourceLifetime::Retained,
            GpuMemoryIntent::Device,
            GpuReconstruction::SourceBacked,
            provenance(value),
        )
        .unwrap()
    }

    fn transfer_data(value: &str, byte_len: usize) -> PreparedGpuData<TransferData> {
        PreparedGpuData::<TransferData>::from_pod_transfer(
            value,
            &vec![0_u8; byte_len],
            provenance(value),
        )
        .unwrap()
    }

    fn texture_descriptor(value: &str) -> GpuTextureDescriptor {
        let label = label(value);
        let extent = GpuTextureExtent::new(&label, GpuTextureDimension::D2, 8, 8, 1).unwrap();
        let usages = GpuTextureUsages::new(&label, [GpuTextureUsage::Sampled]).unwrap();
        GpuTextureDescriptor::new(
            common(value),
            GpuTextureDimension::D2,
            extent,
            1,
            1,
            GpuTextureFormat::Rgba8Unorm,
            usages,
            GpuTextureInitialization::Uninitialized,
        )
        .unwrap()
    }

    fn buffer_descriptor(value: &str) -> GpuBufferDescriptor {
        GpuBufferDescriptor::new(
            common(value),
            16,
            GpuBufferUsages::new(&label(value), [GpuBufferUsage::Storage]).unwrap(),
            GpuBufferInitialization::Uninitialized,
        )
        .unwrap()
    }

    fn export_relationship(
        resource: GpuResourceRef,
        export_key: &str,
        required_final_access: GpuResourceAccessIntent,
        provenance: GpuResourceProvenance,
    ) -> GpuExportRelationship {
        GpuExportRelationship::new(
            resource,
            GpuExportKey::new(export_key).unwrap(),
            required_final_access,
            provenance,
        )
    }

    fn buffer_resource_refs() -> (GpuResourceRef, GpuResourceRef) {
        let mut allocator =
            GpuWorkResourceIdAllocator::for_owner_scope(NonZeroU64::new(7).unwrap());
        let first = allocator
            .allocate_buffer_handle(buffer_descriptor("first buffer"))
            .unwrap();
        let second = allocator
            .allocate_buffer_handle(buffer_descriptor("second buffer"))
            .unwrap();
        (first.into(), second.into())
    }

    #[test]
    fn buffer_rejects_zero_overflow_empty_usage_and_initialization_mismatch() {
        let buffer_label = label("buffer");
        assert!(GpuBufferUsages::new(&buffer_label, []).is_err());
        let usages = GpuBufferUsages::new(&buffer_label, [GpuBufferUsage::Storage]).unwrap();
        assert!(
            GpuBufferDescriptor::new(
                common("buffer"),
                0,
                usages.clone(),
                GpuBufferInitialization::Uninitialized
            )
            .is_err()
        );
        assert!(
            GpuBufferDescriptor::for_elements(
                common("buffer"),
                u64::MAX,
                2,
                usages,
                GpuBufferInitialization::Uninitialized
            )
            .is_err()
        );
        let initialized_usages = GpuBufferUsages::new(
            &buffer_label,
            [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
        )
        .unwrap();
        let mismatch = GpuBufferDescriptor::new(
            common("buffer"),
            8,
            initialized_usages,
            GpuBufferInitialization::Prepared(transfer_data("buffer bytes", 4)),
        )
        .unwrap_err();
        assert_eq!(
            mismatch.cause(),
            GpuResourceDescriptorCause::InitializationLengthMismatch
        );

        let upload_common = GpuResourceCommon::owned(
            label("prepared upload"),
            GpuResourceLifetime::Transient,
            GpuMemoryIntent::Upload,
            GpuReconstruction::SourceBacked,
            provenance("prepared upload"),
        )
        .unwrap();
        let upload_usages = GpuBufferUsages::new(
            &label("prepared upload"),
            [GpuBufferUsage::CopySource, GpuBufferUsage::CopyDestination],
        )
        .unwrap();
        let invalid_prepared_upload = GpuBufferDescriptor::new(
            upload_common,
            4,
            upload_usages,
            GpuBufferInitialization::Prepared(transfer_data("prepared upload", 4)),
        )
        .unwrap_err();
        assert_eq!(
            invalid_prepared_upload.cause(),
            GpuResourceDescriptorCause::InvalidMemoryIntent
        );
    }

    #[test]
    fn ownership_lifetime_reconstruction_and_memory_are_independent_and_checked() {
        assert!(
            GpuResourceCommon::owned(
                label("risk"),
                GpuResourceLifetime::Retained,
                GpuMemoryIntent::Device,
                GpuReconstruction::NonReconstructable,
                provenance("risk"),
            )
            .is_err()
        );
        assert!(
            GpuResourceCommon::owned_retained_non_reconstructable(
                label("risk"),
                GpuMemoryIntent::Device,
                provenance("risk")
            )
            .retained_non_reconstructable_risk_accepted()
        );

        let imported_label = label("imported");
        let imported = GpuResourceCommon::imported(
            imported_label.clone(),
            GpuResourceLifetime::Retained,
            provenance("imported"),
        );
        let imported_usages =
            GpuBufferUsages::new(&imported_label, [GpuBufferUsage::CopySource]).unwrap();
        assert!(
            GpuBufferDescriptor::new(
                imported,
                4,
                imported_usages,
                GpuBufferInitialization::Zeroed,
            )
            .is_err()
        );

        let surface_label = label("surface");
        let surface =
            GpuResourceCommon::surface_acquired(surface_label.clone(), provenance("surface"));
        let surface_usages =
            GpuBufferUsages::new(&surface_label, [GpuBufferUsage::Storage]).unwrap();
        let error = GpuBufferDescriptor::new(
            surface,
            4,
            surface_usages,
            GpuBufferInitialization::Uninitialized,
        )
        .unwrap_err();
        assert_eq!(error.cause(), GpuResourceDescriptorCause::InvalidOwnership);

        let upload = GpuResourceCommon::owned(
            label("upload"),
            GpuResourceLifetime::Transient,
            GpuMemoryIntent::Upload,
            GpuReconstruction::SourceBacked,
            provenance("upload"),
        )
        .unwrap();
        let upload_usages =
            GpuBufferUsages::new(&label("upload"), [GpuBufferUsage::Storage]).unwrap();
        assert!(
            GpuBufferDescriptor::new(
                upload,
                4,
                upload_usages,
                GpuBufferInitialization::Uninitialized,
            )
            .is_err()
        );
    }

    #[test]
    fn texture_extent_format_sample_aspect_and_row_layout_are_checked() {
        let label = label("texture");
        assert!(GpuTextureExtent::new(&label, GpuTextureDimension::D1, 8, 2, 1).is_err());
        let layout = GpuDataLayout::new("texture", 16, 1, 1, 16).unwrap();
        let bytes = PreparedGpuData::<TransferData>::from_transfer_bytes_for_adapter(
            "texture",
            vec![0; 16],
            layout,
            provenance("texture"),
            None,
        )
        .unwrap();
        let extent = GpuTextureExtent::new(&label, GpuTextureDimension::D2, 8, 8, 1).unwrap();
        assert!(
            GpuPreparedTextureData::new(&label, bytes, GpuTextureFormat::Rgba8Unorm, extent, 4, 8)
                .is_err()
        );

        let d2_extent = GpuTextureExtent::new(&label, GpuTextureDimension::D2, 8, 8, 2).unwrap();
        let sampled = GpuTextureUsages::new(&label, [GpuTextureUsage::Sampled]).unwrap();
        assert!(
            GpuTextureDescriptor::new(
                common("texture"),
                GpuTextureDimension::D1,
                d2_extent,
                1,
                1,
                GpuTextureFormat::Rgba8Unorm,
                sampled.clone(),
                GpuTextureInitialization::Uninitialized,
            )
            .is_err()
        );
        assert!(
            GpuTextureDescriptor::new(
                common("texture"),
                GpuTextureDimension::D2,
                d2_extent,
                0,
                1,
                GpuTextureFormat::Rgba8Unorm,
                sampled.clone(),
                GpuTextureInitialization::Uninitialized,
            )
            .is_err()
        );
        assert!(
            GpuTextureDescriptor::new(
                common("texture"),
                GpuTextureDimension::D2,
                d2_extent,
                1,
                3,
                GpuTextureFormat::Rgba8Unorm,
                sampled,
                GpuTextureInitialization::Uninitialized,
            )
            .is_err()
        );
        let storage = GpuTextureUsages::new(&label, [GpuTextureUsage::StorageWrite]).unwrap();
        assert!(
            GpuTextureDescriptor::new(
                common("texture"),
                GpuTextureDimension::D2,
                d2_extent,
                1,
                1,
                GpuTextureFormat::Rgba8UnormSrgb,
                storage,
                GpuTextureInitialization::Uninitialized,
            )
            .is_err()
        );
        assert!(
            validate_aspect(
                "texture",
                GpuTextureFormat::Depth32Float,
                GpuTextureAspect::Color,
            )
            .is_err()
        );

        let single_extent =
            GpuTextureExtent::new(&label, GpuTextureDimension::D2, 8, 8, 1).unwrap();
        assert!(
            GpuPreparedTextureData::new(
                &label,
                transfer_data("single texture", 256),
                GpuTextureFormat::Rgba8Unorm,
                single_extent,
                32,
                0,
            )
            .is_ok()
        );

        let multisample_prepared = GpuPreparedTextureData::new(
            &label,
            transfer_data("multisample prepared texture", 256),
            GpuTextureFormat::Rgba8Unorm,
            single_extent,
            32,
            0,
        )
        .unwrap();
        let multisample_usages = GpuTextureUsages::new(
            &label,
            [
                GpuTextureUsage::ColorAttachment,
                GpuTextureUsage::CopyDestination,
            ],
        )
        .unwrap();
        let invalid_multisample_prepared = GpuTextureDescriptor::new(
            common("multisample prepared texture"),
            GpuTextureDimension::D2,
            single_extent,
            1,
            4,
            GpuTextureFormat::Rgba8Unorm,
            multisample_usages,
            GpuTextureInitialization::Prepared(multisample_prepared),
        )
        .unwrap_err();
        assert_eq!(
            invalid_multisample_prepared.cause(),
            GpuResourceDescriptorCause::InvalidSampleCount
        );

        let zeroed_multisample_usages =
            GpuTextureUsages::new(&label, [GpuTextureUsage::ColorAttachment]).unwrap();
        assert!(
            GpuTextureDescriptor::new(
                common("zeroed multisample texture"),
                GpuTextureDimension::D2,
                single_extent,
                1,
                4,
                GpuTextureFormat::Rgba8Unorm,
                zeroed_multisample_usages,
                GpuTextureInitialization::Zeroed,
            )
            .is_ok()
        );
    }

    #[test]
    fn texture_view_is_bounded_by_parent_descriptor_and_lease() {
        let descriptor = texture_descriptor("parent");
        let mut allocator =
            GpuWorkResourceIdAllocator::for_owner_scope(NonZeroU64::new(1).unwrap());
        let handle = allocator.allocate_texture_handle(descriptor).unwrap();
        let view_label = label("view");
        let range =
            GpuTextureSubresourceRange::new(&view_label, 1, 1, 0, 1, GpuTextureAspect::Color)
                .unwrap();
        assert!(
            GpuTextureViewDescriptor::new(
                common("view"),
                &handle,
                None,
                GpuTextureDimension::D2,
                range
            )
            .is_err()
        );

        let valid_range =
            GpuTextureSubresourceRange::new(&view_label, 0, 1, 0, 1, GpuTextureAspect::Color)
                .unwrap();
        assert!(
            GpuTextureViewDescriptor::new(
                common("view"),
                &handle,
                Some(GpuTextureFormat::Rgba8UnormSrgb),
                GpuTextureDimension::D2,
                valid_range,
            )
            .is_ok()
        );
        let imported_common = GpuResourceCommon::imported(
            label("imported view"),
            GpuResourceLifetime::Retained,
            provenance("imported view"),
        );
        assert!(
            GpuTextureViewDescriptor::new(
                imported_common,
                &handle,
                None,
                GpuTextureDimension::D2,
                valid_range,
            )
            .is_err()
        );
        assert!(
            GpuTextureSubresourceRange::new(
                &view_label,
                u32::MAX,
                2,
                0,
                1,
                GpuTextureAspect::Color,
            )
            .is_err()
        );

        let transient_label = label("transient parent");
        let transient_common = GpuResourceCommon::owned(
            transient_label.clone(),
            GpuResourceLifetime::Transient,
            GpuMemoryIntent::Device,
            GpuReconstruction::SourceBacked,
            provenance("transient parent"),
        )
        .unwrap();
        let transient_extent =
            GpuTextureExtent::new(&transient_label, GpuTextureDimension::D2, 4, 4, 1).unwrap();
        let transient_usages =
            GpuTextureUsages::new(&transient_label, [GpuTextureUsage::Sampled]).unwrap();
        let transient_descriptor = GpuTextureDescriptor::new(
            transient_common,
            GpuTextureDimension::D2,
            transient_extent,
            1,
            1,
            GpuTextureFormat::Rgba8Unorm,
            transient_usages,
            GpuTextureInitialization::Uninitialized,
        )
        .unwrap();
        let transient_handle = allocator
            .allocate_texture_handle(transient_descriptor)
            .unwrap();
        assert!(
            GpuTextureViewDescriptor::new(
                common("retained child"),
                &transient_handle,
                None,
                GpuTextureDimension::D2,
                valid_range,
            )
            .is_err()
        );
    }

    #[test]
    fn sampler_and_query_validation_is_fallible() {
        assert!(
            GpuSamplerDescriptor::new(
                common("sampler"),
                GpuAddressMode::ClampToEdge,
                GpuAddressMode::ClampToEdge,
                GpuAddressMode::ClampToEdge,
                GpuFilterMode::Nearest,
                GpuFilterMode::Nearest,
                GpuFilterMode::Nearest,
                2.0,
                1.0,
                None,
            )
            .is_err()
        );
        assert!(GpuQuerySetDescriptor::new(common("queries"), GpuQueryKind::Timestamp, 0).is_err());

        let upload_sampler = GpuResourceCommon::owned(
            label("upload sampler"),
            GpuResourceLifetime::Transient,
            GpuMemoryIntent::Upload,
            GpuReconstruction::SourceBacked,
            provenance("upload sampler"),
        )
        .unwrap();
        assert!(
            GpuSamplerDescriptor::new(
                upload_sampler,
                GpuAddressMode::ClampToEdge,
                GpuAddressMode::ClampToEdge,
                GpuAddressMode::ClampToEdge,
                GpuFilterMode::Nearest,
                GpuFilterMode::Nearest,
                GpuFilterMode::Nearest,
                0.0,
                1.0,
                None,
            )
            .is_err()
        );
    }

    #[test]
    fn labels_and_provenance_do_not_change_descriptor_semantics() {
        let first = GpuBufferDescriptor::new(
            common("first"),
            16,
            GpuBufferUsages::new(&label("first"), [GpuBufferUsage::Storage]).unwrap(),
            GpuBufferInitialization::Uninitialized,
        )
        .unwrap();
        let second = GpuBufferDescriptor::new(
            common("second"),
            16,
            GpuBufferUsages::new(&label("second"), [GpuBufferUsage::Storage]).unwrap(),
            GpuBufferInitialization::Uninitialized,
        )
        .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn export_equality_excludes_all_provenance_fields() {
        let (resource, _) = buffer_resource_refs();
        let first = export_relationship(
            resource.clone(),
            "consumer.output",
            GpuResourceAccessIntent::Read,
            GpuResourceProvenance::new(label("first producer"), Some(3), Some(label("revision a"))),
        );
        let second = export_relationship(
            resource,
            "consumer.output",
            GpuResourceAccessIntent::Read,
            GpuResourceProvenance::new(
                label("second producer"),
                Some(99),
                Some(label("revision b")),
            ),
        );

        assert_eq!(first, second);
        assert_ne!(first.provenance(), second.provenance());
    }

    #[test]
    fn export_equality_includes_consumer_owned_key() {
        let (resource, _) = buffer_resource_refs();
        let first = export_relationship(
            resource.clone(),
            "consumer.first",
            GpuResourceAccessIntent::Read,
            provenance("producer"),
        );
        let second = export_relationship(
            resource,
            "consumer.second",
            GpuResourceAccessIntent::Read,
            provenance("producer"),
        );

        assert_ne!(first, second);
    }

    #[test]
    fn export_equality_includes_required_final_access() {
        let (resource, _) = buffer_resource_refs();
        let first = export_relationship(
            resource.clone(),
            "consumer.output",
            GpuResourceAccessIntent::Read,
            provenance("producer"),
        );
        let second = export_relationship(
            resource,
            "consumer.output",
            GpuResourceAccessIntent::Write,
            provenance("producer"),
        );

        assert_ne!(first, second);
    }

    #[test]
    fn export_equality_includes_kind_preserving_resource_reference() {
        let (first_resource, second_resource) = buffer_resource_refs();
        let first = export_relationship(
            first_resource,
            "consumer.output",
            GpuResourceAccessIntent::Read,
            provenance("producer"),
        );
        let second = export_relationship(
            second_resource,
            "consumer.output",
            GpuResourceAccessIntent::Read,
            provenance("producer"),
        );

        assert_ne!(first, second);
    }

    #[test]
    fn structured_error_display_names_operation_label_cause_and_correction() {
        let error = GpuResourceLabel::new("   ").unwrap_err().to_string();
        assert!(error.contains("construct GPU resource label"));
        assert!(error.contains("<empty>"));
        assert!(error.contains("EmptyLabel"));
        assert!(error.contains("correction"));
    }
}
