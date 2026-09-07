use super::{
    GpuDataPreparationCause, GpuDataPreparationError, GpuReadbackDecodeError,
    GpuResourceProvenance, GpuTextureFormat,
};
use core::fmt;
use core::marker::PhantomData;
use std::sync::Arc;

mod sealed {
    pub trait Sealed {}
}

/// A sealed marker for data prepared for a uniform binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniformData {}

/// A sealed marker for data prepared for a storage binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageData {}

/// A sealed marker for data prepared for a vertex input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexData {}

/// A sealed marker for data prepared for an indirect command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndirectData {}

/// A sealed marker for bytes whose representation is explicitly transfer-safe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferData {}

impl sealed::Sealed for UniformData {}
impl sealed::Sealed for StorageData {}
impl sealed::Sealed for VertexData {}
impl sealed::Sealed for IndirectData {}
impl sealed::Sealed for TransferData {}

/// Purpose bound used by prepared GPU data.
///
/// The private supertrait prevents downstream crates from inventing a purpose
/// that bypasses the five normalized representation categories.
///
/// ```compile_fail
/// use runen_gpu::GpuDataPurpose;
/// struct UncheckedPurpose;
/// impl GpuDataPurpose for UncheckedPurpose {}
/// ```
pub trait GpuDataPurpose: sealed::Sealed + Send + Sync + 'static {}

impl GpuDataPurpose for UniformData {}
impl GpuDataPurpose for StorageData {}
impl GpuDataPurpose for VertexData {}
impl GpuDataPurpose for IndirectData {}
impl GpuDataPurpose for TransferData {}

/// Checked metadata describing one immutable prepared byte representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuDataLayout {
    byte_len: u64,
    alignment: u64,
    stride: u64,
    element_count: u64,
}

impl GpuDataLayout {
    pub fn new(
        label: impl Into<String>,
        byte_len: u64,
        alignment: u64,
        stride: u64,
        element_count: u64,
    ) -> Result<Self, GpuDataPreparationError> {
        let label = label.into();
        if byte_len == 0 {
            return Err(GpuDataPreparationError::invalid(
                "construct GPU data layout",
                label,
                GpuDataPreparationCause::ZeroLength,
                "provide a nonzero prepared byte length",
            ));
        }
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(GpuDataPreparationError::invalid(
                "construct GPU data layout",
                label,
                GpuDataPreparationCause::InvalidAlignment,
                "provide a nonzero power-of-two alignment",
            ));
        }
        if stride == 0 || !stride.is_multiple_of(alignment) {
            return Err(GpuDataPreparationError::invalid(
                "construct GPU data layout",
                label,
                GpuDataPreparationCause::InvalidStride,
                "provide a nonzero stride that is a multiple of alignment",
            ));
        }
        if element_count == 0 {
            return Err(GpuDataPreparationError::invalid(
                "construct GPU data layout",
                label,
                GpuDataPreparationCause::InvalidElementCount,
                "provide a nonzero element count",
            ));
        }
        let expected_len = stride.checked_mul(element_count).ok_or_else(|| {
            GpuDataPreparationError::invalid(
                "construct GPU data layout",
                label.clone(),
                GpuDataPreparationCause::ArithmeticOverflow,
                "reduce stride or element count",
            )
        })?;
        if expected_len != byte_len {
            return Err(GpuDataPreparationError::invalid(
                "construct GPU data layout",
                label,
                GpuDataPreparationCause::LengthMismatch,
                "make byte length equal checked stride times element count",
            ));
        }
        Ok(Self {
            byte_len,
            alignment,
            stride,
            element_count,
        })
    }

    pub const fn byte_len(self) -> u64 {
        self.byte_len
    }

    pub const fn alignment(self) -> u64 {
        self.alignment
    }

    pub const fn stride(self) -> u64 {
        self.stride
    }

    pub const fn element_count(self) -> u64 {
        self.element_count
    }
}

/// Immutable bytes prepared for one explicit GPU purpose.
///
/// The optional Rust type name is process-local diagnostic evidence only. It
/// does not participate in semantic equality, layout, binding, persistence,
/// replay, wire, cache, or shader-interface authority.
///
/// Arbitrary bytes cannot bypass a purpose-specific encoder:
///
/// ```compile_fail
/// use runen_gpu::{PreparedGpuData, UniformData};
/// let _ = PreparedGpuData::<UniformData>::from_bytes(vec![0; 4]);
/// ```
///
/// ```compile_fail
/// use runen_gpu::{PreparedGpuData, StorageData};
/// let _ = PreparedGpuData::<StorageData>::from_bytes(vec![0; 4]);
/// ```
///
/// ```compile_fail
/// use runen_gpu::{PreparedGpuData, VertexData};
/// let _ = PreparedGpuData::<VertexData>::from_bytes(vec![0; 4]);
/// ```
///
/// ```compile_fail
/// use runen_gpu::{IndirectData, PreparedGpuData};
/// let _ = PreparedGpuData::<IndirectData>::from_bytes(vec![0; 4]);
/// ```
pub struct PreparedGpuData<Purpose: GpuDataPurpose> {
    bytes: Arc<[u8]>,
    layout: GpuDataLayout,
    provenance: GpuResourceProvenance,
    diagnostic_type_name: Option<&'static str>,
    purpose: PhantomData<Purpose>,
}

impl<Purpose: GpuDataPurpose> Clone for PreparedGpuData<Purpose> {
    fn clone(&self) -> Self {
        Self {
            bytes: Arc::clone(&self.bytes),
            layout: self.layout,
            provenance: self.provenance.clone(),
            diagnostic_type_name: self.diagnostic_type_name,
            purpose: PhantomData,
        }
    }
}

impl<Purpose: GpuDataPurpose> fmt::Debug for PreparedGpuData<Purpose> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedGpuData")
            .field("byte_len", &self.bytes.len())
            .field("layout", &self.layout)
            .field("provenance", &self.provenance)
            .field("diagnostic_type_name", &self.diagnostic_type_name)
            .finish_non_exhaustive()
    }
}

impl<Purpose: GpuDataPurpose> PartialEq for PreparedGpuData<Purpose> {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes && self.layout == other.layout
    }
}

impl<Purpose: GpuDataPurpose> Eq for PreparedGpuData<Purpose> {}

impl<Purpose: GpuDataPurpose> PreparedGpuData<Purpose> {
    fn from_encoded_parts(
        label: &str,
        bytes: Vec<u8>,
        layout: GpuDataLayout,
        provenance: GpuResourceProvenance,
        diagnostic_type_name: Option<&'static str>,
    ) -> Result<Self, GpuDataPreparationError> {
        if u64::try_from(bytes.len()).ok() != Some(layout.byte_len()) {
            return Err(GpuDataPreparationError::invalid(
                "prepare GPU data",
                label,
                GpuDataPreparationCause::LengthMismatch,
                "make encoded byte length equal the checked layout byte length",
            ));
        }
        Ok(Self {
            bytes: bytes.into(),
            layout,
            provenance,
            diagnostic_type_name,
            purpose: PhantomData,
        })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub const fn layout(&self) -> GpuDataLayout {
        self.layout
    }

    pub fn provenance(&self) -> &GpuResourceProvenance {
        &self.provenance
    }

    pub const fn diagnostic_type_name(&self) -> Option<&'static str> {
        self.diagnostic_type_name
    }
}

impl PreparedGpuData<TransferData> {
    /// Copies a nonempty slice of [`bytemuck::Pod`] values into an immutable,
    /// explicitly transfer-purpose representation.
    ///
    /// ```
    /// use runen_gpu::*;
    /// let label = GpuResourceLabel::new("indices")?;
    /// let provenance = GpuResourceProvenance::new(label, None, None);
    /// let prepared = PreparedGpuData::<TransferData>::from_pod_transfer(
    ///     "indices", &[1_u32, 2, 3], provenance,
    /// )?;
    /// assert_eq!(prepared.layout().element_count(), 3);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_pod_transfer<Source: bytemuck::Pod>(
        label: impl Into<String>,
        values: &[Source],
        provenance: GpuResourceProvenance,
    ) -> Result<Self, GpuDataPreparationError> {
        let label = label.into();
        let stride = u64::try_from(core::mem::size_of::<Source>()).map_err(|_| {
            GpuDataPreparationError::invalid(
                "prepare Pod transfer data",
                label.clone(),
                GpuDataPreparationCause::ArithmeticOverflow,
                "use a source element representable by the platform",
            )
        })?;
        let alignment = u64::try_from(core::mem::align_of::<Source>()).map_err(|_| {
            GpuDataPreparationError::invalid(
                "prepare Pod transfer data",
                label.clone(),
                GpuDataPreparationCause::ArithmeticOverflow,
                "use a source alignment representable by the platform",
            )
        })?;
        let element_count = u64::try_from(values.len()).map_err(|_| {
            GpuDataPreparationError::invalid(
                "prepare Pod transfer data",
                label.clone(),
                GpuDataPreparationCause::ArithmeticOverflow,
                "reduce the source element count",
            )
        })?;
        let byte_len = stride.checked_mul(element_count).ok_or_else(|| {
            GpuDataPreparationError::invalid(
                "prepare Pod transfer data",
                label.clone(),
                GpuDataPreparationCause::ArithmeticOverflow,
                "reduce the source element count",
            )
        })?;
        let layout = GpuDataLayout::new(&label, byte_len, alignment, stride, element_count)?;
        Self::from_encoded_parts(
            &label,
            bytemuck::cast_slice(values).to_vec(),
            layout,
            provenance,
            Some(core::any::type_name::<Source>()),
        )
    }

    #[cfg(test)]
    pub(crate) fn from_transfer_bytes_for_adapter(
        label: impl Into<String>,
        bytes: Vec<u8>,
        layout: GpuDataLayout,
        provenance: GpuResourceProvenance,
        diagnostic_type_name: Option<&'static str>,
    ) -> Result<Self, GpuDataPreparationError> {
        let label = label.into();
        Self::from_encoded_parts(&label, bytes, layout, provenance, diagnostic_type_name)
    }
}

/// Encodes one source type for one explicit GPU purpose.
///
/// Implementations return owned bytes plus checked layout metadata. Only
/// [`prepare_gpu_data`] can turn those pieces into a `PreparedGpuData`, which
/// prevents a public arbitrary-byte constructor from bypassing the encoder.
pub trait GpuDataEncoder<Purpose: GpuDataPurpose, Source> {
    fn encode(
        &self,
        label: &str,
        source: &Source,
    ) -> Result<(Vec<u8>, GpuDataLayout), GpuDataPreparationError>;

    fn diagnostic_type_name(&self) -> Option<&'static str> {
        None
    }
}

/// Uses an explicit purpose/source encoder to create immutable prepared data.
///
/// ```
/// use runen_gpu::*;
///
/// struct UniformEncoder;
/// impl GpuDataEncoder<UniformData, u32> for UniformEncoder {
///     fn encode(&self, label: &str, value: &u32)
///         -> Result<(Vec<u8>, GpuDataLayout), GpuDataPreparationError>
///     {
///         Ok((value.to_le_bytes().to_vec(), GpuDataLayout::new(label, 4, 4, 4, 1)?))
///     }
/// }
/// struct StorageEncoder;
/// impl GpuDataEncoder<StorageData, u32> for StorageEncoder {
///     fn encode(&self, label: &str, value: &u32)
///         -> Result<(Vec<u8>, GpuDataLayout), GpuDataPreparationError>
///     {
///         Ok((value.to_le_bytes().to_vec(), GpuDataLayout::new(label, 4, 4, 4, 1)?))
///     }
/// }
/// let producer = GpuResourceLabel::new("example encoder")?;
/// let provenance = GpuResourceProvenance::new(producer, None, None);
/// let uniform: PreparedGpuData<UniformData> =
///     prepare_gpu_data("uniform", &7, provenance.clone(), &UniformEncoder)?;
/// let storage: PreparedGpuData<StorageData> =
///     prepare_gpu_data("storage", &7, provenance, &StorageEncoder)?;
/// assert_eq!(uniform.as_bytes(), storage.as_bytes());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Purpose mismatches fail to compile:
///
/// ```compile_fail
/// # use runen_gpu::*;
/// # struct UniformEncoder;
/// # impl GpuDataEncoder<UniformData, u32> for UniformEncoder {
/// #   fn encode(&self, label: &str, value: &u32) -> Result<(Vec<u8>, GpuDataLayout), GpuDataPreparationError> {
/// #     Ok((value.to_le_bytes().to_vec(), GpuDataLayout::new(label, 4, 4, 4, 1)?))
/// #   }
/// # }
/// # let producer = GpuResourceLabel::new("example")?;
/// # let provenance = GpuResourceProvenance::new(producer, None, None);
/// let storage: PreparedGpuData<StorageData> =
///     prepare_gpu_data("wrong purpose", &7, provenance, &UniformEncoder)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn prepare_gpu_data<Purpose, Source, Encoder>(
    label: impl Into<String>,
    source: &Source,
    provenance: GpuResourceProvenance,
    encoder: &Encoder,
) -> Result<PreparedGpuData<Purpose>, GpuDataPreparationError>
where
    Purpose: GpuDataPurpose,
    Encoder: GpuDataEncoder<Purpose, Source>,
{
    let label = label.into();
    let (bytes, layout) = encoder.encode(&label, source)?;
    PreparedGpuData::from_encoded_parts(
        &label,
        bytes,
        layout,
        provenance,
        encoder.diagnostic_type_name(),
    )
}

/// Immutable, normalized bytes returned by an accepted GPU readback operation.
#[derive(Debug, Clone)]
pub struct GpuReadbackBytes {
    bytes: Arc<[u8]>,
    layout: GpuDataLayout,
    texture_format: Option<GpuTextureFormat>,
    provenance: GpuResourceProvenance,
}

impl GpuReadbackBytes {
    pub(crate) fn from_normalized_bytes(
        label: &str,
        bytes: Vec<u8>,
        layout: GpuDataLayout,
        texture_format: Option<GpuTextureFormat>,
        provenance: GpuResourceProvenance,
    ) -> Result<Self, GpuDataPreparationError> {
        if u64::try_from(bytes.len()).ok() != Some(layout.byte_len()) {
            return Err(GpuDataPreparationError::invalid(
                "construct normalized GPU readback bytes",
                label,
                GpuDataPreparationCause::LengthMismatch,
                "make normalized byte length equal the checked readback layout",
            ));
        }
        Ok(Self {
            bytes: bytes.into(),
            layout,
            texture_format,
            provenance,
        })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub const fn layout(&self) -> GpuDataLayout {
        self.layout
    }

    pub const fn texture_format(&self) -> Option<GpuTextureFormat> {
        self.texture_format
    }

    pub fn provenance(&self) -> &GpuResourceProvenance {
        &self.provenance
    }
}

pub trait GpuReadbackDecoder<Output> {
    fn decode(
        &self,
        label: &str,
        bytes: &GpuReadbackBytes,
    ) -> Result<Output, GpuReadbackDecodeError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuResourceLabel;

    fn provenance(producer: &str) -> GpuResourceProvenance {
        let producer = GpuResourceLabel::new(producer).unwrap();
        GpuResourceProvenance::new(producer, None, None)
    }

    #[test]
    fn layout_checks_zero_alignment_stride_count_overflow_and_length() {
        assert!(GpuDataLayout::new("zero", 0, 1, 1, 1).is_err());
        assert!(GpuDataLayout::new("alignment", 4, 3, 4, 1).is_err());
        assert!(GpuDataLayout::new("stride", 6, 4, 6, 1).is_err());
        assert!(GpuDataLayout::new("count", 4, 4, 4, 0).is_err());
        assert!(GpuDataLayout::new("overflow", u64::MAX, 1, u64::MAX, 2).is_err());
        assert!(GpuDataLayout::new("length", 8, 4, 4, 1).is_err());
    }

    #[test]
    fn pod_transfer_carries_checked_explicit_layout() {
        let prepared = PreparedGpuData::<TransferData>::from_pod_transfer(
            "indices",
            &[3_u32, 5, 8],
            provenance("test"),
        )
        .unwrap();
        assert_eq!(prepared.layout().byte_len(), 12);
        assert_eq!(prepared.layout().alignment(), 4);
        assert_eq!(prepared.layout().stride(), 4);
        assert_eq!(prepared.layout().element_count(), 3);
        assert_eq!(
            prepared.as_bytes(),
            bytemuck::cast_slice::<u32, u8>(&[3_u32, 5, 8])
        );
    }

    #[test]
    fn diagnostic_type_and_provenance_do_not_change_semantic_equality() {
        let layout = GpuDataLayout::new("value", 4, 4, 4, 1).unwrap();
        let first = PreparedGpuData::<TransferData>::from_transfer_bytes_for_adapter(
            "first",
            vec![1, 2, 3, 4],
            layout,
            provenance("first"),
            Some("first::Type"),
        )
        .unwrap();
        let second = PreparedGpuData::<TransferData>::from_transfer_bytes_for_adapter(
            "second",
            vec![1, 2, 3, 4],
            layout,
            provenance("second"),
            Some("second::Type"),
        )
        .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn prepared_data_rejects_encoder_length_disagreement() {
        struct BadEncoder;
        impl GpuDataEncoder<UniformData, ()> for BadEncoder {
            fn encode(
                &self,
                _label: &str,
                _source: &(),
            ) -> Result<(Vec<u8>, GpuDataLayout), GpuDataPreparationError> {
                Ok((vec![0; 3], GpuDataLayout::new("bad", 4, 4, 4, 1)?))
            }
        }

        assert!(
            prepare_gpu_data::<UniformData, _, _>("bad", &(), provenance("test"), &BadEncoder)
                .is_err()
        );
    }
}
