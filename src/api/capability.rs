use super::{
    GpuCapabilityAdmissionCause, GpuCapabilityAdmissionError, GpuCapabilityRequirementCause,
    GpuCapabilityRequirementError, GpuTextureAspect,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuCapabilityFeature {
    Compute,
    RenderPipeline,
    Copy,
    IndirectExecution,
    StorageTexture,
    TextureBindingArray,
    BufferBindingArray,
    StorageResourceBindingArray,
    UniformBufferBindingArray,
    DepthAttachment,
    TimestampQuery,
    Presentation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuPreferredFallback {
    ContinueWithoutFeature,
    DisableInstrumentation,
    SelectAlternativeWork,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuCapabilityRequirement {
    Required(GpuCapabilityFeature),
    Preferred {
        feature: GpuCapabilityFeature,
        fallback: GpuPreferredFallback,
    },
    Disabled(GpuCapabilityFeature),
}

impl GpuCapabilityRequirement {
    pub const fn feature(self) -> GpuCapabilityFeature {
        match self {
            Self::Required(feature) | Self::Preferred { feature, .. } | Self::Disabled(feature) => {
                feature
            }
        }
    }
}

/// Deterministically merged capability requirements.
///
/// ```
/// use runen_gpu::{
///     GpuCapabilityFeature, GpuCapabilityRequirement, GpuCapabilityRequirements,
/// };
///
/// let mut requirements = GpuCapabilityRequirements::new();
/// requirements
///     .insert(GpuCapabilityRequirement::Required(
///         GpuCapabilityFeature::Compute,
///     ))?;
/// assert!(requirements.get(GpuCapabilityFeature::Compute).is_some());
/// # Ok::<(), runen_gpu::GpuCapabilityRequirementError>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GpuCapabilityRequirements {
    entries: BTreeMap<GpuCapabilityFeature, GpuCapabilityRequirement>,
}

impl GpuCapabilityRequirements {
    pub const fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    pub fn insert(
        &mut self,
        requirement: GpuCapabilityRequirement,
    ) -> Result<(), GpuCapabilityRequirementError> {
        let feature = requirement.feature();
        let Some(existing) = self.entries.get(&feature).copied() else {
            self.entries.insert(feature, requirement);
            return Ok(());
        };
        let merged = merge_requirement(existing, requirement).map_err(|cause| {
            GpuCapabilityRequirementError::Invalid {
                operation: "merge GPU capability requirement",
                label: format!("{feature:?}"),
                cause,
                correction: "remove the conflicting consumer constraint or choose one preferred fallback",
            }
        })?;
        self.entries.insert(feature, merged);
        Ok(())
    }

    pub fn get(&self, feature: GpuCapabilityFeature) -> Option<GpuCapabilityRequirement> {
        self.entries.get(&feature).copied()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = GpuCapabilityRequirement> + '_ {
        self.entries.values().copied()
    }

    pub fn merge(&self, other: &Self) -> Result<Self, GpuCapabilityRequirementError> {
        let mut merged = self.clone();
        for requirement in other.iter() {
            merged.insert(requirement)?;
        }
        Ok(merged)
    }
}

fn merge_requirement(
    left: GpuCapabilityRequirement,
    right: GpuCapabilityRequirement,
) -> Result<GpuCapabilityRequirement, GpuCapabilityRequirementCause> {
    use GpuCapabilityRequirement::{Disabled, Preferred, Required};
    match (left, right) {
        (Required(feature), Required(_))
        | (Required(feature), Preferred { .. })
        | (Preferred { feature, .. }, Required(_)) => Ok(Required(feature)),
        (
            Preferred {
                feature,
                fallback: left,
            },
            Preferred {
                fallback: right, ..
            },
        ) if left == right => Ok(Preferred {
            feature,
            fallback: left,
        }),
        (Preferred { .. }, Preferred { .. }) => {
            Err(GpuCapabilityRequirementCause::AmbiguousPreferredFallback)
        }
        (Disabled(feature), Disabled(_)) => Ok(Disabled(feature)),
        (Required(_), Disabled(_))
        | (Disabled(_), Required(_))
        | (Preferred { .. }, Disabled(_))
        | (Disabled(_), Preferred { .. }) => {
            Err(GpuCapabilityRequirementCause::ConflictingStrength)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuCapabilityProfile {
    ComputeBaseline,
    OffscreenGraphicsBaseline,
    DesktopPresentationBaseline,
}

/// A profile produces ordinary requirements that can be inspected and merged.
///
/// ```
/// use runen_gpu::{GpuCapabilityFeature, GpuCapabilityProfile};
/// let requirements = GpuCapabilityProfile::ComputeBaseline.requirements();
/// assert!(requirements.get(GpuCapabilityFeature::Compute).is_some());
/// ```
impl GpuCapabilityProfile {
    pub fn requirements(self) -> GpuCapabilityRequirements {
        let entries = match self {
            Self::ComputeBaseline => [
                (
                    GpuCapabilityFeature::Compute,
                    GpuCapabilityRequirement::Required(GpuCapabilityFeature::Compute),
                ),
                (
                    GpuCapabilityFeature::Copy,
                    GpuCapabilityRequirement::Required(GpuCapabilityFeature::Copy),
                ),
            ]
            .into_iter()
            .collect(),
            Self::OffscreenGraphicsBaseline => [
                (
                    GpuCapabilityFeature::Copy,
                    GpuCapabilityRequirement::Required(GpuCapabilityFeature::Copy),
                ),
                (
                    GpuCapabilityFeature::RenderPipeline,
                    GpuCapabilityRequirement::Required(GpuCapabilityFeature::RenderPipeline),
                ),
            ]
            .into_iter()
            .collect(),
            Self::DesktopPresentationBaseline => [
                (
                    GpuCapabilityFeature::Copy,
                    GpuCapabilityRequirement::Required(GpuCapabilityFeature::Copy),
                ),
                (
                    GpuCapabilityFeature::Presentation,
                    GpuCapabilityRequirement::Required(GpuCapabilityFeature::Presentation),
                ),
                (
                    GpuCapabilityFeature::RenderPipeline,
                    GpuCapabilityRequirement::Required(GpuCapabilityFeature::RenderPipeline),
                ),
            ]
            .into_iter()
            .collect(),
        };
        GpuCapabilityRequirements { entries }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuTextureFormat {
    R8Unorm,
    R8Snorm,
    R8Uint,
    R8Sint,
    Rg8Unorm,
    Rg8Snorm,
    Rg8Uint,
    Rg8Sint,
    R16Uint,
    R16Sint,
    R16Float,
    Rg16Uint,
    Rg16Sint,
    Rg16Float,
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Rgba8Snorm,
    Rgba8Uint,
    Rgba8Sint,
    Bgra8Unorm,
    Bgra8UnormSrgb,
    R32Uint,
    R32Sint,
    R32Float,
    Rg32Uint,
    Rg32Sint,
    Rg32Float,
    Rgba32Uint,
    Rgba32Sint,
    Rgba32Float,
    Rgba16Uint,
    Rgba16Sint,
    Rgba16Float,
    Depth32Float,
}

impl GpuTextureFormat {
    pub const fn block_dimensions(self) -> (u32, u32) {
        super::texture_format::block_dimensions(self)
    }

    pub const fn copy_block_size(self, aspect: GpuTextureAspect) -> Option<u32> {
        super::texture_format::copy_block_size(self, aspect)
    }

    pub const fn is_depth(self) -> bool {
        super::texture_format::is_depth(self)
    }

    pub const fn is_stencil(self) -> bool {
        super::texture_format::is_stencil(self)
    }

    pub const fn is_srgb(self) -> bool {
        super::texture_format::is_srgb(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuTextureFormatCapabilities {
    pub sampled: bool,
    pub filterable: bool,
    pub storage_read: bool,
    pub storage_write: bool,
    pub color_attachment: bool,
    pub depth_stencil: bool,
    pub copy_source: bool,
    pub copy_destination: bool,
    /// Normalized structural texture-block dimensions for this format.
    pub block_dimensions: Option<(u32, u32)>,
    /// Normalized copy bytes per texture block when the format has one unambiguous copy aspect.
    pub block_copy_size: Option<u32>,
}

impl GpuTextureFormatCapabilities {
    pub const fn none() -> Self {
        Self {
            sampled: false,
            filterable: false,
            storage_read: false,
            storage_write: false,
            color_attachment: false,
            depth_stencil: false,
            copy_source: false,
            copy_destination: false,
            block_dimensions: None,
            block_copy_size: None,
        }
    }
}

/// Closed normalized limit vocabulary required through G5A.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuLimits {
    max_uniform_buffer_binding_size: u64,
    max_storage_buffer_binding_size: u64,
    max_color_attachments: u32,
    max_vertex_buffers: u32,
    max_bindings_per_group: u32,
    max_texture_dimension_2d: u32,
    max_bind_groups: u32,
    max_bind_groups_plus_vertex_buffers: u32,
    max_dynamic_uniform_buffers_per_pipeline_layout: u32,
    max_dynamic_storage_buffers_per_pipeline_layout: u32,
    max_compute_workgroups_per_dimension: u32,
    max_buffer_size: u64,
    max_texture_dimension_1d: u32,
    max_texture_dimension_3d: u32,
    max_texture_array_layers: u32,
    max_vertex_attributes: u32,
    max_vertex_buffer_array_stride: u32,
}

impl GpuLimits {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_uniform_buffer_binding_size: u64,
        max_storage_buffer_binding_size: u64,
        max_color_attachments: u32,
        max_vertex_buffers: u32,
        max_bindings_per_group: u32,
        max_texture_dimension_2d: u32,
        max_bind_groups: u32,
        max_bind_groups_plus_vertex_buffers: u32,
        max_dynamic_uniform_buffers_per_pipeline_layout: u32,
        max_dynamic_storage_buffers_per_pipeline_layout: u32,
        max_compute_workgroups_per_dimension: u32,
        max_buffer_size: u64,
        max_texture_dimension_1d: u32,
        max_texture_dimension_3d: u32,
        max_texture_array_layers: u32,
        max_vertex_attributes: u32,
        max_vertex_buffer_array_stride: u32,
    ) -> Result<Self, GpuCapabilityAdmissionError> {
        if max_uniform_buffer_binding_size == 0
            || max_storage_buffer_binding_size == 0
            || max_color_attachments == 0
            || max_vertex_buffers == 0
            || max_bindings_per_group == 0
            || max_texture_dimension_2d == 0
            || max_bind_groups == 0
            || max_bind_groups_plus_vertex_buffers == 0
            || max_compute_workgroups_per_dimension == 0
            || max_buffer_size == 0
            || max_texture_dimension_1d == 0
            || max_texture_dimension_3d == 0
            || max_texture_array_layers == 0
            || max_vertex_attributes == 0
            || max_vertex_buffer_array_stride == 0
            || max_bind_groups_plus_vertex_buffers < max_bind_groups
            || max_bind_groups_plus_vertex_buffers < max_vertex_buffers
        {
            return Err(GpuCapabilityAdmissionError::Rejected {
                operation: "construct normalized GPU limits",
                label: "GPU limits".to_string(),
                cause: GpuCapabilityAdmissionCause::InvalidLimit,
                feature: None,
                correction: "provide internally consistent normalized limits; dynamic-buffer maxima may be zero when unsupported",
            });
        }
        Ok(Self {
            max_uniform_buffer_binding_size,
            max_storage_buffer_binding_size,
            max_color_attachments,
            max_vertex_buffers,
            max_bindings_per_group,
            max_texture_dimension_2d,
            max_bind_groups,
            max_bind_groups_plus_vertex_buffers,
            max_dynamic_uniform_buffers_per_pipeline_layout,
            max_dynamic_storage_buffers_per_pipeline_layout,
            max_compute_workgroups_per_dimension,
            max_buffer_size,
            max_texture_dimension_1d,
            max_texture_dimension_3d,
            max_texture_array_layers,
            max_vertex_attributes,
            max_vertex_buffer_array_stride,
        })
    }

    pub const fn max_uniform_buffer_binding_size(self) -> u64 {
        self.max_uniform_buffer_binding_size
    }
    pub const fn max_storage_buffer_binding_size(self) -> u64 {
        self.max_storage_buffer_binding_size
    }
    pub const fn max_color_attachments(self) -> u32 {
        self.max_color_attachments
    }
    pub const fn max_vertex_buffers(self) -> u32 {
        self.max_vertex_buffers
    }
    pub const fn max_bindings_per_group(self) -> u32 {
        self.max_bindings_per_group
    }
    pub const fn max_texture_dimension_2d(self) -> u32 {
        self.max_texture_dimension_2d
    }
    pub const fn max_bind_groups(self) -> u32 {
        self.max_bind_groups
    }
    pub const fn max_bind_groups_plus_vertex_buffers(self) -> u32 {
        self.max_bind_groups_plus_vertex_buffers
    }
    pub const fn max_dynamic_uniform_buffers_per_pipeline_layout(self) -> u32 {
        self.max_dynamic_uniform_buffers_per_pipeline_layout
    }
    pub const fn max_dynamic_storage_buffers_per_pipeline_layout(self) -> u32 {
        self.max_dynamic_storage_buffers_per_pipeline_layout
    }
    pub const fn max_compute_workgroups_per_dimension(self) -> u32 {
        self.max_compute_workgroups_per_dimension
    }
    pub const fn max_buffer_size(self) -> u64 {
        self.max_buffer_size
    }
    pub const fn max_texture_dimension_1d(self) -> u32 {
        self.max_texture_dimension_1d
    }
    pub const fn max_texture_dimension_3d(self) -> u32 {
        self.max_texture_dimension_3d
    }
    pub const fn max_texture_array_layers(self) -> u32 {
        self.max_texture_array_layers
    }
    pub const fn max_vertex_attributes(self) -> u32 {
        self.max_vertex_attributes
    }
    pub const fn max_vertex_buffer_array_stride(self) -> u32 {
        self.max_vertex_buffer_array_stride
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) const fn from_validated_adapter_facts(
        max_uniform_buffer_binding_size: u64,
        max_storage_buffer_binding_size: u64,
        max_color_attachments: u32,
        max_vertex_buffers: u32,
        max_bindings_per_group: u32,
        max_texture_dimension_2d: u32,
        max_bind_groups: u32,
        max_bind_groups_plus_vertex_buffers: u32,
        max_dynamic_uniform_buffers_per_pipeline_layout: u32,
        max_dynamic_storage_buffers_per_pipeline_layout: u32,
        max_compute_workgroups_per_dimension: u32,
        max_buffer_size: u64,
        max_texture_dimension_1d: u32,
        max_texture_dimension_3d: u32,
        max_texture_array_layers: u32,
        max_vertex_attributes: u32,
        max_vertex_buffer_array_stride: u32,
    ) -> Self {
        Self {
            max_uniform_buffer_binding_size,
            max_storage_buffer_binding_size,
            max_color_attachments,
            max_vertex_buffers,
            max_bindings_per_group,
            max_texture_dimension_2d,
            max_bind_groups,
            max_bind_groups_plus_vertex_buffers,
            max_dynamic_uniform_buffers_per_pipeline_layout,
            max_dynamic_storage_buffers_per_pipeline_layout,
            max_compute_workgroups_per_dimension,
            max_buffer_size,
            max_texture_dimension_1d,
            max_texture_dimension_3d,
            max_texture_array_layers,
            max_vertex_attributes,
            max_vertex_buffer_array_stride,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuCapabilities {
    features: BTreeSet<GpuCapabilityFeature>,
    limits: GpuLimits,
    formats: BTreeMap<GpuTextureFormat, GpuTextureFormatCapabilities>,
}

impl GpuCapabilities {
    pub fn from_normalized_facts(
        features: impl IntoIterator<Item = GpuCapabilityFeature>,
        limits: GpuLimits,
        formats: impl IntoIterator<Item = (GpuTextureFormat, GpuTextureFormatCapabilities)>,
    ) -> Self {
        let formats = formats
            .into_iter()
            .map(|(format, mut facts)| {
                facts.block_dimensions = Some(super::texture_format::block_dimensions(format));
                facts.block_copy_size = super::texture_format::default_copy_block_size(format);
                (format, facts)
            })
            .collect();
        Self {
            features: features.into_iter().collect(),
            limits,
            formats,
        }
    }

    pub fn supports(&self, feature: GpuCapabilityFeature) -> bool {
        self.features.contains(&feature)
    }

    /// Generic logical capability limits. G4A additionally exposes typed adapter,
    /// device, and workload facts so this value is never published as device truth.
    pub const fn limits(&self) -> GpuLimits {
        self.limits
    }

    pub fn format(&self, format: GpuTextureFormat) -> Option<GpuTextureFormatCapabilities> {
        self.formats.get(&format).copied()
    }

    pub fn features(&self) -> impl ExactSizeIterator<Item = GpuCapabilityFeature> + '_ {
        self.features.iter().copied()
    }

    pub fn formats(
        &self,
    ) -> impl ExactSizeIterator<Item = (GpuTextureFormat, GpuTextureFormatCapabilities)> + '_ {
        self.formats.iter().map(|(format, facts)| (*format, *facts))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuPreferredAvailability {
    pub feature: GpuCapabilityFeature,
    pub available: bool,
    pub enabled: bool,
    pub fallback: GpuPreferredFallback,
}

#[derive(Debug, Clone)]
pub struct GpuCapabilityAdmission {
    granted_required: Vec<GpuCapabilityFeature>,
    preferred: Vec<GpuPreferredAvailability>,
    verified_disabled: Vec<GpuCapabilityFeature>,
    diagnostics: Vec<String>,
}

impl PartialEq for GpuCapabilityAdmission {
    fn eq(&self, other: &Self) -> bool {
        self.granted_required == other.granted_required
            && self.preferred == other.preferred
            && self.verified_disabled == other.verified_disabled
    }
}

impl Eq for GpuCapabilityAdmission {}

impl GpuCapabilityAdmission {
    pub fn evaluate(
        label: impl Into<String>,
        requirements: &GpuCapabilityRequirements,
        capabilities: &GpuCapabilities,
        enabled_features: impl IntoIterator<Item = GpuCapabilityFeature>,
    ) -> Result<Self, GpuCapabilityAdmissionError> {
        let label = label.into();
        let enabled_features = enabled_features.into_iter().collect::<BTreeSet<_>>();
        if let Some(feature) = enabled_features
            .iter()
            .find(|feature| !capabilities.supports(**feature))
        {
            return Err(GpuCapabilityAdmissionError::Rejected {
                operation: "admit GPU capability requirements",
                label: format!("{label}::{feature:?}"),
                cause: GpuCapabilityAdmissionCause::EnabledUnavailable,
                feature: Some(*feature),
                correction: "enable only normalized features reported by the admitted backend",
            });
        }
        let mut granted_required = Vec::new();
        let mut preferred = Vec::new();
        let mut verified_disabled = Vec::new();
        let mut diagnostics = Vec::new();
        for requirement in requirements.iter() {
            match requirement {
                GpuCapabilityRequirement::Required(feature) => {
                    if !capabilities.supports(feature) {
                        return Err(GpuCapabilityAdmissionError::Rejected {
                            operation: "admit GPU capability requirements",
                            label,
                            cause: GpuCapabilityAdmissionCause::RequiredUnavailable,
                            feature: Some(feature),
                            correction: "select a capable backend or remove the required workload",
                        });
                    }
                    if !enabled_features.contains(&feature) {
                        return Err(GpuCapabilityAdmissionError::Rejected {
                            operation: "admit GPU capability requirements",
                            label,
                            cause: GpuCapabilityAdmissionCause::RequiredNotEnabled,
                            feature: Some(feature),
                            correction: "enable the required feature during backend admission",
                        });
                    }
                    granted_required.push(feature);
                }
                GpuCapabilityRequirement::Preferred { feature, fallback } => {
                    let available = capabilities.supports(feature);
                    let enabled = enabled_features.contains(&feature);
                    preferred.push(GpuPreferredAvailability {
                        feature,
                        available,
                        enabled,
                        fallback,
                    });
                    if !available {
                        diagnostics.push(format!(
                            "preferred {feature:?} unavailable; apply {fallback:?}"
                        ));
                    } else if !enabled {
                        diagnostics.push(format!(
                            "preferred {feature:?} not enabled; apply {fallback:?}"
                        ));
                    }
                }
                GpuCapabilityRequirement::Disabled(feature) => {
                    if enabled_features.contains(&feature) {
                        return Err(GpuCapabilityAdmissionError::Rejected {
                            operation: "admit GPU capability requirements",
                            label,
                            cause: GpuCapabilityAdmissionCause::DisabledEnabled,
                            feature: Some(feature),
                            correction: "disable the feature path before admission",
                        });
                    }
                    verified_disabled.push(feature);
                }
            }
        }
        Ok(Self {
            granted_required,
            preferred,
            verified_disabled,
            diagnostics,
        })
    }

    pub fn granted_required(&self) -> &[GpuCapabilityFeature] {
        &self.granted_required
    }
    pub fn preferred(&self) -> &[GpuPreferredAvailability] {
        &self.preferred
    }
    pub fn verified_disabled(&self) -> &[GpuCapabilityFeature] {
        &self.verified_disabled
    }
    pub fn diagnostics(&self) -> &[String] {
        &self.diagnostics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn r32float_exposes_truthful_block_copy_structure() {
        let format = GpuTextureFormat::R32Float;
        assert_eq!(format.block_dimensions(), (1, 1));
        assert_eq!(format.copy_block_size(GpuTextureAspect::All), Some(4));
        assert_eq!(format.copy_block_size(GpuTextureAspect::Color), Some(4));
        assert_eq!(format.copy_block_size(GpuTextureAspect::DepthOnly), None);
        assert!(!format.is_depth());
        assert!(!format.is_srgb());
    }

    #[test]
    fn format_capability_construction_normalizes_structural_fields() {
        let input = GpuTextureFormatCapabilities {
            sampled: true,
            filterable: false,
            storage_read: true,
            storage_write: false,
            color_attachment: false,
            depth_stencil: false,
            copy_source: true,
            copy_destination: false,
            block_dimensions: Some((99, 77)),
            block_copy_size: Some(123),
        };
        let capabilities = GpuCapabilities::from_normalized_facts(
            [],
            test_limits(),
            [(GpuTextureFormat::R32Float, input)],
        );
        let normalized = capabilities.format(GpuTextureFormat::R32Float).unwrap();
        assert_eq!(normalized.block_dimensions, Some((1, 1)));
        assert_eq!(normalized.block_copy_size, Some(4));
        assert_eq!(normalized.sampled, input.sampled);
        assert_eq!(normalized.filterable, input.filterable);
        assert_eq!(normalized.storage_read, input.storage_read);
        assert_eq!(normalized.storage_write, input.storage_write);
        assert_eq!(normalized.color_attachment, input.color_attachment);
        assert_eq!(normalized.depth_stencil, input.depth_stencil);
        assert_eq!(normalized.copy_source, input.copy_source);
        assert_eq!(normalized.copy_destination, input.copy_destination);
    }

    #[test]
    fn insertion_lookup_iteration_and_merge_are_deterministic() {
        let mut left = GpuCapabilityRequirements::new();
        left.insert(GpuCapabilityRequirement::Preferred {
            feature: GpuCapabilityFeature::TimestampQuery,
            fallback: GpuPreferredFallback::DisableInstrumentation,
        })
        .unwrap();
        let mut right = GpuCapabilityRequirements::new();
        right
            .insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::Compute,
            ))
            .unwrap();

        let left_right = left.merge(&right).unwrap();
        let right_left = right.merge(&left).unwrap();
        assert_eq!(left_right, right_left);
        assert_eq!(
            left_right.iter().collect::<Vec<_>>(),
            right_left.iter().collect::<Vec<_>>()
        );
        assert_eq!(
            left_right.get(GpuCapabilityFeature::Compute),
            Some(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::Compute
            ))
        );
    }

    #[test]
    fn conflicts_are_explicit() {
        let mut requirements = GpuCapabilityRequirements::new();
        requirements
            .insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::Compute,
            ))
            .unwrap();
        let error = requirements
            .insert(GpuCapabilityRequirement::Disabled(
                GpuCapabilityFeature::Compute,
            ))
            .unwrap_err();
        assert!(matches!(
            error,
            GpuCapabilityRequirementError::Invalid {
                cause: GpuCapabilityRequirementCause::ConflictingStrength,
                ..
            }
        ));

        let mut preferred = GpuCapabilityRequirements::new();
        preferred
            .insert(GpuCapabilityRequirement::Preferred {
                feature: GpuCapabilityFeature::TimestampQuery,
                fallback: GpuPreferredFallback::DisableInstrumentation,
            })
            .unwrap();
        let mut alternative = GpuCapabilityRequirements::new();
        alternative
            .insert(GpuCapabilityRequirement::Preferred {
                feature: GpuCapabilityFeature::TimestampQuery,
                fallback: GpuPreferredFallback::ContinueWithoutFeature,
            })
            .unwrap();
        assert!(matches!(
            preferred.merge(&alternative),
            Err(GpuCapabilityRequirementError::Invalid {
                cause: GpuCapabilityRequirementCause::AmbiguousPreferredFallback,
                ..
            })
        ));
        assert!(matches!(
            alternative.merge(&preferred),
            Err(GpuCapabilityRequirementError::Invalid {
                cause: GpuCapabilityRequirementCause::AmbiguousPreferredFallback,
                ..
            })
        ));
    }

    #[test]
    fn compatible_strength_merge_is_commutative() {
        let mut required = GpuCapabilityRequirements::new();
        required
            .insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::Compute,
            ))
            .unwrap();
        let mut preferred = GpuCapabilityRequirements::new();
        preferred
            .insert(GpuCapabilityRequirement::Preferred {
                feature: GpuCapabilityFeature::Compute,
                fallback: GpuPreferredFallback::SelectAlternativeWork,
            })
            .unwrap();

        assert_eq!(required.merge(&preferred), preferred.merge(&required));
        assert_eq!(
            required
                .merge(&preferred)
                .unwrap()
                .get(GpuCapabilityFeature::Compute),
            Some(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::Compute
            ))
        );
    }

    #[test]
    fn profiles_are_ordinary_requirements() {
        let requirements = GpuCapabilityProfile::ComputeBaseline.requirements();
        assert_eq!(requirements.iter().len(), 2);
        assert!(matches!(
            requirements.get(GpuCapabilityFeature::Compute),
            Some(GpuCapabilityRequirement::Required(_))
        ));
    }

    #[test]
    fn portable_baseline_profiles_do_not_require_native_binding_arrays() {
        let binding_array_features = [
            GpuCapabilityFeature::TextureBindingArray,
            GpuCapabilityFeature::BufferBindingArray,
            GpuCapabilityFeature::StorageResourceBindingArray,
            GpuCapabilityFeature::UniformBufferBindingArray,
        ];

        for profile in [
            GpuCapabilityProfile::ComputeBaseline,
            GpuCapabilityProfile::OffscreenGraphicsBaseline,
            GpuCapabilityProfile::DesktopPresentationBaseline,
        ] {
            let requirements = profile.requirements();
            for feature in binding_array_features {
                assert!(
                    requirements.get(feature).is_none(),
                    "{profile:?} must not require native binding-array feature {feature:?}"
                );
            }
        }
    }

    #[test]
    fn normalized_execution_limits_are_complete_and_consistent() {
        let limits = GpuLimits::new(
            1,
            2,
            3,
            4,
            5,
            8192,
            4,
            24,
            8,
            4,
            65535,
            256 * 1024 * 1024,
            8192,
            2048,
            256,
            16,
            2048,
        )
        .unwrap();
        assert_eq!(limits.max_texture_dimension_2d(), 8192);
        assert_eq!(limits.max_bind_groups(), 4);
        assert_eq!(limits.max_bind_groups_plus_vertex_buffers(), 24);
        assert_eq!(limits.max_dynamic_uniform_buffers_per_pipeline_layout(), 8);
        assert_eq!(limits.max_dynamic_storage_buffers_per_pipeline_layout(), 4);
        assert_eq!(limits.max_compute_workgroups_per_dimension(), 65535);
        assert_eq!(limits.max_buffer_size(), 256 * 1024 * 1024);
        assert_eq!(limits.max_texture_dimension_1d(), 8192);
        assert_eq!(limits.max_texture_dimension_3d(), 2048);
        assert_eq!(limits.max_texture_array_layers(), 256);
        assert_eq!(limits.max_vertex_attributes(), 16);
        assert_eq!(limits.max_vertex_buffer_array_stride(), 2048);
        let no_dynamic_buffers = GpuLimits::new(
            1,
            2,
            3,
            4,
            5,
            8192,
            4,
            24,
            0,
            0,
            65535,
            256 * 1024 * 1024,
            8192,
            2048,
            256,
            16,
            2048,
        )
        .unwrap();
        assert_eq!(
            no_dynamic_buffers.max_dynamic_uniform_buffers_per_pipeline_layout(),
            0
        );
        assert_eq!(
            no_dynamic_buffers.max_dynamic_storage_buffers_per_pipeline_layout(),
            0
        );
        assert!(
            GpuLimits::new(
                1,
                2,
                3,
                8,
                5,
                8192,
                4,
                4,
                8,
                4,
                65535,
                256 * 1024 * 1024,
                8192,
                2048,
                256,
                16,
                2048
            )
            .is_err()
        );
    }

    #[test]
    fn admission_rejections_keep_feature_evidence_typed() {
        let mut required = GpuCapabilityRequirements::new();
        required
            .insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::Compute,
            ))
            .unwrap();

        let unavailable = GpuCapabilityAdmission::evaluate(
            "required unavailable",
            &required,
            &GpuCapabilities::from_normalized_facts([], test_limits(), []),
            [],
        )
        .unwrap_err();
        assert_eq!(
            unavailable.cause(),
            GpuCapabilityAdmissionCause::RequiredUnavailable
        );
        assert_eq!(unavailable.feature(), Some(GpuCapabilityFeature::Compute));

        let not_enabled = GpuCapabilityAdmission::evaluate(
            "required not enabled",
            &required,
            &GpuCapabilities::from_normalized_facts(
                [GpuCapabilityFeature::Compute],
                test_limits(),
                [],
            ),
            [],
        )
        .unwrap_err();
        assert_eq!(
            not_enabled.cause(),
            GpuCapabilityAdmissionCause::RequiredNotEnabled
        );
        assert_eq!(not_enabled.feature(), Some(GpuCapabilityFeature::Compute));

        let disabled = {
            let mut requirements = GpuCapabilityRequirements::new();
            requirements
                .insert(GpuCapabilityRequirement::Disabled(
                    GpuCapabilityFeature::TimestampQuery,
                ))
                .unwrap();
            GpuCapabilityAdmission::evaluate(
                "disabled enabled",
                &requirements,
                &GpuCapabilities::from_normalized_facts(
                    [GpuCapabilityFeature::TimestampQuery],
                    test_limits(),
                    [],
                ),
                [GpuCapabilityFeature::TimestampQuery],
            )
            .unwrap_err()
        };
        assert_eq!(
            disabled.cause(),
            GpuCapabilityAdmissionCause::DisabledEnabled
        );
        assert_eq!(
            disabled.feature(),
            Some(GpuCapabilityFeature::TimestampQuery)
        );

        let enabled_unavailable = GpuCapabilityAdmission::evaluate(
            "enabled unavailable",
            &GpuCapabilityRequirements::new(),
            &GpuCapabilities::from_normalized_facts([], test_limits(), []),
            [GpuCapabilityFeature::TimestampQuery],
        )
        .unwrap_err();
        assert_eq!(
            enabled_unavailable.cause(),
            GpuCapabilityAdmissionCause::EnabledUnavailable
        );
        assert_eq!(
            enabled_unavailable.feature(),
            Some(GpuCapabilityFeature::TimestampQuery)
        );

        let invalid_limit = GpuLimits::new(
            0,
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
        .unwrap_err();
        assert_eq!(
            invalid_limit.cause(),
            GpuCapabilityAdmissionCause::InvalidLimit
        );
        assert_eq!(invalid_limit.feature(), None);
    }

    #[test]
    fn array_features_remain_unavailable_without_context_enablement() {
        let capabilities = GpuCapabilities::from_normalized_facts(
            [GpuCapabilityFeature::TextureBindingArray],
            test_limits(),
            [],
        );
        let mut requirements = GpuCapabilityRequirements::new();
        requirements
            .insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::TextureBindingArray,
            ))
            .unwrap();

        assert!(matches!(
            GpuCapabilityAdmission::evaluate("array program", &requirements, &capabilities, []),
            Err(GpuCapabilityAdmissionError::Rejected {
                cause: GpuCapabilityAdmissionCause::RequiredNotEnabled,
                ..
            })
        ));
    }

    #[test]
    fn admission_distinguishes_availability_from_enablement() {
        let capabilities = GpuCapabilities::from_normalized_facts(
            [
                GpuCapabilityFeature::Compute,
                GpuCapabilityFeature::TimestampQuery,
            ],
            test_limits(),
            [],
        );
        let mut requirements = GpuCapabilityRequirements::new();
        requirements
            .insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::Compute,
            ))
            .unwrap();
        requirements
            .insert(GpuCapabilityRequirement::Disabled(
                GpuCapabilityFeature::TimestampQuery,
            ))
            .unwrap();

        let admission = GpuCapabilityAdmission::evaluate(
            "compute",
            &requirements,
            &capabilities,
            [GpuCapabilityFeature::Compute],
        )
        .unwrap();
        assert_eq!(
            admission.verified_disabled(),
            &[GpuCapabilityFeature::TimestampQuery]
        );
        assert!(matches!(
            GpuCapabilityAdmission::evaluate(
                "compute",
                &requirements,
                &capabilities,
                [
                    GpuCapabilityFeature::Compute,
                    GpuCapabilityFeature::TimestampQuery,
                ],
            ),
            Err(GpuCapabilityAdmissionError::Rejected {
                cause: GpuCapabilityAdmissionCause::DisabledEnabled,
                ..
            })
        ));
        assert!(matches!(
            GpuCapabilityAdmission::evaluate("compute", &requirements, &capabilities, []),
            Err(GpuCapabilityAdmissionError::Rejected {
                cause: GpuCapabilityAdmissionCause::RequiredNotEnabled,
                ..
            })
        ));
    }

    #[test]
    fn admission_diagnostics_do_not_change_semantic_equality() {
        let facts = GpuPreferredAvailability {
            feature: GpuCapabilityFeature::TimestampQuery,
            available: false,
            enabled: false,
            fallback: GpuPreferredFallback::DisableInstrumentation,
        };
        let first = GpuCapabilityAdmission {
            granted_required: vec![GpuCapabilityFeature::Compute],
            preferred: vec![facts],
            verified_disabled: vec![GpuCapabilityFeature::Presentation],
            diagnostics: vec!["first diagnostic wording".to_string()],
        };
        let second = GpuCapabilityAdmission {
            granted_required: vec![GpuCapabilityFeature::Compute],
            preferred: vec![facts],
            verified_disabled: vec![GpuCapabilityFeature::Presentation],
            diagnostics: vec!["different diagnostic wording".to_string()],
        };

        assert_eq!(first, second);
        assert_ne!(first.diagnostics(), second.diagnostics());
    }
}
