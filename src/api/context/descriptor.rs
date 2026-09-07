use super::diagnostics::{GpuContextRequestError, GpuContextRequestErrorCategory};
use super::facts::GpuAlignmentFacts;
use super::sanitized_diagnostic;
use super::selection::GpuCandidateId;
use crate::{
    GpuCapabilityFeature, GpuCapabilityRequirement, GpuCapabilityRequirements,
    GpuPreferredFallback, GpuTextureFormat,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuBackendFamily {
    Vulkan,
    Metal,
    Direct3D12,
    OpenGl,
    BrowserWebGpu,
    UnknownBackend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuAdapterClass {
    Discrete,
    Integrated,
    Virtual,
    Cpu,
    Other,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuPowerPreference {
    HighPerformance,
    LowPower,
    NoPreference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuSoftwareFallbackPolicy {
    Allow,
    Require,
    Forbid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuPortabilityPolicy {
    AllowBackendSpecialized,
    RequirePortableBaseline,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuAlignmentKind {
    UniformDynamicOffset,
    StorageDynamicOffset,
    CopyBufferOffset,
    BytesPerRow,
    QueryResolveDestination,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuLimitKind {
    MaxUniformBufferBindingSize,
    MaxStorageBufferBindingSize,
    MaxColorAttachments,
    MaxVertexBuffers,
    MaxBindingsPerGroup,
    MaxTextureDimension2d,
    MaxBindGroups,
    MaxBindGroupsPlusVertexBuffers,
    MaxDynamicUniformBuffersPerPipelineLayout,
    MaxDynamicStorageBuffersPerPipelineLayout,
    MaxComputeWorkgroupsPerDimension,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuFormatRole {
    Sampled,
    Filterable,
    StorageRead,
    StorageWrite,
    ColorAttachment,
    DepthStencil,
    CopySource,
    CopyDestination,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuLimitConstraint {
    pub minimum: Option<u64>,
    pub maximum: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuContextDescriptor {
    label: Option<String>,
    provenance: Option<String>,
    power_preference: GpuPowerPreference,
    fallback_policy: GpuSoftwareFallbackPolicy,
    allowed_backends: BTreeSet<GpuBackendFamily>,
    backend_preference: BTreeMap<GpuBackendFamily, u8>,
    allowed_adapter_classes: BTreeSet<GpuAdapterClass>,
    portability_policy: GpuPortabilityPolicy,
    exact_candidate: Option<GpuCandidateId>,
    requirements: GpuCapabilityRequirements,
    pub(super) limits: BTreeMap<GpuLimitKind, GpuLimitConstraint>,
    pub(super) format_roles: BTreeSet<(GpuTextureFormat, GpuFormatRole)>,
    pub(super) alignments: BTreeMap<GpuAlignmentKind, u64>,
}

/// Request authority retained only while validating an opaque process-local retry token.
/// Diagnostic text and the retry token itself are intentionally excluded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GpuDescriptorRetryIdentity {
    power_preference: GpuPowerPreference,
    fallback_policy: GpuSoftwareFallbackPolicy,
    allowed_backends: BTreeSet<GpuBackendFamily>,
    backend_preference: BTreeMap<GpuBackendFamily, u8>,
    allowed_adapter_classes: BTreeSet<GpuAdapterClass>,
    portability_policy: GpuPortabilityPolicy,
    requirements: Vec<GpuCapabilityRequirement>,
    limits: Vec<(GpuLimitKind, GpuLimitConstraint)>,
    format_roles: BTreeSet<(GpuTextureFormat, GpuFormatRole)>,
    alignments: BTreeMap<GpuAlignmentKind, u64>,
}

impl GpuContextDescriptor {
    pub fn new(requirements: GpuCapabilityRequirements) -> Self {
        Self {
            label: None,
            provenance: None,
            power_preference: GpuPowerPreference::NoPreference,
            fallback_policy: GpuSoftwareFallbackPolicy::Allow,
            allowed_backends: BTreeSet::new(),
            backend_preference: BTreeMap::new(),
            allowed_adapter_classes: BTreeSet::new(),
            portability_policy: GpuPortabilityPolicy::AllowBackendSpecialized,
            exact_candidate: None,
            requirements,
            limits: BTreeMap::new(),
            format_roles: BTreeSet::new(),
            alignments: BTreeMap::new(),
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = sanitized_diagnostic(label.into());
        self
    }

    pub fn with_provenance(mut self, provenance: impl Into<String>) -> Self {
        self.provenance = sanitized_diagnostic(provenance.into());
        self
    }

    pub const fn with_power_preference(mut self, preference: GpuPowerPreference) -> Self {
        self.power_preference = preference;
        self
    }

    pub const fn with_fallback_policy(mut self, policy: GpuSoftwareFallbackPolicy) -> Self {
        self.fallback_policy = policy;
        self
    }

    pub fn with_allowed_backends(
        mut self,
        backends: impl IntoIterator<Item = GpuBackendFamily>,
    ) -> Self {
        self.allowed_backends = backends.into_iter().collect();
        self
    }

    /// Orders permitted backend families without treating backend enumeration as authority.
    pub fn with_backend_preference(
        mut self,
        backends: impl IntoIterator<Item = GpuBackendFamily>,
    ) -> Self {
        self.backend_preference.clear();
        for (priority, backend) in backends.into_iter().enumerate() {
            self.backend_preference
                .entry(backend)
                .or_insert(u8::try_from(priority).unwrap_or(u8::MAX));
        }
        self
    }

    pub fn with_allowed_adapter_classes(
        mut self,
        classes: impl IntoIterator<Item = GpuAdapterClass>,
    ) -> Self {
        self.allowed_adapter_classes = classes.into_iter().collect();
        self
    }

    pub const fn with_portability_policy(mut self, policy: GpuPortabilityPolicy) -> Self {
        self.portability_policy = policy;
        self
    }

    /// Retries one candidate disclosed by an ambiguous process-local admission report.
    /// It is intentionally neither persistent nor a hardware identity promise.
    pub const fn with_exact_candidate(mut self, candidate: GpuCandidateId) -> Self {
        self.exact_candidate = Some(candidate);
        self
    }

    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    pub fn provenance(&self) -> Option<&str> {
        self.provenance.as_deref()
    }

    pub const fn power_preference(&self) -> GpuPowerPreference {
        self.power_preference
    }

    pub const fn fallback_policy(&self) -> GpuSoftwareFallbackPolicy {
        self.fallback_policy
    }

    pub fn requirements(&self) -> &GpuCapabilityRequirements {
        &self.requirements
    }

    pub const fn exact_candidate(&self) -> Option<GpuCandidateId> {
        self.exact_candidate
    }

    pub(crate) fn retry_identity(&self) -> GpuDescriptorRetryIdentity {
        GpuDescriptorRetryIdentity {
            power_preference: self.power_preference,
            fallback_policy: self.fallback_policy,
            allowed_backends: self.allowed_backends.clone(),
            backend_preference: self.backend_preference.clone(),
            allowed_adapter_classes: self.allowed_adapter_classes.clone(),
            portability_policy: self.portability_policy,
            requirements: self.requirements.iter().collect(),
            limits: self
                .limits
                .iter()
                .map(|(kind, constraint)| (*kind, *constraint))
                .collect(),
            format_roles: self.format_roles.clone(),
            alignments: self.alignments.clone(),
        }
    }

    pub(crate) fn allowed_backends(&self) -> &BTreeSet<GpuBackendFamily> {
        &self.allowed_backends
    }

    pub(crate) fn backend_preference(&self) -> &BTreeMap<GpuBackendFamily, u8> {
        &self.backend_preference
    }

    pub(crate) fn allowed_adapter_classes(&self) -> &BTreeSet<GpuAdapterClass> {
        &self.allowed_adapter_classes
    }

    pub const fn portability_policy(&self) -> GpuPortabilityPolicy {
        self.portability_policy
    }

    /// Merges independently authored normalized requests without making diagnostics semantic.
    pub fn merge(&self, other: &Self) -> Result<Self, GpuContextRequestError> {
        let requirements = self
            .requirements
            .merge(&other.requirements)
            .map_err(|error| {
                GpuContextRequestError::new(
                    GpuContextRequestErrorCategory::ContradictoryRequest,
                    error.to_string(),
                )
            })?;
        let power_preference =
            merge_power_preference(self.power_preference, other.power_preference)?;
        let fallback_policy = merge_fallback_policy(self.fallback_policy, other.fallback_policy)?;
        let allowed_backends = merge_allowlist(&self.allowed_backends, &other.allowed_backends)?;
        let backend_preference =
            merge_backend_preference(&self.backend_preference, &other.backend_preference)?;
        let mut ordered_backend_preference = backend_preference.into_iter().collect::<Vec<_>>();
        ordered_backend_preference.sort_by_key(|(_, priority)| *priority);
        let allowed_adapter_classes = merge_allowlist(
            &self.allowed_adapter_classes,
            &other.allowed_adapter_classes,
        )?;
        let exact_candidate = match (self.exact_candidate, other.exact_candidate) {
            (None, value) | (value, None) => value,
            (Some(left), Some(right)) if left == right => Some(left),
            (Some(_), Some(_)) => {
                return Err(GpuContextRequestError::new(
                    GpuContextRequestErrorCategory::ContradictoryRequest,
                    "exact candidate retries conflict",
                ));
            }
        };
        let mut merged = Self::new(requirements)
            .with_power_preference(power_preference)
            .with_fallback_policy(fallback_policy)
            .with_allowed_backends(allowed_backends)
            .with_backend_preference(
                ordered_backend_preference
                    .into_iter()
                    .map(|(backend, _)| backend),
            )
            .with_allowed_adapter_classes(allowed_adapter_classes)
            .with_portability_policy(match (self.portability_policy, other.portability_policy) {
                (GpuPortabilityPolicy::RequirePortableBaseline, _)
                | (_, GpuPortabilityPolicy::RequirePortableBaseline) => {
                    GpuPortabilityPolicy::RequirePortableBaseline
                }
                _ => GpuPortabilityPolicy::AllowBackendSpecialized,
            });
        if let Some(candidate) = exact_candidate {
            merged = merged.with_exact_candidate(candidate);
        }
        for descriptor in [self, other] {
            for (&kind, constraint) in &descriptor.limits {
                if let Some(minimum) = constraint.minimum {
                    merged = merged.require_limit(kind, minimum);
                }
                if let Some(maximum) = constraint.maximum {
                    merged = merged.permit_limit(kind, maximum);
                }
            }
            for &(format, role) in &descriptor.format_roles {
                merged = merged.require_format_role(format, role);
            }
            for (&kind, &maximum) in &descriptor.alignments {
                merged = merged.require_alignment(kind, maximum);
            }
        }
        validate_descriptor_semantics(&merged)?;
        Ok(merged)
    }

    /// Compares only request authority, intentionally excluding diagnostic text.
    pub fn semantically_eq(&self, other: &Self) -> bool {
        self.power_preference == other.power_preference
            && self.fallback_policy == other.fallback_policy
            && self.allowed_backends == other.allowed_backends
            && self.backend_preference == other.backend_preference
            && self.allowed_adapter_classes == other.allowed_adapter_classes
            && self.portability_policy == other.portability_policy
            && self.exact_candidate == other.exact_candidate
            && self.requirements == other.requirements
            && self.limits == other.limits
            && self.format_roles == other.format_roles
            && self.alignments == other.alignments
    }

    pub fn require_limit(mut self, kind: GpuLimitKind, minimum: u64) -> Self {
        let entry = self.limits.entry(kind).or_insert(GpuLimitConstraint {
            minimum: None,
            maximum: None,
        });
        entry.minimum = Some(entry.minimum.unwrap_or(0).max(minimum));
        self
    }

    pub fn permit_limit(mut self, kind: GpuLimitKind, maximum: u64) -> Self {
        let entry = self.limits.entry(kind).or_insert(GpuLimitConstraint {
            minimum: None,
            maximum: None,
        });
        entry.maximum = Some(
            entry
                .maximum
                .map_or(maximum, |current| current.min(maximum)),
        );
        self
    }

    pub fn require_format_role(mut self, format: GpuTextureFormat, role: GpuFormatRole) -> Self {
        self.format_roles.insert((format, role));
        self
    }

    /// Requires a device alignment no larger than `maximum`.
    pub fn require_alignment(mut self, kind: GpuAlignmentKind, maximum: u64) -> Self {
        self.alignments
            .entry(kind)
            .and_modify(|current| *current = (*current).min(maximum))
            .or_insert(maximum);
        self
    }
}

fn merge_power_preference(
    left: GpuPowerPreference,
    right: GpuPowerPreference,
) -> Result<GpuPowerPreference, GpuContextRequestError> {
    match (left, right) {
        (GpuPowerPreference::NoPreference, value) | (value, GpuPowerPreference::NoPreference) => {
            Ok(value)
        }
        (left, right) if left == right => Ok(left),
        _ => Err(GpuContextRequestError::new(
            GpuContextRequestErrorCategory::ContradictoryRequest,
            "power preferences conflict",
        )),
    }
}

fn merge_fallback_policy(
    left: GpuSoftwareFallbackPolicy,
    right: GpuSoftwareFallbackPolicy,
) -> Result<GpuSoftwareFallbackPolicy, GpuContextRequestError> {
    match (left, right) {
        (GpuSoftwareFallbackPolicy::Allow, value) | (value, GpuSoftwareFallbackPolicy::Allow) => {
            Ok(value)
        }
        (left, right) if left == right => Ok(left),
        _ => Err(GpuContextRequestError::new(
            GpuContextRequestErrorCategory::ContradictoryRequest,
            "software fallback policies conflict",
        )),
    }
}

fn merge_allowlist<T: Ord + Copy>(
    left: &BTreeSet<T>,
    right: &BTreeSet<T>,
) -> Result<BTreeSet<T>, GpuContextRequestError> {
    let merged = if left.is_empty() {
        right.clone()
    } else if right.is_empty() {
        left.clone()
    } else {
        left.intersection(right).copied().collect()
    };
    if !left.is_empty() && !right.is_empty() && merged.is_empty() {
        return Err(GpuContextRequestError::new(
            GpuContextRequestErrorCategory::ContradictoryRequest,
            "allowlists have no common candidate",
        ));
    }
    Ok(merged)
}

fn merge_backend_preference(
    left: &BTreeMap<GpuBackendFamily, u8>,
    right: &BTreeMap<GpuBackendFamily, u8>,
) -> Result<BTreeMap<GpuBackendFamily, u8>, GpuContextRequestError> {
    if left.is_empty() {
        return Ok(right.clone());
    }
    if right.is_empty() || left == right {
        return Ok(left.clone());
    }
    Err(GpuContextRequestError::new(
        GpuContextRequestErrorCategory::ContradictoryRequest,
        "backend preferences conflict",
    ))
}

pub(crate) fn validate_descriptor_semantics(
    descriptor: &GpuContextDescriptor,
) -> Result<(), GpuContextRequestError> {
    for requirement in descriptor.requirements.iter() {
        if let GpuCapabilityRequirement::Preferred { feature, fallback } = requirement {
            if !preferred_degradation_is_valid(feature, fallback) {
                return Err(GpuContextRequestError::new(
                    GpuContextRequestErrorCategory::InvalidDegradation,
                    format!("{feature:?} cannot use the {fallback:?} preferred degradation"),
                ));
            }
        }
    }
    for (kind, constraint) in &descriptor.limits {
        if constraint.minimum.is_some_and(|minimum| minimum == 0)
            || constraint.maximum.is_some_and(|maximum| maximum == 0)
            || matches!((constraint.minimum, constraint.maximum), (Some(minimum), Some(maximum)) if minimum > maximum)
        {
            return Err(GpuContextRequestError::new(
                GpuContextRequestErrorCategory::ContradictoryRequest,
                format!("invalid limit constraint for {kind:?}"),
            ));
        }
    }
    if descriptor.alignments.values().any(|maximum| *maximum == 0) {
        return Err(GpuContextRequestError::new(
            GpuContextRequestErrorCategory::ContradictoryRequest,
            "alignment requirements must be nonzero",
        ));
    }
    if !descriptor.allowed_backends.is_empty()
        && descriptor
            .backend_preference
            .keys()
            .any(|backend| !descriptor.allowed_backends.contains(backend))
    {
        return Err(GpuContextRequestError::new(
            GpuContextRequestErrorCategory::ContradictoryRequest,
            "backend preference contains a forbidden backend",
        ));
    }
    Ok(())
}

/// G4A owns this explicit workload-level compatibility table. Generic G2/G3 capability
/// requirements stay representable; only a context descriptor may admit their degradation.
pub(crate) const fn preferred_degradation_is_valid(
    feature: GpuCapabilityFeature,
    fallback: GpuPreferredFallback,
) -> bool {
    match feature {
        GpuCapabilityFeature::Compute
        | GpuCapabilityFeature::RenderPipeline
        | GpuCapabilityFeature::Copy
        | GpuCapabilityFeature::IndirectExecution
        | GpuCapabilityFeature::StorageTexture
        | GpuCapabilityFeature::TextureBindingArray
        | GpuCapabilityFeature::BufferBindingArray
        | GpuCapabilityFeature::StorageResourceBindingArray
        | GpuCapabilityFeature::UniformBufferBindingArray
        | GpuCapabilityFeature::DepthAttachment => {
            matches!(fallback, GpuPreferredFallback::SelectAlternativeWork)
        }
        GpuCapabilityFeature::TimestampQuery => {
            matches!(fallback, GpuPreferredFallback::DisableInstrumentation)
        }
        GpuCapabilityFeature::Presentation => {
            matches!(fallback, GpuPreferredFallback::ContinueWithoutFeature)
        }
    }
}

pub(crate) fn alignment_value(facts: GpuAlignmentFacts, kind: GpuAlignmentKind) -> Option<u64> {
    match kind {
        GpuAlignmentKind::UniformDynamicOffset => facts.uniform_dynamic_offset,
        GpuAlignmentKind::StorageDynamicOffset => facts.storage_dynamic_offset,
        GpuAlignmentKind::CopyBufferOffset => facts.copy_buffer_offset,
        GpuAlignmentKind::BytesPerRow => facts.bytes_per_row,
        GpuAlignmentKind::QueryResolveDestination => facts.query_resolve_destination,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FEATURES: [GpuCapabilityFeature; 12] = [
        GpuCapabilityFeature::Compute,
        GpuCapabilityFeature::RenderPipeline,
        GpuCapabilityFeature::Copy,
        GpuCapabilityFeature::IndirectExecution,
        GpuCapabilityFeature::StorageTexture,
        GpuCapabilityFeature::TextureBindingArray,
        GpuCapabilityFeature::BufferBindingArray,
        GpuCapabilityFeature::StorageResourceBindingArray,
        GpuCapabilityFeature::UniformBufferBindingArray,
        GpuCapabilityFeature::DepthAttachment,
        GpuCapabilityFeature::TimestampQuery,
        GpuCapabilityFeature::Presentation,
    ];
    const FALLBACKS: [GpuPreferredFallback; 3] = [
        GpuPreferredFallback::ContinueWithoutFeature,
        GpuPreferredFallback::DisableInstrumentation,
        GpuPreferredFallback::SelectAlternativeWork,
    ];

    #[test]
    fn preferred_degradation_mapping_accepts_only_the_explicit_feature_pair() {
        for feature in FEATURES {
            for fallback in FALLBACKS {
                let mut requirements = GpuCapabilityRequirements::new();
                requirements
                    .insert(GpuCapabilityRequirement::Preferred { feature, fallback })
                    .unwrap();
                let result =
                    validate_descriptor_semantics(&GpuContextDescriptor::new(requirements));
                assert_eq!(
                    result.is_ok(),
                    preferred_degradation_is_valid(feature, fallback),
                    "{feature:?} with {fallback:?}"
                );
                if !preferred_degradation_is_valid(feature, fallback) {
                    assert_eq!(
                        result.unwrap_err().category(),
                        GpuContextRequestErrorCategory::InvalidDegradation
                    );
                }
            }
        }
    }
}
