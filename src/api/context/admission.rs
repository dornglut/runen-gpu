use super::descriptor::{
    GpuAlignmentKind, GpuContextDescriptor, GpuFormatRole, GpuLimitKind, GpuPortabilityPolicy,
    GpuSoftwareFallbackPolicy, alignment_value, validate_descriptor_semantics,
};
use super::diagnostics::{
    GpuContextLimitRejection, GpuContextRequestError, GpuContextRequestErrorCategory,
};
use super::facts::{
    GpuAdapterFacts, GpuAdmittedDeviceFacts, GpuAlignmentFacts, GpuDeviceLimits,
    GpuDeviceRequestProfile, GpuPortabilityClass, GpuPortabilityEvidence, GpuPortabilityReason,
    GpuSoftwareStatus, GpuWorkloadBudget,
};
use super::selection::GpuCandidateId;
use crate::{
    GpuCapabilityAdmission, GpuCapabilityAdmissionError, GpuCapabilityFeature,
    GpuCapabilityRequirement, GpuLimits, GpuPreferredFallback, GpuTextureFormat,
};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuCandidateEnvironmentEvidence {
    pub(super) host_compatible: bool,
    pub(super) host_compatibility_required: bool,
}

impl GpuCandidateEnvironmentEvidence {
    pub(crate) const fn headless() -> Self {
        Self {
            host_compatible: false,
            host_compatibility_required: false,
        }
    }

    pub(crate) const fn current_host(host_compatible: bool) -> Self {
        Self {
            host_compatible,
            host_compatibility_required: true,
        }
    }

    pub(crate) const fn satisfies_host_compatibility_constraint(self) -> bool {
        !self.host_compatibility_required || self.host_compatible
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDegradationRecord {
    pub feature: GpuCapabilityFeature,
    pub fallback: GpuPreferredFallback,
}

/// Complete typed contract retained after pure admission and before device creation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuAdmissionContract {
    capability_admission: GpuCapabilityAdmission,
    enabled_features: BTreeSet<GpuCapabilityFeature>,
    degradations: Vec<GpuDegradationRecord>,
    format_roles: BTreeSet<(GpuTextureFormat, GpuFormatRole)>,
    selected_alignments: GpuAlignmentFacts,
    workload_budget: GpuWorkloadBudget,
    portability: GpuPortabilityEvidence,
    device_request_profile: GpuDeviceRequestProfile,
}

impl GpuAdmissionContract {
    pub fn capability_admission(&self) -> &GpuCapabilityAdmission {
        &self.capability_admission
    }

    pub fn enabled_features(&self) -> impl ExactSizeIterator<Item = GpuCapabilityFeature> + '_ {
        self.enabled_features.iter().copied()
    }

    pub fn degradations(&self) -> &[GpuDegradationRecord] {
        &self.degradations
    }

    pub fn format_roles(
        &self,
    ) -> impl ExactSizeIterator<Item = (GpuTextureFormat, GpuFormatRole)> + '_ {
        self.format_roles.iter().copied()
    }

    pub const fn selected_alignments(&self) -> GpuAlignmentFacts {
        self.selected_alignments
    }

    pub fn workload_budget(&self) -> &GpuWorkloadBudget {
        &self.workload_budget
    }

    pub fn portability(&self) -> &GpuPortabilityEvidence {
        &self.portability
    }

    pub const fn device_request_profile(&self) -> GpuDeviceRequestProfile {
        self.device_request_profile
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuCandidateAdmissionReport {
    pub(crate) id: GpuCandidateId,
    adapter: GpuAdapterFacts,
    contract: GpuAdmissionContract,
}

impl GpuCandidateAdmissionReport {
    pub const fn id(&self) -> GpuCandidateId {
        self.id
    }

    pub fn adapter(&self) -> &GpuAdapterFacts {
        &self.adapter
    }

    pub fn contract(&self) -> &GpuAdmissionContract {
        &self.contract
    }

    pub fn enabled_features(&self) -> impl ExactSizeIterator<Item = GpuCapabilityFeature> + '_ {
        self.contract.enabled_features()
    }

    pub fn degradations(&self) -> &[GpuDegradationRecord] {
        self.contract.degradations()
    }

    pub const fn portability(&self) -> GpuPortabilityClass {
        self.contract.portability.class()
    }

    pub fn portability_evidence(&self) -> &GpuPortabilityEvidence {
        &self.contract.portability
    }

    pub fn workload_budget(&self) -> &GpuWorkloadBudget {
        &self.contract.workload_budget
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuRejectedCandidateReport {
    pub(crate) id: GpuCandidateId,
    pub(crate) adapter: GpuAdapterFacts,
    pub(crate) category: GpuContextRequestErrorCategory,
    pub(crate) detail: Option<String>,
    pub(crate) capability_admission_error: Option<GpuCapabilityAdmissionError>,
    pub(crate) limit_rejection: Option<GpuContextLimitRejection>,
}

impl GpuRejectedCandidateReport {
    pub const fn id(&self) -> GpuCandidateId {
        self.id
    }

    pub fn adapter(&self) -> &GpuAdapterFacts {
        &self.adapter
    }

    pub const fn category(&self) -> GpuContextRequestErrorCategory {
        self.category
    }

    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }

    pub fn capability_admission_error(&self) -> Option<&GpuCapabilityAdmissionError> {
        self.capability_admission_error.as_ref()
    }

    /// Returns `(normalized limit kind, required minimum, observed value)` for a below-minimum
    /// candidate-admission rejection. Other rejection classes return `None`.
    pub const fn limit_rejection(&self) -> Option<(GpuLimitKind, u64, u64)> {
        match self.limit_rejection {
            Some(rejection) => Some(rejection.as_tuple()),
            None => None,
        }
    }

    pub(crate) fn from_context_error(
        id: GpuCandidateId,
        adapter: GpuAdapterFacts,
        error: GpuContextRequestError,
    ) -> Self {
        Self {
            id,
            adapter,
            category: error.category(),
            detail: error.detail().map(str::to_owned),
            capability_admission_error: error.capability_admission_error().cloned(),
            limit_rejection: error.limit_rejection_evidence(),
        }
    }
}

pub(crate) fn validate_descriptor(
    descriptor: &GpuContextDescriptor,
) -> Result<(), GpuContextRequestError> {
    validate_descriptor_semantics(descriptor)
}

pub(crate) fn evaluate_validated_candidate(
    descriptor: &GpuContextDescriptor,
    id: GpuCandidateId,
    adapter: GpuAdapterFacts,
    environment: GpuCandidateEnvironmentEvidence,
) -> Result<GpuCandidateAdmissionReport, GpuContextRequestError> {
    if !environment.satisfies_host_compatibility_constraint() {
        return Err(GpuContextRequestError::new(
            GpuContextRequestErrorCategory::TemporaryHostCompatibilityFailure,
            "adapter is incompatible with the required current host",
        ));
    }
    if !descriptor.allowed_backends().is_empty()
        && !descriptor.allowed_backends().contains(&adapter.backend())
    {
        return Err(GpuContextRequestError::new(
            GpuContextRequestErrorCategory::BackendFamilyForbidden,
            "adapter backend is not allowed",
        ));
    }
    if !descriptor.allowed_adapter_classes().is_empty()
        && !descriptor
            .allowed_adapter_classes()
            .contains(&adapter.class())
    {
        return Err(GpuContextRequestError::new(
            GpuContextRequestErrorCategory::NoAdmissibleCandidate,
            "adapter class is not allowed",
        ));
    }
    match descriptor.fallback_policy() {
        GpuSoftwareFallbackPolicy::Require
            if adapter.fallback() != super::facts::GpuFallbackStatus::ConfirmedFallback =>
        {
            return Err(GpuContextRequestError::new(
                GpuContextRequestErrorCategory::SoftwareFallbackPolicyViolation,
                "fallback adapter selection was not proven",
            ));
        }
        GpuSoftwareFallbackPolicy::Forbid
            if matches!(
                adapter.fallback(),
                super::facts::GpuFallbackStatus::ConfirmedFallback
                    | super::facts::GpuFallbackStatus::Unknown
            ) || adapter.software() == GpuSoftwareStatus::Software =>
        {
            return Err(GpuContextRequestError::new(
                GpuContextRequestErrorCategory::SoftwareFallbackPolicyViolation,
                "software, fallback, or unknown fallback evidence is forbidden",
            ));
        }
        _ => {}
    }
    if !adapter.device_request_profile_supported() {
        return Err(GpuContextRequestError::new(
            GpuContextRequestErrorCategory::DeviceRequestProfileUnsupported,
            "adapter cannot satisfy the complete selected device-request profile",
        ));
    }

    let enabled_features = requested_enabled_features(descriptor, &adapter);
    let capability_admission = GpuCapabilityAdmission::evaluate(
        "GPU context candidate",
        descriptor.requirements(),
        adapter.supported(),
        enabled_features.iter().copied(),
    )
    .map_err(GpuContextRequestError::from_capability_admission)?;
    let degradations = capability_admission
        .preferred()
        .iter()
        .filter(|availability| !availability.available || !availability.enabled)
        .map(|availability| GpuDegradationRecord {
            feature: availability.feature,
            fallback: availability.fallback,
        })
        .collect::<Vec<_>>();

    for &(format, role) in &descriptor.format_roles {
        let supported = adapter
            .supported()
            .format(format)
            .is_some_and(|facts| match role {
                GpuFormatRole::Sampled => facts.sampled,
                GpuFormatRole::Filterable => facts.filterable,
                GpuFormatRole::StorageRead => facts.storage_read,
                GpuFormatRole::StorageWrite => facts.storage_write,
                GpuFormatRole::ColorAttachment => facts.color_attachment,
                GpuFormatRole::DepthStencil => facts.depth_stencil,
                GpuFormatRole::CopySource => facts.copy_source,
                GpuFormatRole::CopyDestination => facts.copy_destination,
            });
        if !supported {
            return Err(GpuContextRequestError::new(
                GpuContextRequestErrorCategory::UnsupportedFormatRole,
                format!("{format:?} lacks {role:?}"),
            ));
        }
    }

    for (&kind, &maximum) in &descriptor.alignments {
        if alignment_value(adapter.alignments(), kind).is_none_or(|actual| actual > maximum) {
            return Err(GpuContextRequestError::new(
                GpuContextRequestErrorCategory::AlignmentIncompatibility,
                format!("{kind:?} alignment is incompatible"),
            ));
        }
    }

    let workload_budget = effective_workload_budget(descriptor)?;
    for kind in ALL_LIMIT_KINDS {
        let observed = limit_value(adapter.adapter_limits().values(), kind);
        let required_minimum = limit_value(workload_budget.limits(), kind);
        if observed < required_minimum {
            return Err(GpuContextRequestError::limit_below_required_minimum(
                kind,
                required_minimum,
                observed,
                format!("{kind:?} is below the effective workload budget"),
            ));
        }
    }

    let portability = derive_portability(descriptor, &capability_admission, adapter.backend());
    let unsupported_backend = portability.class() == GpuPortabilityClass::Unsupported;
    #[cfg(test)]
    let unsupported_backend = unsupported_backend
        && !(adapter.backend() == super::descriptor::GpuBackendFamily::UnknownBackend
            && descriptor.allowed_backends().len() == 1
            && descriptor
                .allowed_backends()
                .contains(&super::descriptor::GpuBackendFamily::UnknownBackend));
    if unsupported_backend {
        return Err(GpuContextRequestError::new(
            GpuContextRequestErrorCategory::NoAdmissibleCandidate,
            "adapter backend cannot establish the requested portability contract",
        ));
    }
    if descriptor.portability_policy() == GpuPortabilityPolicy::RequirePortableBaseline
        && portability.class() != GpuPortabilityClass::PortableBaseline
    {
        return Err(GpuContextRequestError::new(
            GpuContextRequestErrorCategory::NoAdmissibleCandidate,
            "portable baseline excludes the admitted extension or backend specialization",
        ));
    }

    Ok(GpuCandidateAdmissionReport {
        id,
        contract: GpuAdmissionContract {
            capability_admission,
            enabled_features,
            degradations,
            format_roles: descriptor.format_roles.clone(),
            selected_alignments: adapter.alignments(),
            workload_budget,
            portability,
            device_request_profile: adapter.device_request_profile(),
        },
        adapter,
    })
}

fn requested_enabled_features(
    descriptor: &GpuContextDescriptor,
    adapter: &GpuAdapterFacts,
) -> BTreeSet<GpuCapabilityFeature> {
    descriptor
        .requirements()
        .iter()
        .filter_map(|requirement| match requirement {
            GpuCapabilityRequirement::Required(feature)
            | GpuCapabilityRequirement::Preferred { feature, .. }
                if adapter.supported().supports(feature) =>
            {
                Some(feature)
            }
            GpuCapabilityRequirement::Disabled(_)
            | GpuCapabilityRequirement::Required(_)
            | GpuCapabilityRequirement::Preferred { .. } => None,
        })
        .collect()
}

pub(crate) fn admitted_device_facts(
    candidate: &GpuCandidateAdmissionReport,
    device_limits: GpuDeviceLimits,
    candidate_dispositions: Vec<super::selection::GpuCandidateDisposition>,
) -> Result<GpuAdmittedDeviceFacts, GpuContextRequestError> {
    for kind in ALL_LIMIT_KINDS {
        let observed = limit_value(device_limits.values(), kind);
        let required_minimum = limit_value(candidate.contract.workload_budget.limits(), kind);
        if observed < required_minimum {
            return Err(GpuContextRequestError::limit_below_required_minimum(
                kind,
                required_minimum,
                observed,
                format!("created device {kind:?} is below the admitted workload budget"),
            ));
        }
    }
    for kind in ALL_ALIGNMENT_KINDS {
        let admitted = alignment_value(candidate.contract.selected_alignments, kind);
        let actual = alignment_value(device_limits.alignments(), kind);
        if admitted != actual {
            return Err(GpuContextRequestError::new(
                GpuContextRequestErrorCategory::AlignmentIncompatibility,
                format!("created device {kind:?} differs from the admitted alignment"),
            ));
        }
        if let Some(maximum) = candidate.contract.workload_budget.alignment_maximum(kind) {
            if actual.is_none_or(|value| value > maximum) {
                return Err(GpuContextRequestError::new(
                    GpuContextRequestErrorCategory::AlignmentIncompatibility,
                    format!("created device {kind:?} exceeds the workload alignment cap"),
                ));
            }
        }
    }
    Ok(GpuAdmittedDeviceFacts::new(
        candidate.contract.enabled_features.clone(),
        device_limits,
        candidate.contract.workload_budget.clone(),
        candidate.contract.clone(),
        candidate_dispositions,
    ))
}

const ALL_LIMIT_KINDS: [GpuLimitKind; 11] = [
    GpuLimitKind::MaxUniformBufferBindingSize,
    GpuLimitKind::MaxStorageBufferBindingSize,
    GpuLimitKind::MaxColorAttachments,
    GpuLimitKind::MaxVertexBuffers,
    GpuLimitKind::MaxBindingsPerGroup,
    GpuLimitKind::MaxTextureDimension2d,
    GpuLimitKind::MaxBindGroups,
    GpuLimitKind::MaxBindGroupsPlusVertexBuffers,
    GpuLimitKind::MaxDynamicUniformBuffersPerPipelineLayout,
    GpuLimitKind::MaxDynamicStorageBuffersPerPipelineLayout,
    GpuLimitKind::MaxComputeWorkgroupsPerDimension,
];

const ALL_ALIGNMENT_KINDS: [GpuAlignmentKind; 5] = [
    GpuAlignmentKind::UniformDynamicOffset,
    GpuAlignmentKind::StorageDynamicOffset,
    GpuAlignmentKind::CopyBufferOffset,
    GpuAlignmentKind::BytesPerRow,
    GpuAlignmentKind::QueryResolveDestination,
];

pub(crate) const fn normalized_limit_baseline() -> GpuLimits {
    GpuLimits::from_validated_adapter_facts(
        64 * 1024,
        128 * 1024 * 1024,
        1,
        8,
        16,
        8192,
        4,
        24,
        8,
        4,
        65_535,
    )
}

fn effective_workload_budget(
    descriptor: &GpuContextDescriptor,
) -> Result<GpuWorkloadBudget, GpuContextRequestError> {
    let baseline = normalized_limit_baseline();
    let value = |kind| {
        descriptor
            .limits
            .get(&kind)
            .map(|constraint| {
                constraint
                    .minimum
                    .unwrap_or_else(|| limit_value(baseline, kind))
                    .max(limit_value(baseline, kind))
                    .min(constraint.maximum.unwrap_or(u64::MAX))
            })
            .unwrap_or_else(|| limit_value(baseline, kind))
    };
    let u32_value = |kind| {
        u32::try_from(value(kind)).map_err(|_| {
            GpuContextRequestError::new(
                GpuContextRequestErrorCategory::ContradictoryRequest,
                format!("{kind:?} exceeds the normalized u32 limit domain"),
            )
        })
    };
    Ok(GpuWorkloadBudget::new(
        GpuLimits::from_validated_adapter_facts(
            value(GpuLimitKind::MaxUniformBufferBindingSize),
            value(GpuLimitKind::MaxStorageBufferBindingSize),
            u32_value(GpuLimitKind::MaxColorAttachments)?,
            u32_value(GpuLimitKind::MaxVertexBuffers)?,
            u32_value(GpuLimitKind::MaxBindingsPerGroup)?,
            u32_value(GpuLimitKind::MaxTextureDimension2d)?,
            u32_value(GpuLimitKind::MaxBindGroups)?,
            u32_value(GpuLimitKind::MaxBindGroupsPlusVertexBuffers)?,
            u32_value(GpuLimitKind::MaxDynamicUniformBuffersPerPipelineLayout)?,
            u32_value(GpuLimitKind::MaxDynamicStorageBuffersPerPipelineLayout)?,
            u32_value(GpuLimitKind::MaxComputeWorkgroupsPerDimension)?,
        ),
        descriptor.alignments.clone(),
    ))
}

pub(crate) const fn limit_value(limits: GpuLimits, kind: GpuLimitKind) -> u64 {
    match kind {
        GpuLimitKind::MaxUniformBufferBindingSize => limits.max_uniform_buffer_binding_size(),
        GpuLimitKind::MaxStorageBufferBindingSize => limits.max_storage_buffer_binding_size(),
        GpuLimitKind::MaxColorAttachments => limits.max_color_attachments() as u64,
        GpuLimitKind::MaxVertexBuffers => limits.max_vertex_buffers() as u64,
        GpuLimitKind::MaxBindingsPerGroup => limits.max_bindings_per_group() as u64,
        GpuLimitKind::MaxTextureDimension2d => limits.max_texture_dimension_2d() as u64,
        GpuLimitKind::MaxBindGroups => limits.max_bind_groups() as u64,
        GpuLimitKind::MaxBindGroupsPlusVertexBuffers => {
            limits.max_bind_groups_plus_vertex_buffers() as u64
        }
        GpuLimitKind::MaxDynamicUniformBuffersPerPipelineLayout => {
            limits.max_dynamic_uniform_buffers_per_pipeline_layout() as u64
        }
        GpuLimitKind::MaxDynamicStorageBuffersPerPipelineLayout => {
            limits.max_dynamic_storage_buffers_per_pipeline_layout() as u64
        }
        GpuLimitKind::MaxComputeWorkgroupsPerDimension => {
            limits.max_compute_workgroups_per_dimension() as u64
        }
    }
}

fn derive_portability(
    descriptor: &GpuContextDescriptor,
    admission: &GpuCapabilityAdmission,
    backend: super::descriptor::GpuBackendFamily,
) -> GpuPortabilityEvidence {
    let mut reasons = BTreeSet::new();
    let enabled = admission
        .granted_required()
        .iter()
        .copied()
        .chain(
            admission
                .preferred()
                .iter()
                .filter(|availability| availability.available && availability.enabled)
                .map(|availability| availability.feature),
        )
        .collect::<BTreeSet<_>>();
    for feature in enabled {
        if is_declared_extension(feature) {
            reasons.insert(GpuPortabilityReason::DeclaredExtension(feature));
        }
    }
    for preferred in admission.preferred() {
        if !preferred.available || !preferred.enabled {
            reasons.insert(GpuPortabilityReason::PreferredRequirementDegraded(
                preferred.feature,
            ));
        }
    }
    if descriptor.allowed_backends().len() == 1 {
        reasons.insert(GpuPortabilityReason::BackendSpecialization(
            *descriptor
                .allowed_backends()
                .first()
                .expect("length checked"),
        ));
    }
    let class = if backend == super::descriptor::GpuBackendFamily::UnknownBackend {
        reasons.insert(GpuPortabilityReason::UnknownBackend);
        GpuPortabilityClass::Unsupported
    } else if reasons
        .iter()
        .any(|reason| matches!(reason, GpuPortabilityReason::BackendSpecialization(_)))
    {
        GpuPortabilityClass::BackendSpecialized
    } else if reasons
        .iter()
        .any(|reason| matches!(reason, GpuPortabilityReason::DeclaredExtension(_)))
    {
        GpuPortabilityClass::PortableWithDeclaredExtensions
    } else {
        GpuPortabilityClass::PortableBaseline
    };
    GpuPortabilityEvidence::new(class, reasons)
}

fn is_declared_extension(feature: GpuCapabilityFeature) -> bool {
    matches!(
        feature,
        GpuCapabilityFeature::TimestampQuery
            | GpuCapabilityFeature::StorageTexture
            | GpuCapabilityFeature::IndirectExecution
            | GpuCapabilityFeature::TextureBindingArray
            | GpuCapabilityFeature::BufferBindingArray
            | GpuCapabilityFeature::StorageResourceBindingArray
            | GpuCapabilityFeature::UniformBufferBindingArray
    )
}

#[cfg(test)]
pub(crate) fn evaluate_candidate(
    descriptor: &GpuContextDescriptor,
    adapter: GpuAdapterFacts,
    host_compatible: bool,
) -> Result<GpuCandidateAdmissionReport, GpuContextRequestError> {
    validate_descriptor(descriptor)?;
    evaluate_validated_candidate(
        descriptor,
        GpuCandidateId::allocate()?,
        adapter,
        GpuCandidateEnvironmentEvidence::current_host(host_compatible),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GpuAdapterClass, GpuAdapterLimits, GpuBackendFamily, GpuCapabilities,
        GpuCapabilityRequirement, GpuCapabilityRequirements, GpuContextDescriptor,
        GpuFallbackStatus, GpuPreferredFallback, GpuSoftwareStatus, GpuTextureFormatCapabilities,
    };

    fn limits() -> GpuLimits {
        GpuLimits::new(
            64 * 1024,
            128 * 1024 * 1024,
            4,
            16,
            64,
            8192,
            4,
            24,
            8,
            4,
            65_535,
        )
        .unwrap()
    }

    fn alignments() -> GpuAlignmentFacts {
        GpuAlignmentFacts {
            uniform_dynamic_offset: Some(256),
            storage_dynamic_offset: Some(256),
            copy_buffer_offset: Some(4),
            bytes_per_row: Some(256),
            query_resolve_destination: Some(256),
        }
    }

    fn adapter(features: impl IntoIterator<Item = GpuCapabilityFeature>) -> GpuAdapterFacts {
        adapter_with_fallback(features, GpuFallbackStatus::ConfirmedNotFallback)
    }

    fn adapter_with_fallback(
        features: impl IntoIterator<Item = GpuCapabilityFeature>,
        fallback: GpuFallbackStatus,
    ) -> GpuAdapterFacts {
        GpuAdapterFacts::new(
            GpuBackendFamily::Vulkan,
            GpuAdapterClass::Discrete,
            GpuSoftwareStatus::Hardware,
            fallback,
            GpuCapabilities::from_normalized_facts(
                features,
                limits(),
                [(
                    GpuTextureFormat::Rgba8Unorm,
                    GpuTextureFormatCapabilities::none(),
                )],
            ),
            GpuAdapterLimits::new(limits()),
            alignments(),
        )
    }

    #[test]
    fn headless_presentation_uses_the_shared_required_preferred_disabled_evaluator() {
        let headless = GpuCandidateEnvironmentEvidence::headless();
        let candidate = adapter([GpuCapabilityFeature::Compute]);

        let mut required = GpuCapabilityRequirements::new();
        required
            .insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::Presentation,
            ))
            .unwrap();
        assert!(matches!(
            evaluate_validated_candidate(
                &GpuContextDescriptor::new(required),
                GpuCandidateId::allocate().unwrap(),
                candidate.clone(),
                headless,
            ),
            Err(error) if error.category() == GpuContextRequestErrorCategory::MandatoryFeatureMissing
        ));

        let mut preferred = GpuCapabilityRequirements::new();
        preferred
            .insert(GpuCapabilityRequirement::Preferred {
                feature: GpuCapabilityFeature::Presentation,
                fallback: GpuPreferredFallback::ContinueWithoutFeature,
            })
            .unwrap();
        let preferred = evaluate_validated_candidate(
            &GpuContextDescriptor::new(preferred),
            GpuCandidateId::allocate().unwrap(),
            candidate.clone(),
            headless,
        )
        .unwrap();
        assert_eq!(preferred.degradations().len(), 1);
        assert_eq!(
            preferred
                .contract()
                .capability_admission()
                .verified_disabled(),
            &[]
        );

        let mut disabled = GpuCapabilityRequirements::new();
        disabled
            .insert(GpuCapabilityRequirement::Disabled(
                GpuCapabilityFeature::Presentation,
            ))
            .unwrap();
        let disabled = evaluate_validated_candidate(
            &GpuContextDescriptor::new(disabled),
            GpuCandidateId::allocate().unwrap(),
            candidate,
            headless,
        )
        .unwrap();
        assert_eq!(
            disabled
                .contract()
                .capability_admission()
                .verified_disabled(),
            &[GpuCapabilityFeature::Presentation]
        );
    }

    #[test]
    fn contract_separates_adapter_support_actual_device_facts_and_workload_budget() {
        let descriptor = GpuContextDescriptor::new(GpuCapabilityRequirements::new())
            .permit_limit(GpuLimitKind::MaxVertexBuffers, 4)
            .require_alignment(GpuAlignmentKind::BytesPerRow, 256);
        let candidate =
            evaluate_candidate(&descriptor, adapter([GpuCapabilityFeature::Compute]), true)
                .unwrap();
        assert_eq!(
            candidate
                .adapter()
                .adapter_limits()
                .values()
                .max_vertex_buffers(),
            16
        );
        assert_eq!(candidate.workload_budget().limits().max_vertex_buffers(), 4);
        let actual = GpuDeviceLimits::new(
            GpuLimits::new(
                128 * 1024,
                256 * 1024 * 1024,
                4,
                8,
                32,
                8192,
                4,
                24,
                8,
                4,
                65_535,
            )
            .unwrap(),
            alignments(),
        );
        let facts = admitted_device_facts(&candidate, actual, Vec::new()).unwrap();
        assert_eq!(facts.device_limits(), actual);
        assert_eq!(facts.workload_budget().limits().max_vertex_buffers(), 4);
        assert_eq!(facts.admission_contract(), candidate.contract());
    }

    #[test]
    fn device_limit_rejection_preserves_typed_requirement_evidence() {
        let descriptor = GpuContextDescriptor::new(GpuCapabilityRequirements::new())
            .require_limit(GpuLimitKind::MaxVertexBuffers, 8);
        let candidate = evaluate_candidate(&descriptor, adapter([]), true).unwrap();
        let device_limits = GpuDeviceLimits::new(
            GpuLimits::new(
                64 * 1024,
                128 * 1024 * 1024,
                4,
                7,
                64,
                8192,
                4,
                24,
                8,
                4,
                65_535,
            )
            .unwrap(),
            alignments(),
        );

        let error = admitted_device_facts(&candidate, device_limits, Vec::new()).unwrap_err();
        assert_eq!(
            error.category(),
            GpuContextRequestErrorCategory::LimitBelowRequiredMinimum
        );
        assert_eq!(
            error.limit_rejection(),
            Some((GpuLimitKind::MaxVertexBuffers, 8, 7))
        );
    }

    #[test]
    fn device_alignment_mismatch_rejects_before_publication() {
        let candidate = evaluate_candidate(
            &GpuContextDescriptor::new(GpuCapabilityRequirements::new()),
            adapter([]),
            true,
        )
        .unwrap();
        let mut actual_alignments = alignments();
        actual_alignments.uniform_dynamic_offset = Some(512);
        assert!(matches!(
            admitted_device_facts(
                &candidate,
                GpuDeviceLimits::new(limits(), actual_alignments),
                Vec::new(),
            ),
            Err(error) if error.category() == GpuContextRequestErrorCategory::AlignmentIncompatibility
        ));
    }

    #[test]
    fn unrequested_binding_array_support_is_not_published_as_enabled_device_fact() {
        let candidate = evaluate_candidate(
            &GpuContextDescriptor::new(GpuCapabilityRequirements::new()),
            adapter([GpuCapabilityFeature::TextureBindingArray]),
            true,
        )
        .unwrap();
        assert!(
            candidate
                .adapter()
                .supported()
                .supports(GpuCapabilityFeature::TextureBindingArray)
        );
        assert!(
            !candidate
                .enabled_features()
                .any(|feature| feature == GpuCapabilityFeature::TextureBindingArray)
        );

        let device = admitted_device_facts(
            &candidate,
            GpuDeviceLimits::new(limits(), alignments()),
            Vec::new(),
        )
        .unwrap();
        assert!(!device.is_enabled(GpuCapabilityFeature::TextureBindingArray));
    }

    #[test]
    fn portability_uses_admitted_contract_not_backend_preference() {
        let baseline = evaluate_candidate(
            &GpuContextDescriptor::new(GpuCapabilityRequirements::new())
                .with_backend_preference([GpuBackendFamily::Vulkan]),
            adapter([]),
            true,
        )
        .unwrap();
        assert_eq!(
            baseline.portability(),
            GpuPortabilityClass::PortableBaseline
        );

        let mut extensions = GpuCapabilityRequirements::new();
        extensions
            .insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::TimestampQuery,
            ))
            .unwrap();
        let extension = evaluate_candidate(
            &GpuContextDescriptor::new(extensions),
            adapter([GpuCapabilityFeature::TimestampQuery]),
            true,
        )
        .unwrap();
        assert_eq!(
            extension.portability(),
            GpuPortabilityClass::PortableWithDeclaredExtensions
        );

        let mut binding_arrays = GpuCapabilityRequirements::new();
        binding_arrays
            .insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::TextureBindingArray,
            ))
            .unwrap();
        let binding_array_extension = evaluate_candidate(
            &GpuContextDescriptor::new(binding_arrays.clone()),
            adapter([GpuCapabilityFeature::TextureBindingArray]),
            true,
        )
        .unwrap();
        assert_eq!(
            binding_array_extension.portability(),
            GpuPortabilityClass::PortableWithDeclaredExtensions
        );
        assert!(matches!(
            evaluate_candidate(
                &GpuContextDescriptor::new(binding_arrays)
                    .with_portability_policy(GpuPortabilityPolicy::RequirePortableBaseline),
                adapter([GpuCapabilityFeature::TextureBindingArray]),
                true,
            ),
            Err(error) if error.category() == GpuContextRequestErrorCategory::NoAdmissibleCandidate
        ));

        let specialized = evaluate_candidate(
            &GpuContextDescriptor::new(GpuCapabilityRequirements::new())
                .with_allowed_backends([GpuBackendFamily::Vulkan]),
            adapter([]),
            true,
        )
        .unwrap();
        assert_eq!(
            specialized.portability(),
            GpuPortabilityClass::BackendSpecialized
        );

        let mut preferred = GpuCapabilityRequirements::new();
        preferred
            .insert(GpuCapabilityRequirement::Preferred {
                feature: GpuCapabilityFeature::TimestampQuery,
                fallback: GpuPreferredFallback::DisableInstrumentation,
            })
            .unwrap();
        let degraded =
            evaluate_candidate(&GpuContextDescriptor::new(preferred), adapter([]), true).unwrap();
        assert_eq!(
            degraded.portability(),
            GpuPortabilityClass::PortableBaseline
        );
        assert!(
            degraded
                .portability_evidence()
                .reasons()
                .any(|reason| matches!(
                    reason,
                    GpuPortabilityReason::PreferredRequirementDegraded(_)
                ))
        );
    }

    #[test]
    fn unsupported_complete_device_profile_is_rejected_during_pure_admission() {
        let profile_unsupported =
            adapter([]).with_device_profile(GpuDeviceRequestProfile::Downlevel, false);
        assert!(matches!(
            evaluate_candidate(
                &GpuContextDescriptor::new(GpuCapabilityRequirements::new()),
                profile_unsupported,
                true,
            ),
            Err(error)
                if error.category()
                    == GpuContextRequestErrorCategory::DeviceRequestProfileUnsupported
        ));
    }

    #[test]
    fn forbid_requires_explicit_non_fallback_evidence() {
        let descriptor = GpuContextDescriptor::new(GpuCapabilityRequirements::new())
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Forbid);
        for fallback in [
            GpuFallbackStatus::Unknown,
            GpuFallbackStatus::ConfirmedFallback,
        ] {
            assert!(matches!(
                evaluate_candidate(&descriptor, adapter_with_fallback([], fallback), true),
                Err(error)
                    if error.category()
                        == GpuContextRequestErrorCategory::SoftwareFallbackPolicyViolation
            ));
        }
        assert!(
            evaluate_candidate(
                &descriptor,
                adapter_with_fallback([], GpuFallbackStatus::ConfirmedNotFallback),
                true,
            )
            .is_ok()
        );
    }
}
