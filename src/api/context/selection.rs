use super::admission::{
    GpuCandidateAdmissionReport, GpuCandidateEnvironmentEvidence, GpuRejectedCandidateReport,
    evaluate_validated_candidate,
};
use super::descriptor::{GpuContextDescriptor, GpuDescriptorRetryIdentity, GpuPowerPreference};
use super::diagnostics::{GpuContextRequestError, GpuContextRequestErrorCategory};
use super::facts::{GpuAdapterFacts, GpuFallbackStatus, GpuPortabilityClass, GpuSoftwareStatus};
use crate::{GpuCapabilityFeature, GpuTextureFormat, GpuTextureFormatCapabilities};
use std::{
    collections::BTreeMap,
    num::NonZeroU64,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

/// Opaque, nonzero process-local candidate correlation value.
///
/// A value is accepted by `GpuContextDescriptor::with_exact_candidate` only when it was
/// disclosed by a `GpuContextRequestErrorCategory::AmbiguousAdapterSelection` report and its
/// bounded retry binding is still retained by this process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuCandidateId(NonZeroU64);

impl GpuCandidateId {
    pub const fn is_nonzero(self) -> bool {
        self.0.get() != 0
    }

    pub(crate) fn allocate() -> Result<Self, GpuContextRequestError> {
        let value = NEXT_CANDIDATE_RETRY_TOKEN
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| {
                GpuContextRequestError::new(
                    GpuContextRequestErrorCategory::CandidateRetryTokenExhausted,
                    "the process-local candidate retry-token allocator is exhausted",
                )
            })?;
        Ok(Self(
            NonZeroU64::new(value).expect("candidate retry-token allocation starts nonzero"),
        ))
    }
}

static NEXT_CANDIDATE_RETRY_TOKEN: AtomicU64 = AtomicU64::new(1);
static CANDIDATE_RETRY_BINDINGS: OnceLock<
    Mutex<BTreeMap<GpuCandidateId, GpuCandidateRetryBinding>>,
> = OnceLock::new();
const MAX_RETAINED_CANDIDATE_RETRY_TOKENS: usize = 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
struct GpuCandidateRetryBinding {
    descriptor: GpuDescriptorRetryIdentity,
    candidate_set: Vec<GpuCanonicalCandidateInputKey>,
    target: GpuCanonicalCandidateInputKey,
}

fn candidate_retry_bindings() -> &'static Mutex<BTreeMap<GpuCandidateId, GpuCandidateRetryBinding>>
{
    CANDIDATE_RETRY_BINDINGS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn retain_candidate_retry_bindings(
    bindings_to_retain: impl IntoIterator<Item = (GpuCandidateId, GpuCandidateRetryBinding)>,
) -> Result<(), GpuContextRequestError> {
    let bindings_to_retain = bindings_to_retain.into_iter().collect::<Vec<_>>();
    if bindings_to_retain.len() > MAX_RETAINED_CANDIDATE_RETRY_TOKENS {
        return Err(GpuContextRequestError::new(
            GpuContextRequestErrorCategory::CandidateRetryTokenExhausted,
            "one ambiguous candidate set exceeds the bounded retry-token registry",
        ));
    }

    let mut retained = candidate_retry_bindings()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for (id, binding) in bindings_to_retain {
        retained.insert(id, binding);
    }
    while retained.len() > MAX_RETAINED_CANDIDATE_RETRY_TOKENS {
        let Some(oldest) = retained.keys().next().copied() else {
            break;
        };
        retained.remove(&oldest);
    }
    Ok(())
}

#[cfg(test)]
fn retain_candidate_retry_binding(id: GpuCandidateId, binding: GpuCandidateRetryBinding) {
    retain_candidate_retry_bindings([(id, binding)])
        .expect("one test retry binding must fit the bounded registry");
}

fn candidate_retry_binding(id: GpuCandidateId) -> Option<GpuCandidateRetryBinding> {
    candidate_retry_bindings()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&id)
        .cloned()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuCandidateDisposition {
    Accepted(Box<GpuCandidateAdmissionReport>),
    Rejected(Box<GpuRejectedCandidateReport>),
}

impl GpuCandidateDisposition {
    pub const fn id(&self) -> GpuCandidateId {
        match self {
            Self::Accepted(report) => report.id(),
            Self::Rejected(report) => report.id(),
        }
    }

    pub fn adapter(&self) -> &GpuAdapterFacts {
        match self {
            Self::Accepted(report) => report.adapter(),
            Self::Rejected(report) => report.adapter(),
        }
    }

    pub(crate) const fn is_rejected(&self) -> bool {
        matches!(self, Self::Rejected(_))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct GpuCandidateRankEvidence {
    fallback: u8,
    power: u8,
    portability: u8,
    adapter_class: u8,
    backend_preference: u8,
    vendor: Option<u32>,
    device: Option<u32>,
}

impl GpuCandidateRankEvidence {
    pub const fn fallback_priority(&self) -> u8 {
        self.fallback
    }

    pub const fn power_priority(&self) -> u8 {
        self.power
    }

    pub const fn portability_priority(&self) -> u8 {
        self.portability
    }

    pub const fn adapter_class_priority(&self) -> u8 {
        self.adapter_class
    }

    pub const fn backend_preference_priority(&self) -> u8 {
        self.backend_preference
    }

    pub const fn vendor(&self) -> Option<u32> {
        self.vendor
    }

    pub const fn device(&self) -> Option<u32> {
        self.device
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuCandidateSelectionEvidence {
    rank: GpuCandidateRankEvidence,
    reason: &'static str,
}

impl GpuCandidateSelectionEvidence {
    pub fn rank(&self) -> &GpuCandidateRankEvidence {
        &self.rank
    }

    pub const fn reason(&self) -> &'static str {
        self.reason
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuCandidateSelectionKind {
    BackendSelectedCandidate,
    DeterministicallyRanked,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuContextAdmissionReport {
    pub(crate) selected: GpuCandidateSelectionKind,
    pub(crate) candidate: GpuCandidateAdmissionReport,
    pub(crate) candidate_dispositions: Vec<GpuCandidateDisposition>,
    pub(crate) selection_evidence: GpuCandidateSelectionEvidence,
}

impl GpuContextAdmissionReport {
    pub fn candidate(&self) -> &GpuCandidateAdmissionReport {
        &self.candidate
    }

    pub const fn selection_kind(&self) -> GpuCandidateSelectionKind {
        self.selected
    }

    pub fn candidate_dispositions(&self) -> &[GpuCandidateDisposition] {
        &self.candidate_dispositions
    }

    pub fn candidate_dispositions_are_canonically_ordered(&self) -> bool {
        self.candidate_dispositions
            .windows(2)
            .all(|pair| canonical_disposition_key(&pair[0]) <= canonical_disposition_key(&pair[1]))
    }

    pub fn selection_evidence(&self) -> &GpuCandidateSelectionEvidence {
        &self.selection_evidence
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GpuCandidateInput {
    pub(crate) id: GpuCandidateId,
    pub(crate) adapter: GpuAdapterFacts,
    pub(crate) environment: GpuCandidateEnvironmentEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GpuCandidateSelection {
    pub(crate) candidate: GpuCandidateAdmissionReport,
    /// The newly enumerated backend input selected for this request. This stays private so an
    /// exact retry can retain its disclosed public token without reconstructing a raw adapter.
    pub(crate) backend_candidate_id: GpuCandidateId,
    pub(crate) dispositions: Vec<GpuCandidateDisposition>,
    pub(crate) evidence: GpuCandidateSelectionEvidence,
}

#[cfg(any(test, target_arch = "wasm32"))]
pub(crate) fn select_candidate(
    descriptor: &GpuContextDescriptor,
    candidates: impl IntoIterator<Item = GpuAdapterFacts>,
    host_compatible: bool,
) -> Result<GpuCandidateSelection, GpuContextRequestError> {
    select_candidate_with_host_evidence(
        descriptor,
        candidates.into_iter().map(|candidate| {
            (
                candidate,
                GpuCandidateEnvironmentEvidence::current_host(host_compatible),
            )
        }),
    )
}

#[cfg(any(test, target_arch = "wasm32"))]
pub(crate) fn select_candidate_with_host_evidence(
    descriptor: &GpuContextDescriptor,
    candidates: impl IntoIterator<Item = (GpuAdapterFacts, GpuCandidateEnvironmentEvidence)>,
) -> Result<GpuCandidateSelection, GpuContextRequestError> {
    let mut candidates = candidates
        .into_iter()
        .map(|(adapter, environment)| {
            Ok(GpuCandidateInput {
                id: GpuCandidateId::allocate()?,
                adapter,
                environment,
            })
        })
        .collect::<Result<Vec<_>, GpuContextRequestError>>()?;
    canonicalize_candidate_inputs(&mut candidates);
    select_candidate_inputs(descriptor, candidates)
}

pub(crate) fn select_candidate_inputs(
    descriptor: &GpuContextDescriptor,
    candidates: impl IntoIterator<Item = GpuCandidateInput>,
) -> Result<GpuCandidateSelection, GpuContextRequestError> {
    super::admission::validate_descriptor(descriptor)?;
    let candidates = candidates.into_iter().collect::<Vec<_>>();
    // Snapshot an existing retry binding before this request can publish any fresh tokens.
    // The clone remains valid for this request even if another thread later evicts the registry
    // entry while candidate evaluation is in progress.
    let exact_retry = descriptor
        .exact_candidate()
        .map(|token| (token, candidate_retry_binding(token)));
    let descriptor_retry_identity = descriptor.retry_identity();
    let candidate_set = canonical_candidate_set(&candidates);
    let identities_by_id = candidates
        .iter()
        .map(|candidate| {
            (
                candidate.id,
                canonical_candidate_input_key(&candidate.adapter, candidate.environment),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut dispositions = candidates
        .into_iter()
        .map(|candidate| {
            match evaluate_validated_candidate(
                descriptor,
                candidate.id,
                candidate.adapter.clone(),
                candidate.environment,
            ) {
                Ok(report) => GpuCandidateDisposition::Accepted(Box::new(report)),
                Err(error) => GpuCandidateDisposition::Rejected(Box::new(
                    GpuRejectedCandidateReport::from_context_error(
                        candidate.id,
                        candidate.adapter,
                        error,
                    ),
                )),
            }
        })
        .collect::<Vec<_>>();
    dispositions.sort_by_key(canonical_disposition_key);
    let mut admitted = dispositions
        .iter()
        .filter_map(|disposition| match disposition {
            GpuCandidateDisposition::Accepted(report) => Some(report.as_ref().clone()),
            GpuCandidateDisposition::Rejected(_) => None,
        })
        .collect::<Vec<_>>();
    if let Some((exact, binding)) = exact_retry {
        let Some(binding) = binding else {
            return Err(stale_candidate_retry_error(
                "the disclosed process-local candidate retry token is unknown or expired",
                dispositions,
            ));
        };
        if binding.descriptor != descriptor_retry_identity || binding.candidate_set != candidate_set
        {
            return Err(stale_candidate_retry_error(
                "the normalized request or observed candidate set changed since token disclosure",
                dispositions,
            ));
        }
        let matching = admitted
            .iter()
            .filter(|candidate| {
                identities_by_id
                    .get(&candidate.id())
                    .is_some_and(|identity| identity == &binding.target)
            })
            .collect::<Vec<_>>();
        return match matching.as_slice() {
            [] => Err(stale_candidate_retry_error(
                "the disclosed candidate is no longer admissible for this unchanged retry set",
                dispositions,
            )),
            [candidate] => {
                let backend_candidate_id = candidate.id();
                let mut candidate = (*candidate).clone();
                candidate.id = exact;
                retoken_disposition(&mut dispositions, backend_candidate_id, exact);
                Ok(GpuCandidateSelection {
                    evidence: GpuCandidateSelectionEvidence {
                        rank: candidate_rank(descriptor, matching[0]),
                        reason: "exact process-local candidate retry",
                    },
                    candidate,
                    backend_candidate_id,
                    dispositions,
                })
            }
            _ => Err(GpuContextRequestError::new(
                GpuContextRequestErrorCategory::AmbiguousAdapterSelection,
                "the disclosed retry token still matches indistinguishable candidates",
            )
            .with_candidate_dispositions(dispositions)),
        };
    }
    if dispositions.is_empty() {
        return Err(GpuContextRequestError::new(
            GpuContextRequestErrorCategory::NoAdapterAvailable,
            "the backend reported no adapter candidates",
        ));
    }
    if admitted.is_empty() {
        return Err(GpuContextRequestError::new(
            GpuContextRequestErrorCategory::NoAdmissibleCandidate,
            "no observed adapter satisfied the normalized request",
        )
        .with_candidate_dispositions(dispositions));
    }
    admitted.sort_by_key(|candidate| candidate_rank(descriptor, candidate));
    let best = admitted
        .first()
        .cloned()
        .expect("nonempty admitted candidates checked above");
    if admitted.get(1).is_some_and(|second| {
        candidate_rank(descriptor, second) == candidate_rank(descriptor, &best)
    }) {
        // Only an ambiguity report authorizes exact retry. Register every accepted candidate in
        // one bounded transaction immediately before returning that report, so the report cannot
        // expose a partially retained candidate set.
        if let Err(error) = retain_ambiguous_retry_tokens(
            descriptor_retry_identity,
            candidate_set,
            &identities_by_id,
            &admitted,
        ) {
            return Err(error.with_candidate_dispositions(dispositions));
        }
        return Err(GpuContextRequestError::new(
            GpuContextRequestErrorCategory::AmbiguousAdapterSelection,
            "best candidates remain indistinguishable after normalized ranking",
        )
        .with_candidate_dispositions(dispositions));
    }
    Ok(GpuCandidateSelection {
        evidence: GpuCandidateSelectionEvidence {
            rank: candidate_rank(descriptor, &best),
            reason: "lowest complete normalized rank",
        },
        backend_candidate_id: best.id(),
        candidate: best,
        dispositions,
    })
}

#[cfg(any(test, target_arch = "wasm32"))]
pub(crate) fn canonicalize_candidate_inputs(candidates: &mut [GpuCandidateInput]) {
    candidates.sort_by_key(|candidate| {
        canonical_candidate_input_key(&candidate.adapter, candidate.environment)
    });
}

fn canonical_candidate_set(candidates: &[GpuCandidateInput]) -> Vec<GpuCanonicalCandidateInputKey> {
    let mut candidate_set = candidates
        .iter()
        .map(|candidate| canonical_candidate_input_key(&candidate.adapter, candidate.environment))
        .collect::<Vec<_>>();
    candidate_set.sort();
    candidate_set
}

fn retain_ambiguous_retry_tokens(
    descriptor: GpuDescriptorRetryIdentity,
    candidate_set: Vec<GpuCanonicalCandidateInputKey>,
    identities_by_id: &BTreeMap<GpuCandidateId, GpuCanonicalCandidateInputKey>,
    admitted: &[GpuCandidateAdmissionReport],
) -> Result<(), GpuContextRequestError> {
    let bindings = admitted.iter().map(|candidate| {
        let target = identities_by_id
            .get(&candidate.id())
            .cloned()
            .expect("every admitted candidate retains its canonical input identity");
        (
            candidate.id(),
            GpuCandidateRetryBinding {
                descriptor: descriptor.clone(),
                candidate_set: candidate_set.clone(),
                target,
            },
        )
    });
    retain_candidate_retry_bindings(bindings)
}

fn stale_candidate_retry_error(
    detail: &'static str,
    dispositions: Vec<GpuCandidateDisposition>,
) -> GpuContextRequestError {
    GpuContextRequestError::new(
        GpuContextRequestErrorCategory::StaleCandidateRetryToken,
        detail,
    )
    .with_candidate_dispositions(dispositions)
}

fn retoken_disposition(
    dispositions: &mut [GpuCandidateDisposition],
    backend_candidate_id: GpuCandidateId,
    retry_token: GpuCandidateId,
) {
    for disposition in dispositions {
        if disposition.id() != backend_candidate_id {
            continue;
        }
        match disposition {
            GpuCandidateDisposition::Accepted(report) => report.id = retry_token,
            GpuCandidateDisposition::Rejected(report) => report.id = retry_token,
        }
        return;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct GpuCanonicalCandidateInputKey {
    adapter: GpuCanonicalAdapterKey,
    host_compatible: bool,
    host_compatibility_required: bool,
}

pub(crate) fn canonical_candidate_input_key(
    adapter: &GpuAdapterFacts,
    environment: GpuCandidateEnvironmentEvidence,
) -> GpuCanonicalCandidateInputKey {
    GpuCanonicalCandidateInputKey {
        adapter: canonical_adapter_key(adapter),
        host_compatible: environment.host_compatible,
        host_compatibility_required: environment.host_compatibility_required,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct GpuCanonicalFormatCapabilities {
    sampled: bool,
    filterable: bool,
    storage_read: bool,
    storage_write: bool,
    color_attachment: bool,
    depth_stencil: bool,
    copy_source: bool,
    copy_destination: bool,
    block_dimensions: Option<(u32, u32)>,
    block_copy_size: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct GpuCanonicalAdapterKey {
    backend: super::descriptor::GpuBackendFamily,
    class: super::descriptor::GpuAdapterClass,
    software: GpuSoftwareStatus,
    fallback: GpuFallbackStatus,
    supported_features: Vec<GpuCapabilityFeature>,
    adapter_limits: (u64, u64, u32, u32, u32, u32, u32, u32, u32, u32, u32),
    supported_formats: Vec<(GpuTextureFormat, GpuCanonicalFormatCapabilities)>,
    alignments: super::facts::GpuAlignmentFacts,
    vendor: Option<u32>,
    device: Option<u32>,
    profile: super::facts::GpuDeviceRequestProfile,
    profile_supported: bool,
}

fn canonical_adapter_key(adapter: &GpuAdapterFacts) -> GpuCanonicalAdapterKey {
    let limits = adapter.adapter_limits().values();
    GpuCanonicalAdapterKey {
        backend: adapter.backend(),
        class: adapter.class(),
        software: adapter.software(),
        fallback: adapter.fallback(),
        supported_features: adapter.supported().features().collect(),
        adapter_limits: (
            limits.max_uniform_buffer_binding_size(),
            limits.max_storage_buffer_binding_size(),
            limits.max_color_attachments(),
            limits.max_vertex_buffers(),
            limits.max_bindings_per_group(),
            limits.max_texture_dimension_2d(),
            limits.max_bind_groups(),
            limits.max_bind_groups_plus_vertex_buffers(),
            limits.max_dynamic_uniform_buffers_per_pipeline_layout(),
            limits.max_dynamic_storage_buffers_per_pipeline_layout(),
            limits.max_compute_workgroups_per_dimension(),
        ),
        supported_formats: adapter
            .supported()
            .formats()
            .map(|(format, capabilities)| (format, canonical_format_capabilities(capabilities)))
            .collect(),
        alignments: adapter.alignments(),
        vendor: adapter.vendor(),
        device: adapter.device(),
        profile: adapter.device_request_profile(),
        profile_supported: adapter.device_request_profile_supported(),
    }
}

fn canonical_format_capabilities(
    capabilities: GpuTextureFormatCapabilities,
) -> GpuCanonicalFormatCapabilities {
    GpuCanonicalFormatCapabilities {
        sampled: capabilities.sampled,
        filterable: capabilities.filterable,
        storage_read: capabilities.storage_read,
        storage_write: capabilities.storage_write,
        color_attachment: capabilities.color_attachment,
        depth_stencil: capabilities.depth_stencil,
        copy_source: capabilities.copy_source,
        copy_destination: capabilities.copy_destination,
        block_dimensions: capabilities.block_dimensions,
        block_copy_size: capabilities.block_copy_size,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct GpuCanonicalDispositionKey {
    adapter: GpuCanonicalAdapterKey,
    disposition_category: u8,
}

fn canonical_disposition_key(disposition: &GpuCandidateDisposition) -> GpuCanonicalDispositionKey {
    let (adapter, disposition_category) = match disposition {
        GpuCandidateDisposition::Accepted(report) => (report.adapter(), 0),
        GpuCandidateDisposition::Rejected(report) => {
            (report.adapter(), 1 + report.category().stable_order())
        }
    };
    GpuCanonicalDispositionKey {
        adapter: canonical_adapter_key(adapter),
        disposition_category,
    }
}

fn candidate_rank(
    descriptor: &GpuContextDescriptor,
    candidate: &GpuCandidateAdmissionReport,
) -> GpuCandidateRankEvidence {
    let adapter = candidate.adapter();
    let fallback = match adapter.fallback() {
        GpuFallbackStatus::ConfirmedNotFallback => 0,
        GpuFallbackStatus::Unknown => 1,
        GpuFallbackStatus::ConfirmedFallback => 2,
    };
    let power = match (descriptor.power_preference(), adapter.class()) {
        (GpuPowerPreference::HighPerformance, super::descriptor::GpuAdapterClass::Discrete)
        | (GpuPowerPreference::LowPower, super::descriptor::GpuAdapterClass::Integrated) => 0,
        (_, super::descriptor::GpuAdapterClass::Discrete) => 1,
        (_, super::descriptor::GpuAdapterClass::Integrated) => 2,
        _ => 3,
    };
    let portability = match candidate.portability() {
        GpuPortabilityClass::PortableBaseline => 0,
        GpuPortabilityClass::PortableWithDeclaredExtensions => 1,
        GpuPortabilityClass::BackendSpecialized => 2,
        GpuPortabilityClass::Unsupported => 3,
    };
    let class = match adapter.class() {
        super::descriptor::GpuAdapterClass::Discrete => 0,
        super::descriptor::GpuAdapterClass::Integrated => 1,
        super::descriptor::GpuAdapterClass::Virtual => 2,
        super::descriptor::GpuAdapterClass::Cpu => 3,
        super::descriptor::GpuAdapterClass::Other => 4,
        super::descriptor::GpuAdapterClass::Unknown => 5,
    };
    let backend = descriptor
        .backend_preference()
        .get(&adapter.backend())
        .copied()
        .unwrap_or(u8::MAX);
    GpuCandidateRankEvidence {
        fallback,
        power,
        portability,
        adapter_class: class,
        backend_preference: backend,
        vendor: adapter.vendor(),
        device: adapter.device(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GpuAdapterClass, GpuAdapterFacts, GpuAdapterLimits, GpuAlignmentFacts, GpuBackendFamily,
        GpuCapabilities, GpuCapabilityAdmissionCause, GpuCapabilityFeature,
        GpuCapabilityRequirement, GpuCapabilityRequirements, GpuContextDescriptor,
        GpuFallbackStatus, GpuLimitKind, GpuLimits, GpuSoftwareStatus,
    };

    static RETRY_REGISTRY_TEST_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct IsolatedRetryRegistry {
        _serial: std::sync::MutexGuard<'static, ()>,
    }

    impl Drop for IsolatedRetryRegistry {
        fn drop(&mut self) {
            candidate_retry_bindings()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clear();
        }
    }

    fn isolated_retry_registry() -> IsolatedRetryRegistry {
        let serial = RETRY_REGISTRY_TEST_SERIAL
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        candidate_retry_bindings()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        IsolatedRetryRegistry { _serial: serial }
    }

    fn test_limits() -> GpuLimits {
        GpuLimits::new(
            64 * 1024,
            128 * 1024 * 1024,
            4,
            8,
            16,
            8192,
            4,
            24,
            8,
            4,
            65_535,
        )
        .expect("complete test limits are valid")
    }

    fn adapter_with(features: impl IntoIterator<Item = GpuCapabilityFeature>) -> GpuAdapterFacts {
        let limits = test_limits();
        GpuAdapterFacts::new(
            GpuBackendFamily::Vulkan,
            GpuAdapterClass::Discrete,
            GpuSoftwareStatus::Hardware,
            GpuFallbackStatus::ConfirmedNotFallback,
            GpuCapabilities::from_normalized_facts(features, limits, []),
            GpuAdapterLimits::new(limits),
            GpuAlignmentFacts {
                uniform_dynamic_offset: Some(256),
                storage_dynamic_offset: Some(256),
                copy_buffer_offset: Some(4),
                bytes_per_row: Some(256),
                query_resolve_destination: Some(256),
            },
        )
    }

    fn adapter() -> GpuAdapterFacts {
        adapter_with([])
    }

    fn with_diagnostics(
        adapter: GpuAdapterFacts,
        name: &str,
        driver: &str,
        driver_info: &str,
    ) -> GpuAdapterFacts {
        adapter.with_diagnostics(
            name.to_owned(),
            7,
            9,
            driver.to_owned(),
            driver_info.to_owned(),
        )
    }

    #[test]
    fn reports_canonically_order_all_dispositions_and_split_absence_from_rejection() {
        let descriptor = GpuContextDescriptor::new(GpuCapabilityRequirements::new());
        let empty = select_candidate(&descriptor, [], true).unwrap_err();
        assert_eq!(
            empty.category(),
            GpuContextRequestErrorCategory::NoAdapterAvailable
        );

        let only_metal = descriptor
            .clone()
            .with_allowed_backends([GpuBackendFamily::Metal]);
        let rejected = select_candidate(&only_metal, [adapter()], true).unwrap_err();
        assert_eq!(
            rejected.category(),
            GpuContextRequestErrorCategory::NoAdmissibleCandidate
        );
        assert_eq!(rejected.candidate_dispositions().len(), 1);
        assert!(matches!(
            rejected.candidate_dispositions(),
            [GpuCandidateDisposition::Rejected(report)]
                if report.category() == GpuContextRequestErrorCategory::BackendFamilyForbidden
        ));
    }

    #[test]
    fn rejected_candidate_preserves_typed_capability_admission_evidence() {
        let mut requirements = GpuCapabilityRequirements::new();
        requirements
            .insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::Compute,
            ))
            .unwrap();
        let error = select_candidate(&GpuContextDescriptor::new(requirements), [adapter()], true)
            .unwrap_err();
        assert_eq!(
            error.category(),
            GpuContextRequestErrorCategory::NoAdmissibleCandidate
        );
        let [GpuCandidateDisposition::Rejected(report)] = error.candidate_dispositions() else {
            panic!("one rejected candidate should be retained");
        };
        let capability = report
            .capability_admission_error()
            .expect("capability rejection should remain typed");
        assert_eq!(
            capability.cause(),
            GpuCapabilityAdmissionCause::RequiredUnavailable
        );
        assert_eq!(capability.feature(), Some(GpuCapabilityFeature::Compute));
        assert_eq!(report.limit_rejection(), None);
    }

    #[test]
    fn rejected_candidate_preserves_typed_limit_requirement_evidence() {
        let descriptor = GpuContextDescriptor::new(GpuCapabilityRequirements::new())
            .require_limit(GpuLimitKind::MaxVertexBuffers, 9);
        let error = select_candidate(&descriptor, [adapter()], true).unwrap_err();
        assert_eq!(
            error.category(),
            GpuContextRequestErrorCategory::NoAdmissibleCandidate
        );
        let [GpuCandidateDisposition::Rejected(report)] = error.candidate_dispositions() else {
            panic!("one rejected candidate should be retained");
        };
        assert_eq!(
            report.category(),
            GpuContextRequestErrorCategory::LimitBelowRequiredMinimum
        );
        assert_eq!(
            report.limit_rejection(),
            Some((GpuLimitKind::MaxVertexBuffers, 9, 8))
        );
        assert!(report.capability_admission_error().is_none());
    }

    #[test]
    fn exact_retry_tokens_bind_one_disclosed_candidate_without_diagnostic_identity() {
        let _retry_registry = isolated_retry_registry();
        let first = with_diagnostics(
            adapter_with([GpuCapabilityFeature::Compute]),
            "alpha diagnostic",
            "driver alpha",
            "driver-info alpha",
        );
        let second = with_diagnostics(
            adapter_with([GpuCapabilityFeature::TimestampQuery]),
            "beta diagnostic",
            "driver beta",
            "driver-info beta",
        );
        let descriptor = GpuContextDescriptor::new(GpuCapabilityRequirements::new());
        let ambiguous =
            select_candidate(&descriptor, [second.clone(), first.clone()], true).unwrap_err();
        assert_eq!(
            ambiguous.category(),
            GpuContextRequestErrorCategory::AmbiguousAdapterSelection
        );
        let retry_token = ambiguous
            .candidate_dispositions()
            .iter()
            .find(|disposition| {
                disposition
                    .adapter()
                    .supported()
                    .supports(GpuCapabilityFeature::Compute)
            })
            .map(GpuCandidateDisposition::id)
            .expect("compute candidate should disclose a retry token");
        let alternate_retry_token = ambiguous
            .candidate_dispositions()
            .iter()
            .find(|disposition| {
                disposition
                    .adapter()
                    .supported()
                    .supports(GpuCapabilityFeature::TimestampQuery)
            })
            .map(GpuCandidateDisposition::id)
            .expect("timestamp candidate should disclose a retry token");
        assert!(retry_token.is_nonzero());
        assert!(alternate_retry_token.is_nonzero());

        let retry = select_candidate(
            &descriptor
                .clone()
                .with_label("new request diagnostic")
                .with_exact_candidate(retry_token),
            [
                with_diagnostics(
                    adapter_with([GpuCapabilityFeature::Compute]),
                    "renamed alpha diagnostic",
                    "different driver alpha",
                    "different driver-info alpha",
                ),
                with_diagnostics(
                    adapter_with([GpuCapabilityFeature::TimestampQuery]),
                    "renamed beta diagnostic",
                    "different driver beta",
                    "different driver-info beta",
                ),
            ],
            true,
        )
        .unwrap();
        assert_eq!(retry.candidate.id(), retry_token);
        assert!(
            retry
                .candidate
                .adapter()
                .supported()
                .supports(GpuCapabilityFeature::Compute)
        );
        assert_eq!(
            retry.evidence.reason(),
            "exact process-local candidate retry"
        );

        let alternate_retry = select_candidate(
            &descriptor
                .clone()
                .with_exact_candidate(alternate_retry_token),
            [
                with_diagnostics(
                    adapter_with([GpuCapabilityFeature::Compute]),
                    "another alpha diagnostic",
                    "another driver alpha",
                    "another driver-info alpha",
                ),
                with_diagnostics(
                    adapter_with([GpuCapabilityFeature::TimestampQuery]),
                    "another beta diagnostic",
                    "another driver beta",
                    "another driver-info beta",
                ),
            ],
            true,
        )
        .unwrap();
        assert_eq!(alternate_retry.candidate.id(), alternate_retry_token);
        assert!(
            alternate_retry
                .candidate
                .adapter()
                .supported()
                .supports(GpuCapabilityFeature::TimestampQuery)
        );

        let stale = select_candidate(
            &descriptor.clone().with_exact_candidate(retry_token),
            [second.clone()],
            true,
        )
        .unwrap_err();
        assert_eq!(
            stale.category(),
            GpuContextRequestErrorCategory::StaleCandidateRetryToken
        );

        let unknown = select_candidate(
            &descriptor
                .clone()
                .with_exact_candidate(GpuCandidateId::allocate().unwrap()),
            [second],
            true,
        )
        .unwrap_err();
        assert_eq!(
            unknown.category(),
            GpuContextRequestErrorCategory::StaleCandidateRetryToken
        );
    }

    #[test]
    fn deterministic_success_does_not_create_an_exact_retry_binding() {
        let _retry_registry = isolated_retry_registry();
        let descriptor = GpuContextDescriptor::new(GpuCapabilityRequirements::new());
        let selected = select_candidate(&descriptor, [adapter()], true).unwrap();

        assert!(candidate_retry_binding(selected.candidate.id()).is_none());
    }

    #[test]
    fn exact_retry_cannot_evict_its_own_binding_when_registry_is_full() {
        let _retry_registry = isolated_retry_registry();
        let first = adapter_with([GpuCapabilityFeature::Compute]);
        let second = adapter_with([GpuCapabilityFeature::TimestampQuery]);
        let descriptor = GpuContextDescriptor::new(GpuCapabilityRequirements::new());
        let ambiguous =
            select_candidate(&descriptor, [first.clone(), second.clone()], true).unwrap_err();
        let retry_token = ambiguous
            .candidate_dispositions()
            .iter()
            .map(GpuCandidateDisposition::id)
            .min()
            .expect("ambiguity should disclose at least one retry token");
        let binding = candidate_retry_binding(retry_token)
            .expect("the ambiguity token must be retained before it is returned");

        let retained_count = candidate_retry_bindings()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len();
        for _ in retained_count..MAX_RETAINED_CANDIDATE_RETRY_TOKENS {
            retain_candidate_retry_binding(GpuCandidateId::allocate().unwrap(), binding.clone());
        }
        assert_eq!(
            candidate_retry_bindings()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .len(),
            MAX_RETAINED_CANDIDATE_RETRY_TOKENS
        );
        assert!(candidate_retry_binding(retry_token).is_some());

        let retry = select_candidate(
            &descriptor.clone().with_exact_candidate(retry_token),
            [second, first],
            true,
        )
        .unwrap();

        assert_eq!(retry.candidate.id(), retry_token);
        assert!(
            candidate_retry_binding(retry_token).is_some(),
            "an exact retry must not evict the binding it is currently consuming"
        );
    }

    #[test]
    fn genuinely_indistinguishable_candidates_remain_ambiguous_after_retry() {
        let _retry_registry = isolated_retry_registry();
        let first = with_diagnostics(adapter(), "alpha", "driver alpha", "info alpha");
        let second = with_diagnostics(adapter(), "beta", "driver beta", "info beta");
        assert_eq!(
            canonical_candidate_input_key(
                &first,
                GpuCandidateEnvironmentEvidence::current_host(true)
            ),
            canonical_candidate_input_key(
                &second,
                GpuCandidateEnvironmentEvidence::current_host(true)
            ),
            "diagnostic names and driver facts must not create retry identity"
        );
        let descriptor = GpuContextDescriptor::new(GpuCapabilityRequirements::new());
        let ambiguous =
            select_candidate(&descriptor, [first.clone(), second.clone()], true).unwrap_err();
        let token = ambiguous.candidate_dispositions()[0].id();
        let retry = select_candidate(
            &descriptor.clone().with_exact_candidate(token),
            [second, first],
            true,
        )
        .unwrap_err();
        assert_eq!(
            retry.category(),
            GpuContextRequestErrorCategory::AmbiguousAdapterSelection
        );
    }

    #[test]
    fn full_disposition_collection_is_invariant_to_input_order() {
        let accepted = with_diagnostics(adapter(), "accepted", "driver", "info");
        let limits = test_limits();
        let rejected = GpuAdapterFacts::new(
            GpuBackendFamily::UnknownBackend,
            GpuAdapterClass::Unknown,
            GpuSoftwareStatus::Unknown,
            GpuFallbackStatus::Unknown,
            GpuCapabilities::from_normalized_facts([], limits, []),
            GpuAdapterLimits::new(limits),
            accepted.alignments(),
        )
        .with_diagnostics(
            "rejected".to_owned(),
            2,
            2,
            "driver".to_owned(),
            "info".to_owned(),
        );
        let descriptor = GpuContextDescriptor::new(GpuCapabilityRequirements::new());
        let forward =
            select_candidate(&descriptor, [rejected.clone(), accepted.clone()], true).unwrap();
        let reverse = select_candidate(&descriptor, [accepted, rejected], true).unwrap();
        assert_eq!(
            forward
                .dispositions
                .iter()
                .map(canonical_disposition_key)
                .collect::<Vec<_>>(),
            reverse
                .dispositions
                .iter()
                .map(canonical_disposition_key)
                .collect::<Vec<_>>()
        );
        assert!(
            GpuContextAdmissionReport {
                selected: GpuCandidateSelectionKind::DeterministicallyRanked,
                candidate: forward.candidate.clone(),
                candidate_dispositions: forward.dispositions.clone(),
                selection_evidence: forward.evidence.clone(),
            }
            .candidate_dispositions_are_canonically_ordered()
        );
    }
}
