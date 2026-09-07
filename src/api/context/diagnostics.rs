use super::descriptor::GpuLimitKind;
use super::selection::GpuCandidateDisposition;
use crate::GpuCapabilityAdmissionError;
use std::collections::BTreeMap;
use std::fmt;

pub(crate) const MAX_DIAGNOSTIC_BYTES: usize = 256;
const MAX_ERROR_DISPLAY_BYTES: usize = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GpuContextRequestErrorCategory {
    NoAdapterAvailable,
    NoAdmissibleCandidate,
    AmbiguousAdapterSelection,
    StaleCandidateRetryToken,
    CandidateRetryTokenExhausted,
    BackendFamilyForbidden,
    SoftwareFallbackPolicyViolation,
    MandatoryFeatureMissing,
    LimitBelowRequiredMinimum,
    LimitAbovePermittedMaximum,
    UnsupportedFormatRole,
    AlignmentIncompatibility,
    DeviceRequestProfileUnsupported,
    ContradictoryRequest,
    BackendAdapterRequestFailure,
    BackendDeviceRequestFailure,
    SurfaceCreationFailure,
    SurfaceCompatibilityFailure,
    TemporaryHostCompatibilityFailure,
    IdentityExhausted,
    InvalidDegradation,
}

impl GpuContextRequestErrorCategory {
    const fn correction(self) -> &'static str {
        match self {
            Self::NoAdapterAvailable => {
                "provide a supported GPU environment or defer the optional workload"
            }
            Self::NoAdmissibleCandidate => {
                "adjust the request or inspect the retained candidate dispositions"
            }
            Self::AmbiguousAdapterSelection => {
                "retry with one disclosed process-local candidate ID"
            }
            Self::StaleCandidateRetryToken => {
                "rediscover candidates and retry with a newly disclosed process-local token"
            }
            Self::CandidateRetryTokenExhausted => {
                "restart the process before issuing another candidate retry token"
            }
            Self::BackendFamilyForbidden => {
                "permit the observed backend or choose a compatible adapter"
            }
            Self::SoftwareFallbackPolicyViolation => {
                "adjust the fallback policy or select a proven adapter path"
            }
            Self::MandatoryFeatureMissing => "remove the requirement or select a capable adapter",
            Self::LimitBelowRequiredMinimum => {
                "lower the workload minimum or select a more capable adapter"
            }
            Self::LimitAbovePermittedMaximum => {
                "raise the workload maximum or remove the contradiction"
            }
            Self::UnsupportedFormatRole => "choose a supported normalized format role",
            Self::AlignmentIncompatibility => {
                "relax the alignment cap or select a compatible device"
            }
            Self::DeviceRequestProfileUnsupported => {
                "select a device that supports the complete profile"
            }
            Self::ContradictoryRequest => "make the merged context requirements consistent",
            Self::BackendAdapterRequestFailure => {
                "inspect the bounded backend detail and host environment"
            }
            Self::BackendDeviceRequestFailure => {
                "inspect the bounded backend detail and admitted profile"
            }
            Self::SurfaceCreationFailure => {
                "provide a valid safe host target supported by the current platform"
            }
            Self::SurfaceCompatibilityFailure => {
                "select a surface-compatible adapter or use a compatible presentation target"
            }
            Self::TemporaryHostCompatibilityFailure => {
                "use a surface-compatible adapter for the current host"
            }
            Self::IdentityExhausted => "restart the process before creating another context",
            Self::InvalidDegradation => "declare a valid preferred degradation for the workload",
        }
    }

    pub(crate) const fn stable_order(self) -> u8 {
        match self {
            Self::NoAdapterAvailable => 0,
            Self::NoAdmissibleCandidate => 1,
            Self::AmbiguousAdapterSelection => 2,
            Self::StaleCandidateRetryToken => 3,
            Self::CandidateRetryTokenExhausted => 4,
            Self::BackendFamilyForbidden => 5,
            Self::SoftwareFallbackPolicyViolation => 6,
            Self::MandatoryFeatureMissing => 7,
            Self::LimitBelowRequiredMinimum => 8,
            Self::LimitAbovePermittedMaximum => 9,
            Self::UnsupportedFormatRole => 10,
            Self::AlignmentIncompatibility => 11,
            Self::DeviceRequestProfileUnsupported => 12,
            Self::ContradictoryRequest => 13,
            Self::BackendAdapterRequestFailure => 14,
            Self::BackendDeviceRequestFailure => 15,
            Self::SurfaceCreationFailure => 16,
            Self::SurfaceCompatibilityFailure => 17,
            Self::TemporaryHostCompatibilityFailure => 18,
            Self::IdentityExhausted => 19,
            Self::InvalidDegradation => 20,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuContextLimitRejection {
    kind: GpuLimitKind,
    required_minimum: u64,
    observed: u64,
}

impl GpuContextLimitRejection {
    pub(crate) const fn new(kind: GpuLimitKind, required_minimum: u64, observed: u64) -> Self {
        Self {
            kind,
            required_minimum,
            observed,
        }
    }

    pub(crate) const fn as_tuple(self) -> (GpuLimitKind, u64, u64) {
        (self.kind, self.required_minimum, self.observed)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuContextRequestError {
    category: GpuContextRequestErrorCategory,
    detail: Option<String>,
    capability_admission_error: Option<Box<GpuCapabilityAdmissionError>>,
    limit_rejection: Option<GpuContextLimitRejection>,
    candidate_dispositions: Vec<GpuCandidateDisposition>,
}

impl GpuContextRequestError {
    pub(crate) fn new(category: GpuContextRequestErrorCategory, detail: impl Into<String>) -> Self {
        Self {
            category,
            detail: sanitized_diagnostic(detail.into()),
            capability_admission_error: None,
            limit_rejection: None,
            candidate_dispositions: Vec::new(),
        }
    }

    pub(crate) fn from_capability_admission(error: GpuCapabilityAdmissionError) -> Self {
        let detail = sanitized_diagnostic(error.to_string());
        Self {
            category: GpuContextRequestErrorCategory::MandatoryFeatureMissing,
            detail,
            capability_admission_error: Some(Box::new(error)),
            limit_rejection: None,
            candidate_dispositions: Vec::new(),
        }
    }

    pub(crate) fn limit_below_required_minimum(
        kind: GpuLimitKind,
        required_minimum: u64,
        observed: u64,
        detail: impl Into<String>,
    ) -> Self {
        debug_assert!(observed < required_minimum);
        Self {
            category: GpuContextRequestErrorCategory::LimitBelowRequiredMinimum,
            detail: sanitized_diagnostic(detail.into()),
            capability_admission_error: None,
            limit_rejection: Some(GpuContextLimitRejection::new(
                kind,
                required_minimum,
                observed,
            )),
            candidate_dispositions: Vec::new(),
        }
    }

    pub const fn category(&self) -> GpuContextRequestErrorCategory {
        self.category
    }

    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }

    pub fn capability_admission_error(&self) -> Option<&GpuCapabilityAdmissionError> {
        self.capability_admission_error.as_deref()
    }

    /// Returns `(normalized limit kind, required minimum, observed value)` for a below-minimum
    /// context-admission rejection. Other rejection classes return `None`.
    pub const fn limit_rejection(&self) -> Option<(GpuLimitKind, u64, u64)> {
        match self.limit_rejection {
            Some(rejection) => Some(rejection.as_tuple()),
            None => None,
        }
    }

    pub(crate) const fn limit_rejection_evidence(&self) -> Option<GpuContextLimitRejection> {
        self.limit_rejection
    }

    pub fn candidate_dispositions(&self) -> &[GpuCandidateDisposition] {
        &self.candidate_dispositions
    }

    pub(crate) fn with_candidate_dispositions(
        mut self,
        dispositions: Vec<GpuCandidateDisposition>,
    ) -> Self {
        self.candidate_dispositions = dispositions;
        self
    }
}

impl fmt::Display for GpuContextRequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut rendered = format!("GPU context request failed ({:?})", self.category);
        if let Some(detail) = &self.detail {
            append_bounded(&mut rendered, ": ");
            append_bounded(&mut rendered, detail);
        }
        if !self.candidate_dispositions.is_empty() {
            let rejected = self
                .candidate_dispositions
                .iter()
                .filter(|disposition| disposition.is_rejected())
                .count();
            let accepted = self.candidate_dispositions.len() - rejected;
            append_bounded(
                &mut rendered,
                &format!("; candidate dispositions: {accepted} accepted, {rejected} rejected"),
            );
            let mut category_counts = BTreeMap::<GpuContextRequestErrorCategory, usize>::new();
            let mut representative: Option<((u8, String), String)> = None;
            for disposition in &self.candidate_dispositions {
                let GpuCandidateDisposition::Rejected(report) = disposition else {
                    continue;
                };
                *category_counts.entry(report.category()).or_default() += 1;
                let reason = report.detail().map_or_else(
                    || format!("{:?}", report.category()),
                    |detail| format!("{:?}: {detail}", report.category()),
                );
                let key = (report.category().stable_order(), reason.clone());
                let replace = match &representative {
                    None => true,
                    Some((current, _)) => {
                        key.0 < current.0 || (key.0 == current.0 && key.1 < current.1)
                    }
                };
                if replace {
                    representative = Some((key, reason));
                }
            }
            if !category_counts.is_empty() {
                let mut category_counts = category_counts.into_iter().collect::<Vec<_>>();
                category_counts.sort_by_key(|(category, _)| category.stable_order());
                let counts = category_counts
                    .into_iter()
                    .map(|(category, count)| format!("{category:?}={count}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                append_bounded(&mut rendered, "; rejection categories: ");
                append_bounded(&mut rendered, &counts);
            }
            append_bounded(&mut rendered, "; correction: ");
            append_bounded(&mut rendered, self.category.correction());
            if let Some((_, reason)) = representative {
                append_bounded(&mut rendered, "; representative rejection: ");
                append_bounded(&mut rendered, &reason);
            }
        } else {
            append_bounded(&mut rendered, "; correction: ");
            append_bounded(&mut rendered, self.category.correction());
        }
        f.write_str(&rendered)
    }
}

impl std::error::Error for GpuContextRequestError {}

pub(crate) fn sanitized_diagnostic(value: String) -> Option<String> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let mut bounded = String::new();
    for character in value.chars() {
        if bounded.len() + character.len_utf8() > MAX_DIAGNOSTIC_BYTES {
            break;
        }
        bounded.push(character);
    }
    Some(bounded)
}

fn append_bounded(output: &mut String, value: &str) {
    for character in value.chars() {
        if output.len() + character.len_utf8() > MAX_ERROR_DISPLAY_BYTES {
            break;
        }
        output.push(character);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_keeps_utf8_detail_bounded_and_actionable() {
        let detail = sanitized_diagnostic("é".repeat(129)).unwrap();
        assert_eq!(detail.len(), MAX_DIAGNOSTIC_BYTES);
        assert!(detail.is_char_boundary(detail.len()));
        let error = GpuContextRequestError::new(
            GpuContextRequestErrorCategory::BackendDeviceRequestFailure,
            detail,
        );
        let rendered = error.to_string();
        assert!(rendered.contains("BackendDeviceRequestFailure"));
        assert!(rendered.contains("correction:"));
        assert!(rendered.len() <= MAX_ERROR_DISPLAY_BYTES);
        assert!(rendered.is_char_boundary(rendered.len()));
    }

    #[test]
    fn display_reports_deterministic_rejection_counts_and_a_safe_reason() {
        use crate::{
            GpuAdapterClass, GpuAdapterFacts, GpuAdapterLimits, GpuAlignmentFacts,
            GpuBackendFamily, GpuCapabilities, GpuFallbackStatus, GpuLimits,
            GpuRejectedCandidateReport, GpuSoftwareStatus,
        };

        let limits = || {
            GpuLimits::new(
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
            .unwrap()
        };
        let adapter = || {
            GpuAdapterFacts::new(
                GpuBackendFamily::Vulkan,
                GpuAdapterClass::Discrete,
                GpuSoftwareStatus::Hardware,
                GpuFallbackStatus::Unknown,
                GpuCapabilities::from_normalized_facts([], limits(), []),
                GpuAdapterLimits::new(limits()),
                GpuAlignmentFacts {
                    uniform_dynamic_offset: Some(256),
                    storage_dynamic_offset: Some(256),
                    copy_buffer_offset: Some(4),
                    bytes_per_row: Some(256),
                    query_resolve_destination: Some(256),
                },
            )
        };
        let rejected = |category: GpuContextRequestErrorCategory, detail: &str| {
            GpuCandidateDisposition::Rejected(Box::new(GpuRejectedCandidateReport {
                id: super::super::selection::GpuCandidateId::allocate().unwrap(),
                adapter: adapter(),
                category,
                detail: sanitized_diagnostic(detail.to_owned()),
                capability_admission_error: None,
                limit_rejection: None,
            }))
        };
        let rendered = GpuContextRequestError::new(
            GpuContextRequestErrorCategory::NoAdmissibleCandidate,
            "no candidate passed admission",
        )
        .with_candidate_dispositions(vec![
            rejected(
                GpuContextRequestErrorCategory::MandatoryFeatureMissing,
                "missing compute",
            ),
            rejected(
                GpuContextRequestErrorCategory::BackendFamilyForbidden,
                "second backend reason",
            ),
            rejected(
                GpuContextRequestErrorCategory::BackendFamilyForbidden,
                "first backend reason",
            ),
        ])
        .to_string();
        assert!(
            rendered.contains(
                "rejection categories: BackendFamilyForbidden=2, MandatoryFeatureMissing=1"
            )
        );
        assert!(
            rendered
                .contains("representative rejection: BackendFamilyForbidden: first backend reason")
        );
        assert!(rendered.len() <= MAX_ERROR_DISPLAY_BYTES);
        assert!(rendered.is_char_boundary(rendered.len()));
    }
}
