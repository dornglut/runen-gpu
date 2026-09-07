use super::{
    GpuContextAffinity, GpuInitialCoverage, GpuProgramContractError, GpuReadbackBytes,
    GpuReadbackId, GpuResourceRef, GpuSurfaceLeaseError, GpuWorkOperationError,
};
use core::fmt;
use core::num::{NonZeroU64, NonZeroUsize};
use std::sync::{Arc, Mutex, Weak};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuExecutionPolicy {
    max_prepared_submissions: NonZeroUsize,
    max_in_flight_submissions: NonZeroUsize,
    max_upload_bytes_in_flight: u64,
    max_readback_bytes_in_flight: u64,
    max_pending_readbacks: usize,
}

impl GpuExecutionPolicy {
    pub const fn new(
        max_prepared_submissions: NonZeroUsize,
        max_in_flight_submissions: NonZeroUsize,
        max_upload_bytes_in_flight: u64,
        max_readback_bytes_in_flight: u64,
        max_pending_readbacks: usize,
    ) -> Self {
        Self {
            max_prepared_submissions,
            max_in_flight_submissions,
            max_upload_bytes_in_flight,
            max_readback_bytes_in_flight,
            max_pending_readbacks,
        }
    }

    pub const fn max_prepared_submissions(self) -> NonZeroUsize {
        self.max_prepared_submissions
    }

    pub const fn max_in_flight_submissions(self) -> NonZeroUsize {
        self.max_in_flight_submissions
    }

    pub const fn max_upload_bytes_in_flight(self) -> u64 {
        self.max_upload_bytes_in_flight
    }

    pub const fn max_readback_bytes_in_flight(self) -> u64 {
        self.max_readback_bytes_in_flight
    }

    pub const fn max_pending_readbacks(self) -> usize {
        self.max_pending_readbacks
    }
}

impl Default for GpuExecutionPolicy {
    fn default() -> Self {
        Self::new(
            NonZeroUsize::new(64).unwrap_or(NonZeroUsize::MIN),
            NonZeroUsize::new(32).unwrap_or(NonZeroUsize::MIN),
            64 * 1024 * 1024,
            64 * 1024 * 1024,
            64,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuExecutionStats {
    prepared_submissions: usize,
    in_flight_submissions: usize,
    upload_bytes_in_flight: u64,
    readback_bytes_in_flight: u64,
    pending_readbacks: usize,
}

impl GpuExecutionStats {
    pub(crate) const fn new(
        prepared_submissions: usize,
        in_flight_submissions: usize,
        upload_bytes_in_flight: u64,
        readback_bytes_in_flight: u64,
        pending_readbacks: usize,
    ) -> Self {
        Self {
            prepared_submissions,
            in_flight_submissions,
            upload_bytes_in_flight,
            readback_bytes_in_flight,
            pending_readbacks,
        }
    }

    pub const fn prepared_submissions(self) -> usize {
        self.prepared_submissions
    }

    pub const fn in_flight_submissions(self) -> usize {
        self.in_flight_submissions
    }

    pub const fn upload_bytes_in_flight(self) -> u64 {
        self.upload_bytes_in_flight
    }

    pub const fn readback_bytes_in_flight(self) -> u64 {
        self.readback_bytes_in_flight
    }

    pub const fn pending_readbacks(self) -> usize {
        self.pending_readbacks
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuExecutionLifecycleState {
    Running,
    ShuttingDown,
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuSubmissionId(NonZeroU64);

impl GpuSubmissionId {
    pub(crate) const fn from_nonzero(value: NonZeroU64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

impl fmt::Display for GpuSubmissionId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.get().fmt(formatter)
    }
}

/// Opaque current-content continuity for one retained logical storage resource.
///
/// This fact deliberately does not imply initialized coverage or reconstructability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuOpaqueContentContinuity {
    /// No completed retained-state write has established coherent current content.
    Unestablished,
    /// The last successfully completed retained-state write establishing current content.
    Established {
        last_completed_write: GpuSubmissionId,
    },
    /// A submitted write may have executed without completed success proof, so current content is indeterminate.
    Unknown,
}

/// Typed lifecycle reason why retained opaque content is currently indeterminate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuOpaqueContentIndeterminateReason {
    /// An accepted retained-state write may execute and has not terminalized yet.
    AcceptedWritePending,
    /// A retained-state write may have executed before the submission failed, so prior coherent content was revoked.
    PossibleWriteFailure,
}

/// Point-in-time retained-state continuity owned by one context/device generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuRetainedResourceContinuity {
    affinity: GpuContextAffinity,
    resource: GpuResourceRef,
    initialized_coverage: Option<GpuInitialCoverage>,
    opaque_content: GpuOpaqueContentContinuity,
    opaque_content_indeterminate_reason: Option<GpuOpaqueContentIndeterminateReason>,
}

impl GpuRetainedResourceContinuity {
    pub(crate) fn new(
        affinity: GpuContextAffinity,
        resource: GpuResourceRef,
        initialized_coverage: Option<GpuInitialCoverage>,
        opaque_content: GpuOpaqueContentContinuity,
        opaque_content_indeterminate_reason: Option<GpuOpaqueContentIndeterminateReason>,
    ) -> Self {
        debug_assert_eq!(
            matches!(opaque_content, GpuOpaqueContentContinuity::Unknown),
            opaque_content_indeterminate_reason.is_some()
        );
        Self {
            affinity,
            resource,
            initialized_coverage,
            opaque_content,
            opaque_content_indeterminate_reason,
        }
    }

    pub const fn affinity(&self) -> GpuContextAffinity {
        self.affinity
    }

    pub fn resource(&self) -> &GpuResourceRef {
        &self.resource
    }

    pub fn initialized_coverage(&self) -> Option<&GpuInitialCoverage> {
        self.initialized_coverage.as_ref()
    }

    pub const fn opaque_content(&self) -> GpuOpaqueContentContinuity {
        self.opaque_content
    }

    pub const fn opaque_content_indeterminate_reason(
        &self,
    ) -> Option<GpuOpaqueContentIndeterminateReason> {
        self.opaque_content_indeterminate_reason
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuSubmissionFailureKind {
    BackendValidation,
    BackendResourceExhaustion,
    ContextOrDeviceUnavailableOrLost,
    SurfaceLease,
    ReadbackMapping,
    ContextDropped,
    InternalInvariant,
}

#[derive(Debug, Clone)]
pub struct GpuSubmissionFailure {
    kind: GpuSubmissionFailureKind,
    detail: String,
    surface_error: Option<Box<GpuSurfaceLeaseError>>,
}

impl GpuSubmissionFailure {
    pub(crate) fn new(kind: GpuSubmissionFailureKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
            surface_error: None,
        }
    }

    pub(crate) fn from_surface_lease(error: GpuSurfaceLeaseError) -> Self {
        Self {
            kind: GpuSubmissionFailureKind::SurfaceLease,
            detail: error.to_string(),
            surface_error: Some(Box::new(error)),
        }
    }

    pub const fn kind(&self) -> GpuSubmissionFailureKind {
        self.kind
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }

    pub fn surface_error(&self) -> Option<&GpuSurfaceLeaseError> {
        self.surface_error.as_deref()
    }
}

impl PartialEq for GpuSubmissionFailure {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}

impl Eq for GpuSubmissionFailure {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuSubmissionStatus {
    Accepted,
    Completed,
    Failed(GpuSubmissionFailure),
}

#[derive(Debug, Clone)]
pub enum GpuReadbackStatus {
    Pending,
    Ready(GpuReadbackBytes),
    Failed(GpuSubmissionFailure),
}

#[derive(Clone)]
pub struct GpuReadback {
    id: GpuReadbackId,
    status: Arc<Mutex<GpuReadbackStatus>>,
}

impl GpuReadback {
    pub(crate) fn new(id: GpuReadbackId, status: Arc<Mutex<GpuReadbackStatus>>) -> Self {
        Self { id, status }
    }

    pub const fn id(&self) -> GpuReadbackId {
        self.id
    }

    pub fn status(&self) -> GpuReadbackStatus {
        self.status
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl fmt::Debug for GpuReadback {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuReadback")
            .field("id", &self.id)
            .field("status", &self.status())
            .finish()
    }
}

#[derive(Clone)]
pub struct GpuSubmission {
    id: GpuSubmissionId,
    affinity: GpuContextAffinity,
    status: Arc<Mutex<GpuSubmissionStatus>>,
    readbacks: Arc<[GpuReadback]>,
}

impl GpuSubmission {
    pub(crate) fn new(
        id: GpuSubmissionId,
        affinity: GpuContextAffinity,
        status: Arc<Mutex<GpuSubmissionStatus>>,
        readbacks: Vec<GpuReadback>,
    ) -> Self {
        Self {
            id,
            affinity,
            status,
            readbacks: readbacks.into(),
        }
    }

    pub const fn id(&self) -> GpuSubmissionId {
        self.id
    }

    pub const fn affinity(&self) -> GpuContextAffinity {
        self.affinity
    }

    pub fn status(&self) -> GpuSubmissionStatus {
        self.status
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub fn readbacks(&self) -> &[GpuReadback] {
        &self.readbacks
    }

    pub fn readback(&self, id: GpuReadbackId) -> Option<&GpuReadback> {
        self.readbacks.iter().find(|readback| readback.id() == id)
    }
}

impl fmt::Debug for GpuSubmission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuSubmission")
            .field("id", &self.id)
            .field("affinity", &self.affinity)
            .field("status", &self.status())
            .field("readbacks", &self.readbacks)
            .finish()
    }
}

/// Original typed authority for one contextual `WorkNotAdmitted` rejection.
///
/// This enum transports existing owner-specific errors and does not define new rejection causes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuWorkNotAdmittedSource {
    ProgramContract(GpuProgramContractError),
    Operation(GpuWorkOperationError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuSubmissionPreparationErrorKind {
    CapabilityNotAdmitted,
    WorkNotAdmitted,
    PreparedCapacityExceeded,
    UploadDemandExceedsPolicy,
    ReadbackDemandExceedsPolicy,
    PendingReadbacksExceedPolicy,
    UnsupportedOperation,
    TransferAlignmentNotAdmitted,
    ResourceRealizationFailed,
    ProgramBindingRealizationFailed,
    PipelineRealizationFailed,
    DynamicOffsetNotEncodable,
    ExecutionNotRunning,
    ContextOrDeviceUnavailableOrLost,
    SurfaceLease,
    IdentityExhausted,
    InternalInvariant,
}

#[derive(Debug, Clone)]
pub struct GpuSubmissionPreparationError {
    kind: GpuSubmissionPreparationErrorKind,
    detail: String,
    work_not_admitted_source: Option<Box<GpuWorkNotAdmittedSource>>,
    surface_error: Option<Box<GpuSurfaceLeaseError>>,
}

impl GpuSubmissionPreparationError {
    pub(crate) fn new(kind: GpuSubmissionPreparationErrorKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
            work_not_admitted_source: None,
            surface_error: None,
        }
    }

    pub(crate) fn work_not_admitted(
        detail: impl Into<String>,
        source: GpuWorkNotAdmittedSource,
    ) -> Self {
        Self {
            kind: GpuSubmissionPreparationErrorKind::WorkNotAdmitted,
            detail: detail.into(),
            work_not_admitted_source: Some(Box::new(source)),
            surface_error: None,
        }
    }

    pub(crate) fn from_surface_lease(error: GpuSurfaceLeaseError) -> Self {
        Self {
            kind: GpuSubmissionPreparationErrorKind::SurfaceLease,
            detail: error.to_string(),
            work_not_admitted_source: None,
            surface_error: Some(Box::new(error)),
        }
    }

    pub const fn kind(&self) -> GpuSubmissionPreparationErrorKind {
        self.kind
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }

    pub fn work_not_admitted_source(&self) -> Option<&GpuWorkNotAdmittedSource> {
        self.work_not_admitted_source.as_deref()
    }

    pub fn surface_error(&self) -> Option<&GpuSurfaceLeaseError> {
        self.surface_error.as_deref()
    }
}

impl PartialEq for GpuSubmissionPreparationError {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}

impl Eq for GpuSubmissionPreparationError {}

impl fmt::Display for GpuSubmissionPreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "GPU submission preparation rejected ({:?}): {}",
            self.kind, self.detail
        )
    }
}

impl std::error::Error for GpuSubmissionPreparationError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuSubmissionRejectionKind {
    ForeignContext,
    StaleDeviceGeneration,
    RetainedContinuityChanged,
    PreparedRecordUnavailable,
    InFlightCapacityExceeded,
    UploadBytesInFlightExceeded,
    ReadbackBytesInFlightExceeded,
    PendingReadbacksExceeded,
    ExecutionNotRunning,
    ContextOrDeviceUnavailableOrLost,
    SurfaceLease,
    IdentityExhausted,
}

/// Typed current pressure and policy for one rejected additional in-flight submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuInFlightCapacityEvidence {
    current_in_flight_submissions: usize,
    policy: GpuExecutionPolicy,
}

impl GpuInFlightCapacityEvidence {
    const fn new(current_in_flight_submissions: usize, policy: GpuExecutionPolicy) -> Self {
        Self {
            current_in_flight_submissions,
            policy,
        }
    }

    pub const fn current_in_flight_submissions(self) -> usize {
        self.current_in_flight_submissions
    }

    pub const fn policy(self) -> GpuExecutionPolicy {
        self.policy
    }
}

#[derive(Debug, Clone)]
pub struct GpuSubmissionRejectionReason {
    kind: GpuSubmissionRejectionKind,
    detail: String,
    in_flight_capacity_evidence: Option<Box<GpuInFlightCapacityEvidence>>,
    surface_error: Option<Box<GpuSurfaceLeaseError>>,
}

impl GpuSubmissionRejectionReason {
    pub(crate) fn new(kind: GpuSubmissionRejectionKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
            in_flight_capacity_evidence: None,
            surface_error: None,
        }
    }

    pub(crate) fn from_surface_lease(error: GpuSurfaceLeaseError) -> Self {
        Self {
            kind: GpuSubmissionRejectionKind::SurfaceLease,
            detail: error.to_string(),
            in_flight_capacity_evidence: None,
            surface_error: Some(Box::new(error)),
        }
    }

    fn attach_in_flight_capacity_evidence(
        &mut self,
        current_in_flight_submissions: usize,
        policy: GpuExecutionPolicy,
    ) {
        debug_assert_eq!(
            self.kind,
            GpuSubmissionRejectionKind::InFlightCapacityExceeded
        );
        self.in_flight_capacity_evidence = Some(Box::new(GpuInFlightCapacityEvidence::new(
            current_in_flight_submissions,
            policy,
        )));
    }

    pub const fn kind(&self) -> GpuSubmissionRejectionKind {
        self.kind
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }

    pub fn in_flight_capacity_evidence(&self) -> Option<&GpuInFlightCapacityEvidence> {
        self.in_flight_capacity_evidence.as_deref()
    }

    pub fn surface_error(&self) -> Option<&GpuSurfaceLeaseError> {
        self.surface_error.as_deref()
    }
}

impl PartialEq for GpuSubmissionRejectionReason {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}

impl Eq for GpuSubmissionRejectionReason {}

pub struct GpuPreparedSubmission {
    pub(crate) ticket: NonZeroU64,
    pub(crate) affinity: GpuContextAffinity,
    pub(crate) execution: Weak<crate::backend::WgpuExecutionState>,
    pub(crate) armed: bool,
    planned_readbacks: Arc<[GpuReadbackId]>,
}

impl GpuPreparedSubmission {
    pub(crate) fn new(
        ticket: NonZeroU64,
        affinity: GpuContextAffinity,
        execution: Weak<crate::backend::WgpuExecutionState>,
        planned_readbacks: Vec<GpuReadbackId>,
    ) -> Self {
        Self {
            ticket,
            affinity,
            execution,
            armed: true,
            planned_readbacks: planned_readbacks.into(),
        }
    }

    pub const fn affinity(&self) -> GpuContextAffinity {
        self.affinity
    }

    pub fn planned_readbacks(&self) -> &[GpuReadbackId] {
        &self.planned_readbacks
    }

    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl fmt::Debug for GpuPreparedSubmission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuPreparedSubmission")
            .field("affinity", &self.affinity)
            .field("planned_readbacks", &self.planned_readbacks)
            .finish_non_exhaustive()
    }
}

impl Drop for GpuPreparedSubmission {
    fn drop(&mut self) {
        if self.armed {
            if let Some(execution) = self.execution.upgrade() {
                execution.release_prepared(self.ticket);
            }
        }
    }
}

pub struct GpuPreparedSubmissionRejected {
    prepared: GpuPreparedSubmission,
    reason: GpuSubmissionRejectionReason,
}

impl GpuPreparedSubmissionRejected {
    pub(crate) fn new(
        prepared: GpuPreparedSubmission,
        reason: GpuSubmissionRejectionReason,
    ) -> Self {
        let mut reason = reason;
        if reason.kind() == GpuSubmissionRejectionKind::InFlightCapacityExceeded {
            if let Some(execution) = prepared.execution.upgrade() {
                let stats = execution.stats();
                reason.attach_in_flight_capacity_evidence(
                    stats.in_flight_submissions(),
                    execution.policy(),
                );
            }
        }
        Self { prepared, reason }
    }

    pub fn prepared(&self) -> &GpuPreparedSubmission {
        &self.prepared
    }

    pub fn reason(&self) -> &GpuSubmissionRejectionReason {
        &self.reason
    }

    pub fn into_parts(self) -> (GpuPreparedSubmission, GpuSubmissionRejectionReason) {
        (self.prepared, self.reason)
    }
}

impl fmt::Debug for GpuPreparedSubmissionRejected {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuPreparedSubmissionRejected")
            .field("prepared", &self.prepared)
            .field("reason", &self.reason)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_detail_does_not_define_execution_error_equality() {
        let failure_a = GpuSubmissionFailure::new(
            GpuSubmissionFailureKind::BackendValidation,
            "backend diagnostic A",
        );
        let failure_b = GpuSubmissionFailure::new(
            GpuSubmissionFailureKind::BackendValidation,
            "backend diagnostic B",
        );
        let failure_other = GpuSubmissionFailure::new(
            GpuSubmissionFailureKind::BackendResourceExhaustion,
            "backend diagnostic A",
        );
        assert_eq!(failure_a, failure_b);
        assert_ne!(failure_a, failure_other);

        let preparation_a = GpuSubmissionPreparationError::new(
            GpuSubmissionPreparationErrorKind::ContextOrDeviceUnavailableOrLost,
            "backend diagnostic A",
        );
        let preparation_b = GpuSubmissionPreparationError::new(
            GpuSubmissionPreparationErrorKind::ContextOrDeviceUnavailableOrLost,
            "backend diagnostic B",
        );
        assert_eq!(preparation_a, preparation_b);

        let rejection_a = GpuSubmissionRejectionReason::new(
            GpuSubmissionRejectionKind::ContextOrDeviceUnavailableOrLost,
            "backend diagnostic A",
        );
        let rejection_b = GpuSubmissionRejectionReason::new(
            GpuSubmissionRejectionKind::ContextOrDeviceUnavailableOrLost,
            "backend diagnostic B",
        );
        assert_eq!(rejection_a, rejection_b);

        assert_eq!(
            GpuSubmissionStatus::Failed(failure_a),
            GpuSubmissionStatus::Failed(failure_b)
        );
    }
}
