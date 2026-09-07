use super::source::{GpuProgramSourceDigest, GpuProgramSourceIdentity};
use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuProgramSourceCause {
    InvalidSourceOwner,
    InvalidSourceKey,
    InvalidSourceRevision,
    InvalidProvenance,
    EmptyCanonicalWgsl,
    SourceAdmissionCapacityExceeded,
    SourceRevisionConflict,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuProgramSourceError {
    Invalid {
        operation: &'static str,
        label: String,
        cause: GpuProgramSourceCause,
        correction: &'static str,
    },
    CapacityExceeded {
        operation: &'static str,
        label: String,
        cause: GpuProgramSourceCause,
        max_records: usize,
        max_retained_source_bytes: usize,
        retained_records: usize,
        retained_source_bytes: usize,
        attempted_source_bytes: usize,
        correction: &'static str,
    },
    RevisionConflict {
        operation: &'static str,
        identity: GpuProgramSourceIdentity,
        cause: GpuProgramSourceCause,
        existing_digest: GpuProgramSourceDigest,
        attempted_digest: GpuProgramSourceDigest,
        correction: &'static str,
    },
}

impl GpuProgramSourceError {
    pub(crate) fn invalid(
        operation: &'static str,
        label: impl Into<String>,
        cause: GpuProgramSourceCause,
        correction: &'static str,
    ) -> Self {
        Self::Invalid {
            operation,
            label: label.into(),
            cause,
            correction,
        }
    }

    pub(crate) fn capacity_exceeded(
        label: impl Into<String>,
        max_records: usize,
        max_retained_source_bytes: usize,
        retained_records: usize,
        retained_source_bytes: usize,
        attempted_source_bytes: usize,
    ) -> Self {
        Self::CapacityExceeded {
            operation: "admit canonical GPU program source",
            label: label.into(),
            cause: GpuProgramSourceCause::SourceAdmissionCapacityExceeded,
            max_records,
            max_retained_source_bytes,
            retained_records,
            retained_source_bytes,
            attempted_source_bytes,
            correction: "increase the explicit registry bounds or release an unneeded admitted source",
        }
    }

    pub(crate) fn revision_conflict(
        identity: GpuProgramSourceIdentity,
        existing_digest: GpuProgramSourceDigest,
        attempted_digest: GpuProgramSourceDigest,
    ) -> Self {
        Self::RevisionConflict {
            operation: "admit canonical GPU program source",
            identity,
            cause: GpuProgramSourceCause::SourceRevisionConflict,
            existing_digest,
            attempted_digest,
            correction: "allocate a new source revision for different canonical WGSL",
        }
    }

    pub const fn cause(&self) -> GpuProgramSourceCause {
        match self {
            Self::Invalid { cause, .. }
            | Self::CapacityExceeded { cause, .. }
            | Self::RevisionConflict { cause, .. } => *cause,
        }
    }
}

impl fmt::Display for GpuProgramSourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid {
                operation,
                label,
                cause,
                correction,
            } => write!(
                formatter,
                "cannot {operation} '{label}': {cause:?}; correction: {correction}"
            ),
            Self::CapacityExceeded {
                operation,
                label,
                cause,
                max_records,
                max_retained_source_bytes,
                retained_records,
                retained_source_bytes,
                attempted_source_bytes,
                correction,
            } => write!(
                formatter,
                "cannot {operation} '{label}': {cause:?}; registry retains {retained_records}/{max_records} records and {retained_source_bytes}/{max_retained_source_bytes} source bytes, attempted source uses {attempted_source_bytes} bytes; correction: {correction}"
            ),
            Self::RevisionConflict {
                operation,
                identity,
                cause,
                existing_digest,
                attempted_digest,
                correction,
            } => write!(
                formatter,
                "cannot {operation} '{}': {cause:?}; existing digest {existing_digest} differs from attempted digest {attempted_digest}; correction: {correction}",
                identity.diagnostic_label()
            ),
        }
    }
}

impl std::error::Error for GpuProgramSourceError {}
