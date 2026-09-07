use super::super::diagnostics::{GpuProgramSourceCause, GpuProgramSourceError};
use core::fmt;
use core::num::NonZeroU64;
use core::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};

const MAX_SOURCE_KEY_BYTES: usize = 256;
const MAX_PROVENANCE_FIELD_BYTES: usize = 256;
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Process-local identity for one authority that supplies canonical program sources.
///
/// The raw value is deliberately not a persistence, wire, ABI, cache, or
/// filesystem identity.
///
/// ```compile_fail
/// use runen_gpu::GpuProgramSourceOwnerId;
/// let _ = GpuProgramSourceOwnerId(1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuProgramSourceOwnerId(NonZeroU64);

impl GpuProgramSourceOwnerId {
    pub fn allocate() -> Result<Self, GpuProgramSourceError> {
        allocate_source_owner_id(&PRODUCTION_SOURCE_OWNER_IDS)
    }

    /// Returns process-local diagnostic evidence only.
    pub const fn diagnostic_raw(self) -> u64 {
        self.0.get()
    }
}

fn allocate_source_owner_id(
    allocator: &AtomicU64,
) -> Result<GpuProgramSourceOwnerId, GpuProgramSourceError> {
    let value = allocator
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            (current != 0).then_some(if current == u64::MAX { 0 } else { current + 1 })
        })
        .map_err(|_| {
            GpuProgramSourceError::invalid(
                "allocate GPU program source owner identity",
                "source-owner allocator",
                GpuProgramSourceCause::InvalidSourceOwner,
                "restart the process after the process-local identity space is exhausted",
            )
        })?;
    let Some(value) = NonZeroU64::new(value) else {
        return Err(GpuProgramSourceError::invalid(
            "allocate GPU program source owner identity",
            "source-owner allocator",
            GpuProgramSourceCause::InvalidSourceOwner,
            "restart the process after the process-local identity space is exhausted",
        ));
    };
    Ok(GpuProgramSourceOwnerId(value))
}

static PRODUCTION_SOURCE_OWNER_IDS: AtomicU64 = AtomicU64::new(1);

impl fmt::Display for GpuProgramSourceOwnerId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Validated semantic key scoped by one source owner.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuProgramSourceKey(String);

impl GpuProgramSourceKey {
    pub fn new(value: impl Into<String>) -> Result<Self, GpuProgramSourceError> {
        let value = value.into();
        validate_bounded_text(
            "construct GPU program source key",
            &value,
            MAX_SOURCE_KEY_BYTES,
            GpuProgramSourceCause::InvalidSourceKey,
            "provide a non-empty key without leading/trailing whitespace or control characters",
        )?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl AsRef<str> for GpuProgramSourceKey {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for GpuProgramSourceKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl FromStr for GpuProgramSourceKey {
    type Err = GpuProgramSourceError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

/// Nonzero revision meaningful only for one owner/key pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuProgramSourceRevision(NonZeroU64);

impl GpuProgramSourceRevision {
    pub fn try_from_raw(raw: u64) -> Result<Self, GpuProgramSourceError> {
        NonZeroU64::new(raw).map(Self).ok_or_else(|| {
            GpuProgramSourceError::invalid(
                "construct GPU program source revision",
                raw.to_string(),
                GpuProgramSourceCause::InvalidSourceRevision,
                "provide a nonzero owner-local source revision",
            )
        })
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

impl fmt::Display for GpuProgramSourceRevision {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Complete semantic lookup identity for one admitted source revision.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuProgramSourceIdentity {
    owner: GpuProgramSourceOwnerId,
    key: GpuProgramSourceKey,
    revision: GpuProgramSourceRevision,
}

impl GpuProgramSourceIdentity {
    pub fn new(
        owner: GpuProgramSourceOwnerId,
        key: GpuProgramSourceKey,
        revision: GpuProgramSourceRevision,
    ) -> Self {
        Self {
            owner,
            key,
            revision,
        }
    }

    pub const fn owner(&self) -> GpuProgramSourceOwnerId {
        self.owner
    }

    pub fn key(&self) -> &GpuProgramSourceKey {
        &self.key
    }

    pub const fn revision(&self) -> GpuProgramSourceRevision {
        self.revision
    }

    pub fn diagnostic_label(&self) -> String {
        format!("{}:{}@{}", self.owner, self.key, self.revision)
    }
}

/// Bounded, diagnostic-only source provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuProgramSourceProvenance {
    producer: String,
    detail: Option<String>,
}

impl GpuProgramSourceProvenance {
    pub fn new(
        producer: impl Into<String>,
        detail: Option<String>,
    ) -> Result<Self, GpuProgramSourceError> {
        let producer = producer.into();
        validate_bounded_text(
            "construct GPU program source provenance",
            &producer,
            MAX_PROVENANCE_FIELD_BYTES,
            GpuProgramSourceCause::InvalidProvenance,
            "provide a bounded non-empty producer without control characters",
        )?;
        if let Some(detail) = detail.as_deref() {
            validate_bounded_text(
                "construct GPU program source provenance detail",
                detail,
                MAX_PROVENANCE_FIELD_BYTES,
                GpuProgramSourceCause::InvalidProvenance,
                "provide bounded non-empty provenance detail without control characters",
            )?;
        }
        Ok(Self { producer, detail })
    }

    pub fn producer(&self) -> &str {
        self.producer.as_str()
    }

    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }
}

/// Deterministic process-local digest used only to accelerate comparison and diagnostics.
///
/// Full canonical WGSL equality remains authoritative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuProgramSourceDigest(u64);

impl GpuProgramSourceDigest {
    pub(crate) fn from_canonical_wgsl(source: &str) -> Self {
        let mut digest = FNV_OFFSET_BASIS;
        for byte in source.as_bytes() {
            digest ^= u64::from(*byte);
            digest = digest.wrapping_mul(FNV_PRIME);
        }
        Self(digest)
    }

    pub const fn diagnostic_raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for GpuProgramSourceDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:016x}", self.0)
    }
}

fn validate_bounded_text(
    operation: &'static str,
    value: &str,
    max_bytes: usize,
    cause: GpuProgramSourceCause,
    correction: &'static str,
) -> Result<(), GpuProgramSourceError> {
    if value.is_empty()
        || value.len() > max_bytes
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        return Err(GpuProgramSourceError::invalid(
            operation,
            if value.is_empty() { "<empty>" } else { value },
            cause,
            correction,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_owner_identity_is_unique_and_exhaustion_is_explicit() {
        let first = GpuProgramSourceOwnerId::allocate().unwrap();
        let second = GpuProgramSourceOwnerId::allocate().unwrap();
        assert_ne!(first, second);
        assert_ne!(first.diagnostic_raw(), 0);

        let isolated = AtomicU64::new(u64::MAX);
        assert!(allocate_source_owner_id(&isolated).is_ok());
        assert_eq!(
            allocate_source_owner_id(&isolated).unwrap_err().cause(),
            GpuProgramSourceCause::InvalidSourceOwner
        );
    }

    #[test]
    fn source_key_revision_and_provenance_reject_ambiguous_values() {
        assert_eq!(
            GpuProgramSourceRevision::try_from_raw(0)
                .unwrap_err()
                .cause(),
            GpuProgramSourceCause::InvalidSourceRevision
        );
        for invalid in ["", " leading", "trailing ", "line\nbreak"] {
            assert_eq!(
                GpuProgramSourceKey::new(invalid).unwrap_err().cause(),
                GpuProgramSourceCause::InvalidSourceKey
            );
        }
        assert_eq!(
            GpuProgramSourceProvenance::new("", None)
                .unwrap_err()
                .cause(),
            GpuProgramSourceCause::InvalidProvenance
        );
    }

    #[test]
    fn digest_is_deterministic() {
        let left = GpuProgramSourceDigest::from_canonical_wgsl("abc");
        let same = GpuProgramSourceDigest::from_canonical_wgsl("abc");
        let right = GpuProgramSourceDigest::from_canonical_wgsl("abd");
        assert_eq!(left, same);
        assert_ne!(left, right);
    }
}
