use super::super::diagnostics::{GpuProgramSourceCause, GpuProgramSourceError};
use super::{GpuProgramSourceDigest, GpuProgramSourceIdentity, GpuProgramSourceProvenance};
use core::fmt;
use core::hash::Hash;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug)]
struct GpuAdmittedProgramSourceInner {
    identity: GpuProgramSourceIdentity,
    canonical_wgsl: Arc<str>,
    digest: GpuProgramSourceDigest,
    provenance: GpuProgramSourceProvenance,
}

/// Immutable admitted source retained by descriptors and later realization records.
#[derive(Clone)]
pub struct GpuAdmittedProgramSource(Arc<GpuAdmittedProgramSourceInner>);

impl GpuAdmittedProgramSource {
    pub fn identity(&self) -> &GpuProgramSourceIdentity {
        &self.0.identity
    }

    pub fn canonical_wgsl(&self) -> &str {
        self.0.canonical_wgsl.as_ref()
    }

    pub fn digest(&self) -> GpuProgramSourceDigest {
        self.0.digest
    }

    pub fn provenance(&self) -> &GpuProgramSourceProvenance {
        &self.0.provenance
    }

    pub fn is_same_record(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl fmt::Debug for GpuAdmittedProgramSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuAdmittedProgramSource")
            .field("identity", &self.0.identity)
            .field("digest", &self.0.digest)
            .field("source_bytes", &self.0.canonical_wgsl.len())
            .field("provenance", &self.0.provenance)
            .finish()
    }
}

impl PartialEq for GpuAdmittedProgramSource {
    fn eq(&self, other: &Self) -> bool {
        self.0.identity == other.0.identity && self.0.canonical_wgsl == other.0.canonical_wgsl
    }
}

impl Eq for GpuAdmittedProgramSource {}

impl Hash for GpuAdmittedProgramSource {
    fn hash<State: core::hash::Hasher>(&self, state: &mut State) {
        self.0.identity.hash(state);
        self.0.canonical_wgsl.hash(state);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuProgramSourceRegistryStats {
    retained_records: usize,
    retained_source_bytes: usize,
    max_records: usize,
    max_retained_source_bytes: usize,
}

impl GpuProgramSourceRegistryStats {
    pub const fn retained_records(self) -> usize {
        self.retained_records
    }

    pub const fn retained_source_bytes(self) -> usize {
        self.retained_source_bytes
    }

    pub const fn max_records(self) -> usize {
        self.max_records
    }

    pub const fn max_retained_source_bytes(self) -> usize {
        self.max_retained_source_bytes
    }
}

/// Bounded process-local owner of admitted source consistency.
///
/// The registry retains no internal callback and invokes no consumer code while
/// mutating its map. Accepted descriptors keep records alive through cloned
/// `GpuAdmittedProgramSource` values.
pub struct GpuProgramSourceRegistry {
    max_records: usize,
    max_retained_source_bytes: usize,
    retained_source_bytes: usize,
    records: BTreeMap<GpuProgramSourceIdentity, Arc<GpuAdmittedProgramSourceInner>>,
}

impl fmt::Debug for GpuProgramSourceRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuProgramSourceRegistry")
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

impl GpuProgramSourceRegistry {
    pub fn new(
        max_records: usize,
        max_retained_source_bytes: usize,
    ) -> Result<Self, GpuProgramSourceError> {
        if max_records == 0 {
            return Err(GpuProgramSourceError::invalid(
                "construct GPU program source registry",
                "max_records=0",
                GpuProgramSourceCause::SourceAdmissionCapacityExceeded,
                "provide a nonzero maximum record count",
            ));
        }
        if max_retained_source_bytes == 0 {
            return Err(GpuProgramSourceError::invalid(
                "construct GPU program source registry",
                "max_retained_source_bytes=0",
                GpuProgramSourceCause::SourceAdmissionCapacityExceeded,
                "provide a nonzero retained canonical-source byte limit",
            ));
        }
        Ok(Self {
            max_records,
            max_retained_source_bytes,
            retained_source_bytes: 0,
            records: BTreeMap::new(),
        })
    }

    pub fn admit_wgsl(
        &mut self,
        identity: GpuProgramSourceIdentity,
        canonical_wgsl: impl Into<String>,
        provenance: GpuProgramSourceProvenance,
    ) -> Result<GpuAdmittedProgramSource, GpuProgramSourceError> {
        let canonical_wgsl = canonical_wgsl.into();
        if canonical_wgsl.trim().is_empty() {
            return Err(GpuProgramSourceError::invalid(
                "admit canonical GPU program source",
                identity.diagnostic_label(),
                GpuProgramSourceCause::EmptyCanonicalWgsl,
                "provide non-empty canonical WGSL source text",
            ));
        }

        let attempted_digest = GpuProgramSourceDigest::from_canonical_wgsl(&canonical_wgsl);
        if let Some(existing) = self.records.get(&identity) {
            if existing.canonical_wgsl.as_ref() == canonical_wgsl.as_str() {
                return Ok(GpuAdmittedProgramSource(Arc::clone(existing)));
            }
            return Err(GpuProgramSourceError::revision_conflict(
                identity,
                existing.digest,
                attempted_digest,
            ));
        }

        let attempted_source_bytes = canonical_wgsl.len();
        if !self.fits_after_collecting_unretained(attempted_source_bytes) {
            return Err(GpuProgramSourceError::capacity_exceeded(
                identity.diagnostic_label(),
                self.max_records,
                self.max_retained_source_bytes,
                self.records.len(),
                self.retained_source_bytes,
                attempted_source_bytes,
            ));
        }

        if self.current_bounds_would_be_exceeded(attempted_source_bytes) {
            self.collect_unretained();
        }

        let source = Arc::new(GpuAdmittedProgramSourceInner {
            identity: identity.clone(),
            canonical_wgsl: Arc::from(canonical_wgsl),
            digest: attempted_digest,
            provenance,
        });
        self.records.insert(identity, Arc::clone(&source));
        self.retained_source_bytes += attempted_source_bytes;
        Ok(GpuAdmittedProgramSource(source))
    }

    /// Drops lookup-only records that no accepted descriptor or realization retains.
    pub fn collect_unretained(&mut self) -> usize {
        let before = self.records.len();
        let mut reclaimed_source_bytes = 0usize;
        self.records.retain(|_, record| {
            if Arc::strong_count(record) == 1 {
                reclaimed_source_bytes += record.canonical_wgsl.len();
                false
            } else {
                true
            }
        });
        self.retained_source_bytes -= reclaimed_source_bytes;
        before - self.records.len()
    }

    pub fn get(&self, identity: &GpuProgramSourceIdentity) -> Option<GpuAdmittedProgramSource> {
        self.records
            .get(identity)
            .map(|record| GpuAdmittedProgramSource(Arc::clone(record)))
    }

    pub fn stats(&self) -> GpuProgramSourceRegistryStats {
        GpuProgramSourceRegistryStats {
            retained_records: self.records.len(),
            retained_source_bytes: self.retained_source_bytes,
            max_records: self.max_records,
            max_retained_source_bytes: self.max_retained_source_bytes,
        }
    }

    fn current_bounds_would_be_exceeded(&self, attempted_source_bytes: usize) -> bool {
        self.records.len() >= self.max_records
            || self
                .retained_source_bytes
                .checked_add(attempted_source_bytes)
                .is_none_or(|bytes| bytes > self.max_retained_source_bytes)
    }

    fn fits_after_collecting_unretained(&self, attempted_source_bytes: usize) -> bool {
        let mut live_records = 0usize;
        let mut live_source_bytes = Some(0usize);
        for record in self.records.values() {
            if Arc::strong_count(record) > 1 {
                live_records += 1;
                live_source_bytes = live_source_bytes
                    .and_then(|bytes| bytes.checked_add(record.canonical_wgsl.len()));
            }
        }
        live_records
            .checked_add(1)
            .is_some_and(|count| count <= self.max_records)
            && live_source_bytes
                .and_then(|bytes| bytes.checked_add(attempted_source_bytes))
                .is_some_and(|bytes| bytes <= self.max_retained_source_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuProgramSourceKey, GpuProgramSourceOwnerId, GpuProgramSourceRevision};

    fn identity(
        owner: GpuProgramSourceOwnerId,
        key: &str,
        revision: u64,
    ) -> GpuProgramSourceIdentity {
        GpuProgramSourceIdentity::new(
            owner,
            GpuProgramSourceKey::new(key).unwrap(),
            GpuProgramSourceRevision::try_from_raw(revision).unwrap(),
        )
    }

    fn provenance(detail: Option<&str>) -> GpuProgramSourceProvenance {
        GpuProgramSourceProvenance::new("source-test", detail.map(str::to_owned)).unwrap()
    }

    #[test]
    fn admission_is_idempotent_for_identical_source_and_ignores_diagnostic_provenance() {
        let mut registry = GpuProgramSourceRegistry::new(4, 4096).unwrap();
        let owner = GpuProgramSourceOwnerId::allocate().unwrap();
        let identity = identity(owner, "compute.particles", 7);
        let first = registry
            .admit_wgsl(
                identity.clone(),
                "@compute @workgroup_size(1) fn main() {}",
                provenance(Some("first path")),
            )
            .unwrap();
        let second = registry
            .admit_wgsl(
                identity,
                "@compute @workgroup_size(1) fn main() {}",
                provenance(Some("rediscovered path")),
            )
            .unwrap();

        assert!(first.is_same_record(&second));
        assert_eq!(first.provenance().detail(), Some("first path"));
        assert_eq!(registry.stats().retained_records(), 1);
    }

    #[test]
    fn conflicting_source_revision_is_rejected_without_partial_publication() {
        let mut registry = GpuProgramSourceRegistry::new(4, 4096).unwrap();
        let owner = GpuProgramSourceOwnerId::allocate().unwrap();
        let identity = identity(owner, "compute.particles", 7);
        let accepted = registry
            .admit_wgsl(
                identity.clone(),
                "@compute @workgroup_size(1) fn main() {}",
                provenance(None),
            )
            .unwrap();
        let before = registry.stats();
        let error = registry
            .admit_wgsl(
                identity.clone(),
                "@compute @workgroup_size(2) fn main() {}",
                provenance(None),
            )
            .unwrap_err();

        assert_eq!(error.cause(), GpuProgramSourceCause::SourceRevisionConflict);
        assert_eq!(registry.stats(), before);
        assert!(registry.get(&identity).unwrap().is_same_record(&accepted));
    }

    #[test]
    fn live_records_are_not_evicted_under_capacity_pressure() {
        let source = "@compute @workgroup_size(1) fn main() {}";
        let mut registry = GpuProgramSourceRegistry::new(1, source.len()).unwrap();
        let owner = GpuProgramSourceOwnerId::allocate().unwrap();
        let retained = registry
            .admit_wgsl(identity(owner, "first", 1), source, provenance(None))
            .unwrap();
        let error = registry
            .admit_wgsl(identity(owner, "second", 1), source, provenance(None))
            .unwrap_err();

        assert_eq!(
            error.cause(),
            GpuProgramSourceCause::SourceAdmissionCapacityExceeded
        );
        assert_eq!(registry.stats().retained_records(), 1);
        drop(retained);
        assert_eq!(registry.collect_unretained(), 1);
        assert_eq!(registry.stats().retained_records(), 0);
        registry
            .admit_wgsl(identity(owner, "second", 1), source, provenance(None))
            .unwrap();
    }

    #[test]
    fn dead_revisions_can_exceed_historical_capacity_over_time() {
        let mut registry = GpuProgramSourceRegistry::new(2, 4096).unwrap();
        let owner = GpuProgramSourceOwnerId::allocate().unwrap();

        for revision in 1..=6 {
            let source = registry
                .admit_wgsl(
                    identity(owner, "compute.hot-reload", revision),
                    format!("@compute @workgroup_size({revision}) fn main() {{}}"),
                    provenance(None),
                )
                .unwrap();
            drop(source);
        }

        assert!(registry.stats().retained_records() <= 2);
        assert_eq!(registry.collect_unretained(), 2);
        assert_eq!(registry.stats().retained_records(), 0);
    }

    #[test]
    fn failed_capacity_admission_does_not_collect_existing_lookup_state() {
        let source = "@compute @workgroup_size(1) fn main() {}";
        let mut registry = GpuProgramSourceRegistry::new(2, source.len()).unwrap();
        let owner = GpuProgramSourceOwnerId::allocate().unwrap();
        registry
            .admit_wgsl(identity(owner, "first", 1), source, provenance(None))
            .unwrap();
        let before = registry.stats();

        let error = registry
            .admit_wgsl(
                identity(owner, "too-large", 1),
                format!("{source} extra"),
                provenance(None),
            )
            .unwrap_err();

        assert_eq!(
            error.cause(),
            GpuProgramSourceCause::SourceAdmissionCapacityExceeded
        );
        assert_eq!(registry.stats(), before);
    }
}
