use core::{
    fmt,
    hash::{Hash, Hasher},
    num::NonZeroU64,
};
use std::sync::Arc;

/// Fragment-local opaque work identity.
///
/// ```compile_fail
/// use runen_gpu::GpuWorkNodeId;
///
/// let _ = GpuWorkNodeId::from_raw(1);
/// ```
#[derive(Clone)]
pub struct GpuWorkNodeId {
    pub(super) fragment_identity: Arc<()>,
    pub(super) local: NonZeroU64,
}

impl GpuWorkNodeId {
    pub(super) fn new(fragment_identity: &Arc<()>, local: NonZeroU64) -> Self {
        Self {
            fragment_identity: Arc::clone(fragment_identity),
            local,
        }
    }

    pub const fn diagnostic_local(&self) -> u64 {
        self.local.get()
    }

    pub(super) fn belongs_to(&self, fragment_identity: &Arc<()>) -> bool {
        Arc::ptr_eq(&self.fragment_identity, fragment_identity)
    }
}

impl fmt::Debug for GpuWorkNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuWorkNodeId")
            .field("local", &self.local)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for GpuWorkNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "local-node:{}", self.local)
    }
}

impl PartialEq for GpuWorkNodeId {
    fn eq(&self, other: &Self) -> bool {
        self.local == other.local && Arc::ptr_eq(&self.fragment_identity, &other.fragment_identity)
    }
}

impl Eq for GpuWorkNodeId {}

impl Hash for GpuWorkNodeId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.local.hash(state);
        Arc::as_ptr(&self.fragment_identity).hash(state);
    }
}

/// Deterministic process-local prepared identity. No raw reconstruction API is
/// exposed because this is not a persistence, replay, cache, or wire key.
///
/// ```compile_fail
/// use runen_gpu::GpuPreparedWorkNodeId;
///
/// let _ = GpuPreparedWorkNodeId::from_raw(0, 1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuPreparedWorkNodeId {
    fragment_ordinal: u32,
    local_node: NonZeroU64,
}

impl GpuPreparedWorkNodeId {
    pub(super) fn new(fragment_ordinal: u32, local_node: NonZeroU64) -> Self {
        Self {
            fragment_ordinal,
            local_node,
        }
    }

    pub const fn fragment_ordinal(self) -> u32 {
        self.fragment_ordinal
    }

    pub const fn local_node(self) -> u64 {
        self.local_node.get()
    }
}

impl fmt::Display for GpuPreparedWorkNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.fragment_ordinal, self.local_node)
    }
}
