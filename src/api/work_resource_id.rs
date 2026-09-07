use core::fmt;
use core::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct GpuWorkResourceOwnerScope(NonZeroU64);

/// Identifies one logical resource inside one bounded GPU-work owner scope.
///
/// The numeric diagnostic components are intentionally not stable persistence,
/// replay, network, wire, ABI, or cache identities.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuWorkResourceId {
    owner_scope: GpuWorkResourceOwnerScope,
    local: NonZeroU64,
}

impl GpuWorkResourceId {
    const fn from_parts(owner_scope: GpuWorkResourceOwnerScope, local: NonZeroU64) -> Self {
        Self { owner_scope, local }
    }

    /// Returns diagnostic-only owner and local components.
    ///
    /// No stable-format guarantee is attached to these values.
    pub const fn diagnostic_parts(self) -> (u64, u64) {
        (self.owner_scope.0.get(), self.local.get())
    }
}

impl fmt::Debug for GpuWorkResourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (owner_scope, local) = self.diagnostic_parts();
        f.debug_struct("GpuWorkResourceId")
            .field("owner_scope", &owner_scope)
            .field("local", &local)
            .finish()
    }
}

impl fmt::Display for GpuWorkResourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (owner_scope, local) = self.diagnostic_parts();
        write!(f, "{owner_scope}:{local}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuWorkResourceIdAllocationError {
    Exhausted,
}

impl fmt::Display for GpuWorkResourceIdAllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exhausted => f.write_str("GPU work-resource identity space is exhausted"),
        }
    }
}

impl std::error::Error for GpuWorkResourceIdAllocationError {}

/// Private process-local authority for logical GPU-work owner scopes.
///
/// A zero value is a terminal exhaustion sentinel only. It is never returned as
/// an owner scope.
#[derive(Debug)]
struct GpuWorkResourceOwnerScopeAllocator {
    next: AtomicU64,
}

impl GpuWorkResourceOwnerScopeAllocator {
    const fn new(next: NonZeroU64) -> Self {
        Self {
            next: AtomicU64::new(next.get()),
        }
    }

    fn allocate(&self) -> Result<GpuWorkResourceOwnerScope, GpuWorkResourceIdAllocationError> {
        let value = self
            .next
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                (current != 0).then_some(if current == u64::MAX { 0 } else { current + 1 })
            })
            .map_err(|_| GpuWorkResourceIdAllocationError::Exhausted)?;
        Ok(GpuWorkResourceOwnerScope(
            NonZeroU64::new(value).expect("owner-scope allocator never returns zero"),
        ))
    }
}

static PRODUCTION_OWNER_SCOPES: GpuWorkResourceOwnerScopeAllocator =
    GpuWorkResourceOwnerScopeAllocator::new(NonZeroU64::MIN);

/// Allocates logical GPU-work resource identities.
///
/// Production allocators acquire one opaque RunenGPU-owned owner scope lazily on
/// their first successful allocation. Callers cannot choose or inject that scope.
#[derive(Debug)]
pub struct GpuWorkResourceIdAllocator {
    owner_scope: Option<GpuWorkResourceOwnerScope>,
    next_local: Option<NonZeroU64>,
}

impl GpuWorkResourceIdAllocator {
    /// Creates a scope-free logical resource allocator.
    ///
    /// ```compile_fail
    /// use runen_gpu::GpuWorkResourceIdAllocator;
    /// use std::num::NonZeroU64;
    ///
    /// let _ = GpuWorkResourceIdAllocator::for_owner_scope(NonZeroU64::MIN);
    /// ```
    pub const fn new() -> Self {
        Self {
            owner_scope: None,
            next_local: NonZeroU64::new(1),
        }
    }

    pub fn allocate(&mut self) -> Result<GpuWorkResourceId, GpuWorkResourceIdAllocationError> {
        let local = self
            .next_local
            .ok_or(GpuWorkResourceIdAllocationError::Exhausted)?;
        let owner_scope = match self.owner_scope {
            Some(owner_scope) => owner_scope,
            None => PRODUCTION_OWNER_SCOPES.allocate()?,
        };

        self.owner_scope = Some(owner_scope);
        self.next_local = local.get().checked_add(1).and_then(NonZeroU64::new);
        Ok(GpuWorkResourceId::from_parts(owner_scope, local))
    }

    #[cfg(test)]
    pub(crate) const fn for_owner_scope(owner_scope: NonZeroU64) -> Self {
        Self {
            owner_scope: Some(GpuWorkResourceOwnerScope(owner_scope)),
            next_local: NonZeroU64::new(1),
        }
    }

    #[cfg(test)]
    pub(crate) const fn with_next_local_for_test(
        owner_scope: NonZeroU64,
        next_local: NonZeroU64,
    ) -> Self {
        Self {
            owner_scope: Some(GpuWorkResourceOwnerScope(owner_scope)),
            next_local: Some(next_local),
        }
    }
}

impl Default for GpuWorkResourceIdAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn nonzero(value: u64) -> NonZeroU64 {
        NonZeroU64::new(value).expect("test values are nonzero")
    }

    #[test]
    fn production_allocators_receive_distinct_retained_owner_scopes() {
        let mut first_allocator = GpuWorkResourceIdAllocator::new();
        let mut second_allocator = GpuWorkResourceIdAllocator::new();
        let first = first_allocator
            .allocate()
            .expect("first owner allocation should succeed");
        let second = second_allocator
            .allocate()
            .expect("second owner allocation should succeed");
        let next_first = first_allocator
            .allocate()
            .expect("second local allocation should succeed");

        assert_ne!(first.diagnostic_parts().0, 0);
        assert_ne!(second.diagnostic_parts().0, 0);
        assert_ne!(first.diagnostic_parts().0, second.diagnostic_parts().0);
        assert_eq!(first.diagnostic_parts().1, 1);
        assert_eq!(second.diagnostic_parts().1, 1);
        assert_eq!(first.diagnostic_parts().0, next_first.diagnostic_parts().0);
        assert_eq!(next_first.diagnostic_parts().1, 2);
    }

    #[test]
    fn isolated_owner_scope_allocator_is_monotonic_and_exhausts_without_reuse() {
        let allocator = GpuWorkResourceOwnerScopeAllocator::new(nonzero(u64::MAX));
        let terminal = allocator
            .allocate()
            .expect("maximum owner scope should allocate once");

        assert_eq!(terminal.0.get(), u64::MAX);
        assert_eq!(
            allocator.allocate(),
            Err(GpuWorkResourceIdAllocationError::Exhausted)
        );
        assert_eq!(
            allocator.allocate(),
            Err(GpuWorkResourceIdAllocationError::Exhausted)
        );
    }

    #[test]
    fn isolated_owner_scope_allocators_do_not_mutate_production_authority() {
        let mut first_production = GpuWorkResourceIdAllocator::new();
        let first = first_production
            .allocate()
            .expect("first production owner allocation should succeed");

        let isolated = GpuWorkResourceOwnerScopeAllocator::new(nonzero(1));
        assert_eq!(isolated.allocate().unwrap().0.get(), 1);

        let mut second_production = GpuWorkResourceIdAllocator::new();
        let second = second_production
            .allocate()
            .expect("second production owner allocation should succeed");

        assert_ne!(first.diagnostic_parts().0, second.diagnostic_parts().0);
    }

    #[test]
    fn gpu_work_resource_id_first_allocation_is_nonzero() {
        let mut allocator = GpuWorkResourceIdAllocator::new();
        let id = allocator
            .allocate()
            .expect("first allocation should succeed");

        assert_ne!(id.diagnostic_parts().0, 0);
        assert_eq!(id.diagnostic_parts().1, 1);
    }

    #[test]
    fn gpu_work_resource_id_allocation_is_monotonic_and_distinct() {
        let mut allocator = GpuWorkResourceIdAllocator::for_owner_scope(nonzero(7));
        let first = allocator
            .allocate()
            .expect("first allocation should succeed");
        let second = allocator
            .allocate()
            .expect("second allocation should succeed");

        assert!(first < second);
        assert_ne!(first, second);
        assert_eq!(first.diagnostic_parts(), (7, 1));
        assert_eq!(second.diagnostic_parts(), (7, 2));
    }

    #[test]
    fn gpu_work_resource_id_owner_scope_prevents_local_collisions() {
        let mut first_allocator = GpuWorkResourceIdAllocator::for_owner_scope(nonzero(7));
        let mut second_allocator = GpuWorkResourceIdAllocator::for_owner_scope(nonzero(8));
        let first = first_allocator
            .allocate()
            .expect("first owner allocation should succeed");
        let second = second_allocator
            .allocate()
            .expect("second owner allocation should succeed");

        assert_eq!(first.diagnostic_parts().1, second.diagnostic_parts().1);
        assert_ne!(first, second);
        assert_eq!(HashSet::from([first, second]).len(), 2);
        assert_ne!(format!("{first:?}"), format!("{second:?}"));
        assert_ne!(first.to_string(), second.to_string());
    }

    #[test]
    fn gpu_work_resource_id_exposes_no_arbitrary_scalar_construction() {
        let source =
            std::fs::read_to_string(file!()).expect("work-resource-id source must be readable");
        let identity_impl = source
            .split_once("impl GpuWorkResourceId {")
            .expect("identity implementation must exist")
            .1
            .split_once("impl fmt::Debug for GpuWorkResourceId")
            .expect("identity implementation must end before Debug")
            .0;
        let forbidden = [
            ["try_from", "_raw"].concat(),
            ["from", "_raw"].concat(),
            ["new", "_unchecked"].concat(),
            ["TryFrom", "<u64>"].concat(),
            ["From", "<u64>"].concat(),
            "pub fn new(".to_string(),
            "pub const fn new(".to_string(),
        ];

        for token in forbidden {
            assert!(
                !identity_impl.contains(&token),
                "arbitrary scalar constructor must not be exposed: {token}"
            );
        }
    }

    #[test]
    fn gpu_work_resource_id_exhaustion_is_explicit_without_reuse() {
        let mut allocator =
            GpuWorkResourceIdAllocator::with_next_local_for_test(nonzero(7), nonzero(u64::MAX));
        let last = allocator
            .allocate()
            .expect("maximum local value should allocate once");

        assert_eq!(last.diagnostic_parts(), (7, u64::MAX));
        assert_eq!(
            allocator.allocate(),
            Err(GpuWorkResourceIdAllocationError::Exhausted)
        );
        assert_eq!(
            allocator.allocate(),
            Err(GpuWorkResourceIdAllocationError::Exhausted)
        );
    }
}
