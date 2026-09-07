use super::diagnostics::{GpuContextRequestError, GpuContextRequestErrorCategory};
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

/// Opaque, nonzero, process-local context identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuContextId(pub(super) NonZeroU64);

impl GpuContextId {
    pub const fn is_nonzero(self) -> bool {
        self.0.get() != 0
    }

    #[cfg(test)]
    pub(crate) const fn test_value(value: NonZeroU64) -> Self {
        Self(value)
    }
}

/// Opaque device generation for one process-local logical GPU context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuDeviceGeneration(pub(super) NonZeroU64);

impl GpuDeviceGeneration {
    pub const fn first() -> Self {
        Self(NonZeroU64::MIN)
    }

    pub(crate) fn next(self) -> Option<Self> {
        self.0
            .get()
            .checked_add(1)
            .and_then(NonZeroU64::new)
            .map(Self)
    }

    #[cfg(test)]
    pub(crate) const fn test_value(value: NonZeroU64) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuContextAffinity {
    pub(super) context: GpuContextId,
    pub(super) generation: GpuDeviceGeneration,
}

impl GpuContextAffinity {
    pub(crate) const fn from_context_generation(
        context: GpuContextId,
        generation: GpuDeviceGeneration,
    ) -> Self {
        Self {
            context,
            generation,
        }
    }

    pub const fn context(&self) -> GpuContextId {
        self.context
    }

    pub const fn generation(&self) -> GpuDeviceGeneration {
        self.generation
    }

    #[cfg(test)]
    pub(crate) const fn test_value(context: GpuContextId, generation: GpuDeviceGeneration) -> Self {
        Self::from_context_generation(context, generation)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuContextAffinityError {
    ForeignContext,
    StaleGeneration,
}

pub(crate) fn validate_affinity(
    expected: GpuContextAffinity,
    actual: GpuContextAffinity,
) -> Result<(), GpuContextAffinityError> {
    if actual.context != expected.context {
        Err(GpuContextAffinityError::ForeignContext)
    } else if actual.generation != expected.generation {
        Err(GpuContextAffinityError::StaleGeneration)
    } else {
        Ok(())
    }
}

/// The production allocator is deliberately isolated from test allocators.
#[derive(Debug)]
pub(crate) struct GpuContextIdAllocator {
    next: AtomicU64,
}

impl GpuContextIdAllocator {
    pub(crate) const fn new(next: u64) -> Self {
        Self {
            next: AtomicU64::new(next),
        }
    }

    pub(crate) fn allocate(&self) -> Result<GpuContextId, GpuContextRequestError> {
        let value = self
            .next
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                (current != 0).then_some(if current == u64::MAX { 0 } else { current + 1 })
            })
            .map_err(|_| {
                GpuContextRequestError::new(
                    GpuContextRequestErrorCategory::IdentityExhausted,
                    "context identifier allocator exhausted",
                )
            })?;
        Ok(GpuContextId(
            NonZeroU64::new(value).expect("context identifier allocator never returns zero"),
        ))
    }
}

static PRODUCTION_CONTEXT_IDS: GpuContextIdAllocator = GpuContextIdAllocator::new(1);

pub(crate) fn allocate_context_id() -> Result<GpuContextId, GpuContextRequestError> {
    PRODUCTION_CONTEXT_IDS.allocate()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolated_allocators_prove_nonzero_uniqueness_and_exhaustion_without_global_reset() {
        let first_allocator = GpuContextIdAllocator::new(1);
        let second_allocator = GpuContextIdAllocator::new(1);
        let first = first_allocator.allocate().unwrap();
        let second = first_allocator.allocate().unwrap();
        assert_ne!(first, second);
        assert_eq!(first, second_allocator.allocate().unwrap());

        let exhausted = GpuContextIdAllocator::new(u64::MAX);
        assert!(exhausted.allocate().is_ok());
        assert!(matches!(
            exhausted.allocate(),
            Err(error) if error.category() == GpuContextRequestErrorCategory::IdentityExhausted
        ));
    }

    #[test]
    fn device_generation_advances_monotonically_without_wraparound() {
        let first = GpuDeviceGeneration::first();
        let second = first.next().unwrap();
        assert!(second > first);
        assert_eq!(
            second,
            GpuDeviceGeneration::test_value(NonZeroU64::new(2).unwrap())
        );
        assert!(
            GpuDeviceGeneration::test_value(NonZeroU64::new(u64::MAX).unwrap())
                .next()
                .is_none()
        );
    }

    #[test]
    fn affinity_rejects_foreign_and_stale_values() {
        let allocator = GpuContextIdAllocator::new(1);
        let one = allocator.allocate().unwrap();
        let two = allocator.allocate().unwrap();
        let generation = GpuDeviceGeneration::first();
        assert_eq!(
            validate_affinity(
                GpuContextAffinity {
                    context: one,
                    generation,
                },
                GpuContextAffinity {
                    context: two,
                    generation,
                },
            ),
            Err(GpuContextAffinityError::ForeignContext)
        );
        assert_eq!(
            validate_affinity(
                GpuContextAffinity {
                    context: one,
                    generation,
                },
                GpuContextAffinity {
                    context: one,
                    generation: GpuDeviceGeneration::test_value(NonZeroU64::new(2).unwrap()),
                },
            ),
            Err(GpuContextAffinityError::StaleGeneration)
        );
    }
}
