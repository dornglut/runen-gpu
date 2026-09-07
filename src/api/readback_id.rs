use core::fmt;
use core::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

/// Opaque process-local correlation identity for one logical GPU readback request.
///
/// This identity has no persistence, replay, wire, ABI, cache, or content-identity meaning.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuReadbackId(NonZeroU64);

impl GpuReadbackId {
    pub fn allocate() -> Result<Self, GpuReadbackIdAllocationError> {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let value = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                (current != 0).then_some(if current == u64::MAX { 0 } else { current + 1 })
            })
            .map_err(|_| GpuReadbackIdAllocationError::Exhausted)?;
        let value = NonZeroU64::new(value).ok_or(GpuReadbackIdAllocationError::Exhausted)?;
        Ok(Self(value))
    }
}

impl fmt::Debug for GpuReadbackId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("GpuReadbackId")
            .field(&self.0.get())
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuReadbackIdAllocationError {
    Exhausted,
}

impl fmt::Display for GpuReadbackIdAllocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exhausted => formatter.write_str("GPU readback identity space is exhausted"),
        }
    }
}

impl std::error::Error for GpuReadbackIdAllocationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_readback_ids_are_process_local_and_distinct() {
        let first = GpuReadbackId::allocate().unwrap();
        let second = GpuReadbackId::allocate().unwrap();
        assert_ne!(first, second);
    }
}
