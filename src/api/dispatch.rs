use super::{
    GpuBufferAccess, GpuBufferAccessKind, GpuBufferHandle, GpuBufferRange, GpuDispatchSize,
    GpuLimits, GpuWorkOperationCause, GpuWorkOperationError,
};

const INDIRECT_DISPATCH_ARGUMENT_BYTES: u64 = 12;
const INDIRECT_DISPATCH_ALIGNMENT: u64 = 4;

/// Complete backend-neutral compute-dispatch intent.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuDispatchIntent {
    Direct(GpuDispatchSize),
    Indirect(GpuBufferAccess),
}

impl GpuDispatchIntent {
    pub const fn direct(size: GpuDispatchSize) -> Self {
        Self::Direct(size)
    }

    pub fn indirect(
        arguments: &GpuBufferHandle,
        offset: u64,
    ) -> Result<Self, GpuWorkOperationError> {
        if !offset.is_multiple_of(INDIRECT_DISPATCH_ALIGNMENT) {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU indirect dispatch intent",
                arguments.descriptor().common().label().as_str(),
                Some(arguments.diagnostic_identity()),
                GpuWorkOperationCause::OperationAccessContradiction,
                "use a four-byte-aligned indirect dispatch argument offset",
            ));
        }
        let range = GpuBufferRange::new(arguments, offset, INDIRECT_DISPATCH_ARGUMENT_BYTES)
            .map_err(|source| {
                GpuWorkOperationError::from_access(
                    "construct GPU indirect dispatch intent",
                    arguments.descriptor().common().label().as_str(),
                    GpuWorkOperationCause::OperationAccessContradiction,
                    "keep the exact 12-byte indirect dispatch record inside an Indirect buffer",
                    source,
                )
            })?;
        let access = GpuBufferAccess::new(arguments, range, GpuBufferAccessKind::IndirectRead)
            .map_err(|source| {
                GpuWorkOperationError::from_access(
                    "construct GPU indirect dispatch intent",
                    arguments.descriptor().common().label().as_str(),
                    GpuWorkOperationCause::OperationAccessContradiction,
                    "declare Indirect usage for the exact dispatch argument record",
                    source,
                )
            })?;
        Ok(Self::Indirect(access))
    }

    pub const fn direct_size(&self) -> Option<GpuDispatchSize> {
        match self {
            Self::Direct(size) => Some(*size),
            Self::Indirect(_) => None,
        }
    }

    pub const fn indirect_access(&self) -> Option<&GpuBufferAccess> {
        match self {
            Self::Direct(_) => None,
            Self::Indirect(access) => Some(access),
        }
    }

    pub const fn is_indirect(&self) -> bool {
        matches!(self, Self::Indirect(_))
    }

    pub(crate) fn validate_limits(&self, limits: GpuLimits) -> Result<(), GpuWorkOperationError> {
        let Self::Direct(size) = self else {
            return Ok(());
        };
        let maximum = limits.max_compute_workgroups_per_dimension();
        if size.as_array().into_iter().any(|value| value > maximum) {
            return Err(GpuWorkOperationError::invalid(
                "validate GPU direct dispatch against admitted device limits",
                "dispatch size",
                None,
                GpuWorkOperationCause::MechanicalCapabilityContradiction,
                "keep every direct workgroup dimension within the admitted compute-workgroups-per-dimension limit",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GpuBufferDescriptor, GpuBufferInitialization, GpuBufferUsage, GpuBufferUsages,
        GpuMemoryIntent, GpuReconstruction, GpuResourceCommon, GpuResourceLabel,
        GpuResourceLifetime, GpuResourceProvenance, GpuWorkResourceIdAllocator,
    };
    use core::num::NonZeroU64;

    fn limits(max_compute_workgroups_per_dimension: u32) -> GpuLimits {
        GpuLimits::new(
            1,
            1,
            1,
            1,
            1,
            8192,
            4,
            24,
            0,
            0,
            max_compute_workgroups_per_dimension,
        )
        .unwrap()
    }

    fn indirect_buffer(size: u64) -> GpuBufferHandle {
        let label = GpuResourceLabel::new("indirect-dispatch").unwrap();
        let common = GpuResourceCommon::owned(
            label.clone(),
            GpuResourceLifetime::Transient,
            GpuMemoryIntent::Device,
            GpuReconstruction::SourceBacked,
            GpuResourceProvenance::new(label.clone(), None, None),
        )
        .unwrap();
        let usages = GpuBufferUsages::new(&label, [GpuBufferUsage::Indirect]).unwrap();
        let descriptor =
            GpuBufferDescriptor::new(common, size, usages, GpuBufferInitialization::Uninitialized)
                .unwrap();
        let mut allocator =
            GpuWorkResourceIdAllocator::for_owner_scope(NonZeroU64::new(23).unwrap());
        allocator.allocate_buffer_handle(descriptor).unwrap()
    }

    #[test]
    fn direct_dispatch_is_logical_and_explicit_limit_validation_is_deterministic() {
        let zero = GpuDispatchSize::new(0, 4, 1);
        assert!(
            GpuDispatchIntent::direct(zero)
                .validate_limits(limits(8))
                .is_ok()
        );

        let too_large = GpuDispatchSize::new(9, 1, 1);
        assert_eq!(
            GpuDispatchIntent::direct(too_large)
                .validate_limits(limits(8))
                .unwrap_err()
                .cause(),
            GpuWorkOperationCause::MechanicalCapabilityContradiction
        );
    }

    #[test]
    fn indirect_dispatch_owns_one_exact_aligned_twelve_byte_read() {
        let buffer = indirect_buffer(32);
        let intent = GpuDispatchIntent::indirect(&buffer, 4).unwrap();
        let access = intent.indirect_access().unwrap();
        assert_eq!(access.kind(), GpuBufferAccessKind::IndirectRead);
        assert_eq!(access.range().offset(), 4);
        assert_eq!(access.range().size(), 12);
        assert!(GpuDispatchIntent::indirect(&buffer, 2).is_err());
        assert!(GpuDispatchIntent::indirect(&buffer, 24).is_err());
    }
}
