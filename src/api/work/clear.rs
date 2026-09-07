use super::super::{
    GpuBufferAccess, GpuBufferAccessKind, GpuWorkOperationCause, GpuWorkOperationError,
};
use super::GpuBufferRegion;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuClearOperation {
    BufferZero(GpuBufferRegion),
}

impl GpuClearOperation {
    pub fn buffer_zero(region: GpuBufferRegion) -> Result<Self, GpuWorkOperationError> {
        GpuBufferAccess::new(
            region.buffer(),
            region.range(),
            GpuBufferAccessKind::CopyDestination,
        )
        .map_err(|source| {
            GpuWorkOperationError::from_access(
                "construct GPU buffer-zero operation",
                region.buffer().descriptor().common().label().as_str(),
                GpuWorkOperationCause::InvalidBufferZero,
                "declare CopyDestination usage and a checked nonempty buffer range",
                source,
            )
        })?;
        Ok(Self::BufferZero(region))
    }
}
