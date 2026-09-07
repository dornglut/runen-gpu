#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuStorageBufferAccess {
    ReadOnly,
    ReadWrite,
}
