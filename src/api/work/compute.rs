use super::super::{
    GpuComputePipelineDescriptor, GpuDispatchIntent, GpuRuntimeBindingSet, GpuTimestampWrites,
    GpuWorkOperationCause, GpuWorkOperationError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuDispatchSize {
    x: u32,
    y: u32,
    z: u32,
}

impl GpuDispatchSize {
    /// Constructs an exact logical direct-dispatch size.
    ///
    /// Zero dimensions are valid logical no-op dimensions. Device-specific maximums are checked
    /// later against the admitted context, so construction itself has no failure condition.
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    pub const fn x(self) -> u32 {
        self.x
    }
    pub const fn y(self) -> u32 {
        self.y
    }
    pub const fn z(self) -> u32 {
        self.z
    }
    pub const fn as_array(self) -> [u32; 3] {
        [self.x, self.y, self.z]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuComputeOperation {
    pipeline: GpuComputePipelineDescriptor,
    bindings: GpuRuntimeBindingSet,
    dispatch: GpuDispatchIntent,
    timestamp_writes: Option<GpuTimestampWrites>,
}

impl GpuComputeOperation {
    pub fn new(
        pipeline: GpuComputePipelineDescriptor,
        bindings: GpuRuntimeBindingSet,
        dispatch: GpuDispatchIntent,
    ) -> Result<Self, GpuWorkOperationError> {
        if pipeline.layout() != bindings.layout() {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU compute operation",
                "pipeline bindings",
                None,
                GpuWorkOperationCause::OperationAccessContradiction,
                "use a runtime binding set constructed for the exact compute pipeline layout",
            ));
        }
        Ok(Self {
            pipeline,
            bindings,
            dispatch,
            timestamp_writes: None,
        })
    }

    pub fn with_timestamp_writes(mut self, timestamp_writes: GpuTimestampWrites) -> Self {
        self.timestamp_writes = Some(timestamp_writes);
        self
    }

    pub fn pipeline(&self) -> &GpuComputePipelineDescriptor {
        &self.pipeline
    }

    pub fn bindings(&self) -> &GpuRuntimeBindingSet {
        &self.bindings
    }

    pub fn dispatch(&self) -> &GpuDispatchIntent {
        &self.dispatch
    }

    pub fn timestamp_writes(&self) -> Option<&GpuTimestampWrites> {
        self.timestamp_writes.as_ref()
    }
}
