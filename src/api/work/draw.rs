use super::super::{
    GpuBufferAccess, GpuBufferAccessKind, GpuBufferHandle, GpuBufferRange, GpuWorkOperationCause,
    GpuWorkOperationError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuDrawRange {
    first: u32,
    count: u32,
}

impl GpuDrawRange {
    pub fn new(first: u32, count: u32) -> Result<Self, GpuWorkOperationError> {
        if count == 0 {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU draw range",
                "draw",
                None,
                GpuWorkOperationCause::ZeroDrawCount,
                "provide a nonzero draw count",
            ));
        }
        first.checked_add(count).ok_or_else(|| {
            GpuWorkOperationError::invalid(
                "construct GPU draw range",
                "draw",
                None,
                GpuWorkOperationCause::InvalidDraw,
                "reduce the first element or count",
            )
        })?;
        Ok(Self { first, count })
    }

    pub const fn first(self) -> u32 {
        self.first
    }

    pub const fn count(self) -> u32 {
        self.count
    }

    pub const fn end(self) -> u32 {
        self.first + self.count
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuDrawIntent {
    Direct {
        vertices: GpuDrawRange,
        instances: GpuDrawRange,
    },
    Indexed {
        indices: GpuDrawRange,
        base_vertex: i32,
        instances: GpuDrawRange,
    },
    Indirect {
        arguments: GpuBufferHandle,
        range: GpuBufferRange,
        indexed: bool,
    },
}

impl GpuDrawIntent {
    pub fn direct(vertices: GpuDrawRange, instances: GpuDrawRange) -> Self {
        Self::Direct {
            vertices,
            instances,
        }
    }

    pub fn indexed(indices: GpuDrawRange, base_vertex: i32, instances: GpuDrawRange) -> Self {
        Self::Indexed {
            indices,
            base_vertex,
            instances,
        }
    }

    pub fn indirect(
        arguments: &GpuBufferHandle,
        range: GpuBufferRange,
        indexed: bool,
    ) -> Result<Self, GpuWorkOperationError> {
        let expected_size = if indexed { 20 } else { 16 };
        if !range.offset().is_multiple_of(4) || range.size() != expected_size {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU indirect draw intent",
                arguments.descriptor().common().label().as_str(),
                Some(arguments.diagnostic_identity()),
                GpuWorkOperationCause::InvalidDraw,
                "use one four-byte-aligned direct (16-byte) or indexed (20-byte) argument record",
            ));
        }
        GpuBufferAccess::new(arguments, range, GpuBufferAccessKind::IndirectRead).map_err(
            |source| {
                GpuWorkOperationError::from_access(
                    "construct GPU indirect draw intent",
                    arguments.descriptor().common().label().as_str(),
                    GpuWorkOperationCause::InvalidDraw,
                    "declare Indirect usage and a checked argument record",
                    source,
                )
            },
        )?;
        Ok(Self::Indirect {
            arguments: arguments.clone(),
            range,
            indexed,
        })
    }

    pub const fn is_indexed(&self) -> bool {
        matches!(
            self,
            Self::Indexed { .. } | Self::Indirect { indexed: true, .. }
        )
    }

    pub fn derived_access(&self) -> Result<Option<GpuBufferAccess>, GpuWorkOperationError> {
        match self {
            Self::Indirect {
                arguments, range, ..
            } => GpuBufferAccess::new(arguments, *range, GpuBufferAccessKind::IndirectRead)
                .map(Some)
                .map_err(|source| {
                    GpuWorkOperationError::from_access(
                        "derive GPU indirect draw access",
                        arguments.descriptor().common().label().as_str(),
                        GpuWorkOperationCause::OperationAccessContradiction,
                        "construct indirect draws through the checked constructor",
                        source,
                    )
                }),
            Self::Direct { .. } | Self::Indexed { .. } => Ok(None),
        }
    }
}
