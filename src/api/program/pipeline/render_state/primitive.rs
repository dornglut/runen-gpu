use super::super::super::contract_diagnostics::{GpuProgramContractCause, GpuProgramContractError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuPrimitiveTopology {
    TriangleList,
    TriangleStrip,
    LineList,
    LineStrip,
    PointList,
}

impl GpuPrimitiveTopology {
    pub const fn is_strip(self) -> bool {
        matches!(self, Self::TriangleStrip | Self::LineStrip)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuFrontFace {
    CounterClockwise,
    Clockwise,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuCullMode {
    None,
    Front,
    Back,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuIndexFormat {
    Uint16,
    Uint32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuPrimitiveStateDescriptor {
    topology: GpuPrimitiveTopology,
    strip_index_format: Option<GpuIndexFormat>,
    front_face: GpuFrontFace,
    cull_mode: GpuCullMode,
}

impl GpuPrimitiveStateDescriptor {
    pub fn new(
        topology: GpuPrimitiveTopology,
        strip_index_format: Option<GpuIndexFormat>,
        front_face: GpuFrontFace,
        cull_mode: GpuCullMode,
    ) -> Result<Self, GpuProgramContractError> {
        if strip_index_format.is_some() && !topology.is_strip() {
            return Err(GpuProgramContractError::invalid(
                "construct GPU primitive state",
                format!("topology={topology:?}, strip_index_format={strip_index_format:?}"),
                GpuProgramContractCause::RenderPrimitiveStateInvalid,
                "declare a strip index format only for triangle-strip or line-strip topology",
            ));
        }

        Ok(Self {
            topology,
            strip_index_format,
            front_face,
            cull_mode,
        })
    }

    pub const fn topology(self) -> GpuPrimitiveTopology {
        self.topology
    }

    pub const fn strip_index_format(self) -> Option<GpuIndexFormat> {
        self.strip_index_format
    }

    pub const fn front_face(self) -> GpuFrontFace {
        self.front_face
    }

    pub const fn cull_mode(self) -> GpuCullMode {
        self.cull_mode
    }
}

impl Default for GpuPrimitiveStateDescriptor {
    fn default() -> Self {
        Self {
            topology: GpuPrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: GpuFrontFace::CounterClockwise,
            cull_mode: GpuCullMode::None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuMultisampleStateDescriptor {
    sample_count: u32,
    sample_mask: u64,
    alpha_to_coverage_enabled: bool,
}

impl GpuMultisampleStateDescriptor {
    pub fn new(
        sample_count: u32,
        sample_mask: u64,
        alpha_to_coverage_enabled: bool,
    ) -> Result<Self, GpuProgramContractError> {
        if sample_count == 0 || !sample_count.is_power_of_two() || sample_count > u64::BITS {
            return Err(invalid_multisample_state(
                format!("sample_count={sample_count}"),
                "use a nonzero power-of-two sample count no greater than 64",
            ));
        }

        let valid_mask = if sample_count == u64::BITS {
            u64::MAX
        } else {
            (1_u64 << sample_count) - 1
        };
        if sample_mask & !valid_mask != 0 {
            return Err(invalid_multisample_state(
                format!("sample_count={sample_count}, sample_mask=0x{sample_mask:016x}"),
                "set mask bits only for samples represented by the declared sample count",
            ));
        }

        if alpha_to_coverage_enabled && sample_count == 1 {
            return Err(invalid_multisample_state(
                "sample_count=1, alpha_to_coverage_enabled=true",
                "enable alpha-to-coverage only for multisampled state",
            ));
        }

        Ok(Self {
            sample_count,
            sample_mask,
            alpha_to_coverage_enabled,
        })
    }

    pub const fn sample_count(self) -> u32 {
        self.sample_count
    }

    pub const fn sample_mask(self) -> u64 {
        self.sample_mask
    }

    pub const fn alpha_to_coverage_enabled(self) -> bool {
        self.alpha_to_coverage_enabled
    }
}

impl Default for GpuMultisampleStateDescriptor {
    fn default() -> Self {
        Self {
            sample_count: 1,
            sample_mask: 1,
            alpha_to_coverage_enabled: false,
        }
    }
}

fn invalid_multisample_state(
    label: impl Into<String>,
    correction: &'static str,
) -> GpuProgramContractError {
    GpuProgramContractError::invalid(
        "construct GPU multisample state",
        label,
        GpuProgramContractCause::RenderMultisampleStateInvalid,
        correction,
    )
}
