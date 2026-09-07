use super::super::super::contract_diagnostics::{GpuProgramContractCause, GpuProgramContractError};
use super::super::super::{
    GpuEntryPointName, GpuExpectedVertexInputSignature, GpuShaderIoLocation,
    GpuShaderIoScalarClass, GpuShaderIoValueType,
};

const VERTEX_ALIGNMENT: u64 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuVertexStepMode {
    Vertex,
    Instance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuVertexFormat {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Uint32,
    Uint32x2,
    Uint32x3,
    Uint32x4,
    Sint32,
    Sint32x2,
    Sint32x3,
    Sint32x4,
}

impl GpuVertexFormat {
    pub const fn size_bytes(self) -> u64 {
        match self {
            Self::Float32 | Self::Uint32 | Self::Sint32 => 4,
            Self::Float32x2 | Self::Uint32x2 | Self::Sint32x2 => 8,
            Self::Float32x3 | Self::Uint32x3 | Self::Sint32x3 => 12,
            Self::Float32x4 | Self::Uint32x4 | Self::Sint32x4 => 16,
        }
    }

    pub fn shader_io_type(self) -> GpuShaderIoValueType {
        let (class, width) = match self {
            Self::Float32 => (GpuShaderIoScalarClass::Float, 1),
            Self::Float32x2 => (GpuShaderIoScalarClass::Float, 2),
            Self::Float32x3 => (GpuShaderIoScalarClass::Float, 3),
            Self::Float32x4 => (GpuShaderIoScalarClass::Float, 4),
            Self::Uint32 => (GpuShaderIoScalarClass::Uint, 1),
            Self::Uint32x2 => (GpuShaderIoScalarClass::Uint, 2),
            Self::Uint32x3 => (GpuShaderIoScalarClass::Uint, 3),
            Self::Uint32x4 => (GpuShaderIoScalarClass::Uint, 4),
            Self::Sint32 => (GpuShaderIoScalarClass::Sint, 1),
            Self::Sint32x2 => (GpuShaderIoScalarClass::Sint, 2),
            Self::Sint32x3 => (GpuShaderIoScalarClass::Sint, 3),
            Self::Sint32x4 => (GpuShaderIoScalarClass::Sint, 4),
        };
        GpuShaderIoValueType::try_new(class, width)
            .expect("GPU vertex formats always map to valid shader IO")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuVertexAttribute {
    shader_location: u32,
    offset: u64,
    format: GpuVertexFormat,
}

impl GpuVertexAttribute {
    pub const fn new(shader_location: u32, offset: u64, format: GpuVertexFormat) -> Self {
        Self {
            shader_location,
            offset,
            format,
        }
    }

    pub const fn shader_location(self) -> u32 {
        self.shader_location
    }

    pub const fn offset(self) -> u64 {
        self.offset
    }

    pub const fn format(self) -> GpuVertexFormat {
        self.format
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuVertexBufferLayoutDescriptor {
    slot: u32,
    array_stride: u64,
    step_mode: GpuVertexStepMode,
    attributes: Vec<GpuVertexAttribute>,
}

impl GpuVertexBufferLayoutDescriptor {
    pub fn new(
        slot: u32,
        array_stride: u64,
        step_mode: GpuVertexStepMode,
        attributes: impl IntoIterator<Item = GpuVertexAttribute>,
    ) -> Result<Self, GpuProgramContractError> {
        if array_stride == 0 || !array_stride.is_multiple_of(VERTEX_ALIGNMENT) {
            return Err(invalid_vertex_state(
                format!("slot={slot}, array_stride={array_stride}"),
                "use a nonzero vertex stride aligned to four bytes",
            ));
        }

        let mut attributes = attributes.into_iter().collect::<Vec<_>>();
        if attributes.is_empty() {
            return Err(invalid_vertex_state(
                format!("slot={slot}"),
                "declare at least one vertex attribute or omit the buffer layout",
            ));
        }
        attributes.sort_by_key(|attribute| attribute.shader_location());
        if let Some(location) = duplicate_location(&attributes) {
            return Err(invalid_vertex_state(
                format!("slot={slot}, shader_location={location}"),
                "declare each shader location exactly once",
            ));
        }

        for attribute in &attributes {
            let range_is_valid = attribute.offset().is_multiple_of(VERTEX_ALIGNMENT)
                && attribute
                    .offset()
                    .checked_add(attribute.format().size_bytes())
                    .is_some_and(|end| end <= array_stride);
            if !range_is_valid {
                return Err(invalid_vertex_state(
                    format!(
                        "slot={slot}, shader_location={}, offset={}, format={:?}, array_stride={array_stride}",
                        attribute.shader_location(),
                        attribute.offset(),
                        attribute.format()
                    ),
                    "keep each aligned attribute range inside the vertex stride",
                ));
            }
        }

        Ok(Self {
            slot,
            array_stride,
            step_mode,
            attributes,
        })
    }

    pub const fn slot(&self) -> u32 {
        self.slot
    }

    pub const fn array_stride(&self) -> u64 {
        self.array_stride
    }

    pub const fn step_mode(&self) -> GpuVertexStepMode {
        self.step_mode
    }

    pub fn attributes(&self) -> impl ExactSizeIterator<Item = &GpuVertexAttribute> {
        self.attributes.iter()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuVertexInputStateDescriptor {
    layouts: Vec<GpuVertexBufferLayoutDescriptor>,
}

impl GpuVertexInputStateDescriptor {
    pub fn new(
        layouts: impl IntoIterator<Item = GpuVertexBufferLayoutDescriptor>,
    ) -> Result<Self, GpuProgramContractError> {
        let mut layouts = layouts.into_iter().collect::<Vec<_>>();
        layouts.sort_by_key(GpuVertexBufferLayoutDescriptor::slot);
        if let Some(slot) = layouts
            .windows(2)
            .find(|pair| pair[0].slot() == pair[1].slot())
            .map(|pair| pair[0].slot())
        {
            return Err(invalid_vertex_state(
                format!("slot={slot}"),
                "declare each vertex-buffer slot exactly once",
            ));
        }

        let mut attributes = layouts
            .iter()
            .flat_map(|layout| layout.attributes().copied())
            .collect::<Vec<_>>();
        attributes.sort_by_key(|attribute| attribute.shader_location());
        if let Some(location) = duplicate_location(&attributes) {
            return Err(invalid_vertex_state(
                format!("shader_location={location}"),
                "declare each shader location in exactly one vertex-buffer layout",
            ));
        }

        Ok(Self { layouts })
    }

    pub fn layouts(&self) -> impl ExactSizeIterator<Item = &GpuVertexBufferLayoutDescriptor> {
        self.layouts.iter()
    }

    pub fn layout(&self, slot: u32) -> Option<&GpuVertexBufferLayoutDescriptor> {
        self.layouts
            .binary_search_by_key(&slot, GpuVertexBufferLayoutDescriptor::slot)
            .ok()
            .map(|index| &self.layouts[index])
    }

    pub fn expected_signature(
        &self,
        entry_point: GpuEntryPointName,
    ) -> Result<GpuExpectedVertexInputSignature, GpuProgramContractError> {
        GpuExpectedVertexInputSignature::new(
            entry_point,
            self.layouts.iter().flat_map(|layout| {
                layout.attributes().map(|attribute| {
                    GpuShaderIoLocation::new(
                        attribute.shader_location(),
                        attribute.format().shader_io_type(),
                    )
                })
            }),
        )
    }
}

fn duplicate_location(attributes: &[GpuVertexAttribute]) -> Option<u32> {
    attributes
        .windows(2)
        .find(|pair| pair[0].shader_location() == pair[1].shader_location())
        .map(|pair| pair[0].shader_location())
}

fn invalid_vertex_state(
    label: impl Into<String>,
    correction: &'static str,
) -> GpuProgramContractError {
    GpuProgramContractError::invalid(
        "construct GPU vertex-input state",
        label,
        GpuProgramContractCause::VertexInputStateInvalid,
        correction,
    )
}
