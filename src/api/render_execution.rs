use super::{
    GpuBufferAccess, GpuBufferAccessKind, GpuBufferHandle, GpuBufferRange,
    GpuCapabilityRequirements, GpuDrawIntent, GpuIndexFormat, GpuLimits,
    GpuRenderPipelineDescriptor, GpuResourceAccess, GpuRuntimeBindingSet, GpuVertexStepMode,
    GpuWorkOperationCause, GpuWorkOperationError,
};
use core::hash::Hash;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuVertexBufferBinding {
    slot: u32,
    buffer: GpuBufferHandle,
    range: GpuBufferRange,
    access: GpuBufferAccess,
}

impl GpuVertexBufferBinding {
    pub fn new(
        slot: u32,
        buffer: &GpuBufferHandle,
        range: GpuBufferRange,
    ) -> Result<Self, GpuWorkOperationError> {
        let access = GpuBufferAccess::new(buffer, range, GpuBufferAccessKind::VertexRead).map_err(
            |source| {
                GpuWorkOperationError::from_access(
                    "construct GPU vertex-buffer binding",
                    buffer.descriptor().common().label().as_str(),
                    GpuWorkOperationCause::InvalidDraw,
                    "bind a checked range from a buffer declaring Vertex usage",
                    source,
                )
            },
        )?;
        Ok(Self {
            slot,
            buffer: buffer.clone(),
            range,
            access,
        })
    }

    pub const fn slot(&self) -> u32 {
        self.slot
    }

    pub fn buffer(&self) -> &GpuBufferHandle {
        &self.buffer
    }

    pub const fn range(&self) -> GpuBufferRange {
        self.range
    }

    pub fn access(&self) -> &GpuBufferAccess {
        &self.access
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuIndexBufferBinding {
    buffer: GpuBufferHandle,
    range: GpuBufferRange,
    format: GpuIndexFormat,
    access: GpuBufferAccess,
}

impl GpuIndexBufferBinding {
    pub fn new(
        buffer: &GpuBufferHandle,
        range: GpuBufferRange,
        format: GpuIndexFormat,
    ) -> Result<Self, GpuWorkOperationError> {
        let alignment = match format {
            GpuIndexFormat::Uint16 => 2,
            GpuIndexFormat::Uint32 => 4,
        };
        if !range.offset().is_multiple_of(alignment) || !range.size().is_multiple_of(alignment) {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU index-buffer binding",
                buffer.descriptor().common().label().as_str(),
                Some(buffer.diagnostic_identity()),
                GpuWorkOperationCause::InvalidDraw,
                "align the index-buffer range to the selected index format",
            ));
        }
        let access = GpuBufferAccess::new(buffer, range, GpuBufferAccessKind::IndexRead).map_err(
            |source| {
                GpuWorkOperationError::from_access(
                    "construct GPU index-buffer binding",
                    buffer.descriptor().common().label().as_str(),
                    GpuWorkOperationCause::InvalidDraw,
                    "bind a checked range from a buffer declaring Index usage",
                    source,
                )
            },
        )?;
        Ok(Self {
            buffer: buffer.clone(),
            range,
            format,
            access,
        })
    }

    pub fn buffer(&self) -> &GpuBufferHandle {
        &self.buffer
    }

    pub const fn range(&self) -> GpuBufferRange {
        self.range
    }

    pub const fn format(&self) -> GpuIndexFormat {
        self.format
    }

    pub fn access(&self) -> &GpuBufferAccess {
        &self.access
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuViewport {
    bits: [u32; 6],
}

impl GpuViewport {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) -> Result<Self, GpuWorkOperationError> {
        let values = [x, y, width, height, min_depth, max_depth];
        if values.iter().any(|value| !value.is_finite()) {
            return Err(invalid_dynamic_state(
                "viewport",
                "provide finite viewport coordinates, extent, and depth bounds",
            ));
        }
        let end_x = x + width;
        let end_y = y + height;
        if x < 0.0
            || y < 0.0
            || width < 0.0
            || height < 0.0
            || !end_x.is_finite()
            || !end_y.is_finite()
            || min_depth < 0.0
            || max_depth > 1.0
            || min_depth > max_depth
        {
            return Err(invalid_dynamic_state(
                "viewport",
                "keep viewport coordinates and extent inside the finite nonnegative domain and depth inside 0 through 1",
            ));
        }
        Ok(Self {
            bits: values.map(canonical_f32_bits),
        })
    }

    pub fn values(self) -> [f32; 6] {
        self.bits.map(f32::from_bits)
    }

    pub(crate) fn validate_limits(self, limits: GpuLimits) -> Result<(), GpuWorkOperationError> {
        let [x, y, width, height, _, _] = self.values();
        let max_dimension = limits.max_texture_dimension_2d() as f32;
        if x + width > max_dimension || y + height > max_dimension {
            return Err(invalid_dynamic_state(
                "viewport",
                "keep viewport coordinates and extent inside the admitted 2D dimension bound",
            ));
        }
        Ok(())
    }
}

impl core::fmt::Debug for GpuViewport {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_tuple("GpuViewport")
            .field(&self.values())
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuScissorRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl GpuScissorRect {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Result<Self, GpuWorkOperationError> {
        if x.checked_add(width).is_none() || y.checked_add(height).is_none() {
            return Err(invalid_dynamic_state(
                "scissor",
                "keep scissor origin plus extent inside the u32 coordinate domain",
            ));
        }
        Ok(Self {
            x,
            y,
            width,
            height,
        })
    }

    pub const fn x(self) -> u32 {
        self.x
    }

    pub const fn y(self) -> u32 {
        self.y
    }

    pub const fn width(self) -> u32 {
        self.width
    }

    pub const fn height(self) -> u32 {
        self.height
    }

    pub const fn end_x(self) -> u32 {
        self.x + self.width
    }

    pub const fn end_y(self) -> u32 {
        self.y + self.height
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuBlendConstant {
    bits: [u64; 4],
}

impl GpuBlendConstant {
    pub fn new(red: f64, green: f64, blue: f64, alpha: f64) -> Result<Self, GpuWorkOperationError> {
        let values = [red, green, blue, alpha];
        if values.iter().any(|value| !value.is_finite()) {
            return Err(invalid_dynamic_state(
                "blend constant",
                "provide four finite blend-constant components",
            ));
        }
        Ok(Self {
            bits: values.map(canonical_f64_bits),
        })
    }

    pub fn components(self) -> [f64; 4] {
        self.bits.map(f64::from_bits)
    }
}

impl core::fmt::Debug for GpuBlendConstant {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_tuple("GpuBlendConstant")
            .field(&self.components())
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuRenderDraw {
    pipeline: GpuRenderPipelineDescriptor,
    bindings: GpuRuntimeBindingSet,
    vertex_buffers: Vec<GpuVertexBufferBinding>,
    index_buffer: Option<GpuIndexBufferBinding>,
    draw: GpuDrawIntent,
    viewport: GpuViewport,
    scissor: GpuScissorRect,
    blend_constant: GpuBlendConstant,
    stencil_reference: u32,
    accesses: Vec<GpuResourceAccess>,
}

impl GpuRenderDraw {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pipeline: GpuRenderPipelineDescriptor,
        bindings: GpuRuntimeBindingSet,
        vertex_buffers: impl IntoIterator<Item = GpuVertexBufferBinding>,
        index_buffer: Option<GpuIndexBufferBinding>,
        draw: GpuDrawIntent,
        viewport: GpuViewport,
        scissor: GpuScissorRect,
        blend_constant: GpuBlendConstant,
        stencil_reference: u32,
    ) -> Result<Self, GpuWorkOperationError> {
        if pipeline.layout() != bindings.layout() {
            return Err(invalid_draw(
                "pipeline bindings",
                "use a runtime binding set constructed for the exact render pipeline layout",
            ));
        }

        let mut vertex_buffers = vertex_buffers.into_iter().collect::<Vec<_>>();
        vertex_buffers.sort_by_key(GpuVertexBufferBinding::slot);
        if vertex_buffers
            .windows(2)
            .any(|pair| pair[0].slot() == pair[1].slot())
        {
            return Err(invalid_draw(
                "vertex buffers",
                "bind each vertex-buffer slot exactly once",
            ));
        }

        let vertex_layouts = pipeline
            .state()
            .vertex_input()
            .layouts()
            .collect::<Vec<_>>();
        if vertex_layouts.len() != vertex_buffers.len()
            || vertex_layouts
                .iter()
                .zip(&vertex_buffers)
                .any(|(layout, binding)| layout.slot() != binding.slot())
        {
            return Err(invalid_draw(
                "vertex buffers",
                "bind exactly the vertex-buffer slots declared by the render pipeline",
            ));
        }

        match (&draw, &index_buffer) {
            (GpuDrawIntent::Indexed { indices, .. }, Some(binding)) => {
                validate_indexed_range(*indices, binding)?;
            }
            (GpuDrawIntent::Indirect { indexed: true, .. }, Some(_)) => {}
            (
                GpuDrawIntent::Indexed { .. } | GpuDrawIntent::Indirect { indexed: true, .. },
                None,
            ) => {
                return Err(invalid_draw(
                    "index buffer",
                    "bind an index buffer for indexed direct or indirect draws",
                ));
            }
            (
                GpuDrawIntent::Direct { .. } | GpuDrawIntent::Indirect { indexed: false, .. },
                Some(_),
            ) => {
                return Err(invalid_draw(
                    "index buffer",
                    "omit the index buffer for non-indexed draws",
                ));
            }
            (
                GpuDrawIntent::Direct { .. } | GpuDrawIntent::Indirect { indexed: false, .. },
                None,
            ) => {}
        }
        validate_direct_vertex_ranges(&draw, &pipeline, &vertex_buffers)?;

        let mut accesses = bindings.accesses().to_vec();
        accesses.extend(
            vertex_buffers
                .iter()
                .cloned()
                .map(|binding| GpuResourceAccess::Buffer(binding.access)),
        );
        if let Some(binding) = &index_buffer {
            accesses.push(GpuResourceAccess::Buffer(binding.access().clone()));
        }
        if let Some(access) = draw.derived_access()? {
            accesses.push(GpuResourceAccess::Buffer(access));
        }

        Ok(Self {
            pipeline,
            bindings,
            vertex_buffers,
            index_buffer,
            draw,
            viewport,
            scissor,
            blend_constant,
            stencil_reference,
            accesses,
        })
    }

    pub fn pipeline(&self) -> &GpuRenderPipelineDescriptor {
        &self.pipeline
    }

    pub fn bindings(&self) -> &GpuRuntimeBindingSet {
        &self.bindings
    }

    pub fn vertex_buffers(&self) -> &[GpuVertexBufferBinding] {
        &self.vertex_buffers
    }

    pub fn index_buffer(&self) -> Option<&GpuIndexBufferBinding> {
        self.index_buffer.as_ref()
    }

    pub fn draw(&self) -> &GpuDrawIntent {
        &self.draw
    }

    pub const fn viewport(&self) -> GpuViewport {
        self.viewport
    }

    pub const fn scissor(&self) -> GpuScissorRect {
        self.scissor
    }

    pub const fn blend_constant(&self) -> GpuBlendConstant {
        self.blend_constant
    }

    pub const fn stencil_reference(&self) -> u32 {
        self.stencil_reference
    }

    pub fn accesses(&self) -> &[GpuResourceAccess] {
        &self.accesses
    }

    pub fn requirements(&self) -> &GpuCapabilityRequirements {
        self.pipeline.requirements()
    }

    pub(crate) fn validate_limits(&self, limits: GpuLimits) -> Result<(), GpuWorkOperationError> {
        self.viewport.validate_limits(limits)?;
        let vertex_buffer_slots = self
            .vertex_buffers
            .last()
            .map(|binding| u64::from(binding.slot()) + 1)
            .unwrap_or(0);
        if vertex_buffer_slots > u64::from(limits.max_vertex_buffers())
            || vertex_buffer_slots + self.bindings.required_bind_group_slots()
                > u64::from(limits.max_bind_groups_plus_vertex_buffers())
        {
            return Err(invalid_draw(
                "vertex buffers",
                "keep positional vertex-buffer and bind-group slots inside the admitted execution limits",
            ));
        }
        Ok(())
    }
}

fn validate_indexed_range(
    indices: super::GpuDrawRange,
    binding: &GpuIndexBufferBinding,
) -> Result<(), GpuWorkOperationError> {
    let element_size = match binding.format() {
        GpuIndexFormat::Uint16 => 2_u64,
        GpuIndexFormat::Uint32 => 4_u64,
    };
    let required = u64::from(indices.end())
        .checked_mul(element_size)
        .ok_or_else(|| invalid_draw("indexed draw", "reduce the index range"))?;
    if required > binding.range().size() {
        return Err(invalid_draw(
            "indexed draw",
            "keep the direct index range inside the bound index-buffer slice",
        ));
    }
    Ok(())
}

fn validate_direct_vertex_ranges(
    draw: &GpuDrawIntent,
    pipeline: &GpuRenderPipelineDescriptor,
    bindings: &[GpuVertexBufferBinding],
) -> Result<(), GpuWorkOperationError> {
    let (vertices, instances) = match draw {
        GpuDrawIntent::Direct {
            vertices,
            instances,
        } => (*vertices, *instances),
        GpuDrawIntent::Indexed { instances, .. } => {
            // Indexed draws do not expose the runtime index values used for vertex fetch. Instance
            // coverage is still host-known and can be admitted here; vertex fetch validity remains
            // runtime GPU data and backend robustness territory.
            for (layout, binding) in pipeline.state().vertex_input().layouts().zip(bindings) {
                if layout.step_mode() == GpuVertexStepMode::Instance {
                    validate_stepped_range(instances.end(), layout, binding)?;
                }
            }
            return Ok(());
        }
        GpuDrawIntent::Indirect { .. } => return Ok(()),
    };

    for (layout, binding) in pipeline.state().vertex_input().layouts().zip(bindings) {
        let end = match layout.step_mode() {
            GpuVertexStepMode::Vertex => vertices.end(),
            GpuVertexStepMode::Instance => instances.end(),
        };
        validate_stepped_range(end, layout, binding)?;
    }
    Ok(())
}

fn validate_stepped_range(
    end: u32,
    layout: &super::GpuVertexBufferLayoutDescriptor,
    binding: &GpuVertexBufferBinding,
) -> Result<(), GpuWorkOperationError> {
    let max_attribute_end = layout
        .attributes()
        .map(|attribute| attribute.offset() + attribute.format().size_bytes())
        .max()
        .unwrap_or(0);
    let required = if end == 0 {
        0
    } else {
        u64::from(end - 1)
            .checked_mul(layout.array_stride())
            .and_then(|base| base.checked_add(max_attribute_end))
            .ok_or_else(|| invalid_draw("vertex fetch", "reduce the direct draw range"))?
    };
    if required > binding.range().size() {
        return Err(invalid_draw(
            "vertex fetch",
            "keep direct vertex or instance fetches inside the bound vertex-buffer slice",
        ));
    }
    Ok(())
}

fn invalid_dynamic_state(label: &'static str, correction: &'static str) -> GpuWorkOperationError {
    GpuWorkOperationError::invalid(
        "construct GPU render dynamic state",
        label,
        None,
        GpuWorkOperationCause::InvalidDraw,
        correction,
    )
}

fn invalid_draw(label: &'static str, correction: &'static str) -> GpuWorkOperationError {
    GpuWorkOperationError::invalid(
        "construct GPU render draw",
        label,
        None,
        GpuWorkOperationCause::InvalidDraw,
        correction,
    )
}

const fn canonical_f32_bits(value: f32) -> u32 {
    if value == 0.0 {
        0.0_f32.to_bits()
    } else {
        value.to_bits()
    }
}

const fn canonical_f64_bits(value: f64) -> u64 {
    if value == 0.0 {
        0.0_f64.to_bits()
    } else {
        value.to_bits()
    }
}
