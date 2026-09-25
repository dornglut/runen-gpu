use crate::{
    GpuColorWriteMask, GpuCompareFunction, GpuCullMode, GpuFrontFace, GpuIndexFormat,
    GpuPrimitiveTopology, GpuVertexFormat, GpuVertexStepMode,
};
use wgpu::{
    ColorWrites, CompareFunction, Face, FrontFace, IndexFormat, PrimitiveTopology, VertexFormat,
    VertexStepMode,
};

pub(super) fn color_write_mask(mask: GpuColorWriteMask) -> ColorWrites {
    let mut native = ColorWrites::empty();
    if mask.contains(GpuColorWriteMask::RED) {
        native |= ColorWrites::RED;
    }
    if mask.contains(GpuColorWriteMask::GREEN) {
        native |= ColorWrites::GREEN;
    }
    if mask.contains(GpuColorWriteMask::BLUE) {
        native |= ColorWrites::BLUE;
    }
    if mask.contains(GpuColorWriteMask::ALPHA) {
        native |= ColorWrites::ALPHA;
    }
    native
}

pub(super) const fn vertex_format(value: GpuVertexFormat) -> VertexFormat {
    match value {
        GpuVertexFormat::Uint8 => VertexFormat::Uint8,
        GpuVertexFormat::Uint8x2 => VertexFormat::Uint8x2,
        GpuVertexFormat::Uint8x4 => VertexFormat::Uint8x4,
        GpuVertexFormat::Sint8 => VertexFormat::Sint8,
        GpuVertexFormat::Sint8x2 => VertexFormat::Sint8x2,
        GpuVertexFormat::Sint8x4 => VertexFormat::Sint8x4,
        GpuVertexFormat::Unorm8 => VertexFormat::Unorm8,
        GpuVertexFormat::Unorm8x2 => VertexFormat::Unorm8x2,
        GpuVertexFormat::Unorm8x4 => VertexFormat::Unorm8x4,
        GpuVertexFormat::Snorm8 => VertexFormat::Snorm8,
        GpuVertexFormat::Snorm8x2 => VertexFormat::Snorm8x2,
        GpuVertexFormat::Snorm8x4 => VertexFormat::Snorm8x4,
        GpuVertexFormat::Uint16 => VertexFormat::Uint16,
        GpuVertexFormat::Uint16x2 => VertexFormat::Uint16x2,
        GpuVertexFormat::Uint16x4 => VertexFormat::Uint16x4,
        GpuVertexFormat::Sint16 => VertexFormat::Sint16,
        GpuVertexFormat::Sint16x2 => VertexFormat::Sint16x2,
        GpuVertexFormat::Sint16x4 => VertexFormat::Sint16x4,
        GpuVertexFormat::Unorm16 => VertexFormat::Unorm16,
        GpuVertexFormat::Unorm16x2 => VertexFormat::Unorm16x2,
        GpuVertexFormat::Unorm16x4 => VertexFormat::Unorm16x4,
        GpuVertexFormat::Snorm16 => VertexFormat::Snorm16,
        GpuVertexFormat::Snorm16x2 => VertexFormat::Snorm16x2,
        GpuVertexFormat::Snorm16x4 => VertexFormat::Snorm16x4,
        GpuVertexFormat::Float16 => VertexFormat::Float16,
        GpuVertexFormat::Float16x2 => VertexFormat::Float16x2,
        GpuVertexFormat::Float16x4 => VertexFormat::Float16x4,
        GpuVertexFormat::Unorm10_10_10_2 => VertexFormat::Unorm10_10_10_2,
        GpuVertexFormat::Unorm8x4Bgra => VertexFormat::Unorm8x4Bgra,
        GpuVertexFormat::Float32 => VertexFormat::Float32,
        GpuVertexFormat::Float32x2 => VertexFormat::Float32x2,
        GpuVertexFormat::Float32x3 => VertexFormat::Float32x3,
        GpuVertexFormat::Float32x4 => VertexFormat::Float32x4,
        GpuVertexFormat::Uint32 => VertexFormat::Uint32,
        GpuVertexFormat::Uint32x2 => VertexFormat::Uint32x2,
        GpuVertexFormat::Uint32x3 => VertexFormat::Uint32x3,
        GpuVertexFormat::Uint32x4 => VertexFormat::Uint32x4,
        GpuVertexFormat::Sint32 => VertexFormat::Sint32,
        GpuVertexFormat::Sint32x2 => VertexFormat::Sint32x2,
        GpuVertexFormat::Sint32x3 => VertexFormat::Sint32x3,
        GpuVertexFormat::Sint32x4 => VertexFormat::Sint32x4,
    }
}

pub(super) const fn vertex_step_mode(value: GpuVertexStepMode) -> VertexStepMode {
    match value {
        GpuVertexStepMode::Vertex => VertexStepMode::Vertex,
        GpuVertexStepMode::Instance => VertexStepMode::Instance,
    }
}

pub(super) const fn primitive_topology(value: GpuPrimitiveTopology) -> PrimitiveTopology {
    match value {
        GpuPrimitiveTopology::TriangleList => PrimitiveTopology::TriangleList,
        GpuPrimitiveTopology::TriangleStrip => PrimitiveTopology::TriangleStrip,
        GpuPrimitiveTopology::LineList => PrimitiveTopology::LineList,
        GpuPrimitiveTopology::LineStrip => PrimitiveTopology::LineStrip,
        GpuPrimitiveTopology::PointList => PrimitiveTopology::PointList,
    }
}

pub(super) const fn index_format(value: GpuIndexFormat) -> IndexFormat {
    match value {
        GpuIndexFormat::Uint16 => IndexFormat::Uint16,
        GpuIndexFormat::Uint32 => IndexFormat::Uint32,
    }
}

pub(super) const fn front_face(value: GpuFrontFace) -> FrontFace {
    match value {
        GpuFrontFace::CounterClockwise => FrontFace::Ccw,
        GpuFrontFace::Clockwise => FrontFace::Cw,
    }
}

pub(super) const fn cull_mode(value: GpuCullMode) -> Option<Face> {
    match value {
        GpuCullMode::None => None,
        GpuCullMode::Front => Some(Face::Front),
        GpuCullMode::Back => Some(Face::Back),
    }
}

pub(super) const fn stencil_operation(value: crate::GpuStencilOperation) -> wgpu::StencilOperation {
    match value {
        crate::GpuStencilOperation::Keep => wgpu::StencilOperation::Keep,
        crate::GpuStencilOperation::Zero => wgpu::StencilOperation::Zero,
        crate::GpuStencilOperation::Replace => wgpu::StencilOperation::Replace,
        crate::GpuStencilOperation::Invert => wgpu::StencilOperation::Invert,
        crate::GpuStencilOperation::IncrementClamp => wgpu::StencilOperation::IncrementClamp,
        crate::GpuStencilOperation::DecrementClamp => wgpu::StencilOperation::DecrementClamp,
        crate::GpuStencilOperation::IncrementWrap => wgpu::StencilOperation::IncrementWrap,
        crate::GpuStencilOperation::DecrementWrap => wgpu::StencilOperation::DecrementWrap,
    }
}

pub(super) const fn compare_function(value: GpuCompareFunction) -> CompareFunction {
    match value {
        GpuCompareFunction::Never => CompareFunction::Never,
        GpuCompareFunction::Less => CompareFunction::Less,
        GpuCompareFunction::Equal => CompareFunction::Equal,
        GpuCompareFunction::LessEqual => CompareFunction::LessEqual,
        GpuCompareFunction::Greater => CompareFunction::Greater,
        GpuCompareFunction::NotEqual => CompareFunction::NotEqual,
        GpuCompareFunction::GreaterEqual => CompareFunction::GreaterEqual,
        GpuCompareFunction::Always => CompareFunction::Always,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_mappings_cover_every_current_backend_neutral_family() {
        assert_eq!(
            vertex_format(GpuVertexFormat::Sint32x4),
            VertexFormat::Sint32x4
        );
        for (normalized, native) in [
            (GpuVertexFormat::Uint8, VertexFormat::Uint8),
            (GpuVertexFormat::Uint8x2, VertexFormat::Uint8x2),
            (GpuVertexFormat::Uint8x4, VertexFormat::Uint8x4),
            (GpuVertexFormat::Sint8, VertexFormat::Sint8),
            (GpuVertexFormat::Sint8x2, VertexFormat::Sint8x2),
            (GpuVertexFormat::Sint8x4, VertexFormat::Sint8x4),
            (GpuVertexFormat::Unorm8, VertexFormat::Unorm8),
            (GpuVertexFormat::Unorm8x2, VertexFormat::Unorm8x2),
            (GpuVertexFormat::Unorm8x4, VertexFormat::Unorm8x4),
            (GpuVertexFormat::Snorm8, VertexFormat::Snorm8),
            (GpuVertexFormat::Snorm8x2, VertexFormat::Snorm8x2),
            (GpuVertexFormat::Snorm8x4, VertexFormat::Snorm8x4),
            (GpuVertexFormat::Uint16, VertexFormat::Uint16),
            (GpuVertexFormat::Uint16x2, VertexFormat::Uint16x2),
            (GpuVertexFormat::Uint16x4, VertexFormat::Uint16x4),
            (GpuVertexFormat::Sint16, VertexFormat::Sint16),
            (GpuVertexFormat::Sint16x2, VertexFormat::Sint16x2),
            (GpuVertexFormat::Sint16x4, VertexFormat::Sint16x4),
            (GpuVertexFormat::Unorm16, VertexFormat::Unorm16),
            (GpuVertexFormat::Unorm16x2, VertexFormat::Unorm16x2),
            (GpuVertexFormat::Unorm16x4, VertexFormat::Unorm16x4),
            (GpuVertexFormat::Snorm16, VertexFormat::Snorm16),
            (GpuVertexFormat::Snorm16x2, VertexFormat::Snorm16x2),
            (GpuVertexFormat::Snorm16x4, VertexFormat::Snorm16x4),
            (GpuVertexFormat::Float16, VertexFormat::Float16),
            (GpuVertexFormat::Float16x2, VertexFormat::Float16x2),
            (GpuVertexFormat::Float16x4, VertexFormat::Float16x4),
            (
                GpuVertexFormat::Unorm10_10_10_2,
                VertexFormat::Unorm10_10_10_2,
            ),
            (GpuVertexFormat::Unorm8x4Bgra, VertexFormat::Unorm8x4Bgra),
        ] {
            assert_eq!(vertex_format(normalized), native);
        }
        assert_eq!(
            primitive_topology(GpuPrimitiveTopology::LineStrip),
            PrimitiveTopology::LineStrip
        );
        assert_eq!(front_face(GpuFrontFace::Clockwise), FrontFace::Cw);
        assert_eq!(cull_mode(GpuCullMode::Back), Some(Face::Back));
        assert_eq!(
            compare_function(GpuCompareFunction::LessEqual),
            CompareFunction::LessEqual
        );
        for (normalized, native) in [
            (
                crate::GpuStencilOperation::Keep,
                wgpu::StencilOperation::Keep,
            ),
            (
                crate::GpuStencilOperation::Zero,
                wgpu::StencilOperation::Zero,
            ),
            (
                crate::GpuStencilOperation::Replace,
                wgpu::StencilOperation::Replace,
            ),
            (
                crate::GpuStencilOperation::Invert,
                wgpu::StencilOperation::Invert,
            ),
            (
                crate::GpuStencilOperation::IncrementClamp,
                wgpu::StencilOperation::IncrementClamp,
            ),
            (
                crate::GpuStencilOperation::DecrementClamp,
                wgpu::StencilOperation::DecrementClamp,
            ),
            (
                crate::GpuStencilOperation::IncrementWrap,
                wgpu::StencilOperation::IncrementWrap,
            ),
            (
                crate::GpuStencilOperation::DecrementWrap,
                wgpu::StencilOperation::DecrementWrap,
            ),
        ] {
            assert_eq!(stencil_operation(normalized), native);
        }
        assert_eq!(index_format(GpuIndexFormat::Uint16), IndexFormat::Uint16);
        assert_eq!(
            vertex_step_mode(GpuVertexStepMode::Instance),
            VertexStepMode::Instance
        );
        assert_eq!(color_write_mask(GpuColorWriteMask::ALL), ColorWrites::ALL);
    }
}
