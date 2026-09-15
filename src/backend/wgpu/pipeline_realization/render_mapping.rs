use crate::{
    GpuColorWriteMask, GpuCompareFunction, GpuCullMode, GpuFrontFace, GpuIndexFormat,
    GpuPrimitiveTopology, GpuTextureFormat, GpuVertexFormat, GpuVertexStepMode,
};
use wgpu::{
    ColorWrites, CompareFunction, Face, FrontFace, IndexFormat, PrimitiveTopology, TextureFormat,
    VertexFormat, VertexStepMode,
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

pub(super) const fn texture_format(value: GpuTextureFormat) -> TextureFormat {
    match value {
        GpuTextureFormat::R8Unorm => TextureFormat::R8Unorm,
        GpuTextureFormat::Rgba8Unorm => TextureFormat::Rgba8Unorm,
        GpuTextureFormat::Rgba8UnormSrgb => TextureFormat::Rgba8UnormSrgb,
        GpuTextureFormat::Bgra8Unorm => TextureFormat::Bgra8Unorm,
        GpuTextureFormat::Bgra8UnormSrgb => TextureFormat::Bgra8UnormSrgb,
        GpuTextureFormat::R32Uint => TextureFormat::R32Uint,
        GpuTextureFormat::Depth32Float => TextureFormat::Depth32Float,
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
        assert_eq!(
            texture_format(GpuTextureFormat::Depth32Float),
            TextureFormat::Depth32Float
        );
        assert_eq!(index_format(GpuIndexFormat::Uint16), IndexFormat::Uint16);
        assert_eq!(
            vertex_step_mode(GpuVertexStepMode::Instance),
            VertexStepMode::Instance
        );
        assert_eq!(color_write_mask(GpuColorWriteMask::ALL), ColorWrites::ALL);
    }
}
