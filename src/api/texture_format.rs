use super::{GpuTextureAspect, GpuTextureFormat};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuTextureScalarClass {
    Float,
    Sint,
    Uint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuTextureAspectClass {
    Color,
    Depth,
    // These normalized states are intentionally staged before their public formats land.
    #[cfg_attr(not(test), allow(dead_code))]
    Stencil,
    #[cfg_attr(not(test), allow(dead_code))]
    DepthStencil,
}

impl GpuTextureAspectClass {
    const fn supports(self, aspect: GpuTextureAspect) -> bool {
        matches!(
            (self, aspect),
            (_, GpuTextureAspect::All)
                | (Self::Color, GpuTextureAspect::Color)
                | (Self::Depth, GpuTextureAspect::DepthOnly)
                | (Self::Stencil, GpuTextureAspect::StencilOnly)
                | (Self::DepthStencil, GpuTextureAspect::DepthOnly)
                | (Self::DepthStencil, GpuTextureAspect::StencilOnly)
        )
    }

    const fn canonical(self, aspect: GpuTextureAspect) -> Option<GpuTextureAspect> {
        if !self.supports(aspect) {
            return None;
        }
        match (self, aspect) {
            (Self::Color, GpuTextureAspect::All) => Some(GpuTextureAspect::Color),
            (Self::Depth, GpuTextureAspect::All) => Some(GpuTextureAspect::DepthOnly),
            (Self::Stencil, GpuTextureAspect::All) => Some(GpuTextureAspect::StencilOnly),
            (Self::DepthStencil, GpuTextureAspect::All) => Some(GpuTextureAspect::All),
            (_, explicit) => Some(explicit),
        }
    }

    const fn has_depth(self) -> bool {
        matches!(self, Self::Depth | Self::DepthStencil)
    }

    const fn has_stencil(self) -> bool {
        matches!(self, Self::Stencil | Self::DepthStencil)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GpuTextureFormatSemantics {
    aspect_class: GpuTextureAspectClass,
    block_dimensions: (u32, u32),
    color_copy_block_size: Option<u32>,
    depth_copy_block_size: Option<u32>,
    stencil_copy_block_size: Option<u32>,
    srgb: bool,
    paired_view_format: Option<GpuTextureFormat>,
    scalar_class: GpuTextureScalarClass,
    component_count: u8,
    has_alpha: bool,
}

const fn semantics(format: GpuTextureFormat) -> GpuTextureFormatSemantics {
    match format {
        GpuTextureFormat::R8Unorm => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(1),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Float,
            component_count: 1,
            has_alpha: false,
        },
        GpuTextureFormat::R8Snorm => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(1),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Float,
            component_count: 1,
            has_alpha: false,
        },
        GpuTextureFormat::R8Uint => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(1),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Uint,
            component_count: 1,
            has_alpha: false,
        },
        GpuTextureFormat::R8Sint => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(1),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Sint,
            component_count: 1,
            has_alpha: false,
        },
        GpuTextureFormat::Rg8Unorm => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(2),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Float,
            component_count: 2,
            has_alpha: false,
        },
        GpuTextureFormat::Rg8Snorm => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(2),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Float,
            component_count: 2,
            has_alpha: false,
        },
        GpuTextureFormat::Rg8Uint => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(2),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Uint,
            component_count: 2,
            has_alpha: false,
        },
        GpuTextureFormat::Rg8Sint => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(2),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Sint,
            component_count: 2,
            has_alpha: false,
        },
        GpuTextureFormat::R16Uint => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(2),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Uint,
            component_count: 1,
            has_alpha: false,
        },
        GpuTextureFormat::R16Sint => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(2),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Sint,
            component_count: 1,
            has_alpha: false,
        },
        GpuTextureFormat::R16Float => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(2),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Float,
            component_count: 1,
            has_alpha: false,
        },
        GpuTextureFormat::Rg16Uint => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(4),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Uint,
            component_count: 2,
            has_alpha: false,
        },
        GpuTextureFormat::Rg16Sint => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(4),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Sint,
            component_count: 2,
            has_alpha: false,
        },
        GpuTextureFormat::Rg16Float => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(4),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Float,
            component_count: 2,
            has_alpha: false,
        },
        GpuTextureFormat::Rgba8Unorm => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(4),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: Some(GpuTextureFormat::Rgba8UnormSrgb),
            scalar_class: GpuTextureScalarClass::Float,
            component_count: 4,
            has_alpha: true,
        },
        GpuTextureFormat::Rgba8UnormSrgb => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(4),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: true,
            paired_view_format: Some(GpuTextureFormat::Rgba8Unorm),
            scalar_class: GpuTextureScalarClass::Float,
            component_count: 4,
            has_alpha: true,
        },
        GpuTextureFormat::Rgba8Snorm => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(4),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Float,
            component_count: 4,
            has_alpha: true,
        },
        GpuTextureFormat::Rgba8Uint => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(4),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Uint,
            component_count: 4,
            has_alpha: true,
        },
        GpuTextureFormat::Rgba8Sint => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(4),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Sint,
            component_count: 4,
            has_alpha: true,
        },
        GpuTextureFormat::Bgra8Unorm => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(4),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: Some(GpuTextureFormat::Bgra8UnormSrgb),
            scalar_class: GpuTextureScalarClass::Float,
            component_count: 4,
            has_alpha: true,
        },
        GpuTextureFormat::Bgra8UnormSrgb => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(4),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: true,
            paired_view_format: Some(GpuTextureFormat::Bgra8Unorm),
            scalar_class: GpuTextureScalarClass::Float,
            component_count: 4,
            has_alpha: true,
        },
        GpuTextureFormat::R32Uint => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(4),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Uint,
            component_count: 1,
            has_alpha: false,
        },
        GpuTextureFormat::R32Sint => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(4),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Sint,
            component_count: 1,
            has_alpha: false,
        },
        GpuTextureFormat::R32Float => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(4),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Float,
            component_count: 1,
            has_alpha: false,
        },
        GpuTextureFormat::Rg32Uint => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(8),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Uint,
            component_count: 2,
            has_alpha: false,
        },
        GpuTextureFormat::Rg32Sint => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(8),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Sint,
            component_count: 2,
            has_alpha: false,
        },
        GpuTextureFormat::Rg32Float => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(8),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Float,
            component_count: 2,
            has_alpha: false,
        },
        GpuTextureFormat::Rgba32Uint => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(16),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Uint,
            component_count: 4,
            has_alpha: true,
        },
        GpuTextureFormat::Rgba32Sint => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(16),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Sint,
            component_count: 4,
            has_alpha: true,
        },
        GpuTextureFormat::Rgba32Float => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(16),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Float,
            component_count: 4,
            has_alpha: true,
        },
        GpuTextureFormat::Rgba16Uint => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(8),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Uint,
            component_count: 4,
            has_alpha: true,
        },
        GpuTextureFormat::Rgba16Sint => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(8),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Sint,
            component_count: 4,
            has_alpha: true,
        },
        GpuTextureFormat::Rgba16Float => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Color,
            block_dimensions: (1, 1),
            color_copy_block_size: Some(8),
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Float,
            component_count: 4,
            has_alpha: true,
        },
        GpuTextureFormat::Depth32Float => GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Depth,
            block_dimensions: (1, 1),
            color_copy_block_size: None,
            depth_copy_block_size: Some(4),
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Float,
            component_count: 1,
            has_alpha: false,
        },
    }
}

pub(crate) const fn bytes_per_texel(format: GpuTextureFormat) -> u32 {
    let value = semantics(format);
    match (
        value.color_copy_block_size,
        value.depth_copy_block_size,
        value.stencil_copy_block_size,
    ) {
        (Some(size), None, None) | (None, Some(size), None) | (None, None, Some(size)) => size,
        _ => 0,
    }
}

pub(crate) const fn is_depth(format: GpuTextureFormat) -> bool {
    semantics(format).aspect_class.has_depth()
}

pub(crate) const fn is_stencil(format: GpuTextureFormat) -> bool {
    semantics(format).aspect_class.has_stencil()
}

pub(crate) const fn whole_aspect(format: GpuTextureFormat) -> GpuTextureAspect {
    match semantics(format)
        .aspect_class
        .canonical(GpuTextureAspect::All)
    {
        Some(aspect) => aspect,
        None => GpuTextureAspect::All,
    }
}

pub(crate) const fn canonical_aspect(
    format: GpuTextureFormat,
    aspect: GpuTextureAspect,
) -> Option<GpuTextureAspect> {
    semantics(format).aspect_class.canonical(aspect)
}

pub(crate) const fn is_srgb(format: GpuTextureFormat) -> bool {
    semantics(format).srgb
}

pub(crate) const fn block_dimensions(format: GpuTextureFormat) -> (u32, u32) {
    semantics(format).block_dimensions
}

pub(crate) const fn default_copy_block_size(format: GpuTextureFormat) -> Option<u32> {
    copy_block_size(format, GpuTextureAspect::All)
}

pub(crate) const fn supports_aspect(format: GpuTextureFormat, aspect: GpuTextureAspect) -> bool {
    semantics(format).aspect_class.supports(aspect)
}

pub(crate) const fn canonical_copy_aspect(
    format: GpuTextureFormat,
    aspect: GpuTextureAspect,
) -> Option<GpuTextureAspect> {
    match canonical_aspect(format, aspect) {
        Some(GpuTextureAspect::Color) => Some(GpuTextureAspect::Color),
        Some(GpuTextureAspect::DepthOnly) => Some(GpuTextureAspect::DepthOnly),
        Some(GpuTextureAspect::StencilOnly) => Some(GpuTextureAspect::StencilOnly),
        Some(GpuTextureAspect::All) | None => None,
    }
}

pub(crate) const fn copy_block_size(
    format: GpuTextureFormat,
    aspect: GpuTextureAspect,
) -> Option<u32> {
    let value = semantics(format);
    match canonical_copy_aspect(format, aspect) {
        Some(GpuTextureAspect::Color) => value.color_copy_block_size,
        Some(GpuTextureAspect::DepthOnly) => value.depth_copy_block_size,
        Some(GpuTextureAspect::StencilOnly) => value.stencil_copy_block_size,
        Some(GpuTextureAspect::All) | None => None,
    }
}

pub(crate) const fn paired_view_format(format: GpuTextureFormat) -> Option<GpuTextureFormat> {
    semantics(format).paired_view_format
}

pub(crate) fn view_compatible(parent: GpuTextureFormat, view: GpuTextureFormat) -> bool {
    parent == view || paired_view_format(parent).is_some_and(|paired| paired == view)
}

pub(crate) fn raw_copy_compatible(source: GpuTextureFormat, destination: GpuTextureFormat) -> bool {
    source == destination || paired_view_format(source).is_some_and(|paired| paired == destination)
}

pub(crate) const fn scalar_class(format: GpuTextureFormat) -> GpuTextureScalarClass {
    semantics(format).scalar_class
}

pub(crate) const fn component_count(format: GpuTextureFormat) -> u8 {
    semantics(format).component_count
}

pub(crate) const fn has_alpha(format: GpuTextureFormat) -> bool {
    semantics(format).has_alpha
}

const fn block_count(extent: u32, block: u32) -> u32 {
    extent.div_ceil(block)
}

pub(crate) const fn logical_copy_footprint(
    format: GpuTextureFormat,
    aspect: GpuTextureAspect,
    width: u32,
    height: u32,
) -> Option<(u32, u32)> {
    if width == 0 || height == 0 {
        return None;
    }
    let (block_width, block_height) = block_dimensions(format);
    let bytes_per_block = match copy_block_size(format, aspect) {
        Some(value) => value,
        None => return None,
    };
    let blocks_wide = block_count(width, block_width);
    let block_rows = block_count(height, block_height);
    match blocks_wide.checked_mul(bytes_per_block) {
        Some(row_bytes) => Some((row_bytes, block_rows)),
        None => None,
    }
}

pub(crate) const fn tightly_packed_copy_byte_len(
    format: GpuTextureFormat,
    aspect: GpuTextureAspect,
    width: u32,
    height: u32,
    depth_or_layers: u32,
) -> Option<u64> {
    let (row_bytes, block_rows) = match logical_copy_footprint(format, aspect, width, height) {
        Some(value) => value,
        None => return None,
    };
    let rows = match (row_bytes as u64).checked_mul(block_rows as u64) {
        Some(value) => value,
        None => return None,
    };
    rows.checked_mul(depth_or_layers as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_format_structure_has_one_normalized_authority() {
        let cases = [
            (
                GpuTextureFormat::R8Unorm,
                1,
                false,
                false,
                None,
                GpuTextureScalarClass::Float,
                1,
                false,
            ),
            (
                GpuTextureFormat::R8Snorm,
                1,
                false,
                false,
                None,
                GpuTextureScalarClass::Float,
                1,
                false,
            ),
            (
                GpuTextureFormat::R8Uint,
                1,
                false,
                false,
                None,
                GpuTextureScalarClass::Uint,
                1,
                false,
            ),
            (
                GpuTextureFormat::R8Sint,
                1,
                false,
                false,
                None,
                GpuTextureScalarClass::Sint,
                1,
                false,
            ),
            (
                GpuTextureFormat::Rg8Unorm,
                2,
                false,
                false,
                None,
                GpuTextureScalarClass::Float,
                2,
                false,
            ),
            (
                GpuTextureFormat::Rg8Snorm,
                2,
                false,
                false,
                None,
                GpuTextureScalarClass::Float,
                2,
                false,
            ),
            (
                GpuTextureFormat::Rg8Uint,
                2,
                false,
                false,
                None,
                GpuTextureScalarClass::Uint,
                2,
                false,
            ),
            (
                GpuTextureFormat::Rg8Sint,
                2,
                false,
                false,
                None,
                GpuTextureScalarClass::Sint,
                2,
                false,
            ),
            (
                GpuTextureFormat::R16Uint,
                2,
                false,
                false,
                None,
                GpuTextureScalarClass::Uint,
                1,
                false,
            ),
            (
                GpuTextureFormat::R16Sint,
                2,
                false,
                false,
                None,
                GpuTextureScalarClass::Sint,
                1,
                false,
            ),
            (
                GpuTextureFormat::R16Float,
                2,
                false,
                false,
                None,
                GpuTextureScalarClass::Float,
                1,
                false,
            ),
            (
                GpuTextureFormat::Rg16Uint,
                4,
                false,
                false,
                None,
                GpuTextureScalarClass::Uint,
                2,
                false,
            ),
            (
                GpuTextureFormat::Rg16Sint,
                4,
                false,
                false,
                None,
                GpuTextureScalarClass::Sint,
                2,
                false,
            ),
            (
                GpuTextureFormat::Rg16Float,
                4,
                false,
                false,
                None,
                GpuTextureScalarClass::Float,
                2,
                false,
            ),
            (
                GpuTextureFormat::Rgba8Unorm,
                4,
                false,
                false,
                Some(GpuTextureFormat::Rgba8UnormSrgb),
                GpuTextureScalarClass::Float,
                4,
                true,
            ),
            (
                GpuTextureFormat::Rgba8UnormSrgb,
                4,
                false,
                true,
                Some(GpuTextureFormat::Rgba8Unorm),
                GpuTextureScalarClass::Float,
                4,
                true,
            ),
            (
                GpuTextureFormat::Rgba8Snorm,
                4,
                false,
                false,
                None,
                GpuTextureScalarClass::Float,
                4,
                true,
            ),
            (
                GpuTextureFormat::Rgba8Uint,
                4,
                false,
                false,
                None,
                GpuTextureScalarClass::Uint,
                4,
                true,
            ),
            (
                GpuTextureFormat::Rgba8Sint,
                4,
                false,
                false,
                None,
                GpuTextureScalarClass::Sint,
                4,
                true,
            ),
            (
                GpuTextureFormat::Bgra8Unorm,
                4,
                false,
                false,
                Some(GpuTextureFormat::Bgra8UnormSrgb),
                GpuTextureScalarClass::Float,
                4,
                true,
            ),
            (
                GpuTextureFormat::Bgra8UnormSrgb,
                4,
                false,
                true,
                Some(GpuTextureFormat::Bgra8Unorm),
                GpuTextureScalarClass::Float,
                4,
                true,
            ),
            (
                GpuTextureFormat::R32Uint,
                4,
                false,
                false,
                None,
                GpuTextureScalarClass::Uint,
                1,
                false,
            ),
            (
                GpuTextureFormat::R32Sint,
                4,
                false,
                false,
                None,
                GpuTextureScalarClass::Sint,
                1,
                false,
            ),
            (
                GpuTextureFormat::R32Float,
                4,
                false,
                false,
                None,
                GpuTextureScalarClass::Float,
                1,
                false,
            ),
            (
                GpuTextureFormat::Rg32Uint,
                8,
                false,
                false,
                None,
                GpuTextureScalarClass::Uint,
                2,
                false,
            ),
            (
                GpuTextureFormat::Rg32Sint,
                8,
                false,
                false,
                None,
                GpuTextureScalarClass::Sint,
                2,
                false,
            ),
            (
                GpuTextureFormat::Rg32Float,
                8,
                false,
                false,
                None,
                GpuTextureScalarClass::Float,
                2,
                false,
            ),
            (
                GpuTextureFormat::Rgba32Uint,
                16,
                false,
                false,
                None,
                GpuTextureScalarClass::Uint,
                4,
                true,
            ),
            (
                GpuTextureFormat::Rgba32Sint,
                16,
                false,
                false,
                None,
                GpuTextureScalarClass::Sint,
                4,
                true,
            ),
            (
                GpuTextureFormat::Rgba32Float,
                16,
                false,
                false,
                None,
                GpuTextureScalarClass::Float,
                4,
                true,
            ),
            (
                GpuTextureFormat::Rgba16Uint,
                8,
                false,
                false,
                None,
                GpuTextureScalarClass::Uint,
                4,
                true,
            ),
            (
                GpuTextureFormat::Rgba16Sint,
                8,
                false,
                false,
                None,
                GpuTextureScalarClass::Sint,
                4,
                true,
            ),
            (
                GpuTextureFormat::Rgba16Float,
                8,
                false,
                false,
                None,
                GpuTextureScalarClass::Float,
                4,
                true,
            ),
            (
                GpuTextureFormat::Depth32Float,
                4,
                true,
                false,
                None,
                GpuTextureScalarClass::Float,
                1,
                false,
            ),
        ];

        assert_eq!(cases.len(), 34);
        for (format, bytes, depth, srgb, pair, class, components, alpha) in cases {
            let explicit_aspect = if depth {
                GpuTextureAspect::DepthOnly
            } else {
                GpuTextureAspect::Color
            };
            assert_eq!(block_dimensions(format), (1, 1));
            assert_eq!(copy_block_size(format, GpuTextureAspect::All), Some(bytes));
            assert_eq!(copy_block_size(format, explicit_aspect), Some(bytes));
            assert_eq!(bytes_per_texel(format), bytes);
            assert_eq!(is_depth(format), depth);
            assert_eq!(is_srgb(format), srgb);
            assert_eq!(paired_view_format(format), pair);
            assert_eq!(scalar_class(format), class);
            assert_eq!(component_count(format), components);
            assert_eq!(has_alpha(format), alpha);
            assert!(supports_aspect(format, GpuTextureAspect::All));
            assert_eq!(supports_aspect(format, GpuTextureAspect::Color), !depth);
            assert_eq!(supports_aspect(format, GpuTextureAspect::DepthOnly), depth);
            assert!(!supports_aspect(format, GpuTextureAspect::StencilOnly));
            assert_eq!(
                canonical_copy_aspect(format, GpuTextureAspect::All),
                Some(explicit_aspect)
            );
            assert_eq!(
                logical_copy_footprint(format, GpuTextureAspect::All, 7, 5),
                Some((7 * bytes, 5))
            );
            assert_eq!(
                tightly_packed_copy_byte_len(format, GpuTextureAspect::All, 7, 5, 3),
                Some(u64::from(7 * bytes * 5 * 3))
            );
        }
    }

    #[test]
    fn view_and_raw_copy_compatibility_share_the_same_storage_pairing() {
        for (left, right) in [
            (GpuTextureFormat::R8Unorm, GpuTextureFormat::R8Snorm),
            (GpuTextureFormat::R8Snorm, GpuTextureFormat::R8Uint),
            (GpuTextureFormat::R8Uint, GpuTextureFormat::R8Sint),
            (GpuTextureFormat::Rg8Unorm, GpuTextureFormat::Rg8Snorm),
            (GpuTextureFormat::Rg8Snorm, GpuTextureFormat::Rg8Uint),
            (GpuTextureFormat::Rg8Uint, GpuTextureFormat::Rg8Sint),
            (GpuTextureFormat::R8Unorm, GpuTextureFormat::Rg8Unorm),
            (GpuTextureFormat::R16Uint, GpuTextureFormat::R16Sint),
            (GpuTextureFormat::R16Sint, GpuTextureFormat::R16Float),
            (GpuTextureFormat::Rg16Uint, GpuTextureFormat::Rg16Sint),
            (GpuTextureFormat::Rg16Sint, GpuTextureFormat::Rg16Float),
            (GpuTextureFormat::R16Uint, GpuTextureFormat::Rg8Uint),
            (GpuTextureFormat::Rg16Float, GpuTextureFormat::R32Float),
        ] {
            assert!(!view_compatible(left, right));
            assert!(!view_compatible(right, left));
            assert!(!raw_copy_compatible(left, right));
            assert!(!raw_copy_compatible(right, left));
        }
        for (linear, srgb) in [
            (
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureFormat::Rgba8UnormSrgb,
            ),
            (
                GpuTextureFormat::Bgra8Unorm,
                GpuTextureFormat::Bgra8UnormSrgb,
            ),
        ] {
            assert!(view_compatible(linear, srgb));
            assert!(view_compatible(srgb, linear));
            assert!(raw_copy_compatible(linear, srgb));
            assert!(raw_copy_compatible(srgb, linear));
        }
        assert!(!view_compatible(
            GpuTextureFormat::Rgba8Unorm,
            GpuTextureFormat::Bgra8Unorm
        ));
        assert!(!view_compatible(
            GpuTextureFormat::Rgba8Snorm,
            GpuTextureFormat::Rgba8Unorm
        ));
        assert!(!raw_copy_compatible(
            GpuTextureFormat::Rgba8Uint,
            GpuTextureFormat::Rgba8Sint
        ));
        assert!(!raw_copy_compatible(
            GpuTextureFormat::R32Float,
            GpuTextureFormat::R32Uint
        ));
        assert!(!view_compatible(
            GpuTextureFormat::Rg32Float,
            GpuTextureFormat::Rg32Uint
        ));
        assert!(!raw_copy_compatible(
            GpuTextureFormat::Rgba32Sint,
            GpuTextureFormat::Rgba32Uint
        ));
        assert!(!view_compatible(
            GpuTextureFormat::Rgba16Float,
            GpuTextureFormat::Rgba16Uint
        ));
        assert!(!raw_copy_compatible(
            GpuTextureFormat::Rgba16Sint,
            GpuTextureFormat::Rgba16Uint
        ));
    }

    #[test]
    fn aspects_are_explicit_and_all_canonicalizes_only_when_unambiguous() {
        assert_eq!(
            canonical_copy_aspect(GpuTextureFormat::Rgba8Unorm, GpuTextureAspect::All),
            Some(GpuTextureAspect::Color)
        );
        assert_eq!(
            canonical_copy_aspect(GpuTextureFormat::Rgba8Snorm, GpuTextureAspect::All),
            Some(GpuTextureAspect::Color)
        );
        assert_eq!(
            canonical_copy_aspect(GpuTextureFormat::Rg32Float, GpuTextureAspect::All),
            Some(GpuTextureAspect::Color)
        );
        assert_eq!(
            canonical_copy_aspect(GpuTextureFormat::Rgba16Float, GpuTextureAspect::All),
            Some(GpuTextureAspect::Color)
        );
        assert_eq!(
            canonical_copy_aspect(GpuTextureFormat::Depth32Float, GpuTextureAspect::All),
            Some(GpuTextureAspect::DepthOnly)
        );
        assert!(!supports_aspect(
            GpuTextureFormat::Rgba8Unorm,
            GpuTextureAspect::DepthOnly
        ));
        assert!(!supports_aspect(
            GpuTextureFormat::Depth32Float,
            GpuTextureAspect::Color
        ));
        assert!(!supports_aspect(
            GpuTextureFormat::Depth32Float,
            GpuTextureAspect::StencilOnly
        ));
    }

    #[test]
    fn aspect_class_is_closed_over_exactly_four_normalized_states() {
        for (class, canonical, depth, stencil) in [
            (
                GpuTextureAspectClass::Color,
                GpuTextureAspect::Color,
                false,
                false,
            ),
            (
                GpuTextureAspectClass::Depth,
                GpuTextureAspect::DepthOnly,
                true,
                false,
            ),
            (
                GpuTextureAspectClass::Stencil,
                GpuTextureAspect::StencilOnly,
                false,
                true,
            ),
            (
                GpuTextureAspectClass::DepthStencil,
                GpuTextureAspect::All,
                true,
                true,
            ),
        ] {
            assert_eq!(class.canonical(GpuTextureAspect::All), Some(canonical));
            assert_eq!(class.has_depth(), depth);
            assert_eq!(class.has_stencil(), stencil);
            assert_eq!(
                class.supports(GpuTextureAspect::Color),
                class == GpuTextureAspectClass::Color
            );
            assert_eq!(
                class.supports(GpuTextureAspect::DepthOnly),
                matches!(
                    class,
                    GpuTextureAspectClass::Depth | GpuTextureAspectClass::DepthStencil
                )
            );
            assert_eq!(
                class.supports(GpuTextureAspect::StencilOnly),
                matches!(
                    class,
                    GpuTextureAspectClass::Stencil | GpuTextureAspectClass::DepthStencil
                )
            );
        }
    }

    #[test]
    fn aspect_identity_is_independent_from_copy_footprint() {
        let synthetic = GpuTextureFormatSemantics {
            aspect_class: GpuTextureAspectClass::Depth,
            block_dimensions: (1, 1),
            color_copy_block_size: None,
            depth_copy_block_size: None,
            stencil_copy_block_size: None,
            srgb: false,
            paired_view_format: None,
            scalar_class: GpuTextureScalarClass::Float,
            component_count: 1,
            has_alpha: false,
        };
        assert!(synthetic.aspect_class.supports(GpuTextureAspect::DepthOnly));
        assert_eq!(
            synthetic.aspect_class.canonical(GpuTextureAspect::All),
            Some(GpuTextureAspect::DepthOnly)
        );
        assert_eq!(synthetic.depth_copy_block_size, None);
    }

    #[test]
    fn depth_stencil_aspect_authority_distinguishes_selected_aspects() {
        let class = GpuTextureAspectClass::DepthStencil;
        assert!(class.supports(GpuTextureAspect::All));
        assert!(class.supports(GpuTextureAspect::DepthOnly));
        assert!(class.supports(GpuTextureAspect::StencilOnly));
        assert!(!class.supports(GpuTextureAspect::Color));
        assert_eq!(
            class.canonical(GpuTextureAspect::All),
            Some(GpuTextureAspect::All)
        );
        assert_eq!(
            class.canonical(GpuTextureAspect::DepthOnly),
            Some(GpuTextureAspect::DepthOnly)
        );
        assert_eq!(
            class.canonical(GpuTextureAspect::StencilOnly),
            Some(GpuTextureAspect::StencilOnly)
        );
    }
}
