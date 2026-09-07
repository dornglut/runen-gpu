use super::super::{
    GpuBufferAccess, GpuBufferAccessKind, GpuBufferHandle, GpuBufferRange, GpuTextureAccess,
    GpuTextureAccessKind, GpuTextureAccessResource, GpuTextureAspect, GpuTextureDimension,
    GpuTextureHandle, GpuTextureSubresourceRange, GpuWorkOperationCause, GpuWorkOperationError,
    GpuWorkResourceId, gpu_texture_formats_copy_compatible,
};
use super::mip_extent;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuTextureOrigin {
    x: u32,
    y: u32,
    z: u32,
}

impl GpuTextureOrigin {
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuCopyExtent {
    width: u32,
    height: u32,
    depth_or_layers: u32,
}

impl GpuCopyExtent {
    pub fn new(
        width: u32,
        height: u32,
        depth_or_layers: u32,
    ) -> Result<Self, GpuWorkOperationError> {
        if width == 0 || height == 0 || depth_or_layers == 0 {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU copy extent",
                "copy",
                None,
                GpuWorkOperationCause::InvalidCopyRegion,
                "provide nonzero width, height, and depth-or-layer coverage",
            ));
        }
        Ok(Self {
            width,
            height,
            depth_or_layers,
        })
    }
    pub const fn width(self) -> u32 {
        self.width
    }
    pub const fn height(self) -> u32 {
        self.height
    }
    pub const fn depth_or_layers(self) -> u32 {
        self.depth_or_layers
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuBufferRegion {
    buffer: GpuBufferHandle,
    range: GpuBufferRange,
}

impl GpuBufferRegion {
    pub fn new(
        buffer: &GpuBufferHandle,
        range: GpuBufferRange,
    ) -> Result<Self, GpuWorkOperationError> {
        GpuBufferRange::new(buffer, range.offset(), range.size()).map_err(|source| {
            GpuWorkOperationError::from_access(
                "construct GPU buffer region",
                buffer.descriptor().common().label().as_str(),
                GpuWorkOperationCause::InvalidCopyRegion,
                "provide a checked nonempty buffer region",
                source,
            )
        })?;
        Ok(Self {
            buffer: buffer.clone(),
            range,
        })
    }

    pub fn buffer(&self) -> &GpuBufferHandle {
        &self.buffer
    }

    pub const fn range(&self) -> GpuBufferRange {
        self.range
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuTextureCopyRegion {
    texture: GpuTextureHandle,
    mip_level: u32,
    origin: GpuTextureOrigin,
    aspect: GpuTextureAspect,
    extent: GpuCopyExtent,
    subresources: GpuTextureSubresourceRange,
}

impl GpuTextureCopyRegion {
    pub fn new(
        texture: &GpuTextureHandle,
        mip_level: u32,
        origin: GpuTextureOrigin,
        aspect: GpuTextureAspect,
        extent: GpuCopyExtent,
    ) -> Result<Self, GpuWorkOperationError> {
        let descriptor = texture.descriptor();
        let label = descriptor.common().label().as_str();
        if descriptor.sample_count() != 1 || mip_level >= descriptor.mip_level_count() {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU texture copy region",
                label,
                Some(texture.diagnostic_identity()),
                GpuWorkOperationCause::InvalidCopyRegion,
                "use a valid mip of a single-sampled texture",
            ));
        }
        let (mip_width, mip_height, mip_depth_or_layers) = mip_extent(texture, mip_level);
        let x_end = origin.x().checked_add(extent.width());
        let y_end = origin.y().checked_add(extent.height());
        let z_end = origin.z().checked_add(extent.depth_or_layers());
        let dimension_valid = match descriptor.dimension() {
            GpuTextureDimension::D1 => {
                origin.y() == 0
                    && origin.z() == 0
                    && extent.height() == 1
                    && extent.depth_or_layers() == 1
            }
            GpuTextureDimension::D2 => true,
            GpuTextureDimension::D3 => true,
        };
        let aspect_valid = if descriptor.format().is_depth() {
            matches!(aspect, GpuTextureAspect::All | GpuTextureAspect::DepthOnly)
        } else {
            matches!(aspect, GpuTextureAspect::All | GpuTextureAspect::Color)
        };
        if !dimension_valid
            || !aspect_valid
            || x_end.is_none_or(|end| end > mip_width)
            || y_end.is_none_or(|end| end > mip_height)
            || z_end.is_none_or(|end| end > mip_depth_or_layers)
        {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU texture copy region",
                label,
                Some(texture.diagnostic_identity()),
                GpuWorkOperationCause::InvalidCopyRegion,
                "keep origin, extent, and aspect inside the selected mip",
            ));
        }
        if descriptor.format().is_depth()
            && (origin.x() != 0
                || origin.y() != 0
                || extent.width() != mip_width
                || extent.height() != mip_height)
        {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU texture copy region",
                label,
                Some(texture.diagnostic_identity()),
                GpuWorkOperationCause::InvalidCopyRegion,
                "copy the complete depth mip plane from zero x/y origin",
            ));
        }
        let (base_array_layer, array_layer_count) = match descriptor.dimension() {
            GpuTextureDimension::D2 => (origin.z(), extent.depth_or_layers()),
            GpuTextureDimension::D1 | GpuTextureDimension::D3 => (0, 1),
        };
        let canonical_aspect = if descriptor.format().is_depth() {
            GpuTextureAspect::DepthOnly
        } else {
            GpuTextureAspect::Color
        };
        let subresources = GpuTextureSubresourceRange::new(
            descriptor.common().label(),
            mip_level,
            1,
            base_array_layer,
            array_layer_count,
            canonical_aspect,
        )
        .map_err(|_| {
            GpuWorkOperationError::invalid(
                "construct GPU texture copy region",
                label,
                Some(texture.diagnostic_identity()),
                GpuWorkOperationCause::InvalidCopyRegion,
                "provide a checked texture subresource region",
            )
        })?;
        Ok(Self {
            texture: texture.clone(),
            mip_level,
            origin,
            aspect: canonical_aspect,
            extent,
            subresources,
        })
    }

    pub fn texture(&self) -> &GpuTextureHandle {
        &self.texture
    }
    pub const fn mip_level(&self) -> u32 {
        self.mip_level
    }
    pub const fn origin(&self) -> GpuTextureOrigin {
        self.origin
    }
    pub const fn aspect(&self) -> GpuTextureAspect {
        self.aspect
    }
    pub const fn extent(&self) -> GpuCopyExtent {
        self.extent
    }
    pub const fn subresources(&self) -> GpuTextureSubresourceRange {
        self.subresources
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuBufferTextureLayout {
    buffer: GpuBufferHandle,
    byte_offset: u64,
    bytes_per_row: u32,
    rows_per_image: u32,
}

impl GpuBufferTextureLayout {
    pub fn new(
        buffer: &GpuBufferHandle,
        byte_offset: u64,
        bytes_per_row: u32,
        rows_per_image: u32,
    ) -> Result<Self, GpuWorkOperationError> {
        if bytes_per_row == 0 {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU buffer-texture layout",
                buffer.descriptor().common().label().as_str(),
                Some(buffer.diagnostic_identity()),
                GpuWorkOperationCause::InvalidCopyLayout,
                "provide a nonzero logical bytes-per-row value",
            ));
        }
        if byte_offset >= buffer.descriptor().size_bytes() {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU buffer-texture layout",
                buffer.descriptor().common().label().as_str(),
                Some(buffer.diagnostic_identity()),
                GpuWorkOperationCause::InvalidCopyLayout,
                "keep the byte offset inside the buffer descriptor",
            ));
        }
        Ok(Self {
            buffer: buffer.clone(),
            byte_offset,
            bytes_per_row,
            rows_per_image,
        })
    }

    pub fn buffer(&self) -> &GpuBufferHandle {
        &self.buffer
    }
    pub const fn byte_offset(&self) -> u64 {
        self.byte_offset
    }
    pub const fn bytes_per_row(&self) -> u32 {
        self.bytes_per_row
    }
    pub const fn rows_per_image(&self) -> u32 {
        self.rows_per_image
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuCopyOperation {
    BufferToBuffer {
        source: GpuBufferRegion,
        destination: GpuBufferRegion,
    },
    BufferToTexture {
        source: GpuBufferTextureLayout,
        destination: GpuTextureCopyRegion,
    },
    TextureToBuffer {
        source: GpuTextureCopyRegion,
        destination: GpuBufferTextureLayout,
    },
    TextureToTexture {
        source: GpuTextureCopyRegion,
        destination: GpuTextureCopyRegion,
    },
}

impl GpuCopyOperation {
    pub fn buffer_to_buffer(
        source: GpuBufferRegion,
        destination: GpuBufferRegion,
    ) -> Result<Self, GpuWorkOperationError> {
        if source.range().size() != destination.range().size()
            || (source.buffer() == destination.buffer()
                && source.range().overlaps(destination.range()))
        {
            return Err(copy_error(
                "construct GPU buffer-to-buffer copy",
                source.buffer().diagnostic_identity(),
                "use equal-sized non-overlapping source and destination regions",
            ));
        }
        buffer_access(
            source.buffer(),
            source.range(),
            GpuBufferAccessKind::CopySource,
            "construct GPU buffer-to-buffer copy",
        )?;
        buffer_access(
            destination.buffer(),
            destination.range(),
            GpuBufferAccessKind::CopyDestination,
            "construct GPU buffer-to-buffer copy",
        )?;
        Ok(Self::BufferToBuffer {
            source,
            destination,
        })
    }

    pub fn buffer_to_texture(
        source: GpuBufferTextureLayout,
        destination: GpuTextureCopyRegion,
    ) -> Result<Self, GpuWorkOperationError> {
        validate_buffer_texture_layout(&source, &destination, GpuBufferAccessKind::CopySource)?;
        texture_copy_access(&destination, GpuTextureAccessKind::CopyDestination)?;
        Ok(Self::BufferToTexture {
            source,
            destination,
        })
    }

    pub fn texture_to_buffer(
        source: GpuTextureCopyRegion,
        destination: GpuBufferTextureLayout,
    ) -> Result<Self, GpuWorkOperationError> {
        texture_copy_access(&source, GpuTextureAccessKind::CopySource)?;
        validate_buffer_texture_layout(
            &destination,
            &source,
            GpuBufferAccessKind::CopyDestination,
        )?;
        Ok(Self::TextureToBuffer {
            source,
            destination,
        })
    }

    pub fn texture_to_texture(
        source: GpuTextureCopyRegion,
        destination: GpuTextureCopyRegion,
    ) -> Result<Self, GpuWorkOperationError> {
        let copy_compatible = gpu_texture_formats_copy_compatible(
            source.texture().descriptor().format(),
            destination.texture().descriptor().format(),
        );
        let same_extent = source.extent() == destination.extent();
        let aliases = source.texture() == destination.texture()
            && source
                .subresources()
                .overlaps(destination.subresources(), source.aspect());
        if !copy_compatible || !same_extent || source.aspect() != destination.aspect() || aliases {
            return Err(copy_error(
                "construct GPU texture-to-texture copy",
                source.texture().diagnostic_identity(),
                "use copy-compatible formats, matching aspects/extents, and non-overlapping source/destination storage",
            ));
        }
        texture_copy_access(&source, GpuTextureAccessKind::CopySource)?;
        texture_copy_access(&destination, GpuTextureAccessKind::CopyDestination)?;
        Ok(Self::TextureToTexture {
            source,
            destination,
        })
    }
}

fn buffer_access(
    buffer: &GpuBufferHandle,
    range: GpuBufferRange,
    kind: GpuBufferAccessKind,
    operation: &'static str,
) -> Result<GpuBufferAccess, GpuWorkOperationError> {
    GpuBufferAccess::new(buffer, range, kind).map_err(|source| {
        GpuWorkOperationError::from_access(
            operation,
            buffer.descriptor().common().label().as_str(),
            GpuWorkOperationCause::InvalidCopyRegion,
            "declare matching copy usage and checked coverage",
            source,
        )
    })
}

fn texture_copy_access(
    region: &GpuTextureCopyRegion,
    kind: GpuTextureAccessKind,
) -> Result<GpuTextureAccess, GpuWorkOperationError> {
    GpuTextureAccess::new(
        GpuTextureAccessResource::Texture(region.texture().clone()),
        region.subresources(),
        kind,
    )
    .map_err(|source| {
        GpuWorkOperationError::from_access(
            "construct GPU texture copy",
            region.texture().descriptor().common().label().as_str(),
            GpuWorkOperationCause::InvalidCopyRegion,
            "declare matching texture copy usage and checked coverage",
            source,
        )
    })
}

fn validate_buffer_texture_layout(
    layout: &GpuBufferTextureLayout,
    texture: &GpuTextureCopyRegion,
    kind: GpuBufferAccessKind,
) -> Result<GpuBufferAccess, GpuWorkOperationError> {
    buffer_layout_access(layout, texture, kind)
}

fn buffer_layout_access(
    layout: &GpuBufferTextureLayout,
    texture: &GpuTextureCopyRegion,
    kind: GpuBufferAccessKind,
) -> Result<GpuBufferAccess, GpuWorkOperationError> {
    let extent = texture.extent();
    let logical_row = extent
        .width()
        .checked_mul(texture.texture().descriptor().format().bytes_per_texel())
        .ok_or_else(|| {
            copy_layout_error(
                layout,
                "reduce the copy width so logical row size does not overflow",
            )
        })?;
    if layout.bytes_per_row() < logical_row
        || (extent.depth_or_layers() > 1 && layout.rows_per_image() < extent.height())
        || (extent.depth_or_layers() == 1
            && layout.rows_per_image() != 0
            && layout.rows_per_image() < extent.height())
    {
        return Err(copy_layout_error(
            layout,
            "provide bytes-per-row and rows-per-image covering the complete logical copy",
        ));
    }
    let image_rows = if extent.depth_or_layers() > 1 {
        layout.rows_per_image()
    } else {
        0
    };
    let image_stride = u64::from(layout.bytes_per_row())
        .checked_mul(u64::from(image_rows))
        .ok_or_else(|| copy_layout_error(layout, "reduce the logical image stride"))?;
    let preceding_images = u64::from(extent.depth_or_layers() - 1)
        .checked_mul(image_stride)
        .ok_or_else(|| copy_layout_error(layout, "reduce the copy depth or layer count"))?;
    let preceding_rows = u64::from(extent.height() - 1)
        .checked_mul(u64::from(layout.bytes_per_row()))
        .ok_or_else(|| copy_layout_error(layout, "reduce the copy height"))?;
    let size = preceding_images
        .checked_add(preceding_rows)
        .and_then(|value| value.checked_add(u64::from(logical_row)))
        .ok_or_else(|| copy_layout_error(layout, "reduce the logical copy byte coverage"))?;
    let range =
        GpuBufferRange::new(layout.buffer(), layout.byte_offset(), size).map_err(|source| {
            GpuWorkOperationError::from_access(
                "validate GPU buffer-texture layout",
                layout.buffer().descriptor().common().label().as_str(),
                GpuWorkOperationCause::InvalidCopyLayout,
                "keep the complete logical row and image coverage inside the buffer",
                source,
            )
        })?;
    buffer_access(
        layout.buffer(),
        range,
        kind,
        "construct GPU buffer-texture copy",
    )
}

fn copy_layout_error(
    layout: &GpuBufferTextureLayout,
    correction: &'static str,
) -> GpuWorkOperationError {
    GpuWorkOperationError::invalid(
        "validate GPU buffer-texture layout",
        layout.buffer().descriptor().common().label().as_str(),
        Some(layout.buffer().diagnostic_identity()),
        GpuWorkOperationCause::InvalidCopyLayout,
        correction,
    )
}

fn copy_error(
    operation: &'static str,
    resource: GpuWorkResourceId,
    correction: &'static str,
) -> GpuWorkOperationError {
    GpuWorkOperationError::invalid(
        operation,
        "copy",
        Some(resource),
        GpuWorkOperationCause::InvalidCopyRegion,
        correction,
    )
}
