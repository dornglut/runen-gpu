use super::{
    GpuRenderColorAttachment, GpuRenderDepthStencilAttachment, GpuRenderDraw,
    GpuRenderPipelineDescriptor, GpuScissorRect, GpuTextureDimension, GpuTextureFormat,
    GpuTextureHandle, GpuTextureViewHandle, GpuWorkOperationCause, GpuWorkOperationError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuRenderExtent {
    width: u32,
    height: u32,
}

impl GpuRenderExtent {
    const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub const fn width(self) -> u32 {
        self.width
    }

    pub const fn height(self) -> u32 {
        self.height
    }
}

/// Backend-neutral compatibility signature shared by every draw in one logical render pass.
///
/// This intentionally excludes blend, primitive, vertex, binding, and dynamic state. Those remain
/// draw-local pipeline/execution semantics. The signature contains only the attachment facts that
/// must remain compatible for the lifetime of one render pass.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuRenderPassSignature {
    extent: GpuRenderExtent,
    sample_count: u32,
    color_formats: Vec<GpuTextureFormat>,
    depth_stencil_format: Option<GpuTextureFormat>,
}

impl GpuRenderPassSignature {
    pub fn from_attachments(
        color_attachments: &[GpuRenderColorAttachment],
        depth_stencil_attachment: Option<&GpuRenderDepthStencilAttachment>,
    ) -> Result<Self, GpuWorkOperationError> {
        let first = color_attachments
            .first()
            .map(|attachment| attachment_fact(attachment.source()))
            .or_else(|| depth_stencil_attachment.map(|attachment| attachment_fact(attachment.source())))
            .ok_or_else(|| {
                invalid_attachment_signature(
                    "attachments=0",
                    "derive a render-pass signature only when the pass has a color or depth attachment",
                )
            })?;

        let mut color_formats = Vec::with_capacity(color_attachments.len());
        for attachment in color_attachments {
            let fact = attachment_fact(attachment.source());
            validate_attachment_compatibility(first, fact)?;
            color_formats.push(fact.format);
        }

        let depth_stencil_format = depth_stencil_attachment
            .map(|attachment| {
                let fact = attachment_fact(attachment.source());
                validate_attachment_compatibility(first, fact)?;
                Ok(fact.format)
            })
            .transpose()?;

        Ok(Self {
            extent: first.extent,
            sample_count: first.sample_count,
            color_formats,
            depth_stencil_format,
        })
    }

    pub const fn extent(&self) -> GpuRenderExtent {
        self.extent
    }

    pub const fn sample_count(&self) -> u32 {
        self.sample_count
    }

    pub fn color_formats(&self) -> &[GpuTextureFormat] {
        &self.color_formats
    }

    pub const fn depth_stencil_format(&self) -> Option<GpuTextureFormat> {
        self.depth_stencil_format
    }

    pub fn validate_pipeline(
        &self,
        pipeline: &GpuRenderPipelineDescriptor,
    ) -> Result<(), GpuWorkOperationError> {
        let state = pipeline.state();
        let pipeline_color_formats = state
            .fragment_output()
            .map(|output| {
                output
                    .color_targets()
                    .map(|target| target.format())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let pipeline_depth_format = state.depth_stencil().map(|depth| depth.format());
        let pipeline_sample_count = state.multisample().sample_count();

        if pipeline_color_formats != self.color_formats
            || pipeline_depth_format != self.depth_stencil_format
            || pipeline_sample_count != self.sample_count
        {
            return Err(GpuWorkOperationError::invalid(
                "validate GPU render draw against pass signature",
                format!(
                    "pipeline colors={pipeline_color_formats:?}, depth={pipeline_depth_format:?}, samples={pipeline_sample_count}"
                ),
                None,
                GpuWorkOperationCause::InvalidDraw,
                "use a render pipeline whose ordered color formats, depth format, and sample count match the render pass",
            ));
        }
        Ok(())
    }

    pub fn validate_scissor(&self, scissor: GpuScissorRect) -> Result<(), GpuWorkOperationError> {
        if scissor.end_x() > self.extent.width() || scissor.end_y() > self.extent.height() {
            return Err(GpuWorkOperationError::invalid(
                "validate GPU render scissor against pass signature",
                format!(
                    "scissor=({}, {})..({}, {}), extent={}x{}",
                    scissor.x(),
                    scissor.y(),
                    scissor.end_x(),
                    scissor.end_y(),
                    self.extent.width(),
                    self.extent.height(),
                ),
                None,
                GpuWorkOperationCause::InvalidDraw,
                "keep the scissor rectangle inside the effective render extent",
            ));
        }
        Ok(())
    }

    pub fn validate_draw(&self, draw: &GpuRenderDraw) -> Result<(), GpuWorkOperationError> {
        self.validate_pipeline(draw.pipeline())?;
        self.validate_scissor(draw.scissor())
    }
}

#[derive(Debug, Clone, Copy)]
struct AttachmentFact {
    extent: GpuRenderExtent,
    sample_count: u32,
    format: GpuTextureFormat,
}

fn attachment_fact(view: &GpuTextureViewHandle) -> AttachmentFact {
    let descriptor = view.descriptor();
    let texture = descriptor.texture();
    AttachmentFact {
        extent: render_extent(texture, descriptor.subresources().base_mip_level()),
        sample_count: texture.descriptor().sample_count(),
        format: descriptor
            .format()
            .unwrap_or_else(|| texture.descriptor().format()),
    }
}

fn validate_attachment_compatibility(
    expected: AttachmentFact,
    actual: AttachmentFact,
) -> Result<(), GpuWorkOperationError> {
    if actual.extent != expected.extent || actual.sample_count != expected.sample_count {
        return Err(invalid_attachment_signature(
            format!(
                "expected extent={}x{} samples={}, actual extent={}x{} samples={}",
                expected.extent.width(),
                expected.extent.height(),
                expected.sample_count,
                actual.extent.width(),
                actual.extent.height(),
                actual.sample_count,
            ),
            "use attachments with one effective render extent and sample count",
        ));
    }
    Ok(())
}

fn render_extent(texture: &GpuTextureHandle, mip_level: u32) -> GpuRenderExtent {
    let extent = texture.descriptor().extent();
    let width = (extent.width() >> mip_level).max(1);
    let height = match texture.descriptor().dimension() {
        GpuTextureDimension::D1 => 1,
        GpuTextureDimension::D2 | GpuTextureDimension::D3 => (extent.height() >> mip_level).max(1),
    };
    GpuRenderExtent::new(width, height)
}

fn invalid_attachment_signature(
    label: impl Into<String>,
    correction: &'static str,
) -> GpuWorkOperationError {
    GpuWorkOperationError::invalid(
        "derive GPU render-pass signature",
        label,
        None,
        GpuWorkOperationCause::InvalidAttachment,
        correction,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GpuAttachmentStore, GpuColorAttachmentLoad, GpuColorClearValue, GpuMemoryIntent,
        GpuReconstruction, GpuResourceCommon, GpuResourceLabel, GpuResourceLifetime,
        GpuResourceProvenance, GpuTextureAspect, GpuTextureDescriptor, GpuTextureExtent,
        GpuTextureInitialization, GpuTextureSubresourceRange, GpuTextureUsage, GpuTextureUsages,
        GpuTextureViewDescriptor, GpuWorkResourceIdAllocator,
    };
    use std::num::NonZeroU64;

    fn label(value: &str) -> GpuResourceLabel {
        GpuResourceLabel::new(value).unwrap()
    }

    fn common(value: &str) -> GpuResourceCommon {
        let label = label(value);
        GpuResourceCommon::owned(
            label.clone(),
            GpuResourceLifetime::Transient,
            GpuMemoryIntent::Device,
            GpuReconstruction::SourceBacked,
            GpuResourceProvenance::new(label, None, None),
        )
        .unwrap()
    }

    fn color_texture(
        allocator: &mut GpuWorkResourceIdAllocator,
        name: &str,
        width: u32,
        height: u32,
        mip_levels: u32,
        sample_count: u32,
    ) -> GpuTextureHandle {
        let resource_label = label(name);
        allocator
            .allocate_texture_handle(
                GpuTextureDescriptor::new(
                    common(name),
                    GpuTextureDimension::D2,
                    GpuTextureExtent::new(
                        &resource_label,
                        GpuTextureDimension::D2,
                        width,
                        height,
                        1,
                    )
                    .unwrap(),
                    mip_levels,
                    sample_count,
                    GpuTextureFormat::Rgba8Unorm,
                    GpuTextureUsages::new(&resource_label, [GpuTextureUsage::ColorAttachment])
                        .unwrap(),
                    GpuTextureInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap()
    }

    fn color_attachment(
        allocator: &mut GpuWorkResourceIdAllocator,
        texture: &GpuTextureHandle,
        mip_level: u32,
    ) -> GpuRenderColorAttachment {
        let subresources = GpuTextureSubresourceRange::new(
            texture.descriptor().common().label(),
            mip_level,
            1,
            0,
            1,
            GpuTextureAspect::Color,
        )
        .unwrap();
        let view = allocator
            .allocate_texture_view_handle(
                GpuTextureViewDescriptor::new(
                    common(&format!(
                        "{} view",
                        texture.descriptor().common().label().as_str()
                    )),
                    texture,
                    None,
                    GpuTextureDimension::D2,
                    subresources,
                )
                .unwrap(),
            )
            .unwrap();
        GpuRenderColorAttachment::new(
            view,
            GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
            GpuAttachmentStore::Store,
            None,
        )
        .unwrap()
    }

    #[test]
    fn signature_uses_effective_mip_extent_and_rejects_attachment_mismatch() {
        let mut allocator =
            GpuWorkResourceIdAllocator::for_owner_scope(NonZeroU64::new(91).unwrap());
        let first_texture = color_texture(&mut allocator, "first", 32, 16, 2, 1);
        let first = color_attachment(&mut allocator, &first_texture, 1);
        let second_texture = color_texture(&mut allocator, "second", 16, 8, 1, 1);
        let second = color_attachment(&mut allocator, &second_texture, 0);
        let signature = GpuRenderPassSignature::from_attachments(&[first, second], None).unwrap();
        assert_eq!(signature.extent(), GpuRenderExtent::new(16, 8));
        assert_eq!(signature.sample_count(), 1);
        assert_eq!(
            signature.color_formats(),
            &[GpuTextureFormat::Rgba8Unorm, GpuTextureFormat::Rgba8Unorm]
        );

        let mismatch_texture = color_texture(&mut allocator, "mismatch", 17, 8, 1, 1);
        let mismatch = color_attachment(&mut allocator, &mismatch_texture, 0);
        let reference_texture = color_texture(&mut allocator, "reference", 16, 8, 1, 1);
        let reference = color_attachment(&mut allocator, &reference_texture, 0);
        assert!(GpuRenderPassSignature::from_attachments(&[mismatch, reference], None).is_err());
    }

    #[test]
    fn scissor_is_checked_against_effective_extent_and_zero_area_is_valid() {
        let mut allocator =
            GpuWorkResourceIdAllocator::for_owner_scope(NonZeroU64::new(92).unwrap());
        let target = color_texture(&mut allocator, "target", 16, 8, 1, 1);
        let attachment = color_attachment(&mut allocator, &target, 0);
        let signature = GpuRenderPassSignature::from_attachments(&[attachment], None).unwrap();

        assert!(
            signature
                .validate_scissor(GpuScissorRect::new(15, 7, 1, 1).unwrap())
                .is_ok()
        );
        assert!(
            signature
                .validate_scissor(GpuScissorRect::new(16, 8, 0, 0).unwrap())
                .is_ok()
        );
        assert!(
            signature
                .validate_scissor(GpuScissorRect::new(16, 0, 1, 1).unwrap())
                .is_err()
        );
    }
}
