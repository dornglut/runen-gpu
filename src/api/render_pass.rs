use super::{
    GpuCapabilityFeature, GpuCapabilityRequirement, GpuCapabilityRequirements, GpuLimits,
    GpuRenderColorAttachment, GpuRenderDepthStencilAttachment, GpuRenderDraw,
    GpuRenderPipelineDescriptor, GpuScissorRect, GpuTextureDimension, GpuTextureFormat,
    GpuTextureHandle, GpuTextureViewDimension, GpuTextureViewHandle, GpuWorkOperationCause,
    GpuWorkOperationError,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuMultiviewState {
    view_count: u32,
}

impl GpuMultiviewState {
    pub fn new(view_count: u32) -> Result<Self, GpuWorkOperationError> {
        if !(2..=31).contains(&view_count) {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU multiview state",
                format!("view_count={view_count}"),
                None,
                GpuWorkOperationCause::InvalidMultiview,
                "use a contiguous multiview count from 2 through 31",
            ));
        }
        Ok(Self { view_count })
    }

    pub const fn view_count(self) -> u32 {
        self.view_count
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
    multiview: Option<GpuMultiviewState>,
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
            multiview: first.multiview,
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

    pub const fn multiview(&self) -> Option<GpuMultiviewState> {
        self.multiview
    }

    pub fn requirements(&self) -> GpuCapabilityRequirements {
        let mut requirements = GpuCapabilityRequirements::new();
        if self.multiview.is_some() {
            requirements
                .insert(GpuCapabilityRequirement::Required(
                    GpuCapabilityFeature::Multiview,
                ))
                .expect("one render-pass multiview requirement cannot conflict");
        }
        requirements
    }

    pub fn validate_limits(&self, limits: GpuLimits) -> Result<(), GpuWorkOperationError> {
        if let Some(multiview) = self.multiview {
            let admitted = limits.max_multiview_view_count();
            if multiview.view_count() > admitted {
                return Err(GpuWorkOperationError::invalid(
                    "validate GPU render-pass multiview limits",
                    format!(
                        "view_count={}, admitted_max={admitted}",
                        multiview.view_count()
                    ),
                    None,
                    GpuWorkOperationCause::InvalidMultiview,
                    "request a multiview workload budget at least as large as the render-pass view count",
                ));
            }
        }
        Ok(())
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
        let pipeline_multiview = state.multiview();

        if pipeline_color_formats != self.color_formats
            || pipeline_depth_format != self.depth_stencil_format
            || pipeline_sample_count != self.sample_count
            || pipeline_multiview != self.multiview
        {
            return Err(GpuWorkOperationError::invalid(
                "validate GPU render draw against pass signature",
                format!(
                    "pipeline colors={pipeline_color_formats:?}, depth={pipeline_depth_format:?}, samples={pipeline_sample_count}, multiview={pipeline_multiview:?}"
                ),
                None,
                GpuWorkOperationCause::InvalidDraw,
                "use a render pipeline whose ordered color formats, depth format, sample count, and multiview state match the render pass",
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
    multiview: Option<GpuMultiviewState>,
    format: GpuTextureFormat,
}

fn attachment_fact(view: &GpuTextureViewHandle) -> AttachmentFact {
    let descriptor = view.descriptor();
    let texture = descriptor.texture();
    let multiview = match descriptor.dimension() {
        GpuTextureViewDimension::D2 => None,
        GpuTextureViewDimension::D2Array => Some(
            GpuMultiviewState::new(descriptor.subresources().array_layer_count())
                .expect("validated D2Array render attachments retain a checked multiview count"),
        ),
        _ => unreachable!("render attachments admit only D2 or D2Array views"),
    };
    AttachmentFact {
        extent: render_extent(texture, descriptor.subresources().base_mip_level()),
        sample_count: texture.descriptor().sample_count(),
        multiview,
        format: descriptor
            .format()
            .unwrap_or_else(|| texture.descriptor().format()),
    }
}

fn validate_attachment_compatibility(
    expected: AttachmentFact,
    actual: AttachmentFact,
) -> Result<(), GpuWorkOperationError> {
    if actual.extent != expected.extent
        || actual.sample_count != expected.sample_count
        || actual.multiview != expected.multiview
    {
        return Err(invalid_attachment_signature(
            format!(
                "expected extent={}x{} samples={} multiview={:?}, actual extent={}x{} samples={} multiview={:?}",
                expected.extent.width(),
                expected.extent.height(),
                expected.sample_count,
                expected.multiview,
                actual.extent.width(),
                actual.extent.height(),
                actual.sample_count,
                actual.multiview,
            ),
            "use attachments with one effective render extent, sample count, and multiview state",
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
        GpuTextureViewDescriptor, GpuTextureViewDimension, GpuWorkResourceIdAllocator,
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
                    GpuTextureViewDimension::D2,
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

    fn layered_color_attachment(
        allocator: &mut GpuWorkResourceIdAllocator,
        name: &str,
        base_layer: u32,
        layer_count: u32,
    ) -> GpuRenderColorAttachment {
        let resource_label = label(name);
        let texture = allocator
            .allocate_texture_handle(
                GpuTextureDescriptor::new(
                    common(name),
                    GpuTextureDimension::D2,
                    GpuTextureExtent::new(
                        &resource_label,
                        GpuTextureDimension::D2,
                        16,
                        8,
                        base_layer + layer_count,
                    )
                    .unwrap(),
                    1,
                    1,
                    GpuTextureFormat::Rgba8Unorm,
                    GpuTextureUsages::new(
                        &resource_label,
                        [GpuTextureUsage::ColorAttachment],
                    )
                    .unwrap(),
                    GpuTextureInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();
        let subresources = GpuTextureSubresourceRange::new(
            texture.descriptor().common().label(),
            0,
            1,
            base_layer,
            layer_count,
            GpuTextureAspect::Color,
        )
        .unwrap();
        let view = allocator
            .allocate_texture_view_handle(
                GpuTextureViewDescriptor::new(
                    common(&format!("{name} view")),
                    &texture,
                    None,
                    GpuTextureViewDimension::D2Array,
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
    fn multiview_state_and_signature_enforce_the_normalized_2_through_31_domain() {
        for rejected in [0, 1, 32, u32::MAX] {
            let error = GpuMultiviewState::new(rejected)
                .expect_err("outside-domain multiview count must reject structurally");
            assert_eq!(error.cause(), GpuWorkOperationCause::InvalidMultiview);
        }
        assert_eq!(GpuMultiviewState::new(2).unwrap().view_count(), 2);
        assert_eq!(GpuMultiviewState::new(31).unwrap().view_count(), 31);

        let mut allocator =
            GpuWorkResourceIdAllocator::for_owner_scope(NonZeroU64::new(93).unwrap());
        let first = layered_color_attachment(&mut allocator, "layered first", 1, 2);
        assert_eq!(
            first.source().descriptor().subresources().base_array_layer(),
            1
        );
        assert_eq!(
            first.source().descriptor().subresources().array_layer_count(),
            2
        );
        let second = layered_color_attachment(&mut allocator, "layered second", 4, 2);
        let signature = GpuRenderPassSignature::from_attachments(&[first, second], None).unwrap();
        assert_eq!(
            signature.multiview(),
            Some(GpuMultiviewState::new(2).unwrap())
        );
        assert!(matches!(
            signature
                .requirements()
                .get(GpuCapabilityFeature::Multiview),
            Some(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::Multiview
            ))
        ));
        assert!(
            signature
                .validate_limits(
                    GpuLimits::new(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1,)
                        .unwrap()
                        .with_multiview_limit(1)
                )
                .is_err()
        );
        assert!(
            signature
                .validate_limits(
                    GpuLimits::new(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1,)
                        .unwrap()
                        .with_multiview_limit(2)
                )
                .is_ok()
        );
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
