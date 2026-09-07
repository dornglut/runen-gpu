use super::{
    GpuAttachmentStore, GpuBlendConstant, GpuColorAttachmentLoad, GpuDrawIntent, GpuDrawRange,
    GpuRenderColorAttachment, GpuRenderDraw, GpuRenderOperation, GpuRenderPassSignature,
    GpuRenderPipelineDescriptor, GpuRuntimeBindingSet, GpuScissorRect, GpuTextureViewHandle,
    GpuViewport, GpuWorkOperationCause, GpuWorkOperationError,
};

impl GpuRenderOperation {
    /// Constructs one direct single-color draw covering the complete effective target extent.
    ///
    /// The caller keeps the workload-significant choices explicit: render pipeline, runtime
    /// bindings, target view, attachment load/store semantics, vertex range, and instance range.
    /// This constrained ordinary path derives only administration fixed by the operation shape:
    /// no resolve/depth/timestamp attachments, no host vertex/index buffers, a full-target
    /// viewport/scissor, zero blend constant, and zero stencil reference. The target must be
    /// single-sampled so omitting multisample resolve cannot hide a workload decision.
    ///
    /// Target compatibility, pipeline/pass compatibility, runtime bindings, draw ranges, resource
    /// accesses, and render-pass usage remain validated by the existing canonical constructors.
    /// Indexed/indirect drawing, multisample resolve, partial viewport/scissor state,
    /// depth/timestamp work, host vertex/index buffers, and explicit dynamic blend/stencil state
    /// remain available through [`GpuRenderDraw::new`] and [`GpuRenderOperation::new`].
    pub fn ordinary_color_full_target_direct(
        pipeline: &GpuRenderPipelineDescriptor,
        bindings: GpuRuntimeBindingSet,
        target: &GpuTextureViewHandle,
        load: GpuColorAttachmentLoad,
        store: GpuAttachmentStore,
        vertices: GpuDrawRange,
        instances: GpuDrawRange,
    ) -> Result<Self, GpuWorkOperationError> {
        let attachment = GpuRenderColorAttachment::new(target.clone(), load, store, None)?;
        let signature =
            GpuRenderPassSignature::from_attachments(std::slice::from_ref(&attachment), None)?;
        if signature.sample_count() != 1 {
            return Err(GpuWorkOperationError::invalid(
                "construct ordinary full-target direct GPU color render",
                format!("sample_count={}", signature.sample_count()),
                None,
                GpuWorkOperationCause::InvalidAttachment,
                "use a single-sampled target or construct multisample resolve explicitly",
            ));
        }
        let extent = signature.extent();
        let draw = GpuRenderDraw::new(
            pipeline.clone(),
            bindings,
            [],
            None,
            GpuDrawIntent::direct(vertices, instances),
            GpuViewport::new(
                0.0,
                0.0,
                extent.width() as f32,
                extent.height() as f32,
                0.0,
                1.0,
            )?,
            GpuScissorRect::new(0, 0, extent.width(), extent.height())?,
            GpuBlendConstant::new(0.0, 0.0, 0.0, 0.0)?,
            0,
        )?;
        Self::new([attachment], None, [draw], None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GpuColorClearValue, GpuMemoryIntent, GpuReconstruction, GpuResourceCommon,
        GpuResourceLabel, GpuResourceLifetime, GpuResourceProvenance, GpuResourceScope,
        GpuRuntimeBindingValue, GpuTextureDescriptor, GpuTextureDimension, GpuTextureExtent,
        GpuTextureFormat, GpuTextureInitialization, GpuTextureUsage, GpuTextureUsages,
        GpuTextureViewDescriptor, admit_static_wgsl_sources,
    };

    const RENDER_WGSL: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(positions[vertex_index], 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
"#;

    fn pipeline(format: GpuTextureFormat) -> GpuRenderPipelineDescriptor {
        let [source] =
            admit_static_wgsl_sources([("proof.render.full-target-direct", 1, RENDER_WGSL)])
                .unwrap();
        GpuRenderPipelineDescriptor::ordinary_color(source, "vs_main", "fs_main", format).unwrap()
    }

    fn target(
        resources: &mut GpuResourceScope,
        format: GpuTextureFormat,
        width: u32,
        height: u32,
    ) -> GpuTextureViewHandle {
        let texture = resources
            .texture(
                GpuTextureDescriptor::ordinary_owned_2d(
                    "ordinary render target",
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    width,
                    height,
                    format,
                    [GpuTextureUsage::ColorAttachment],
                    GpuTextureInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();
        resources
            .texture_view(
                GpuTextureViewDescriptor::ordinary_full_owned(
                    "ordinary render target view",
                    &texture,
                )
                .unwrap(),
            )
            .unwrap()
    }

    fn multisample_target(
        resources: &mut GpuResourceScope,
        sample_count: u32,
    ) -> GpuTextureViewHandle {
        let label = GpuResourceLabel::new("multisample ordinary render target").unwrap();
        let common = GpuResourceCommon::owned(
            label.clone(),
            GpuResourceLifetime::Transient,
            GpuMemoryIntent::Device,
            GpuReconstruction::SourceBacked,
            GpuResourceProvenance::new(label.clone(), None, None),
        )
        .unwrap();
        let texture = resources
            .texture(
                GpuTextureDescriptor::new(
                    common,
                    GpuTextureDimension::D2,
                    GpuTextureExtent::new(&label, GpuTextureDimension::D2, 8, 8, 1).unwrap(),
                    1,
                    sample_count,
                    GpuTextureFormat::Rgba8Unorm,
                    GpuTextureUsages::new(&label, [GpuTextureUsage::ColorAttachment]).unwrap(),
                    GpuTextureInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();
        resources
            .texture_view(
                GpuTextureViewDescriptor::ordinary_full_owned(
                    "multisample ordinary render target view",
                    &texture,
                )
                .unwrap(),
            )
            .unwrap()
    }

    fn draw_ranges() -> (GpuDrawRange, GpuDrawRange) {
        (
            GpuDrawRange::new(0, 3).unwrap(),
            GpuDrawRange::new(0, 1).unwrap(),
        )
    }

    #[test]
    fn ordinary_full_target_direct_derives_only_shape_administration() {
        let pipeline = pipeline(GpuTextureFormat::Rgba8Unorm);
        let bindings = pipeline
            .runtime_bindings(std::iter::empty::<GpuRuntimeBindingValue>())
            .unwrap();
        let mut resources = GpuResourceScope::new();
        let target = target(&mut resources, GpuTextureFormat::Rgba8Unorm, 32, 16);
        let clear = GpuColorClearValue::new(0.1, 0.2, 0.3, 1.0).unwrap();
        let (vertices, instances) = draw_ranges();

        let operation = GpuRenderOperation::ordinary_color_full_target_direct(
            &pipeline,
            bindings,
            &target,
            GpuColorAttachmentLoad::Clear(clear),
            GpuAttachmentStore::Store,
            vertices,
            instances,
        )
        .unwrap();

        assert_eq!(operation.color_attachments().len(), 1);
        assert_eq!(operation.color_attachments()[0].source(), &target);
        assert_eq!(
            operation.color_attachments()[0].load(),
            GpuColorAttachmentLoad::Clear(clear)
        );
        assert_eq!(
            operation.color_attachments()[0].store(),
            GpuAttachmentStore::Store
        );
        assert_eq!(operation.color_attachments()[0].resolve_target(), None);
        assert_eq!(operation.depth_stencil_attachment(), None);
        assert_eq!(operation.timestamp_writes(), None);

        let draw = &operation.draws()[0];
        assert_eq!(draw.pipeline(), &pipeline);
        assert!(draw.vertex_buffers().is_empty());
        assert!(draw.index_buffer().is_none());
        assert_eq!(draw.draw(), &GpuDrawIntent::direct(vertices, instances));
        assert_eq!(draw.viewport().values(), [0.0, 0.0, 32.0, 16.0, 0.0, 1.0]);
        assert_eq!(draw.scissor(), GpuScissorRect::new(0, 0, 32, 16).unwrap());
        assert_eq!(draw.blend_constant().components(), [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(draw.stencil_reference(), 0);
    }

    #[test]
    fn ordinary_full_target_direct_retains_canonical_pass_compatibility_validation() {
        let pipeline = pipeline(GpuTextureFormat::Rgba8Unorm);
        let bindings = pipeline
            .runtime_bindings(std::iter::empty::<GpuRuntimeBindingValue>())
            .unwrap();
        let mut resources = GpuResourceScope::new();
        let target = target(&mut resources, GpuTextureFormat::R8Unorm, 8, 8);
        let (vertices, instances) = draw_ranges();

        let error = GpuRenderOperation::ordinary_color_full_target_direct(
            &pipeline,
            bindings,
            &target,
            GpuColorAttachmentLoad::Load,
            GpuAttachmentStore::Store,
            vertices,
            instances,
        )
        .unwrap_err();

        assert_eq!(error.cause(), GpuWorkOperationCause::InvalidDraw);
    }

    #[test]
    fn ordinary_full_target_direct_rejects_hidden_multisample_resolve_choice() {
        let pipeline = pipeline(GpuTextureFormat::Rgba8Unorm);
        let bindings = pipeline
            .runtime_bindings(std::iter::empty::<GpuRuntimeBindingValue>())
            .unwrap();
        let mut resources = GpuResourceScope::new();
        let target = multisample_target(&mut resources, 4);
        let (vertices, instances) = draw_ranges();

        let error = GpuRenderOperation::ordinary_color_full_target_direct(
            &pipeline,
            bindings,
            &target,
            GpuColorAttachmentLoad::Load,
            GpuAttachmentStore::Store,
            vertices,
            instances,
        )
        .unwrap_err();

        assert_eq!(error.cause(), GpuWorkOperationCause::InvalidAttachment);
    }
}
