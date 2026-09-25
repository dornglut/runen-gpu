use runen_gpu::*;
use std::time::{Duration, Instant};

const HEIGHT: u32 = 2;
const REFERENCE: u32 = 91;
const STENCIL_WGSL: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    var positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0),
    );
    return vec4f(positions[index], 0.0, 1.0);
}
"#;

fn label(value: &str) -> GpuResourceLabel {
    GpuResourceLabel::new(value).unwrap()
}

fn provenance(value: &str) -> GpuResourceProvenance {
    GpuResourceProvenance::new(label(value), None, None)
}

fn common(value: &str) -> GpuResourceCommon {
    GpuResourceCommon::owned(
        label(value),
        GpuResourceLifetime::Transient,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        provenance(value),
    )
    .unwrap()
}

fn requirements() -> GpuCapabilityRequirements {
    let mut requirements = GpuCapabilityRequirements::new();
    for feature in [
        GpuCapabilityFeature::RenderPipeline,
        GpuCapabilityFeature::DepthAttachment,
        GpuCapabilityFeature::Copy,
    ] {
        requirements
            .insert(GpuCapabilityRequirement::Required(feature))
            .unwrap();
    }
    requirements
}

fn stencil_context() -> GpuContext {
    let descriptor = GpuContextDescriptor::new(requirements())
        .require_format_role(GpuTextureFormat::Stencil8, GpuFormatRole::DepthStencil)
        .require_format_role(GpuTextureFormat::Stencil8, GpuFormatRole::CopySource)
        .require_format_role(GpuTextureFormat::Stencil8, GpuFormatRole::CopyDestination)
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label("R1 native Stencil8 proof");
    pollster::block_on(GpuContext::request(descriptor))
        .expect("native conformance must provide baseline Stencil8 attachment and copy roles")
}

fn stencil_texture(
    allocator: &mut GpuWorkResourceIdAllocator,
    width: u32,
) -> (GpuTextureHandle, GpuTextureViewHandle) {
    let resource_label = label("Stencil8 target");
    let texture = allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::new(
                common("Stencil8 target"),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, width, HEIGHT, 1)
                    .unwrap(),
                1,
                1,
                GpuTextureFormat::Stencil8,
                GpuTextureUsages::new(
                    &resource_label,
                    [
                        GpuTextureUsage::DepthStencilAttachment,
                        GpuTextureUsage::CopySource,
                        GpuTextureUsage::CopyDestination,
                    ],
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
        0,
        1,
        GpuTextureAspect::StencilOnly,
    )
    .unwrap();
    let view = allocator
        .allocate_texture_view_handle(
            GpuTextureViewDescriptor::new(
                common("Stencil8 target view"),
                &texture,
                None,
                GpuTextureViewDimension::D2,
                subresources,
            )
            .unwrap(),
        )
        .unwrap();
    (texture, view)
}

fn stencil_face(
    compare: GpuCompareFunction,
    pass_op: GpuStencilOperation,
) -> GpuStencilFaceStateDescriptor {
    GpuStencilFaceStateDescriptor::new(
        compare,
        GpuStencilOperation::Keep,
        GpuStencilOperation::Keep,
        pass_op,
    )
}

fn stencil_pipeline(write: bool) -> GpuRenderPipelineDescriptor {
    let vertex = GpuEntryPointName::new("vs_main").unwrap();
    let identity = GpuProgramSourceIdentity::new(
        GpuProgramSourceOwnerId::allocate().unwrap(),
        GpuProgramSourceKey::new(if write {
            "proof.stencil8.write"
        } else {
            "proof.stencil8.read-only"
        })
        .unwrap(),
        GpuProgramSourceRevision::try_from_raw(1).unwrap(),
    );
    let mut sources = GpuProgramSourceRegistry::new(2, 4096).unwrap();
    let source = sources
        .admit_wgsl(
            identity,
            STENCIL_WGSL,
            GpuProgramSourceProvenance::new("Stencil8 native proof", None).unwrap(),
        )
        .unwrap();
    let program = GpuProgramDescriptor::new(
        source,
        [vertex.clone()],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();
    let face = if write {
        stencil_face(GpuCompareFunction::Always, GpuStencilOperation::Replace)
    } else {
        stencil_face(GpuCompareFunction::Equal, GpuStencilOperation::Keep)
    };
    let stencil =
        GpuStencilStateDescriptor::new(face, face, u32::MAX, if write { 0xff } else { 0 });
    let depth_stencil =
        GpuDepthStencilStateDescriptor::new(GpuTextureFormat::Stencil8, None, Some(stencil))
            .unwrap();
    let state = GpuRenderPipelineStateDescriptor::new(
        GpuVertexInputStateDescriptor::new([]).unwrap(),
        None,
        GpuPrimitiveStateDescriptor::default(),
        Some(depth_stencil),
        GpuMultisampleStateDescriptor::default(),
    )
    .unwrap();
    GpuRenderPipelineDescriptor::new(
        program,
        GpuRenderEntryPoints::new(vertex, None),
        state,
        GpuPipelineConfiguration::default(),
    )
    .unwrap()
}

fn draw(
    pipeline: GpuRenderPipelineDescriptor,
    width: u32,
    stencil_reference: u32,
) -> GpuRenderDraw {
    let bindings = GpuRuntimeBindingSet::new(pipeline.layout().clone(), []).unwrap();
    GpuRenderDraw::new(
        pipeline,
        bindings,
        [],
        None,
        GpuDrawIntent::direct(
            GpuDrawRange::new(0, 3).unwrap(),
            GpuDrawRange::new(0, 1).unwrap(),
        ),
        GpuViewport::new(0.0, 0.0, width as f32, HEIGHT as f32, 0.0, 1.0).unwrap(),
        GpuScissorRect::new(0, 0, width, HEIGHT).unwrap(),
        GpuBlendConstant::new(0.0, 0.0, 0.0, 0.0).unwrap(),
        stencil_reference,
    )
    .unwrap()
}

fn wait_for_readback(
    context: &GpuContext,
    submission: &GpuSubmission,
    readback: &GpuReadback,
) -> GpuReadbackBytes {
    let deadline = Instant::now() + Duration::from_secs(15);
    let bytes = loop {
        context.progress();
        match readback.status() {
            GpuReadbackStatus::Ready(bytes) => break bytes,
            GpuReadbackStatus::Failed(error) => panic!("Stencil8 readback failed: {error:?}"),
            GpuReadbackStatus::Pending => {}
        }
        if let GpuSubmissionStatus::Failed(error) = submission.status() {
            panic!("Stencil8 submission failed before readback: {error:?}");
        }
        assert!(Instant::now() < deadline, "Stencil8 readback timed out");
        std::thread::yield_now();
    };
    loop {
        context.progress();
        match submission.status() {
            GpuSubmissionStatus::Completed => break,
            GpuSubmissionStatus::Failed(error) => panic!("Stencil8 submission failed: {error:?}"),
            GpuSubmissionStatus::Accepted => {}
        }
        assert!(
            Instant::now() < deadline,
            "Stencil8 submission did not terminalize"
        );
        std::thread::yield_now();
    }
    bytes
}

fn run_native_stencil8(width: u32) {
    let context = stencil_context();
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let (texture, view) = stencil_texture(&mut allocator, width);

    let write_state = GpuStencilAttachmentState::new(
        GpuDepthStencilAccess::ReadWrite,
        GpuStencilAttachmentLoad::Clear(GpuStencilClearValue::new(0).unwrap()),
        GpuAttachmentStore::Store,
    )
    .unwrap();
    let write_attachment =
        GpuRenderDepthStencilAttachment::new(view.clone(), None, Some(write_state)).unwrap();
    let write_render = GpuRenderOperation::new(
        [],
        Some(write_attachment),
        [draw(stencil_pipeline(true), width, REFERENCE)],
        None,
    )
    .unwrap();

    let read_state = GpuStencilAttachmentState::new(
        GpuDepthStencilAccess::ReadOnly,
        GpuStencilAttachmentLoad::Load,
        GpuAttachmentStore::Store,
    )
    .unwrap();
    let read_attachment =
        GpuRenderDepthStencilAttachment::new(view.clone(), None, Some(read_state)).unwrap();
    let read_render = GpuRenderOperation::new(
        [],
        Some(read_attachment),
        [draw(stencil_pipeline(false), width, REFERENCE)],
        None,
    )
    .unwrap();

    let region = GpuTextureCopyRegion::new(
        &texture,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::StencilOnly,
        GpuCopyExtent::new(width, HEIGHT, 1).unwrap(),
    )
    .unwrap();
    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback_op = GpuReadbackOperation::new(region.into(), readback_id).unwrap();

    let name = format!("Stencil8 {width}x{HEIGHT}");
    let mut builder = GpuWorkFragmentBuilder::new(label(&name), provenance(&name));
    builder.declare_resource(texture.into()).unwrap();
    builder.declare_resource(view.into()).unwrap();
    for (node, operation) in [
        ("Stencil8 write", GpuWorkOperation::Render(write_render)),
        (
            "Stencil8 read-only test",
            GpuWorkOperation::Render(read_render),
        ),
        ("Stencil8 readback", GpuWorkOperation::Readback(readback_op)),
    ] {
        builder
            .add_node(
                label(node),
                operation,
                [],
                GpuCapabilityRequirements::new(),
                GpuExecutionPreference::Automatic,
                provenance(node),
            )
            .unwrap();
    }

    let graph = GpuPreparedWorkGraph::prepare(label(&name), [builder.finish().unwrap()]).unwrap();
    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let readback = prepared
        .readback(readback_id)
        .expect("Stencil8 readback handle must be prepared")
        .clone();
    let submission = context.submit_prepared(prepared).unwrap();
    let bytes = wait_for_readback(&context, &submission, &readback);
    assert_eq!(bytes.texture_format(), Some(GpuTextureFormat::Stencil8));
    assert_eq!(
        bytes.as_bytes().len(),
        usize::try_from(width * HEIGHT).unwrap()
    );
    assert!(
        bytes.as_bytes().iter().all(|byte| *byte == REFERENCE as u8),
        "full-target Replace must write the dynamic stencil reference to every covered pixel"
    );
    println!(
        "Stencil8: EXERCISED clear + Replace(reference={REFERENCE}) + read-only Equal/Keep + exact readback at {width}x{HEIGHT}"
    );
}

#[test]
fn stencil8_structural_and_attachment_contracts_are_explicit() {
    let format = GpuTextureFormat::Stencil8;
    assert!(!format.is_depth());
    assert!(format.is_stencil());
    assert_eq!(format.block_dimensions(), (1, 1));
    assert_eq!(format.copy_block_size(GpuTextureAspect::All), Some(1));
    assert_eq!(
        format.copy_block_size(GpuTextureAspect::StencilOnly),
        Some(1)
    );
    assert_eq!(format.copy_block_size(GpuTextureAspect::DepthOnly), None);
    assert_eq!(format.copy_block_size(GpuTextureAspect::Color), None);
    assert_eq!(GpuStencilClearValue::new(255).unwrap().value(), 255);
    assert!(GpuStencilClearValue::new(256).is_err());

    let keep = stencil_face(GpuCompareFunction::Always, GpuStencilOperation::Keep);
    let replace = stencil_face(GpuCompareFunction::Always, GpuStencilOperation::Replace);
    assert!(!GpuStencilStateDescriptor::new(keep, keep, u32::MAX, u32::MAX).may_write());
    assert!(GpuStencilStateDescriptor::new(replace, replace, u32::MAX, 0xff).may_write());
    assert!(!GpuStencilStateDescriptor::new(replace, replace, u32::MAX, 0).may_write());

    assert!(
        GpuDepthStencilStateDescriptor::new(
            GpuTextureFormat::Stencil8,
            Some(GpuDepthStateDescriptor::new(
                false,
                GpuCompareFunction::Always
            )),
            None,
        )
        .is_err()
    );
    assert!(
        GpuDepthStencilStateDescriptor::new(
            GpuTextureFormat::Depth32Float,
            None,
            Some(GpuStencilStateDescriptor::new(keep, keep, u32::MAX, 0)),
        )
        .is_err()
    );
    assert!(
        GpuStencilAttachmentState::new(
            GpuDepthStencilAccess::ReadOnly,
            GpuStencilAttachmentLoad::Clear(GpuStencilClearValue::new(0).unwrap()),
            GpuAttachmentStore::Store,
        )
        .is_err()
    );
    assert!(
        GpuStencilAttachmentState::new(
            GpuDepthStencilAccess::ReadOnly,
            GpuStencilAttachmentLoad::Load,
            GpuAttachmentStore::Discard,
        )
        .is_err()
    );
}

#[test]
#[ignore = "requires the conformance Vulkan software adapter"]
fn stencil8_native_clear_draw_test_and_readback_cross_copy_alignment_boundary() {
    run_native_stencil8(255);
    run_native_stencil8(256);
}
