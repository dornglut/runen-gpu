use runen_gpu::*;
use std::time::{Duration, Instant};

const HEIGHT: u32 = 2;
const FIRST_REFERENCE: u32 = 91;
const MIXED_REFERENCE: u32 = 123;
const COPIED_DEPTH_REFERENCE: u32 = 177;
const COMBINED_WGSL: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    var positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0),
    );
    return vec4f(positions[index], 0.25, 1.0);
}
"#;

const SAMPLED_WGSL: &str = r#"
@group(0) @binding(0)
var depth_texture: texture_depth_2d;

@group(0) @binding(1)
var stencil_texture: texture_2d<u32>;

var<workgroup> sampled_sink: u32;

@compute @workgroup_size(1)
fn cs_main() {
    let depth_value = textureLoad(depth_texture, vec2<i32>(0, 0), 0);
    let stencil_value = textureLoad(stencil_texture, vec2<i32>(0, 0), 0).x;
    sampled_sink = select(0u, stencil_value, depth_value >= 0.0);
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
        GpuCapabilityFeature::Compute,
    ] {
        requirements
            .insert(GpuCapabilityRequirement::Required(feature))
            .unwrap();
    }
    requirements
}

fn combined_context() -> Option<(GpuContext, bool)> {
    let census = pollster::block_on(GpuContext::request(
        GpuContextDescriptor::new(requirements())
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .with_label("R1 native Depth32FloatStencil8 census"),
    ))
    .expect("native conformance must provide the declared Vulkan fallback environment");
    let facts = census
        .adapter_facts()
        .supported()
        .format(GpuTextureFormat::Depth32FloatStencil8)
        .expect("Depth32FloatStencil8 must remain in the normalized format census");
    if !(facts.depth_stencil && facts.copy_source && facts.copy_destination) {
        println!(
            "Depth32FloatStencil8: SKIPPED (optional backend prerequisite or required attachment/copy roles are absent)"
        );
        return None;
    }

    let mut descriptor = GpuContextDescriptor::new(requirements())
        .require_format_role(
            GpuTextureFormat::Depth32FloatStencil8,
            GpuFormatRole::DepthStencil,
        )
        .require_format_role(
            GpuTextureFormat::Depth32FloatStencil8,
            GpuFormatRole::CopySource,
        )
        .require_format_role(
            GpuTextureFormat::Depth32FloatStencil8,
            GpuFormatRole::CopyDestination,
        );
    if facts.sampled {
        descriptor = descriptor.require_format_role(
            GpuTextureFormat::Depth32FloatStencil8,
            GpuFormatRole::Sampled,
        );
    }
    let context = pollster::block_on(GpuContext::request(
        descriptor
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .with_label("R1 native Depth32FloatStencil8 execution proof"),
    ))
    .expect("advertised native combined roles must admit an execution context");
    Some((context, facts.sampled))
}

fn combined_texture(
    allocator: &mut GpuWorkResourceIdAllocator,
    name: &str,
    width: u32,
    usages: impl IntoIterator<Item = GpuTextureUsage>,
) -> (GpuTextureHandle, GpuTextureViewHandle) {
    let resource_label = label(name);
    let texture = allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::new(
                common(name),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, width, HEIGHT, 1)
                    .unwrap(),
                1,
                1,
                GpuTextureFormat::Depth32FloatStencil8,
                GpuTextureUsages::new(&resource_label, usages).unwrap(),
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
        GpuTextureAspect::All,
    )
    .unwrap();
    let view = allocator
        .allocate_texture_view_handle(
            GpuTextureViewDescriptor::new(
                common(&format!("{name} view")),
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

fn combined_aspect_view(
    allocator: &mut GpuWorkResourceIdAllocator,
    texture: &GpuTextureHandle,
    name: &str,
    aspect: GpuTextureAspect,
) -> GpuTextureViewHandle {
    let subresources =
        GpuTextureSubresourceRange::new(texture.descriptor().common().label(), 0, 1, 0, 1, aspect)
            .unwrap();
    allocator
        .allocate_texture_view_handle(
            GpuTextureViewDescriptor::new(
                common(name),
                texture,
                None,
                GpuTextureViewDimension::D2,
                subresources,
            )
            .unwrap(),
        )
        .unwrap()
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

fn combined_pipeline(
    key: &str,
    depth_write: bool,
    depth_compare: GpuCompareFunction,
    stencil_compare: GpuCompareFunction,
) -> GpuRenderPipelineDescriptor {
    let vertex = GpuEntryPointName::new("vs_main").unwrap();
    let identity = GpuProgramSourceIdentity::new(
        GpuProgramSourceOwnerId::allocate().unwrap(),
        GpuProgramSourceKey::new(key).unwrap(),
        GpuProgramSourceRevision::try_from_raw(1).unwrap(),
    );
    let mut sources = GpuProgramSourceRegistry::new(2, 4096).unwrap();
    let source = sources
        .admit_wgsl(
            identity,
            COMBINED_WGSL,
            GpuProgramSourceProvenance::new("Depth32FloatStencil8 native proof", None).unwrap(),
        )
        .unwrap();
    let program = GpuProgramDescriptor::new(
        source,
        [vertex.clone()],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();
    let face = stencil_face(stencil_compare, GpuStencilOperation::Replace);
    let stencil = GpuStencilStateDescriptor::new(face, face, u32::MAX, 0xff);
    let depth_stencil = GpuDepthStencilStateDescriptor::new(
        GpuTextureFormat::Depth32FloatStencil8,
        Some(GpuDepthStateDescriptor::new(depth_write, depth_compare)),
        Some(stencil),
    )
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

fn sampled_pipeline() -> GpuComputePipelineDescriptor {
    let entry = GpuEntryPointName::new("cs_main").unwrap();
    let identity = GpuProgramSourceIdentity::new(
        GpuProgramSourceOwnerId::allocate().unwrap(),
        GpuProgramSourceKey::new("proof.depth32float-stencil8.sampled").unwrap(),
        GpuProgramSourceRevision::try_from_raw(1).unwrap(),
    );
    let mut sources = GpuProgramSourceRegistry::new(2, 4096).unwrap();
    let source = sources
        .admit_wgsl(
            identity,
            SAMPLED_WGSL,
            GpuProgramSourceProvenance::new("Depth32FloatStencil8 sampled proof", None).unwrap(),
        )
        .unwrap();
    let program = GpuProgramDescriptor::new(
        source,
        [entry.clone()],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();
    GpuComputePipelineDescriptor::new(program, entry, GpuPipelineConfiguration::default()).unwrap()
}

fn sampled_texture_binding(binding: u32, view: &GpuTextureViewHandle) -> GpuRuntimeBindingValue {
    GpuRuntimeBindingValue::new(
        GpuBindingKey::try_new(0, u64::from(binding)).unwrap(),
        [GpuRuntimeBindingResource::TextureView(
            GpuRuntimeTextureViewBinding::new(view.clone()),
        )],
    )
    .unwrap()
}

fn sampled_operation(
    depth_view: &GpuTextureViewHandle,
    stencil_view: &GpuTextureViewHandle,
) -> GpuComputeOperation {
    let pipeline = sampled_pipeline();
    let bindings = pipeline
        .runtime_bindings([
            sampled_texture_binding(0, depth_view),
            sampled_texture_binding(1, stencil_view),
        ])
        .unwrap();
    GpuComputeOperation::new(
        pipeline,
        bindings,
        GpuDispatchIntent::direct(GpuDispatchSize::new(1, 1, 1)),
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

fn seed_attachment(view: GpuTextureViewHandle) -> GpuRenderDepthStencilAttachment {
    let depth = GpuDepthAttachmentState::new(
        GpuDepthStencilAccess::ReadWrite,
        GpuDepthAttachmentLoad::Clear(GpuDepthClearValue::new(1.0).unwrap()),
        GpuAttachmentStore::Store,
    )
    .unwrap();
    let stencil = GpuStencilAttachmentState::new(
        GpuDepthStencilAccess::ReadWrite,
        GpuStencilAttachmentLoad::Clear(GpuStencilClearValue::new(0).unwrap()),
        GpuAttachmentStore::Store,
    )
    .unwrap();
    GpuRenderDepthStencilAttachment::new(view, Some(depth), Some(stencil)).unwrap()
}

fn mixed_attachment(view: GpuTextureViewHandle) -> GpuRenderDepthStencilAttachment {
    let depth = GpuDepthAttachmentState::new(
        GpuDepthStencilAccess::ReadOnly,
        GpuDepthAttachmentLoad::Load,
        GpuAttachmentStore::Store,
    )
    .unwrap();
    let stencil = GpuStencilAttachmentState::new(
        GpuDepthStencilAccess::ReadWrite,
        GpuStencilAttachmentLoad::Load,
        GpuAttachmentStore::Store,
    )
    .unwrap();
    GpuRenderDepthStencilAttachment::new(view, Some(depth), Some(stencil)).unwrap()
}

fn readback_region(
    texture: &GpuTextureHandle,
    width: u32,
    aspect: GpuTextureAspect,
) -> GpuTextureCopyRegion {
    GpuTextureCopyRegion::new(
        texture,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        aspect,
        GpuCopyExtent::new(width, HEIGHT, 1).unwrap(),
    )
    .unwrap()
}

fn wait_for_readback(
    context: &GpuContext,
    submission: &GpuSubmission,
    readback: &GpuReadback,
    name: &str,
) -> GpuReadbackBytes {
    let deadline = Instant::now() + Duration::from_secs(15);
    let bytes = loop {
        context.progress();
        match readback.status() {
            GpuReadbackStatus::Ready(bytes) => break bytes,
            GpuReadbackStatus::Failed(error) => panic!("{name} readback failed: {error:?}"),
            GpuReadbackStatus::Pending => {}
        }
        if let GpuSubmissionStatus::Failed(error) = submission.status() {
            panic!("{name} submission failed before readback: {error:?}");
        }
        assert!(Instant::now() < deadline, "{name} readback timed out");
        std::thread::yield_now();
    };
    loop {
        context.progress();
        match submission.status() {
            GpuSubmissionStatus::Completed => break,
            GpuSubmissionStatus::Failed(error) => {
                panic!("{name} submission failed: {error:?}")
            }
            GpuSubmissionStatus::Accepted => {}
        }
        assert!(
            Instant::now() < deadline,
            "{name} submission did not terminalize"
        );
        std::thread::yield_now();
    }
    bytes
}

fn run_native_combined(width: u32) -> bool {
    let Some((context, sampled)) = combined_context() else {
        return false;
    };
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let (source, source_view) = combined_texture(
        &mut allocator,
        "Depth32FloatStencil8 source",
        width,
        [
            GpuTextureUsage::DepthStencilAttachment,
            GpuTextureUsage::CopySource,
        ],
    );
    let mut destination_usages = vec![
        GpuTextureUsage::DepthStencilAttachment,
        GpuTextureUsage::CopySource,
        GpuTextureUsage::CopyDestination,
    ];
    if sampled {
        destination_usages.push(GpuTextureUsage::Sampled);
    }
    let (destination, destination_view) = combined_texture(
        &mut allocator,
        "Depth32FloatStencil8 destination",
        width,
        destination_usages,
    );
    let sampled_views = sampled.then(|| {
        (
            combined_aspect_view(
                &mut allocator,
                &destination,
                "Depth32FloatStencil8 sampled depth view",
                GpuTextureAspect::DepthOnly,
            ),
            combined_aspect_view(
                &mut allocator,
                &destination,
                "Depth32FloatStencil8 sampled stencil view",
                GpuTextureAspect::StencilOnly,
            ),
        )
    });
    let sampled_compute = sampled_views
        .as_ref()
        .map(|(depth, stencil)| sampled_operation(depth, stencil));

    let seed_render = GpuRenderOperation::new(
        [],
        Some(seed_attachment(source_view.clone())),
        [draw(
            combined_pipeline(
                "proof.depth32float-stencil8.seed",
                true,
                GpuCompareFunction::Always,
                GpuCompareFunction::Always,
            ),
            width,
            FIRST_REFERENCE,
        )],
        None,
    )
    .unwrap();

    let mixed_render = GpuRenderOperation::new(
        [],
        Some(mixed_attachment(source_view.clone())),
        [draw(
            combined_pipeline(
                "proof.depth32float-stencil8.mixed",
                false,
                GpuCompareFunction::Equal,
                GpuCompareFunction::Always,
            ),
            width,
            MIXED_REFERENCE,
        )],
        None,
    )
    .unwrap();

    let extent = GpuCopyExtent::new(width, HEIGHT, 1).unwrap();
    let source_all = GpuTextureCopyRegion::new(
        &source,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::All,
        extent,
    )
    .unwrap();
    let destination_all = GpuTextureCopyRegion::new(
        &destination,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::All,
        extent,
    )
    .unwrap();
    let combined_copy = GpuCopyOperation::texture_to_texture(source_all, destination_all).unwrap();

    let copied_readback_id = GpuReadbackId::allocate().unwrap();
    let copied_readback = GpuReadbackOperation::new(
        readback_region(&destination, width, GpuTextureAspect::StencilOnly).into(),
        copied_readback_id,
    )
    .unwrap();

    let copied_depth_readback_id = GpuReadbackId::allocate().unwrap();
    let copied_depth_readback = GpuReadbackOperation::new(
        readback_region(&destination, width, GpuTextureAspect::DepthOnly).into(),
        copied_depth_readback_id,
    )
    .unwrap();

    let copied_depth_render = GpuRenderOperation::new(
        [],
        Some(mixed_attachment(destination_view.clone())),
        [draw(
            combined_pipeline(
                "proof.depth32float-stencil8.copied-depth",
                false,
                GpuCompareFunction::Equal,
                GpuCompareFunction::Always,
            ),
            width,
            COPIED_DEPTH_REFERENCE,
        )],
        None,
    )
    .unwrap();

    let final_readback_id = GpuReadbackId::allocate().unwrap();
    let final_readback = GpuReadbackOperation::new(
        readback_region(&destination, width, GpuTextureAspect::StencilOnly).into(),
        final_readback_id,
    )
    .unwrap();

    let name = format!("Depth32FloatStencil8 {width}x{HEIGHT}");
    let mut builder = GpuWorkFragmentBuilder::new(label(&name), provenance(&name));
    for resource in [
        source.clone().into(),
        source_view.into(),
        destination.clone().into(),
        destination_view.into(),
    ] {
        builder.declare_resource(resource).unwrap();
    }
    if let Some((depth_view, stencil_view)) = &sampled_views {
        builder.declare_resource(depth_view.clone().into()).unwrap();
        builder
            .declare_resource(stencil_view.clone().into())
            .unwrap();
    }
    for (node, operation) in [
        (
            "combined seed depth + stencil",
            GpuWorkOperation::Render(seed_render),
        ),
        (
            "combined mixed depth-read-only + stencil-write",
            GpuWorkOperation::Render(mixed_render),
        ),
        (
            "combined all-aspect texture copy",
            GpuWorkOperation::Copy(combined_copy),
        ),
        (
            "combined copied stencil snapshot",
            GpuWorkOperation::Readback(copied_readback),
        ),
        (
            "combined copied depth snapshot",
            GpuWorkOperation::Readback(copied_depth_readback),
        ),
        (
            "combined copied depth gate",
            GpuWorkOperation::Render(copied_depth_render),
        ),
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
    if let Some(compute) = sampled_compute {
        builder
            .add_node(
                label("combined aspect-specific sampled dispatch"),
                GpuWorkOperation::Compute(compute),
                [],
                GpuCapabilityRequirements::new(),
                GpuExecutionPreference::Automatic,
                provenance("combined aspect-specific sampled dispatch"),
            )
            .unwrap();
    }
    builder
        .add_node(
            label("combined final stencil readback"),
            GpuWorkOperation::Readback(final_readback),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::Automatic,
            provenance("combined final stencil readback"),
        )
        .unwrap();

    let graph = GpuPreparedWorkGraph::prepare(label(&name), [builder.finish().unwrap()]).unwrap();
    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let copied = submission
        .readback(copied_readback_id)
        .expect("combined copied stencil readback must be submitted")
        .clone();
    let copied_depth = submission
        .readback(copied_depth_readback_id)
        .expect("combined copied depth readback must be submitted")
        .clone();
    let final_bytes = submission
        .readback(final_readback_id)
        .expect("combined final stencil readback must be submitted")
        .clone();

    let copied = wait_for_readback(&context, &submission, &copied, &name);
    assert_eq!(
        copied.texture_format(),
        Some(GpuTextureFormat::Depth32FloatStencil8)
    );
    assert_eq!(
        copied.as_bytes().len(),
        usize::try_from(width * HEIGHT).unwrap()
    );
    assert!(
        copied
            .as_bytes()
            .iter()
            .all(|byte| *byte == MIXED_REFERENCE as u8),
        "all-aspect copy must preserve the mixed-pass stencil result"
    );

    let copied_depth = wait_for_readback(&context, &submission, &copied_depth, &name);
    assert_eq!(
        copied_depth.texture_format(),
        Some(GpuTextureFormat::Depth32FloatStencil8)
    );
    assert_eq!(
        copied_depth.as_bytes().len(),
        usize::try_from(width * HEIGHT * 4).unwrap()
    );
    let expected_depth = 0.25_f32.to_le_bytes();
    assert!(
        copied_depth
            .as_bytes()
            .chunks_exact(4)
            .all(|bytes| bytes == expected_depth),
        "DepthOnly linear readback must preserve the exact copied 32-bit depth result"
    );

    let final_bytes = wait_for_readback(&context, &submission, &final_bytes, &name);
    assert!(
        final_bytes
            .as_bytes()
            .iter()
            .all(|byte| *byte == COPIED_DEPTH_REFERENCE as u8),
        "copied depth must pass Equal and gate the final stencil Replace"
    );
    if sampled {
        println!(
            "Depth32FloatStencil8 Sampled: EXERCISED DepthOnly texture_depth_2d + StencilOnly texture_2d<u32> compute bindings"
        );
    } else {
        println!("Depth32FloatStencil8 Sampled: SKIPPED (sampled role not advertised)");
    }
    println!(
        "Depth32FloatStencil8: EXERCISED depth+stencil seed, mixed depth-read-only/stencil-write, all-aspect copy, DepthOnly 4-byte readback, StencilOnly 1-byte readback, copied-depth gate at {width}x{HEIGHT}"
    );
    true
}

#[test]
fn depth32float_stencil8_copy_operations_preserve_aspect_contract() {
    let format = GpuTextureFormat::Depth32FloatStencil8;
    assert!(format.is_depth());
    assert!(format.is_stencil());
    assert_eq!(format.copy_block_size(GpuTextureAspect::All), None);
    assert_eq!(format.copy_block_size(GpuTextureAspect::DepthOnly), Some(4));
    assert_eq!(
        format.copy_block_size(GpuTextureAspect::StencilOnly),
        Some(1)
    );

    let mut allocator = GpuWorkResourceIdAllocator::new();
    let (source, _) = combined_texture(
        &mut allocator,
        "Depth32FloatStencil8 copy source",
        16,
        [GpuTextureUsage::CopySource],
    );
    let (destination, _) = combined_texture(
        &mut allocator,
        "Depth32FloatStencil8 copy destination",
        16,
        [GpuTextureUsage::CopyDestination],
    );
    let extent = GpuCopyExtent::new(16, 2, 1).unwrap();

    let source_all = GpuTextureCopyRegion::new(
        &source,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::All,
        extent,
    )
    .unwrap();
    let destination_all = GpuTextureCopyRegion::new(
        &destination,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::All,
        extent,
    )
    .unwrap();
    assert!(GpuCopyOperation::texture_to_texture(source_all, destination_all).is_ok());

    for aspect in [GpuTextureAspect::DepthOnly, GpuTextureAspect::StencilOnly] {
        let source_region =
            GpuTextureCopyRegion::new(&source, 0, GpuTextureOrigin::new(0, 0, 0), aspect, extent)
                .unwrap();
        let destination_region = GpuTextureCopyRegion::new(
            &destination,
            0,
            GpuTextureOrigin::new(0, 0, 0),
            aspect,
            extent,
        )
        .unwrap();
        assert!(GpuCopyOperation::texture_to_texture(source_region, destination_region).is_err());
    }

    let buffer_label = label("Depth32FloatStencil8 linear source");
    let buffer = allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common("Depth32FloatStencil8 linear source"),
                4096,
                GpuBufferUsages::new(&buffer_label, [GpuBufferUsage::CopySource]).unwrap(),
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();

    let depth_region = GpuTextureCopyRegion::new(
        &destination,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::DepthOnly,
        extent,
    )
    .unwrap();
    let depth_layout = GpuBufferTextureLayout::new(&buffer, 0, 64, 0).unwrap();
    assert!(GpuCopyOperation::buffer_to_texture(depth_layout, depth_region).is_ok());

    let stencil_region = GpuTextureCopyRegion::new(
        &destination,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::StencilOnly,
        extent,
    )
    .unwrap();
    let stencil_layout = GpuBufferTextureLayout::new(&buffer, 0, 16, 0).unwrap();
    assert!(GpuCopyOperation::buffer_to_texture(stencil_layout, stencil_region).is_ok());

    let all_region = GpuTextureCopyRegion::new(
        &destination,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::All,
        extent,
    )
    .unwrap();
    let all_layout = GpuBufferTextureLayout::new(&buffer, 0, 64, 0).unwrap();
    assert!(GpuCopyOperation::buffer_to_texture(all_layout, all_region).is_err());
}

#[test]
fn combined_pipeline_cannot_use_an_attachment_aspect_that_was_omitted() {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let (_, view) = combined_texture(
        &mut allocator,
        "combined parity target",
        8,
        [GpuTextureUsage::DepthStencilAttachment],
    );
    let pipeline = combined_pipeline(
        "proof.depth32float-stencil8.parity",
        true,
        GpuCompareFunction::Always,
        GpuCompareFunction::Always,
    );

    let depth = GpuDepthAttachmentState::new(
        GpuDepthStencilAccess::ReadWrite,
        GpuDepthAttachmentLoad::Clear(GpuDepthClearValue::new(1.0).unwrap()),
        GpuAttachmentStore::Store,
    )
    .unwrap();
    let depth_only = GpuRenderDepthStencilAttachment::new(view.clone(), Some(depth), None).unwrap();
    assert!(
        GpuRenderOperation::new(
            [],
            Some(depth_only),
            [draw(pipeline.clone(), 8, FIRST_REFERENCE)],
            None,
        )
        .is_err()
    );

    let stencil = GpuStencilAttachmentState::new(
        GpuDepthStencilAccess::ReadWrite,
        GpuStencilAttachmentLoad::Clear(GpuStencilClearValue::new(0).unwrap()),
        GpuAttachmentStore::Store,
    )
    .unwrap();
    let stencil_only = GpuRenderDepthStencilAttachment::new(view, None, Some(stencil)).unwrap();
    assert!(
        GpuRenderOperation::new(
            [],
            Some(stencil_only),
            [draw(pipeline, 8, FIRST_REFERENCE)],
            None,
        )
        .is_err()
    );
}

#[test]
#[ignore = "requires the conformance Vulkan software adapter"]
fn depth32float_stencil8_native_combined_attachment_copy_and_readback() {
    if !run_native_combined(255) {
        return;
    }
    assert!(
        run_native_combined(256),
        "optional Depth32FloatStencil8 support must remain stable across retained widths"
    );
}
