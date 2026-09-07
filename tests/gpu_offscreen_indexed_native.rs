use runen_gpu::*;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const CLEAR_PIXEL: [u8; 4] = [0, 0, 0, 255];
const DRAW_PIXEL: [u8; 4] = [255, 0, 0, 255];
const INDICES: [u32; 3] = [0, 1, 2];
const ARTIFACT_NAME: &str = "known-pattern-indexed-offscreen.png";
const KNOWN_PATTERN_WGSL: &str = include_str!("gpu_offscreen_indexed_native/known_pattern.wgsl");

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

fn native_render_context() -> GpuContext {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .with_label("indexed offscreen known-pattern proof");
    let context = pollster::block_on(GpuContext::request(descriptor))
        .expect("native conformance environment must provide a Vulkan fallback adapter");
    assert_eq!(context.adapter_facts().backend(), GpuBackendFamily::Vulkan);
    assert_eq!(
        context.adapter_facts().fallback(),
        GpuFallbackStatus::ConfirmedFallback,
        "indexed offscreen conformance must execute through the explicitly required fallback path"
    );
    context
}

fn admitted_render_source() -> GpuAdmittedProgramSource {
    let identity = GpuProgramSourceIdentity::new(
        GpuProgramSourceOwnerId::allocate().expect("offscreen proof source owner should allocate"),
        GpuProgramSourceKey::new("proof.offscreen.indexed-known-pattern").unwrap(),
        GpuProgramSourceRevision::try_from_raw(1).unwrap(),
    );
    let mut sources = GpuProgramSourceRegistry::new(4, 16 * 1024).unwrap();
    sources
        .admit_wgsl(
            identity,
            KNOWN_PATTERN_WGSL,
            GpuProgramSourceProvenance::new("indexed-offscreen-known-pattern-proof", None).unwrap(),
        )
        .unwrap()
}

fn render_pipeline() -> GpuRenderPipelineDescriptor {
    let vertex = GpuEntryPointName::new("vs_main").unwrap();
    let fragment = GpuEntryPointName::new("fs_main").unwrap();
    let program = GpuProgramDescriptor::new(
        admitted_render_source(),
        [vertex.clone(), fragment.clone()],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();
    let color_target = GpuColorTargetStateDescriptor::new(
        GpuTextureFormat::Rgba8Unorm,
        GpuBlendMode::Replace,
        GpuColorWriteMask::ALL,
    )
    .unwrap();
    let state = GpuRenderPipelineStateDescriptor::new(
        GpuVertexInputStateDescriptor::new([]).unwrap(),
        Some(GpuFragmentOutputStateDescriptor::new([color_target])),
        GpuPrimitiveStateDescriptor::default(),
        None,
        GpuMultisampleStateDescriptor::default(),
    )
    .unwrap();
    GpuRenderPipelineDescriptor::new(
        program,
        GpuRenderEntryPoints::new(vertex, Some(fragment)),
        state,
        GpuPipelineConfiguration::default(),
    )
    .unwrap()
}

fn render_target(
    allocator: &mut GpuWorkResourceIdAllocator,
) -> (GpuTextureHandle, GpuTextureViewHandle) {
    let texture_label = label("indexed offscreen color target");
    let texture = allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::new(
                common("indexed offscreen color target"),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&texture_label, GpuTextureDimension::D2, WIDTH, HEIGHT, 1)
                    .unwrap(),
                1,
                1,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(
                    &texture_label,
                    [
                        GpuTextureUsage::ColorAttachment,
                        GpuTextureUsage::CopySource,
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
        GpuTextureAspect::Color,
    )
    .unwrap();
    let view = allocator
        .allocate_texture_view_handle(
            GpuTextureViewDescriptor::new(
                common("indexed offscreen color target view"),
                &texture,
                None,
                GpuTextureDimension::D2,
                subresources,
            )
            .unwrap(),
        )
        .unwrap();
    (texture, view)
}

fn prepared_index_buffer(allocator: &mut GpuWorkResourceIdAllocator) -> GpuBufferHandle {
    let resource_label = label("indexed offscreen index buffer");
    let data = PreparedGpuData::<TransferData>::from_pod_transfer(
        "indexed offscreen indices",
        &INDICES,
        provenance("indexed offscreen indices"),
    )
    .unwrap();
    allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common("indexed offscreen index buffer"),
                u64::try_from(core::mem::size_of_val(&INDICES)).unwrap(),
                GpuBufferUsages::new(
                    &resource_label,
                    [GpuBufferUsage::Index, GpuBufferUsage::CopyDestination],
                )
                .unwrap(),
                GpuBufferInitialization::Prepared(data),
            )
            .unwrap(),
        )
        .unwrap()
}

pub(crate) fn render_graph() -> (GpuPreparedWorkGraph, GpuReadbackId) {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let (texture, view) = render_target(&mut allocator);
    let index_buffer = prepared_index_buffer(&mut allocator);
    let pipeline = render_pipeline();
    let bindings = GpuRuntimeBindingSet::new(pipeline.layout().clone(), []).unwrap();
    let index_binding = GpuIndexBufferBinding::new(
        &index_buffer,
        GpuBufferRange::whole(&index_buffer).unwrap(),
        GpuIndexFormat::Uint32,
    )
    .unwrap();
    let draw = GpuRenderDraw::new(
        pipeline,
        bindings,
        [],
        Some(index_binding),
        GpuDrawIntent::indexed(
            GpuDrawRange::new(0, u32::try_from(INDICES.len()).unwrap()).unwrap(),
            0,
            GpuDrawRange::new(0, 1).unwrap(),
        ),
        GpuViewport::new(0.0, 0.0, WIDTH as f32, HEIGHT as f32, 0.0, 1.0).unwrap(),
        GpuScissorRect::new(0, 0, WIDTH / 2, HEIGHT).unwrap(),
        GpuBlendConstant::new(0.0, 0.0, 0.0, 0.0).unwrap(),
        0,
    )
    .unwrap();
    let attachment = GpuRenderColorAttachment::new(
        view.clone(),
        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
        GpuAttachmentStore::Store,
        None,
    )
    .unwrap();
    let render = GpuRenderOperation::new([attachment], None, [draw], None).unwrap();

    assert_eq!(render.color_attachments().len(), 1);
    let proof_attachment = &render.color_attachments()[0];
    match proof_attachment.load() {
        GpuColorAttachmentLoad::Clear(value) => {
            assert_eq!(value.components(), [0.0, 0.0, 0.0, 1.0]);
        }
        GpuColorAttachmentLoad::Load => panic!("known-pattern proof must clear before drawing"),
    }
    assert_eq!(proof_attachment.store(), GpuAttachmentStore::Store);

    let readback_region = GpuTextureCopyRegion::new(
        &texture,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::Color,
        GpuCopyExtent::new(WIDTH, HEIGHT, 1).unwrap(),
    )
    .unwrap();
    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(readback_region.into(), readback_id).unwrap();

    let name = "indexed offscreen known pattern";
    let mut builder = GpuWorkFragmentBuilder::new(label(name), provenance(name));
    builder.declare_resource(texture.into()).unwrap();
    builder.declare_resource(view.into()).unwrap();
    builder.declare_resource(index_buffer.into()).unwrap();
    builder
        .add_node(
            label("indexed offscreen draw"),
            GpuWorkOperation::Render(render),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::Automatic,
            provenance("indexed offscreen draw"),
        )
        .unwrap();
    builder
        .add_node(
            label("indexed offscreen readback"),
            GpuWorkOperation::Readback(readback),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::Automatic,
            provenance("indexed offscreen readback"),
        )
        .unwrap();

    (
        GpuPreparedWorkGraph::prepare(
            label("indexed offscreen known-pattern graph"),
            [builder.finish().unwrap()],
        )
        .unwrap(),
        readback_id,
    )
}

fn progress_to_readback(
    context: &GpuContext,
    submission: &GpuSubmission,
    readback: &GpuReadback,
) -> GpuReadbackBytes {
    let deadline = Instant::now() + Duration::from_secs(15);
    let bytes = loop {
        context.progress();
        match readback.status() {
            GpuReadbackStatus::Ready(bytes) => break bytes,
            GpuReadbackStatus::Failed(failure) => {
                panic!("indexed offscreen readback failed: {failure:?}")
            }
            GpuReadbackStatus::Pending => {}
        }
        if let GpuSubmissionStatus::Failed(failure) = submission.status() {
            panic!("indexed offscreen submission failed before readback: {failure:?}");
        }
        assert!(
            Instant::now() < deadline,
            "indexed offscreen readback timed out"
        );
        std::thread::yield_now();
    };

    loop {
        context.progress();
        match submission.status() {
            GpuSubmissionStatus::Completed => break,
            GpuSubmissionStatus::Failed(failure) => {
                panic!("indexed offscreen submission failed: {failure:?}")
            }
            GpuSubmissionStatus::Accepted => {}
        }
        assert!(
            Instant::now() < deadline,
            "indexed offscreen submission did not terminalize"
        );
        std::thread::yield_now();
    }
    bytes
}

fn pixel_at(bytes: &GpuReadbackBytes, x: u32, y: u32) -> [u8; 4] {
    assert!(x < WIDTH && y < HEIGHT);
    let offset = usize::try_from((y * WIDTH + x) * 4).unwrap();
    bytes.as_bytes()[offset..offset + 4].try_into().unwrap()
}

pub(crate) fn assert_known_pattern(bytes: &GpuReadbackBytes) {
    assert_eq!(bytes.texture_format(), Some(GpuTextureFormat::Rgba8Unorm));
    assert_eq!(
        bytes.as_bytes().len(),
        usize::try_from(WIDTH * HEIGHT * 4).unwrap()
    );

    for (x, y) in [(1, 1), (2, 6)] {
        assert_eq!(
            pixel_at(bytes, x, y),
            DRAW_PIXEL,
            "selected pixel ({x}, {y}) inside the draw scissor must contain indexed draw output"
        );
    }
    for (x, y) in [(5, 1), (6, 6)] {
        assert_eq!(
            pixel_at(bytes, x, y),
            CLEAR_PIXEL,
            "selected pixel ({x}, {y}) outside the draw scissor must preserve the known clear value"
        );
    }
}

fn artifact_directory() -> PathBuf {
    std::env::var_os("RUNEN_GPU_PROOF_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/runengpu-proof-artifacts")
        })
}

fn write_validated_png(bytes: &GpuReadbackBytes) -> PathBuf {
    let directory = artifact_directory();
    std::fs::create_dir_all(&directory).expect("proof artifact directory must be creatable");
    let path = directory.join(ARTIFACT_NAME);
    image::save_buffer_with_format(
        &path,
        bytes.as_bytes(),
        WIDTH,
        HEIGHT,
        image::ColorType::Rgba8,
        image::ImageFormat::Png,
    )
    .expect("validated offscreen bytes must encode as PNG");
    assert!(
        std::fs::metadata(&path)
            .expect("proof PNG metadata must be readable")
            .len()
            > 0,
        "proof PNG must not be empty"
    );
    println!(
        "RunenGPU known-pattern offscreen artifact: {}",
        path.display()
    );
    path
}

#[test]
#[ignore = "requires a real Vulkan fallback adapter; executed by RunenGPU Conformance CI"]
fn indexed_offscreen_draw_matches_known_pattern_and_writes_png() {
    let context = native_render_context();
    let (graph, readback_id) = render_graph();

    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let readback = submission
        .readback(readback_id)
        .expect("accepted indexed offscreen readback must remain observable")
        .clone();
    let bytes = progress_to_readback(&context, &submission, &readback);

    assert_known_pattern(&bytes);
    let artifact = write_validated_png(&bytes);
    assert!(artifact.ends_with(ARTIFACT_NAME));

    let stats = context.execution_stats();
    assert_eq!(stats.prepared_submissions(), 0);
    assert_eq!(stats.in_flight_submissions(), 0);
    assert_eq!(stats.upload_bytes_in_flight(), 0);
    assert_eq!(stats.readback_bytes_in_flight(), 0);
    assert_eq!(stats.pending_readbacks(), 0);
}
