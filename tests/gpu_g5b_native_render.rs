use runen_gpu::*;
use std::time::{Duration, Instant};

const WIDTH: u32 = 4;
const HEIGHT: u32 = 4;
const OPAQUE_RED: [u8; 4] = [255, 0, 0, 255];

const OFFSCREEN_RENDER_WGSL: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4f {
    let positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0),
    );
    return vec4f(positions[vertex_index], 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(1.0, 0.0, 0.0, 1.0);
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

fn native_render_context() -> GpuContext {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .with_label("G5B native offscreen render proof");
    let context = pollster::block_on(GpuContext::request(descriptor))
        .expect("native conformance environment must provide a Vulkan fallback adapter");
    assert_eq!(context.adapter_facts().backend(), GpuBackendFamily::Vulkan);
    assert_eq!(
        context.adapter_facts().fallback(),
        GpuFallbackStatus::ConfirmedFallback,
        "native render conformance must execute through the explicitly required fallback path"
    );
    context
}

fn admitted_render_source() -> GpuAdmittedProgramSource {
    admitted_render_source_from("g5b.native.offscreen-render", OFFSCREEN_RENDER_WGSL)
}

fn admitted_render_source_from(key: &str, wgsl: &str) -> GpuAdmittedProgramSource {
    let identity = GpuProgramSourceIdentity::new(
        GpuProgramSourceOwnerId::allocate().expect("native render source owner should allocate"),
        GpuProgramSourceKey::new(key).unwrap(),
        GpuProgramSourceRevision::try_from_raw(1).unwrap(),
    );
    let mut sources = GpuProgramSourceRegistry::new(4, 16 * 1024).unwrap();
    sources
        .admit_wgsl(
            identity,
            wgsl,
            GpuProgramSourceProvenance::new("g5b-native-offscreen-render-proof", None).unwrap(),
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

fn render_pipeline_with_vertex_input(
    key: &str,
    wgsl: &str,
    vertex_input: GpuVertexInputStateDescriptor,
) -> GpuRenderPipelineDescriptor {
    let vertex = GpuEntryPointName::new("vs_main").unwrap();
    let fragment = GpuEntryPointName::new("fs_main").unwrap();
    let program = GpuProgramDescriptor::new(
        admitted_render_source_from(key, wgsl),
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
        vertex_input,
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

fn vertex_input_wgsl(attribute_count: u32) -> String {
    let fields = (0..attribute_count)
        .map(|location| format!("    @location({location}) value_{location}: f32,\n"))
        .collect::<String>();
    format!(
        "struct VertexInput {{\n{fields}}}\n\n@vertex\nfn vs_main(input: VertexInput) -> @builtin(position) vec4f {{\n    return vec4f(input.value_0 * 0.0, 0.0, 0.0, 1.0);\n}}\n\n@fragment\nfn fs_main() -> @location(0) vec4f {{\n    return vec4f(1.0, 0.0, 0.0, 1.0);\n}}\n"
    )
}

fn render_target(
    allocator: &mut GpuWorkResourceIdAllocator,
) -> (GpuTextureHandle, GpuTextureViewHandle) {
    let texture_label = label("native offscreen render target");
    let texture = allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::new(
                common("native offscreen render target"),
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
                common("native offscreen render target view"),
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

fn render_graph() -> (GpuPreparedWorkGraph, GpuReadbackId) {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let (texture, view) = render_target(&mut allocator);
    let pipeline = render_pipeline();
    let bindings = GpuRuntimeBindingSet::new(pipeline.layout().clone(), []).unwrap();
    let draw = GpuRenderDraw::new(
        pipeline,
        bindings,
        [],
        None,
        GpuDrawIntent::direct(
            GpuDrawRange::new(0, 3).unwrap(),
            GpuDrawRange::new(0, 1).unwrap(),
        ),
        GpuViewport::new(0.0, 0.0, WIDTH as f32, HEIGHT as f32, 0.0, 1.0).unwrap(),
        GpuScissorRect::new(0, 0, WIDTH, HEIGHT).unwrap(),
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

    let name = "native offscreen render";
    let mut builder = GpuWorkFragmentBuilder::new(label(name), provenance(name));
    builder.declare_resource(texture.into()).unwrap();
    builder.declare_resource(view.into()).unwrap();
    builder
        .add_node(
            label("native offscreen render draw"),
            GpuWorkOperation::Render(render),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::Automatic,
            provenance("native offscreen render draw"),
        )
        .unwrap();
    builder
        .add_node(
            label("native offscreen render readback"),
            GpuWorkOperation::Readback(readback),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::Automatic,
            provenance("native offscreen render readback"),
        )
        .unwrap();

    (
        GpuPreparedWorkGraph::prepare(
            label("native offscreen render graph"),
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
                panic!("native G5B render readback failed: {failure:?}")
            }
            GpuReadbackStatus::Pending => {}
        }
        if let GpuSubmissionStatus::Failed(failure) = submission.status() {
            panic!("native G5B render submission failed before readback: {failure:?}");
        }
        assert!(
            Instant::now() < deadline,
            "native G5B render readback timed out"
        );
        std::thread::yield_now();
    };

    loop {
        context.progress();
        match submission.status() {
            GpuSubmissionStatus::Completed => break,
            GpuSubmissionStatus::Failed(failure) => {
                panic!("native G5B render submission failed: {failure:?}")
            }
            GpuSubmissionStatus::Accepted => {}
        }
        assert!(
            Instant::now() < deadline,
            "native G5B render submission did not terminalize"
        );
        std::thread::yield_now();
    }
    bytes
}

fn oversized_texture(
    allocator: &mut GpuWorkResourceIdAllocator,
    name: &str,
    dimension: GpuTextureDimension,
    width: u32,
    height: u32,
    depth_or_layers: u32,
) -> GpuTextureHandle {
    let texture_label = label(name);
    allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::new(
                common(name),
                dimension,
                GpuTextureExtent::new(&texture_label, dimension, width, height, depth_or_layers)
                    .unwrap(),
                1,
                1,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(&texture_label, [GpuTextureUsage::ColorAttachment]).unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap()
}

fn realize_render_pipeline_error(
    context: &GpuContext,
    descriptor: &GpuRenderPipelineDescriptor,
) -> GpuPipelineRealizationError {
    let program = pollster::block_on(context.realize_program(descriptor.program())).unwrap();
    let layout = pollster::block_on(context.realize_pipeline_layout(descriptor.layout())).unwrap();
    pollster::block_on(context.realize_render_pipeline(descriptor, &program, &layout)).unwrap_err()
}

#[test]
#[ignore = "requires a real Vulkan fallback adapter; executed by RunenGPU Native Conformance CI"]
fn native_normalized_public_resource_limits_reject_before_backend_creation() {
    let context = native_render_context();
    let limits = context.device_facts().workload_budget().limits();
    let mut allocator = GpuWorkResourceIdAllocator::new();

    let buffer_common = common("normalized max buffer size rejection");
    let buffer = allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                buffer_common.clone(),
                limits.max_buffer_size().checked_add(1).unwrap(),
                GpuBufferUsages::new(buffer_common.label(), [GpuBufferUsage::CopySource]).unwrap(),
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let buffer_error = context.realize_buffer(&buffer).unwrap_err();
    assert_eq!(
        buffer_error.category(),
        GpuResourceRealizationErrorCategory::FormatOrAlignmentNotAdmitted
    );

    let d1 = oversized_texture(
        &mut allocator,
        "normalized D1 dimension rejection",
        GpuTextureDimension::D1,
        limits.max_texture_dimension_1d().checked_add(1).unwrap(),
        1,
        1,
    );
    let d1_error = context.realize_texture(&d1).unwrap_err();
    assert_eq!(
        d1_error.category(),
        GpuResourceRealizationErrorCategory::FormatOrAlignmentNotAdmitted
    );

    let d3 = oversized_texture(
        &mut allocator,
        "normalized D3 dimension rejection",
        GpuTextureDimension::D3,
        limits.max_texture_dimension_3d().checked_add(1).unwrap(),
        1,
        1,
    );
    let d3_error = context.realize_texture(&d3).unwrap_err();
    assert_eq!(
        d3_error.category(),
        GpuResourceRealizationErrorCategory::FormatOrAlignmentNotAdmitted
    );

    let d2_array = oversized_texture(
        &mut allocator,
        "normalized D2 array layer rejection",
        GpuTextureDimension::D2,
        1,
        1,
        limits.max_texture_array_layers().checked_add(1).unwrap(),
    );
    let d2_array_error = context.realize_texture(&d2_array).unwrap_err();
    assert_eq!(
        d2_array_error.category(),
        GpuResourceRealizationErrorCategory::FormatOrAlignmentNotAdmitted
    );
}

#[test]
#[ignore = "requires a real Vulkan fallback adapter; executed by RunenGPU Native Conformance CI"]
fn native_normalized_vertex_limits_reject_before_render_pipeline_creation() {
    let context = native_render_context();
    let limits = context.device_facts().workload_budget().limits();

    let attribute_count = limits.max_vertex_attributes().checked_add(1).unwrap();
    let attribute_stride = u64::from(attribute_count) * 4;
    assert!(
        attribute_stride <= u64::from(limits.max_vertex_buffer_array_stride()),
        "attribute-count proof must isolate the normalized attribute limit"
    );
    let attributes = (0..attribute_count)
        .map(|location| {
            GpuVertexAttribute::new(location, u64::from(location) * 4, GpuVertexFormat::Float32)
        })
        .collect::<Vec<_>>();
    let attribute_layout = GpuVertexBufferLayoutDescriptor::new(
        0,
        attribute_stride,
        GpuVertexStepMode::Vertex,
        attributes,
    )
    .unwrap();
    let attribute_pipeline = render_pipeline_with_vertex_input(
        "g5b.native.normalized-vertex-attribute-limit",
        &vertex_input_wgsl(attribute_count),
        GpuVertexInputStateDescriptor::new([attribute_layout]).unwrap(),
    );
    let attribute_error = realize_render_pipeline_error(&context, &attribute_pipeline);
    assert_eq!(
        attribute_error.category(),
        GpuPipelineRealizationErrorCategory::FormatOrAlignmentNotAdmitted
    );
    assert!(
        attribute_error
            .detail()
            .is_some_and(|detail| detail.contains("vertex attribute count"))
    );

    let oversized_stride = u64::from(limits.max_vertex_buffer_array_stride())
        .checked_add(4)
        .unwrap();
    let stride_layout = GpuVertexBufferLayoutDescriptor::new(
        0,
        oversized_stride,
        GpuVertexStepMode::Vertex,
        [GpuVertexAttribute::new(0, 0, GpuVertexFormat::Float32)],
    )
    .unwrap();
    let stride_pipeline = render_pipeline_with_vertex_input(
        "g5b.native.normalized-vertex-stride-limit",
        &vertex_input_wgsl(1),
        GpuVertexInputStateDescriptor::new([stride_layout]).unwrap(),
    );
    let stride_error = realize_render_pipeline_error(&context, &stride_pipeline);
    assert_eq!(
        stride_error.category(),
        GpuPipelineRealizationErrorCategory::FormatOrAlignmentNotAdmitted
    );
    assert!(
        stride_error
            .detail()
            .is_some_and(|detail| detail.contains("vertex-buffer stride"))
    );
}

#[test]
#[ignore = "requires a real Vulkan fallback adapter; executed by RunenGPU Native Conformance CI"]
fn native_offscreen_render_executes_shader_and_reads_back_color() {
    let context = native_render_context();
    let (graph, readback_id) = render_graph();

    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let readback = submission
        .readback(readback_id)
        .expect("accepted native render readback must remain observable")
        .clone();
    let bytes = progress_to_readback(&context, &submission, &readback);

    assert_eq!(bytes.texture_format(), Some(GpuTextureFormat::Rgba8Unorm));
    assert_eq!(
        bytes.as_bytes().len(),
        usize::try_from(WIDTH * HEIGHT * 4).unwrap()
    );
    let mut pixel_chunks = bytes.as_bytes().chunks_exact(4);
    assert!(pixel_chunks.remainder().is_empty());
    let pixels = pixel_chunks
        .by_ref()
        .map(|pixel| <[u8; 4]>::try_from(pixel).unwrap())
        .collect::<Vec<_>>();
    for pixel in pixels {
        assert_eq!(
            pixel, OPAQUE_RED,
            "real Vulkan Render execution must replace the black clear with the shader's opaque red output"
        );
    }

    let stats = context.execution_stats();
    assert_eq!(stats.prepared_submissions(), 0);
    assert_eq!(stats.in_flight_submissions(), 0);
    assert_eq!(stats.upload_bytes_in_flight(), 0);
    assert_eq!(stats.readback_bytes_in_flight(), 0);
    assert_eq!(stats.pending_readbacks(), 0);
}
