use bytemuck::{Pod, Zeroable};
use runen_gpu::*;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const WIDTH: u32 = 16;
const HEIGHT: u32 = 16;
const CLEAR_PIXEL: [u8; 4] = [0, 0, 0, 255];
const DRAW_PIXEL: [u8; 4] = [0, 255, 0, 255];
const ARTIFACT_NAME: &str = "compute-generated-indirect-offscreen.png";

const COMPUTE_WGSL: &str = r#"
struct DrawIndirectArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

struct GeneratedVertex {
    position: vec2<f32>,
    _pad: vec2<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0)
var<storage, read_write> draw_args: DrawIndirectArgs;

@group(0) @binding(1)
var<storage, read_write> vertices: array<GeneratedVertex>;

@compute @workgroup_size(1)
fn cs_main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if invocation.x != 0u || invocation.y != 0u || invocation.z != 0u {
        return;
    }

    draw_args.vertex_count = 3u;
    draw_args.instance_count = 1u;
    draw_args.first_vertex = 0u;
    draw_args.first_instance = 0u;

    let green = vec4<f32>(0.0, 1.0, 0.0, 1.0);
    vertices[0] = GeneratedVertex(vec2<f32>(-0.75, -0.75), vec2<f32>(0.0), green);
    vertices[1] = GeneratedVertex(vec2<f32>( 0.75, -0.75), vec2<f32>(0.0), green);
    vertices[2] = GeneratedVertex(vec2<f32>( 0.00,  0.75), vec2<f32>(0.0), green);
}
"#;

const RENDER_WGSL: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
"#;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct DrawIndirectArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GeneratedVertex {
    position: [f32; 2],
    _pad: [f32; 2],
    color: [f32; 4],
}

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

fn native_context() -> GpuContext {
    let mut requirements = GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements();
    for feature in [
        GpuCapabilityFeature::Compute,
        GpuCapabilityFeature::IndirectExecution,
    ] {
        requirements
            .insert(GpuCapabilityRequirement::Required(feature))
            .unwrap();
    }
    let descriptor = GpuContextDescriptor::new(requirements)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label("compute-generated indirect drawing proof");
    let context = pollster::block_on(GpuContext::request(descriptor))
        .expect("native conformance environment must provide a Vulkan fallback adapter");
    assert_eq!(context.adapter_facts().backend(), GpuBackendFamily::Vulkan);
    assert_eq!(
        context.adapter_facts().fallback(),
        GpuFallbackStatus::ConfirmedFallback,
        "compute-generated indirect conformance must execute through the required fallback path"
    );
    context
}

fn pipelines() -> (GpuComputePipelineDescriptor, GpuRenderPipelineDescriptor) {
    let [compute_source, render_source] = admit_static_wgsl_sources([
        ("proof.compute-generated-indirect.compute", 1, COMPUTE_WGSL),
        ("proof.compute-generated-indirect.render", 1, RENDER_WGSL),
    ])
    .unwrap();

    let compute = GpuComputePipelineDescriptor::ordinary(compute_source, "cs_main").unwrap();

    let vertex_entry = GpuEntryPointName::new("vs_main").unwrap();
    let fragment_entry = GpuEntryPointName::new("fs_main").unwrap();
    let render_program = GpuProgramDescriptor::new(
        render_source,
        [vertex_entry.clone(), fragment_entry.clone()],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();
    let vertex_layout = GpuVertexBufferLayoutDescriptor::new(
        0,
        std::mem::size_of::<GeneratedVertex>() as u64,
        GpuVertexStepMode::Vertex,
        [
            GpuVertexAttribute::new(0, 0, GpuVertexFormat::Float32x2),
            GpuVertexAttribute::new(1, 16, GpuVertexFormat::Float32x4),
        ],
    )
    .unwrap();
    let color_target = GpuColorTargetStateDescriptor::new(
        GpuTextureFormat::Rgba8Unorm,
        GpuBlendMode::Replace,
        GpuColorWriteMask::ALL,
    )
    .unwrap();
    let render_state = GpuRenderPipelineStateDescriptor::new(
        GpuVertexInputStateDescriptor::new([vertex_layout]).unwrap(),
        Some(GpuFragmentOutputStateDescriptor::new([color_target])),
        GpuPrimitiveStateDescriptor::default(),
        None,
        GpuMultisampleStateDescriptor::default(),
    )
    .unwrap();
    let render = GpuRenderPipelineDescriptor::new(
        render_program,
        GpuRenderEntryPoints::new(vertex_entry, Some(fragment_entry)),
        render_state,
        GpuPipelineConfiguration::default(),
    )
    .unwrap();
    (compute, render)
}

fn prepared_buffer<T: Pod>(
    scope: &mut GpuResourceScope,
    name: &str,
    values: &[T],
    usages: impl IntoIterator<Item = GpuBufferUsage>,
) -> GpuBufferHandle {
    let prepared = PreparedGpuData::<TransferData>::ordinary_pod_transfer(name, values).unwrap();
    let resource_label = label(name);
    let mut usages = usages.into_iter().collect::<Vec<_>>();
    usages.push(GpuBufferUsage::CopyDestination);
    scope
        .buffer(
            GpuBufferDescriptor::new(
                common(name),
                prepared.layout().byte_len(),
                GpuBufferUsages::new(&resource_label, usages).unwrap(),
                GpuBufferInitialization::Prepared(prepared),
            )
            .unwrap(),
        )
        .unwrap()
}

fn generated_buffers(scope: &mut GpuResourceScope) -> (GpuBufferHandle, GpuBufferHandle) {
    let args = prepared_buffer(
        scope,
        "compute-generated indirect arguments",
        &[DrawIndirectArgs::zeroed()],
        [GpuBufferUsage::Storage, GpuBufferUsage::Indirect],
    );
    let vertices = prepared_buffer(
        scope,
        "compute-generated vertices",
        &[GeneratedVertex::zeroed(); 3],
        [GpuBufferUsage::Storage, GpuBufferUsage::Vertex],
    );
    (args, vertices)
}

fn render_target(scope: &mut GpuResourceScope) -> (GpuTextureHandle, GpuTextureViewHandle) {
    let texture_label = label("compute-generated indirect target");
    let texture = scope
        .texture(
            GpuTextureDescriptor::new(
                common("compute-generated indirect target"),
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
    let view = scope
        .texture_view(
            GpuTextureViewDescriptor::new(
                common("compute-generated indirect target view"),
                &texture,
                None,
                GpuTextureDimension::D2,
                GpuTextureSubresourceRange::whole(&texture).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    (texture, view)
}

fn compute_operation(
    pipeline: &GpuComputePipelineDescriptor,
    args: &GpuBufferHandle,
    vertices: &GpuBufferHandle,
) -> GpuComputeOperation {
    let bindings = pipeline
        .runtime_bindings([
            GpuRuntimeBindingValue::whole_buffer(0, 0, args),
            GpuRuntimeBindingValue::whole_buffer(0, 1, vertices),
        ])
        .unwrap();
    GpuComputeOperation::new(
        pipeline.clone(),
        bindings,
        GpuDispatchIntent::direct(GpuDispatchSize::new(1, 1, 1)),
    )
    .unwrap()
}

fn render_operation(
    pipeline: &GpuRenderPipelineDescriptor,
    args: &GpuBufferHandle,
    vertices: &GpuBufferHandle,
    target: &GpuTextureViewHandle,
) -> GpuRenderOperation {
    let bindings = GpuRuntimeBindingSet::new(pipeline.layout().clone(), []).unwrap();
    let vertex_binding =
        GpuVertexBufferBinding::new(0, vertices, GpuBufferRange::whole(vertices).unwrap()).unwrap();
    let draw = GpuRenderDraw::new(
        pipeline.clone(),
        bindings,
        [vertex_binding],
        None,
        GpuDrawIntent::indirect(args, GpuBufferRange::whole(args).unwrap(), false).unwrap(),
        GpuViewport::new(0.0, 0.0, WIDTH as f32, HEIGHT as f32, 0.0, 1.0).unwrap(),
        GpuScissorRect::new(0, 0, WIDTH, HEIGHT).unwrap(),
        GpuBlendConstant::new(0.0, 0.0, 0.0, 0.0).unwrap(),
        0,
    )
    .unwrap();
    let attachment = GpuRenderColorAttachment::new(
        target.clone(),
        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
        GpuAttachmentStore::Store,
        None,
    )
    .unwrap();
    GpuRenderOperation::new([attachment], None, [draw], None).unwrap()
}

fn graph() -> (
    GpuPreparedWorkGraph,
    GpuReadbackId,
    GpuBufferHandle,
    GpuBufferHandle,
) {
    let mut scope = GpuResourceScope::new();
    let (args, vertices) = generated_buffers(&mut scope);
    let (target, target_view) = render_target(&mut scope);
    let (compute_pipeline, render_pipeline) = pipelines();
    let compute = compute_operation(&compute_pipeline, &args, &vertices);
    let render = render_operation(&render_pipeline, &args, &vertices, &target_view);
    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(
        GpuTextureCopyRegion::new(
            &target,
            0,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::Color,
            GpuCopyExtent::new(WIDTH, HEIGHT, 1).unwrap(),
        )
        .unwrap()
        .into(),
        readback_id,
    )
    .unwrap();

    let fragment = GpuWorkFragment::build("compute-generated indirect drawing", |builder| {
        builder.operation("generate indirect draw data", compute)?;
        builder.operation("consume generated indirect draw", render)?;
        builder.operation("read compute-generated indirect target", readback)?;
        Ok(())
    })
    .unwrap();
    let graph = GpuPreparedWorkGraph::prepare(
        label("compute-generated indirect drawing graph"),
        [fragment],
    )
    .unwrap();
    (graph, readback_id, args, vertices)
}

fn assert_required(requirements: &GpuCapabilityRequirements, feature: GpuCapabilityFeature) {
    assert_eq!(
        requirements.get(feature),
        Some(GpuCapabilityRequirement::Required(feature)),
        "prepared graph must mechanically require {feature:?}"
    );
}

fn assert_prepared_materialization(graph: &GpuPreparedWorkGraph, buffer: &GpuBufferHandle) {
    assert!(
        matches!(
            buffer.descriptor().initialization(),
            GpuBufferInitialization::Prepared(_)
        ),
        "proof buffers must enter through the accepted Prepared initial-content contract"
    );
    let summary = graph
        .initialization()
        .iter()
        .find(|summary| summary.resource().diagnostic_identity() == buffer.diagnostic_identity())
        .expect("prepared buffer must publish initialization evidence");
    assert!(
        summary.initial().is_none(),
        "G5R requires Prepared descriptor metadata alone to establish no initial coverage"
    );
    let final_coverage = summary
        .final_coverage()
        .expect("canonical Prepared materialization must leave readable buffer coverage");
    assert_eq!(
        final_coverage.buffer_values(),
        Some(
            &[GpuBufferCoverage::Dense(
                GpuBufferRange::whole(buffer).unwrap()
            )][..]
        ),
        "canonical initial-content materialization must establish exact whole-buffer coverage before compute mutates it"
    );
}

fn assert_graph_contract(
    graph: &GpuPreparedWorkGraph,
    args: &GpuBufferHandle,
    vertices: &GpuBufferHandle,
) {
    for feature in [
        GpuCapabilityFeature::Compute,
        GpuCapabilityFeature::RenderPipeline,
        GpuCapabilityFeature::IndirectExecution,
    ] {
        assert_required(graph.requirements(), feature);
    }

    let compute = graph
        .nodes()
        .iter()
        .find(|prepared| prepared.node().label().as_str() == "generate indirect draw data")
        .expect("prepared graph must retain the compute producer");
    let render = graph
        .nodes()
        .iter()
        .find(|prepared| prepared.node().label().as_str() == "consume generated indirect draw")
        .expect("prepared graph must retain the render consumer");

    assert!(
        render.node().accesses().iter().any(|access| {
            matches!(
                access,
                GpuResourceAccess::Buffer(access)
                    if access.resource_identity() == args.diagnostic_identity()
                        && access.kind() == GpuBufferAccessKind::IndirectRead
                        && access.range() == GpuBufferRange::whole(args).unwrap()
            )
        }),
        "indirect draw intent must mechanically derive the exact IndirectRead access"
    );
    assert!(
        render.node().accesses().iter().any(|access| {
            matches!(
                access,
                GpuResourceAccess::Buffer(access)
                    if access.resource_identity() == vertices.diagnostic_identity()
                        && access.kind() == GpuBufferAccessKind::VertexRead
                        && access.range() == GpuBufferRange::whole(vertices).unwrap()
            )
        }),
        "vertex-buffer binding must mechanically derive the exact VertexRead access"
    );

    let dependency = graph
        .dependencies()
        .iter()
        .find(|dependency| dependency.before() == compute.id() && dependency.after() == render.id())
        .expect("compute producer must data-depend into the render consumer");
    assert!(
        dependency
            .reasons()
            .iter()
            .all(|reason| !matches!(reason, GpuDependencyReason::ExplicitNonData { .. })),
        "compute -> render composition must not use an explicit non-data ordering edge"
    );
    for buffer in [args, vertices] {
        let whole = GpuBufferRange::whole(buffer).unwrap();
        assert!(
            dependency.reasons().iter().any(|reason| {
                matches!(
                    reason,
                    GpuDependencyReason::ReadAfterWrite {
                        resource,
                        region: GpuDependencyRegion::Buffer(range),
                    } if *resource == buffer.diagnostic_identity() && *range == whole
                )
            }),
            "compute -> render dependency must contain a whole-buffer RAW reason for {}",
            buffer.descriptor().common().label().as_str()
        );
    }

    assert_prepared_materialization(graph, args);
    assert_prepared_materialization(graph, vertices);
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
                panic!("compute-generated indirect readback failed: {failure:?}")
            }
            GpuReadbackStatus::Pending => {}
        }
        if let GpuSubmissionStatus::Failed(failure) = submission.status() {
            panic!("compute-generated indirect submission failed before readback: {failure:?}");
        }
        assert!(
            Instant::now() < deadline,
            "compute-generated indirect readback timed out"
        );
        std::thread::yield_now();
    };

    loop {
        context.progress();
        match submission.status() {
            GpuSubmissionStatus::Completed => break,
            GpuSubmissionStatus::Failed(failure) => {
                panic!("compute-generated indirect submission failed: {failure:?}")
            }
            GpuSubmissionStatus::Accepted => {}
        }
        assert!(
            Instant::now() < deadline,
            "compute-generated indirect submission did not terminalize"
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

fn assert_rendered_pixels(bytes: &GpuReadbackBytes) {
    assert_eq!(bytes.texture_format(), Some(GpuTextureFormat::Rgba8Unorm));
    assert_eq!(
        bytes.as_bytes().len(),
        usize::try_from(WIDTH * HEIGHT * 4).unwrap()
    );
    assert_eq!(
        pixel_at(bytes, WIDTH / 2, HEIGHT / 2),
        DRAW_PIXEL,
        "center pixel must prove the compute-generated indirect draw executed"
    );
    assert_eq!(
        pixel_at(bytes, 0, 0),
        CLEAR_PIXEL,
        "corner pixel must preserve the known clear region"
    );
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
    .expect("validated compute-generated indirect bytes must encode as PNG");
    assert!(
        std::fs::metadata(&path)
            .expect("proof PNG metadata must be readable")
            .len()
            > 0,
        "proof PNG must not be empty"
    );
    println!(
        "RunenGPU compute-generated indirect artifact: {}",
        path.display()
    );
    path
}

#[test]
#[ignore = "requires a real Vulkan fallback adapter; executed by RunenGPU Conformance CI"]
fn compute_generated_indirect_draw_is_data_ordered_and_matches_known_pixels() {
    let context = native_context();
    let (graph, readback_id, args, vertices) = graph();
    assert_graph_contract(&graph, &args, &vertices);

    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let readback = submission
        .readback(readback_id)
        .expect("accepted compute-generated indirect readback must remain observable")
        .clone();
    let bytes = progress_to_readback(&context, &submission, &readback);

    assert_rendered_pixels(&bytes);
    let artifact = write_validated_png(&bytes);
    assert!(artifact.ends_with(ARTIFACT_NAME));

    let stats = context.execution_stats();
    assert_eq!(stats.prepared_submissions(), 0);
    assert_eq!(stats.in_flight_submissions(), 0);
    assert_eq!(stats.upload_bytes_in_flight(), 0);
    assert_eq!(stats.readback_bytes_in_flight(), 0);
    assert_eq!(stats.pending_readbacks(), 0);
}
