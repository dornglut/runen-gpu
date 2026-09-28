use runen_gpu::*;

#[path = "support/readback_wait.rs"]
mod readback_wait;

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const QUERY_COUNT: u32 = 2;
const RESOLVE_BYTES: u64 = 16;

const RENDER_WGSL: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-0.75, -0.75),
        vec2<f32>(0.75, -0.75),
        vec2<f32>(0.0, 0.75),
    );
    return vec4<f32>(positions[vertex_index], 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 1.0, 0.0, 1.0);
}
"#;

fn label(value: impl AsRef<str>) -> GpuResourceLabel {
    GpuResourceLabel::new(value.as_ref()).unwrap()
}

fn provenance(value: impl AsRef<str>) -> GpuResourceProvenance {
    let value = value.as_ref();
    GpuResourceProvenance::new(label(value), None, None)
}

fn common(value: impl AsRef<str>) -> GpuResourceCommon {
    let value = value.as_ref();
    GpuResourceCommon::owned(
        label(value),
        GpuResourceLifetime::Transient,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        provenance(value),
    )
    .unwrap()
}

pub(crate) fn descriptor(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
) -> GpuContextDescriptor {
    let mut descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
            .with_allowed_backends([backend])
            .with_label("R4 occlusion query proof");
    if let Some(fallback) = fallback {
        descriptor = descriptor.with_fallback_policy(fallback);
    }
    descriptor
}

fn render_pipeline() -> GpuRenderPipelineDescriptor {
    let [source] =
        admit_static_wgsl_sources([("proof.r4.occlusion-query", 1, RENDER_WGSL)]).unwrap();
    let vertex = GpuEntryPointName::new("vs_main").unwrap();
    let fragment = GpuEntryPointName::new("fs_main").unwrap();
    let program = GpuProgramDescriptor::new(
        source,
        [vertex.clone(), fragment.clone()],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();
    let target = GpuColorTargetStateDescriptor::new(
        GpuTextureFormat::Rgba8Unorm,
        None,
        GpuColorWriteMask::ALL,
    )
    .unwrap();
    let state = GpuRenderPipelineStateDescriptor::new(
        GpuVertexInputStateDescriptor::new([]).unwrap(),
        Some(GpuFragmentOutputStateDescriptor::new([target])),
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

fn graph() -> (GpuPreparedWorkGraph, GpuReadbackId) {
    let mut scope = GpuResourceScope::new();

    let query_set = scope
        .query_set(
            GpuQuerySetDescriptor::new(
                common("R4 occlusion query set"),
                GpuQueryKind::Occlusion,
                QUERY_COUNT,
            )
            .unwrap(),
        )
        .unwrap();

    let resolve_label = label("R4 occlusion query resolve");
    let resolve = scope
        .buffer(
            GpuBufferDescriptor::new(
                common("R4 occlusion query resolve"),
                RESOLVE_BYTES,
                GpuBufferUsages::new(
                    &resolve_label,
                    [GpuBufferUsage::QueryResolve, GpuBufferUsage::CopySource],
                )
                .unwrap(),
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();

    let target_label = label("R4 occlusion query target");
    let target = scope
        .texture(
            GpuTextureDescriptor::new(
                common("R4 occlusion query target"),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&target_label, GpuTextureDimension::D2, WIDTH, HEIGHT, 1)
                    .unwrap(),
                1,
                1,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(&target_label, [GpuTextureUsage::ColorAttachment]).unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let view = scope
        .texture_view(
            GpuTextureViewDescriptor::new(
                common("R4 occlusion query target view"),
                &target,
                None,
                GpuTextureViewDimension::D2,
                GpuTextureSubresourceRange::whole(&target).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();

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

    let visible = GpuOcclusionQueryScope::new(&query_set, 0, [draw]).unwrap();
    let empty = GpuOcclusionQueryScope::new(&query_set, 1, []).unwrap();
    let attachment = GpuRenderColorAttachment::new(
        view,
        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
        GpuAttachmentStore::Store,
        None,
    )
    .unwrap();
    let render = GpuRenderOperation::new(
        [attachment],
        None,
        [
            GpuRenderPassItem::OcclusionQuery(visible),
            GpuRenderPassItem::OcclusionQuery(empty),
        ],
        None,
    )
    .unwrap();

    let resolve_queries = GpuQueryResolveOperation::new(
        &query_set,
        GpuQueryRange::whole(&query_set).unwrap(),
        &resolve,
        0,
    )
    .unwrap();

    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(
        GpuBufferRegion::new(&resolve, GpuBufferRange::whole(&resolve).unwrap())
            .unwrap()
            .into(),
        readback_id,
    )
    .unwrap();

    let fragment = GpuWorkFragment::build("R4 occlusion query proof", |builder| {
        builder.operation("write occlusion queries", render)?;
        builder.operation("resolve occlusion queries", resolve_queries)?;
        builder.operation("read occlusion query results", readback)?;
        Ok(())
    })
    .unwrap();
    let graph =
        GpuPreparedWorkGraph::prepare(label("R4 occlusion query proof graph"), [fragment]).unwrap();
    assert!(
        graph
            .requirements()
            .get(GpuCapabilityFeature::TimestampQuery)
            .is_none(),
        "baseline occlusion queries must not derive TimestampQuery"
    );
    (graph, readback_id)
}

fn resolved_values(bytes: &GpuReadbackBytes) -> (u64, u64) {
    assert_eq!(
        bytes.as_bytes().len(),
        usize::try_from(RESOLVE_BYTES).unwrap()
    );
    (
        u64::from_le_bytes(bytes.as_bytes()[0..8].try_into().unwrap()),
        u64::from_le_bytes(bytes.as_bytes()[8..16].try_into().unwrap()),
    )
}

pub(crate) fn assert_results(bytes: &GpuReadbackBytes) {
    let (visible, empty) = resolved_values(bytes);
    println!("R4 occlusion query resolved values: visible={visible}, empty={empty}");
    assert_ne!(
        visible, 0,
        "visible queried draw must resolve a nonzero occlusion result"
    );
    assert_eq!(empty, 0, "empty queried scope must resolve zero");
}

async fn run_case_bytes(context: &GpuContext) -> GpuReadbackBytes {
    let (graph, readback_id) = graph();
    let prepared = context.prepare_submission(graph).await.unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    readback_wait::wait_for_readback(
        context,
        &submission,
        readback_id,
        "R4 occlusion query proof",
    )
    .await
}

pub(crate) async fn run_case(context: &GpuContext) {
    let bytes = run_case_bytes(context).await;
    assert_results(&bytes);
}

#[cfg(target_arch = "wasm32")]
pub(crate) async fn run_browser_occlusion_query() -> u32 {
    let context = GpuContext::request(descriptor(GpuBackendFamily::BrowserWebGpu, None))
        .await
        .expect("actual-browser Conformance must admit baseline occlusion queries");
    let bytes = run_case_bytes(&context).await;
    let (visible, empty) = resolved_values(&bytes);
    let mask = u32::from(visible != 0) | (u32::from(empty == 0) << 1);
    println!(
        "OcclusionQuery BrowserWebGpu: result_mask={mask}, visible={visible}, empty={empty}"
    );
    mask
}

#[test]
fn occlusion_query_graph_requires_no_optional_query_capability() {
    let (graph, _) = graph();
    assert!(
        graph
            .requirements()
            .get(GpuCapabilityFeature::TimestampQuery)
            .is_none()
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires the retained Vulkan software adapter"]
fn occlusion_query_native_execution_resolves_visible_and_empty_scopes() {
    let context = pollster::block_on(GpuContext::request(descriptor(
        GpuBackendFamily::Vulkan,
        Some(GpuSoftwareFallbackPolicy::Require),
    )))
    .expect("native Conformance must provide the retained Vulkan fallback adapter");
    pollster::block_on(run_case(&context));
    println!("OcclusionQuery Vulkan: EXERCISED");
}
