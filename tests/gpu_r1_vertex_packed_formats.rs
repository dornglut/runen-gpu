use runen_gpu::*;

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const CLEAR_PIXEL: [u8; 4] = [0, 0, 0, 255];
const DRAW_PIXEL: [u8; 4] = [0, 255, 0, 255];

#[derive(Clone, Copy)]
struct PackedVertexCase {
    name: &'static str,
    format: GpuVertexFormat,
    bytes: [u8; 4],
    condition: &'static str,
}

const CASES: [PackedVertexCase; 2] = [
    PackedVertexCase {
        name: "unorm10_10_10_2",
        format: GpuVertexFormat::Unorm10_10_10_2,
        // x = 1023, y = 256, z = 768, w = 2 -> packed u32 0xb00403ff.
        bytes: [0xff, 0x03, 0x04, 0xb0],
        condition: "value.x > 0.99 && value.y > 0.24 && value.y < 0.26 && value.z > 0.74 && value.z < 0.76 && value.w > 0.65 && value.w < 0.68",
    },
    PackedVertexCase {
        name: "unorm8x4_bgra",
        format: GpuVertexFormat::Unorm8x4Bgra,
        bytes: [32, 96, 224, 160],
        condition: "value.x > 0.87 && value.x < 0.89 && value.y > 0.37 && value.y < 0.39 && value.z > 0.12 && value.z < 0.14 && value.w > 0.62 && value.w < 0.64",
    },
];

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

fn shader_source(case: PackedVertexCase) -> String {
    format!(
        r#"
struct VertexOutput {{
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) value: vec4<f32>
) -> VertexOutput {{
    var position = vec2<f32>(0.0, 0.75);
    if vertex_index == 0u {{
        position = vec2<f32>(-0.75, -0.75);
    }} else if vertex_index == 1u {{
        position = vec2<f32>(0.75, -0.75);
    }}
    let decoded = {};
    var output: VertexOutput;
    output.position = vec4<f32>(position, 0.0, 1.0);
    output.color = vec4<f32>(0.0, select(0.0, 1.0, decoded), 0.0, 1.0);
    return output;
}}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {{
    return input.color;
}}
"#,
        case.condition
    )
}

fn pipeline(case: PackedVertexCase) -> GpuRenderPipelineDescriptor {
    let source_text = shader_source(case);
    let identity = GpuProgramSourceIdentity::new(
        GpuProgramSourceOwnerId::allocate().unwrap(),
        GpuProgramSourceKey::new(format!("r1.vertex_packed.{}", case.name)).unwrap(),
        GpuProgramSourceRevision::try_from_raw(1).unwrap(),
    );
    let mut sources = GpuProgramSourceRegistry::new(2, 16 * 1024).unwrap();
    let source = sources
        .admit_wgsl(
            identity,
            &source_text,
            GpuProgramSourceProvenance::new(format!("R1 packed vertex {} proof", case.name), None)
                .unwrap(),
        )
        .unwrap();
    let vertex = GpuEntryPointName::new("vs_main").unwrap();
    let fragment = GpuEntryPointName::new("fs_main").unwrap();
    let program = GpuProgramDescriptor::new(
        source,
        [vertex.clone(), fragment.clone()],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();
    let layout = GpuVertexBufferLayoutDescriptor::new(
        0,
        4,
        GpuVertexStepMode::Vertex,
        [GpuVertexAttribute::new(0, 0, case.format)],
    )
    .unwrap();
    let target = GpuColorTargetStateDescriptor::new(
        GpuTextureFormat::Rgba8Unorm,
        GpuBlendMode::Replace,
        GpuColorWriteMask::ALL,
    )
    .unwrap();
    let state = GpuRenderPipelineStateDescriptor::new(
        GpuVertexInputStateDescriptor::new([layout]).unwrap(),
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

fn vertex_buffer(scope: &mut GpuResourceScope, case: PackedVertexCase) -> GpuBufferHandle {
    let mut bytes = Vec::with_capacity(12);
    for _ in 0..3 {
        bytes.extend_from_slice(&case.bytes);
    }
    let prepared =
        PreparedGpuData::<TransferData>::ordinary_pod_transfer(case.name, bytes.as_slice())
            .unwrap();
    let name = format!("{} packed vertex buffer", case.name);
    let resource_label = label(&name);
    scope
        .buffer(
            GpuBufferDescriptor::new(
                common(&name),
                prepared.layout().byte_len(),
                GpuBufferUsages::new(
                    &resource_label,
                    [GpuBufferUsage::Vertex, GpuBufferUsage::CopyDestination],
                )
                .unwrap(),
                GpuBufferInitialization::Prepared(prepared),
            )
            .unwrap(),
        )
        .unwrap()
}

fn render_target(
    scope: &mut GpuResourceScope,
    case: PackedVertexCase,
) -> (GpuTextureHandle, GpuTextureViewHandle) {
    let name = format!("{} packed vertex target", case.name);
    let resource_label = label(&name);
    let texture = scope
        .texture(
            GpuTextureDescriptor::new(
                common(&name),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, WIDTH, HEIGHT, 1)
                    .unwrap(),
                1,
                1,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(
                    &resource_label,
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
                common(format!("{name} view")),
                &texture,
                None,
                GpuTextureViewDimension::D2,
                GpuTextureSubresourceRange::whole(&texture).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    (texture, view)
}

fn graph(case: PackedVertexCase) -> (GpuPreparedWorkGraph, GpuReadbackId) {
    let mut scope = GpuResourceScope::new();
    let vertices = vertex_buffer(&mut scope, case);
    let (target, target_view) = render_target(&mut scope, case);
    let pipeline = pipeline(case);
    let bindings = GpuRuntimeBindingSet::new(pipeline.layout().clone(), []).unwrap();
    let vertex_binding =
        GpuVertexBufferBinding::new(0, &vertices, GpuBufferRange::whole(&vertices).unwrap())
            .unwrap();
    let draw = GpuRenderDraw::new(
        pipeline,
        bindings,
        [vertex_binding],
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
        target_view,
        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
        GpuAttachmentStore::Store,
        None,
    )
    .unwrap();
    let render = GpuRenderOperation::new([attachment], None, [draw], None).unwrap();
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
    let graph_name = format!("R1 packed vertex {}", case.name);
    let fragment = GpuWorkFragment::build(&graph_name, |builder| {
        builder.operation("draw packed vertex format", render)?;
        builder.operation("read packed vertex target", readback)?;
        Ok(())
    })
    .unwrap();
    (
        GpuPreparedWorkGraph::prepare(label(format!("{graph_name} graph")), [fragment]).unwrap(),
        readback_id,
    )
}

#[cfg(target_arch = "wasm32")]
struct YieldOnce(bool);

#[cfg(target_arch = "wasm32")]
impl std::future::Future for YieldOnce {
    type Output = ();

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        _context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.0 {
            std::task::Poll::Ready(())
        } else {
            self.0 = true;
            std::task::Poll::Pending
        }
    }
}

async fn progress_yield() {
    #[cfg(target_arch = "wasm32")]
    YieldOnce(false).await;
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::yield_now();
}

async fn wait_for_readback(
    context: &GpuContext,
    submission: &GpuSubmission,
    id: GpuReadbackId,
    case: PackedVertexCase,
) -> GpuReadbackBytes {
    const MAX_PROGRESS_TICKS: usize = 4_000;
    let readback = submission.readback(id).unwrap().clone();
    for _ in 0..MAX_PROGRESS_TICKS {
        context.progress();
        match readback.status() {
            GpuReadbackStatus::Ready(bytes)
                if matches!(submission.status(), GpuSubmissionStatus::Completed) =>
            {
                return bytes;
            }
            GpuReadbackStatus::Ready(_) | GpuReadbackStatus::Pending => {}
            GpuReadbackStatus::Failed(error) => {
                panic!("{} packed vertex readback failed: {error:?}", case.name)
            }
        }
        if let GpuSubmissionStatus::Failed(error) = submission.status() {
            panic!("{} packed vertex submission failed: {error:?}", case.name);
        }
        progress_yield().await;
    }
    panic!(
        "{} packed vertex proof exceeded its bounded progress budget",
        case.name
    )
}

fn pixel_at(bytes: &GpuReadbackBytes, x: u32, y: u32) -> [u8; 4] {
    let offset = usize::try_from((y * WIDTH + x) * 4).unwrap();
    bytes.as_bytes()[offset..offset + 4].try_into().unwrap()
}

async fn run_suite(context: &GpuContext) -> u32 {
    let mut mask = 0_u32;
    for (index, case) in CASES.into_iter().enumerate() {
        let (graph, readback_id) = graph(case);
        let prepared = context.prepare_submission(graph).await.unwrap();
        let submission = context.submit_prepared(prepared).unwrap();
        let bytes = wait_for_readback(context, &submission, readback_id, case).await;
        assert_eq!(bytes.texture_format(), Some(GpuTextureFormat::Rgba8Unorm));
        assert_eq!(pixel_at(&bytes, WIDTH / 2, HEIGHT / 2), DRAW_PIXEL);
        assert_eq!(pixel_at(&bytes, 0, 0), CLEAR_PIXEL);
        println!(
            "{:?} vertex: EXERCISED (non-symmetric packed decode + exact readback)",
            case.format
        );
        mask |= 1 << index;
    }
    assert_eq!(mask, (1_u32 << CASES.len()) - 1);
    mask
}

#[cfg(not(target_arch = "wasm32"))]
fn native_context() -> GpuContext {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .with_label("R1 packed vertex proof");
    let context = pollster::block_on(GpuContext::request(descriptor))
        .expect("native Conformance must provide the retained Vulkan fallback adapter");
    assert_eq!(context.adapter_facts().backend(), GpuBackendFamily::Vulkan);
    assert_eq!(
        context.adapter_facts().fallback(),
        GpuFallbackStatus::ConfirmedFallback
    );
    context
}

#[cfg(target_arch = "wasm32")]
pub(crate) async fn run_browser_vertex_packed() -> u32 {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
            .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
            .with_label("R1 browser packed vertex proof");
    let context = GpuContext::request(descriptor)
        .await
        .expect("actual-browser Conformance must provide WebGPU");
    assert_eq!(
        context.adapter_facts().backend(),
        GpuBackendFamily::BrowserWebGpu
    );
    run_suite(&context).await
}

#[test]
fn packed_vertex_case_census_matches_public_semantics() {
    assert_eq!(CASES.len(), 2);
    for case in CASES {
        assert_eq!(case.format.size_bytes(), 4, "{:?}", case.format);
        assert_eq!(
            case.format.attribute_alignment_bytes(),
            4,
            "{:?}",
            case.format
        );
        let value_type = case.format.shader_io_type();
        assert_eq!(value_type.scalar_class(), GpuShaderIoScalarClass::Float);
        assert_eq!(value_type.vector_width().get(), 4);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires the retained Vulkan software adapter"]
fn packed_vertex_native_pipeline_and_decode_are_backend_proven() {
    let context = native_context();
    let mask = pollster::block_on(run_suite(&context));
    assert_eq!(mask, 0b11);
}
