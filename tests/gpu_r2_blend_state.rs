use runen_gpu::*;

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;

#[derive(Clone, Copy)]
struct BlendCase {
    name: &'static str,
    blend: GpuBlendState,
    clear: [f64; 4],
    source: &'static str,
    blend_constant: [f64; 4],
    expected: [u8; 4],
}

const CUSTOM_BLEND: GpuBlendState = GpuBlendState::new(
    GpuBlendComponent::new(
        GpuBlendFactor::Constant,
        GpuBlendFactor::OneMinusConstant,
        GpuBlendOperation::Subtract,
    ),
    GpuBlendComponent::new(
        GpuBlendFactor::SrcAlpha,
        GpuBlendFactor::OneMinusSrcAlpha,
        GpuBlendOperation::Add,
    ),
);

const MIN_MAX_BLEND: GpuBlendState = GpuBlendState::new(
    GpuBlendComponent::new(
        GpuBlendFactor::Constant,
        GpuBlendFactor::OneMinusConstant,
        GpuBlendOperation::Min,
    ),
    GpuBlendComponent::new(
        GpuBlendFactor::OneMinusConstant,
        GpuBlendFactor::Constant,
        GpuBlendOperation::Max,
    ),
);

const CASES: [BlendCase; 2] = [
    BlendCase {
        name: "independent_subtract",
        blend: CUSTOM_BLEND,
        clear: [0.0, 1.0, 1.0, 0.0],
        source: "vec4<f32>(1.0, 0.0, 0.0, 1.0)",
        blend_constant: [1.0, 0.0, 1.0, 0.0],
        expected: [255, 0, 0, 255],
    },
    BlendCase {
        name: "min_max",
        blend: MIN_MAX_BLEND,
        clear: [0.0, 1.0, 0.0, 1.0],
        source: "vec4<f32>(1.0, 0.0, 1.0, 0.0)",
        blend_constant: [1.0, 0.0, 1.0, 0.0],
        expected: [0, 0, 0, 255],
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

fn shader_source(case: BlendCase) -> String {
    format!(
        r#"
struct VertexOutput {{
    @builtin(position) position: vec4<f32>,
}};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {{
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return output;
}}

@fragment
fn fs_main() -> @location(0) vec4<f32> {{
    return {};
}}
"#,
        case.source
    )
}

fn pipeline(case: BlendCase) -> GpuRenderPipelineDescriptor {
    let source_text = shader_source(case);
    let identity = GpuProgramSourceIdentity::new(
        GpuProgramSourceOwnerId::allocate().unwrap(),
        GpuProgramSourceKey::new(format!("r2.blend.{}", case.name)).unwrap(),
        GpuProgramSourceRevision::try_from_raw(1).unwrap(),
    );
    let mut sources = GpuProgramSourceRegistry::new(2, 16 * 1024).unwrap();
    let source = sources
        .admit_wgsl(
            identity,
            &source_text,
            GpuProgramSourceProvenance::new(format!("R2 blend {} proof", case.name), None).unwrap(),
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
    let target = GpuColorTargetStateDescriptor::new(
        GpuTextureFormat::Rgba8Unorm,
        Some(case.blend),
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

fn render_target(
    scope: &mut GpuResourceScope,
    case: BlendCase,
) -> (GpuTextureHandle, GpuTextureViewHandle) {
    let name = format!("{} blend target", case.name);
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

fn graph(case: BlendCase) -> (GpuPreparedWorkGraph, GpuReadbackId) {
    let mut scope = GpuResourceScope::new();
    let (target, target_view) = render_target(&mut scope, case);
    let pipeline = pipeline(case);
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
        GpuBlendConstant::new(
            case.blend_constant[0],
            case.blend_constant[1],
            case.blend_constant[2],
            case.blend_constant[3],
        )
        .unwrap(),
        0,
    )
    .unwrap();
    let attachment = GpuRenderColorAttachment::new(
        target_view,
        GpuColorAttachmentLoad::Clear(
            GpuColorClearValue::new(case.clear[0], case.clear[1], case.clear[2], case.clear[3])
                .unwrap(),
        ),
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
    let graph_name = format!("R2 blend {}", case.name);
    let fragment = GpuWorkFragment::build(&graph_name, |builder| {
        builder.operation("draw blend proof", render)?;
        builder.operation("read blend target", readback)?;
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
    case: BlendCase,
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
                panic!("{} blend readback failed: {error:?}", case.name)
            }
        }
        if let GpuSubmissionStatus::Failed(error) = submission.status() {
            panic!("{} blend submission failed: {error:?}", case.name);
        }
        progress_yield().await;
    }
    panic!(
        "{} blend proof exceeded its bounded progress budget",
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
        assert_eq!(
            pixel_at(&bytes, WIDTH / 2, HEIGHT / 2),
            case.expected,
            "{} exact blend result",
            case.name
        );
        println!(
            "{} blend: EXERCISED (independent normalized blend state + exact readback)",
            case.name
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
            .with_label("R2 blend proof");
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
pub(crate) async fn run_browser_blend_state() -> u32 {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
            .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
            .with_label("R2 browser blend proof");
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
fn blend_state_census_matches_portable_contract() {
    let factors = [
        GpuBlendFactor::Zero,
        GpuBlendFactor::One,
        GpuBlendFactor::Src,
        GpuBlendFactor::OneMinusSrc,
        GpuBlendFactor::SrcAlpha,
        GpuBlendFactor::OneMinusSrcAlpha,
        GpuBlendFactor::Dst,
        GpuBlendFactor::OneMinusDst,
        GpuBlendFactor::DstAlpha,
        GpuBlendFactor::OneMinusDstAlpha,
        GpuBlendFactor::SrcAlphaSaturated,
        GpuBlendFactor::Constant,
        GpuBlendFactor::OneMinusConstant,
    ];
    let operations = [
        GpuBlendOperation::Add,
        GpuBlendOperation::Subtract,
        GpuBlendOperation::ReverseSubtract,
        GpuBlendOperation::Min,
        GpuBlendOperation::Max,
    ];
    assert_eq!(factors.len(), 13);
    assert_eq!(operations.len(), 5);

    assert_eq!(
        CUSTOM_BLEND.color().operation(),
        GpuBlendOperation::Subtract
    );
    assert_eq!(CUSTOM_BLEND.alpha().operation(), GpuBlendOperation::Add);
    assert_eq!(MIN_MAX_BLEND.color().operation(), GpuBlendOperation::Min);
    assert_eq!(MIN_MAX_BLEND.alpha().operation(), GpuBlendOperation::Max);

    assert!(
        GpuColorTargetStateDescriptor::new(
            GpuTextureFormat::R32Uint,
            Some(CUSTOM_BLEND),
            GpuColorWriteMask::ALL,
        )
        .is_err()
    );
    assert!(
        GpuColorTargetStateDescriptor::new(
            GpuTextureFormat::Rgba8Unorm,
            Some(CUSTOM_BLEND),
            GpuColorWriteMask::ALL,
        )
        .is_ok()
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires the retained Vulkan software adapter"]
fn portable_blend_state_native_execution_is_backend_proven() {
    let context = native_context();
    let mask = pollster::block_on(run_suite(&context));
    assert_eq!(mask, 0b11);
}
