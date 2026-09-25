use runen_gpu::*;

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const RED: [u8; 4] = [255, 0, 0, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];
const BASELINE_MASK: u32 = 0b0111;
const CLAMP_BIT: u32 = 1 << 3;
#[cfg(target_arch = "wasm32")]
const CLAMP_SUPPORTED_BIT: u32 = 1 << 8;

#[derive(Clone, Copy)]
struct BiasCase {
    name: &'static str,
    depths: [f32; 3],
    bias: GpuDepthBiasState,
    expected: [u8; 4],
    requires_clamp: bool,
}

fn baseline_cases() -> [BiasCase; 3] {
    [
        BiasCase {
            name: "neutral",
            depths: [0.49; 3],
            bias: GpuDepthBiasState::default(),
            expected: GREEN,
            requires_clamp: false,
        },
        BiasCase {
            name: "constant",
            depths: [0.49; 3],
            bias: GpuDepthBiasState::new(4096, 0.0, 0.0).unwrap(),
            expected: RED,
            requires_clamp: false,
        },
        BiasCase {
            name: "slope_scale",
            depths: [0.43, 0.51, 0.43],
            bias: GpuDepthBiasState::new(0, 16.0, 0.0).unwrap(),
            expected: RED,
            requires_clamp: false,
        },
    ]
}

fn clamp_case() -> BiasCase {
    BiasCase {
        name: "clamp",
        depths: [0.49; 3],
        bias: GpuDepthBiasState::new(8192, 0.0, 0.005).unwrap(),
        expected: GREEN,
        requires_clamp: true,
    }
}

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

fn shader_source(key: &str, depths: [f32; 3], color: [f32; 4]) -> String {
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
    var depths = array<f32, 3>({:.9}, {:.9}, {:.9});
    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_index], depths[vertex_index], 1.0);
    return output;
}}

@fragment
fn fs_main() -> @location(0) vec4<f32> {{
    return vec4<f32>({:.9}, {:.9}, {:.9}, {:.9});
}}
// {}
"#,
        depths[0], depths[1], depths[2], color[0], color[1], color[2], color[3], key,
    )
}

fn pipeline(
    key: &str,
    depths: [f32; 3],
    color: [f32; 4],
    compare: GpuCompareFunction,
    bias: GpuDepthBiasState,
) -> GpuRenderPipelineDescriptor {
    let source_text = shader_source(key, depths, color);
    let identity = GpuProgramSourceIdentity::new(
        GpuProgramSourceOwnerId::allocate().unwrap(),
        GpuProgramSourceKey::new(format!("r2.depth_bias.{key}")).unwrap(),
        GpuProgramSourceRevision::try_from_raw(1).unwrap(),
    );
    let mut sources = GpuProgramSourceRegistry::new(2, 16 * 1024).unwrap();
    let source = sources
        .admit_wgsl(
            identity,
            &source_text,
            GpuProgramSourceProvenance::new(format!("R2 depth bias {key} proof"), None).unwrap(),
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
    let color_target = GpuColorTargetStateDescriptor::new(
        GpuTextureFormat::Rgba8Unorm,
        None,
        GpuColorWriteMask::ALL,
    )
    .unwrap();
    let depth_stencil = GpuDepthStencilStateDescriptor::new(
        GpuTextureFormat::Depth16Unorm,
        Some(GpuDepthStateDescriptor::new(true, compare)),
        None,
        bias,
    )
    .unwrap();
    let state = GpuRenderPipelineStateDescriptor::new(
        GpuVertexInputStateDescriptor::new([]).unwrap(),
        Some(GpuFragmentOutputStateDescriptor::new([color_target])),
        GpuPrimitiveStateDescriptor::default(),
        Some(depth_stencil),
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

fn color_target(
    scope: &mut GpuResourceScope,
    name: &str,
) -> (GpuTextureHandle, GpuTextureViewHandle) {
    let resource_label = label(name);
    let texture = scope
        .texture(
            GpuTextureDescriptor::new(
                common(name),
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

fn depth_target(scope: &mut GpuResourceScope, name: &str) -> GpuTextureViewHandle {
    let resource_label = label(name);
    let texture = scope
        .texture(
            GpuTextureDescriptor::new(
                common(name),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, WIDTH, HEIGHT, 1)
                    .unwrap(),
                1,
                1,
                GpuTextureFormat::Depth16Unorm,
                GpuTextureUsages::new(&resource_label, [GpuTextureUsage::DepthStencilAttachment])
                    .unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    scope
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
        .unwrap()
}

fn draw(pipeline: GpuRenderPipelineDescriptor) -> GpuRenderDraw {
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
        GpuViewport::new(0.0, 0.0, WIDTH as f32, HEIGHT as f32, 0.0, 1.0).unwrap(),
        GpuScissorRect::new(0, 0, WIDTH, HEIGHT).unwrap(),
        GpuBlendConstant::new(0.0, 0.0, 0.0, 0.0).unwrap(),
        0,
    )
    .unwrap()
}

fn seed_render(color: GpuTextureViewHandle, depth: GpuTextureViewHandle) -> GpuRenderOperation {
    let color_attachment = GpuRenderColorAttachment::new(
        color,
        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
        GpuAttachmentStore::Store,
        None,
    )
    .unwrap();
    let depth_attachment = GpuRenderDepthStencilAttachment::new(
        depth,
        Some(
            GpuDepthAttachmentState::new(
                GpuDepthStencilAccess::ReadWrite,
                GpuDepthAttachmentLoad::Clear(GpuDepthClearValue::new(1.0).unwrap()),
                GpuAttachmentStore::Store,
            )
            .unwrap(),
        ),
        None,
    )
    .unwrap();
    let pipeline = pipeline(
        "seed",
        [0.5; 3],
        [1.0, 0.0, 0.0, 1.0],
        GpuCompareFunction::Always,
        GpuDepthBiasState::default(),
    );
    GpuRenderOperation::new(
        [color_attachment],
        Some(depth_attachment),
        [draw(pipeline)],
        None,
    )
    .unwrap()
}

fn biased_render(
    color: GpuTextureViewHandle,
    depth: GpuTextureViewHandle,
    case: BiasCase,
) -> GpuRenderOperation {
    let color_attachment = GpuRenderColorAttachment::new(
        color,
        GpuColorAttachmentLoad::Load,
        GpuAttachmentStore::Store,
        None,
    )
    .unwrap();
    let depth_attachment = GpuRenderDepthStencilAttachment::new(
        depth,
        Some(
            GpuDepthAttachmentState::new(
                GpuDepthStencilAccess::ReadWrite,
                GpuDepthAttachmentLoad::Load,
                GpuAttachmentStore::Store,
            )
            .unwrap(),
        ),
        None,
    )
    .unwrap();
    let pipeline = pipeline(
        case.name,
        case.depths,
        [0.0, 1.0, 0.0, 1.0],
        GpuCompareFunction::Less,
        case.bias,
    );
    if case.requires_clamp {
        assert!(matches!(
            pipeline
                .requirements()
                .get(GpuCapabilityFeature::DepthBiasClamp),
            Some(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::DepthBiasClamp
            ))
        ));
    } else {
        assert!(
            pipeline
                .requirements()
                .get(GpuCapabilityFeature::DepthBiasClamp)
                .is_none()
        );
    }
    GpuRenderOperation::new(
        [color_attachment],
        Some(depth_attachment),
        [draw(pipeline)],
        None,
    )
    .unwrap()
}

fn graph(case: BiasCase) -> (GpuPreparedWorkGraph, GpuReadbackId) {
    let mut scope = GpuResourceScope::new();
    let (color, color_view) = color_target(&mut scope, &format!("{} depth-bias color", case.name));
    let depth_view = depth_target(&mut scope, &format!("{} depth-bias depth", case.name));
    let seed = seed_render(color_view.clone(), depth_view.clone());
    let biased = biased_render(color_view, depth_view, case);
    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(
        GpuTextureCopyRegion::new(
            &color,
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
    let name = format!("R2 depth bias {}", case.name);
    let fragment = GpuWorkFragment::build(&name, |builder| {
        builder.operation("seed depth bias target", seed)?;
        builder.operation("draw biased triangle", biased)?;
        builder.operation("read depth bias target", readback)?;
        Ok(())
    })
    .unwrap();
    (
        GpuPreparedWorkGraph::prepare(label(format!("{name} graph")), [fragment]).unwrap(),
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
    case: BiasCase,
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
                panic!("{} depth-bias readback failed: {error:?}", case.name)
            }
        }
        if let GpuSubmissionStatus::Failed(error) = submission.status() {
            panic!("{} depth-bias submission failed: {error:?}", case.name);
        }
        progress_yield().await;
    }
    panic!(
        "{} depth-bias proof exceeded its progress budget",
        case.name
    )
}

fn pixel_at(bytes: &GpuReadbackBytes, x: u32, y: u32) -> [u8; 4] {
    let offset = usize::try_from((y * WIDTH + x) * 4).unwrap();
    bytes.as_bytes()[offset..offset + 4].try_into().unwrap()
}

async fn run_case(context: &GpuContext, case: BiasCase) {
    let (graph, readback_id) = graph(case);
    let prepared = context.prepare_submission(graph).await.unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let bytes = wait_for_readback(context, &submission, readback_id, case).await;
    assert_eq!(bytes.texture_format(), Some(GpuTextureFormat::Rgba8Unorm));
    assert_eq!(
        pixel_at(&bytes, WIDTH / 2, HEIGHT / 2),
        case.expected,
        "{} exact depth-bias outcome",
        case.name
    );
    println!("{} depth bias: EXERCISED (exact color readback)", case.name);
}

async fn run_baseline(context: &GpuContext) -> u32 {
    let mut mask = 0_u32;
    for (index, case) in baseline_cases().into_iter().enumerate() {
        run_case(context, case).await;
        mask |= 1 << index;
    }
    assert_eq!(mask, BASELINE_MASK);
    mask
}

fn descriptor(
    backend: GpuBackendFamily,
    require_clamp: bool,
    fallback: Option<GpuSoftwareFallbackPolicy>,
) -> GpuContextDescriptor {
    let mut requirements = GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::DepthAttachment,
        ))
        .unwrap();
    if require_clamp {
        requirements
            .insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::DepthBiasClamp,
            ))
            .unwrap();
    }
    let mut descriptor = GpuContextDescriptor::new(requirements)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
        .require_format_role(GpuTextureFormat::Depth16Unorm, GpuFormatRole::DepthStencil)
        .with_allowed_backends([backend])
        .with_label(if require_clamp {
            "R2 depth bias clamp proof"
        } else {
            "R2 depth bias baseline proof"
        });
    if let Some(fallback) = fallback {
        descriptor = descriptor.with_fallback_policy(fallback);
    }
    descriptor
}

#[cfg(not(target_arch = "wasm32"))]
async fn native_context(require_clamp: bool) -> GpuContext {
    let context = GpuContext::request(descriptor(
        GpuBackendFamily::Vulkan,
        require_clamp,
        Some(GpuSoftwareFallbackPolicy::Require),
    ))
    .await
    .expect("native Conformance must provide retained Vulkan depth-bias support");
    assert_eq!(context.adapter_facts().backend(), GpuBackendFamily::Vulkan);
    assert_eq!(
        context.adapter_facts().fallback(),
        GpuFallbackStatus::ConfirmedFallback
    );
    context
}

#[cfg(target_arch = "wasm32")]
async fn browser_context(require_clamp: bool) -> Result<GpuContext, GpuContextRequestError> {
    GpuContext::request(descriptor(
        GpuBackendFamily::BrowserWebGpu,
        require_clamp,
        None,
    ))
    .await
}

#[cfg(target_arch = "wasm32")]
pub(crate) async fn run_browser_depth_bias() -> u32 {
    let context = browser_context(false)
        .await
        .expect("actual-browser Conformance must provide baseline WebGPU depth bias");
    assert_eq!(
        context.adapter_facts().backend(),
        GpuBackendFamily::BrowserWebGpu
    );
    let mut mask = run_baseline(&context).await;
    if context
        .adapter_facts()
        .supported()
        .supports(GpuCapabilityFeature::DepthBiasClamp)
    {
        let clamp_context = browser_context(true)
            .await
            .expect("advertised browser depth-bias clamp capability must admit a context");
        run_case(&clamp_context, clamp_case()).await;
        mask |= CLAMP_BIT | CLAMP_SUPPORTED_BIT;
    } else {
        println!("clamp depth bias: UNSUPPORTED (normalized capability absent)");
    }
    mask
}

#[test]
fn depth_bias_contract_cases_are_distinguishing() {
    assert_eq!(baseline_cases().len(), 3);
    for case in baseline_cases() {
        assert!(!case.requires_clamp);
    }
    let clamp = clamp_case();
    assert!(clamp.requires_clamp);
    assert_ne!(clamp.bias.clamp(), 0.0);
    assert_eq!(
        GpuDepthBiasState::new(1, -0.0, -0.0).unwrap().slope_scale(),
        0.0
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires the retained Vulkan software adapter"]
fn depth_bias_native_execution_is_backend_proven() {
    let context = pollster::block_on(native_context(false));
    let mut mask = pollster::block_on(run_baseline(&context));
    assert!(
        context
            .adapter_facts()
            .supported()
            .supports(GpuCapabilityFeature::DepthBiasClamp),
        "retained Lavapipe must positively qualify depth-bias clamp"
    );
    let clamp_context = pollster::block_on(native_context(true));
    pollster::block_on(run_case(&clamp_context, clamp_case()));
    mask |= CLAMP_BIT;
    assert_eq!(mask, 0b1111);
}
