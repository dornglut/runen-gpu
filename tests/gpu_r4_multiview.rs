use runen_gpu::*;

#[path = "support/readback_wait.rs"]
mod readback_wait;

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const SENTINEL_PIXEL: [u8; 4] = [0, 0, 255, 255];
const VIEW_ZERO_PIXEL: [u8; 4] = [255, 0, 0, 255];
const VIEW_ONE_PIXEL: [u8; 4] = [0, 255, 0, 255];

const VIEW_INDEX_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return output;
}

@fragment
fn fs_main(@builtin(view_index) view_index: u32) -> @location(0) vec4<f32> {
    return select(
        vec4<f32>(1.0, 0.0, 0.0, 1.0),
        vec4<f32>(0.0, 1.0, 0.0, 1.0),
        view_index == 1u,
    );
}
"#;

const NO_VIEW_INDEX_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return output;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 1.0, 1.0, 1.0);
}
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MultiviewProofOutcome {
    pub supported: bool,
    pub normalized_max: u32,
    pub primary_exercised: bool,
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

fn multiview_pipeline(
    source_key: &'static str,
    source_text: &'static str,
) -> GpuRenderPipelineDescriptor {
    let [source] = admit_static_wgsl_sources([(source_key, 1, source_text)]).unwrap();
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
    .unwrap()
    .with_multiview(GpuMultiviewState::new(2).unwrap())
    .unwrap();
    GpuRenderPipelineDescriptor::new(
        program,
        GpuRenderEntryPoints::new(vertex, Some(fragment)),
        state,
        GpuPipelineConfiguration::default(),
    )
    .unwrap()
}

fn texture_view(
    scope: &mut GpuResourceScope,
    texture: &GpuTextureHandle,
    name: &str,
    dimension: GpuTextureViewDimension,
    base_layer: u32,
    layer_count: u32,
) -> GpuTextureViewHandle {
    let range = GpuTextureSubresourceRange::new(
        texture.descriptor().common().label(),
        0,
        1,
        base_layer,
        layer_count,
        GpuTextureAspect::Color,
    )
    .unwrap();
    scope
        .texture_view(
            GpuTextureViewDescriptor::new(common(name), texture, None, dimension, range).unwrap(),
        )
        .unwrap()
}

fn render_draw(pipeline: GpuRenderPipelineDescriptor) -> GpuRenderDraw {
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

fn primary_graph() -> (GpuPreparedWorkGraph, [GpuReadbackId; 3]) {
    let mut scope = GpuResourceScope::new();
    let texture_label = label("R4 multiview parent");
    let texture = scope
        .texture(
            GpuTextureDescriptor::new(
                common("R4 multiview parent"),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&texture_label, GpuTextureDimension::D2, WIDTH, HEIGHT, 3)
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

    let sentinel_view = texture_view(
        &mut scope,
        &texture,
        "R4 multiview sentinel layer",
        GpuTextureViewDimension::D2,
        0,
        1,
    );
    let multiview_view = texture_view(
        &mut scope,
        &texture,
        "R4 multiview layers 1 2",
        GpuTextureViewDimension::D2Array,
        1,
        2,
    );

    let sentinel = GpuRenderOperation::new(
        [GpuRenderColorAttachment::new(
            sentinel_view,
            GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 1.0, 1.0).unwrap()),
            GpuAttachmentStore::Store,
            None,
        )
        .unwrap()],
        None,
        std::iter::empty::<GpuRenderDraw>(),
        None,
    )
    .unwrap();

    let pipeline = multiview_pipeline("proof.r4.multiview.view-index", VIEW_INDEX_WGSL);
    assert!(matches!(
        pipeline.requirements().get(GpuCapabilityFeature::Multiview),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Multiview
        ))
    ));
    let render = GpuRenderOperation::new(
        [GpuRenderColorAttachment::new(
            multiview_view,
            GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
            GpuAttachmentStore::Store,
            None,
        )
        .unwrap()],
        None,
        [render_draw(pipeline)],
        None,
    )
    .unwrap();
    assert_eq!(
        render.signature().multiview(),
        Some(GpuMultiviewState::new(2).unwrap())
    );

    let ids = [
        GpuReadbackId::allocate().unwrap(),
        GpuReadbackId::allocate().unwrap(),
        GpuReadbackId::allocate().unwrap(),
    ];
    let readbacks = ids.into_iter().enumerate().map(|(layer, id)| {
        let region = GpuTextureCopyRegion::new(
            &texture,
            0,
            GpuTextureOrigin::new(0, 0, u32::try_from(layer).unwrap()),
            GpuTextureAspect::Color,
            GpuCopyExtent::new(WIDTH, HEIGHT, 1).unwrap(),
        )
        .unwrap();
        GpuReadbackOperation::new(region.into(), id).unwrap()
    });

    let fragment = GpuWorkFragment::build("R4 multiview primary oracle", |builder| {
        builder.operation("initialize disjoint parent layer zero", sentinel)?;
        builder.operation("render contiguous views into parent layers one two", render)?;
        for (layer, readback) in readbacks.enumerate() {
            builder.operation(format!("read parent layer {layer}"), readback)?;
        }
        Ok(())
    })
    .unwrap();
    let graph =
        GpuPreparedWorkGraph::prepare(label("R4 multiview primary oracle graph"), [fragment])
            .unwrap();
    assert!(matches!(
        graph.requirements().get(GpuCapabilityFeature::Multiview),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Multiview
        ))
    ));
    (graph, ids)
}

fn pixel_at(bytes: &GpuReadbackBytes, x: u32, y: u32) -> [u8; 4] {
    let offset = usize::try_from((y * WIDTH + x) * 4).unwrap();
    bytes.as_bytes()[offset..offset + 4].try_into().unwrap()
}

async fn run_primary_oracle(context: &GpuContext) {
    let (graph, ids) = primary_graph();
    let prepared = context.prepare_submission(graph).await.unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let mut outputs = Vec::new();
    for (layer, id) in ids.into_iter().enumerate() {
        outputs.push(
            readback_wait::wait_for_readback(
                context,
                &submission,
                id,
                format!("R4 multiview parent layer {layer}"),
            )
            .await,
        );
    }
    assert_eq!(pixel_at(&outputs[0], WIDTH / 2, HEIGHT / 2), SENTINEL_PIXEL);
    assert_eq!(
        pixel_at(&outputs[1], WIDTH / 2, HEIGHT / 2),
        VIEW_ZERO_PIXEL
    );
    assert_eq!(pixel_at(&outputs[2], WIDTH / 2, HEIGHT / 2), VIEW_ONE_PIXEL);
}

fn descriptor(
    backend: GpuBackendFamily,
    require_multiview: bool,
    fallback: Option<GpuSoftwareFallbackPolicy>,
) -> GpuContextDescriptor {
    let mut requirements = GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements();
    if require_multiview {
        requirements
            .insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::Multiview,
            ))
            .unwrap();
    }
    let mut descriptor = GpuContextDescriptor::new(requirements)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
        .with_allowed_backends([backend])
        .with_label(if require_multiview {
            "R4 multiview retained proof"
        } else {
            "R4 multiview census"
        });
    if require_multiview {
        descriptor = descriptor.require_limit(GpuLimitKind::MaxMultiviewViewCount, 2);
    }
    if let Some(fallback) = fallback {
        descriptor = descriptor.with_fallback_policy(fallback);
    }
    descriptor
}

pub(crate) async fn run_on_adapter(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
    census: &GpuContext,
) -> MultiviewProofOutcome {
    let supported = census
        .adapter_facts()
        .supported()
        .supports(GpuCapabilityFeature::Multiview);
    let normalized_max = census.adapter_facts().limits().max_multiview_view_count();

    if !supported {
        assert_eq!(
            normalized_max, 0,
            "unsupported normalized Multiview must expose max view count zero"
        );
        let rejected = GpuContext::request(descriptor(backend, true, fallback))
            .await
            .expect_err("unsupported Multiview must reject a required context");
        assert_eq!(
            rejected.category(),
            GpuContextRequestErrorCategory::NoAdmissibleCandidate
        );
        assert!(rejected.candidate_dispositions().iter().any(|disposition| {
            matches!(
                disposition,
                GpuCandidateDisposition::Rejected(report)
                    if report.capability_admission_error().is_some_and(|error| {
                        error.cause() == GpuCapabilityAdmissionCause::RequiredUnavailable
                            && error.feature() == Some(GpuCapabilityFeature::Multiview)
                    })
            )
        }));
        return MultiviewProofOutcome {
            supported: false,
            normalized_max,
            primary_exercised: false,
        };
    }

    assert!((2..=31).contains(&normalized_max));
    assert!(
        !census
            .device_facts()
            .is_enabled(GpuCapabilityFeature::Multiview),
        "census context must not enable optional Multiview implicitly"
    );

    let (graph, _) = primary_graph();
    let missing_enablement = census.prepare_submission(graph).await.expect_err(
        "multiview work must reject when support exists but the feature is not enabled",
    );
    assert_eq!(
        missing_enablement.kind(),
        GpuSubmissionPreparationErrorKind::CapabilityNotAdmitted
    );

    let context = GpuContext::request(descriptor(backend, true, fallback))
        .await
        .expect("advertised normalized Multiview must admit the retained proof context");
    assert_eq!(context.adapter_facts(), census.adapter_facts());
    assert!(
        context
            .device_facts()
            .is_enabled(GpuCapabilityFeature::Multiview)
    );
    assert_eq!(
        context
            .device_facts()
            .workload_budget()
            .limits()
            .max_multiview_view_count(),
        2,
        "default required Multiview workload budget must be exactly two"
    );

    run_primary_oracle(&context).await;
    MultiviewProofOutcome {
        supported: true,
        normalized_max,
        primary_exercised: true,
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) async fn prove_browser_webgpu_unsupported() {
    let census = GpuContext::request(descriptor(GpuBackendFamily::BrowserWebGpu, false, None))
        .await
        .expect("actual-browser Conformance must provide baseline WebGPU rendering");
    let outcome = run_on_adapter(GpuBackendFamily::BrowserWebGpu, None, &census).await;
    assert!(
        !outcome.supported,
        "pinned BrowserWebGpu must remain normalized unsupported for the first Multiview contract"
    );
    assert_eq!(outcome.normalized_max, 0);
}

#[test]
fn selected_view_index_requires_multiview_but_multiview_shader_may_ignore_it() {
    let [view_source, ordinary_source] = admit_static_wgsl_sources([
        ("proof.r4.multiview.stage-io.view-index", 1, VIEW_INDEX_WGSL),
        (
            "proof.r4.multiview.stage-io.ordinary",
            1,
            NO_VIEW_INDEX_WGSL,
        ),
    ])
    .unwrap();
    let vertex = GpuEntryPointName::new("vs_main").unwrap();
    let fragment = GpuEntryPointName::new("fs_main").unwrap();

    let view_program = GpuProgramDescriptor::new(
        view_source,
        [vertex.clone(), fragment.clone()],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();
    assert!(matches!(
        view_program
            .requirements()
            .get(GpuCapabilityFeature::Multiview),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Multiview
        ))
    ));
    assert!(
        view_program
            .entry_point(GpuShaderStage::Fragment, &fragment)
            .unwrap()
            .uses_view_index()
    );

    let target = GpuColorTargetStateDescriptor::new(
        GpuTextureFormat::Rgba8Unorm,
        None,
        GpuColorWriteMask::ALL,
    )
    .unwrap();
    let ordinary_state = GpuRenderPipelineStateDescriptor::new(
        GpuVertexInputStateDescriptor::new([]).unwrap(),
        Some(GpuFragmentOutputStateDescriptor::new([target])),
        GpuPrimitiveStateDescriptor::default(),
        None,
        GpuMultisampleStateDescriptor::default(),
    )
    .unwrap();
    let error = GpuRenderPipelineDescriptor::new(
        view_program,
        GpuRenderEntryPoints::new(vertex.clone(), Some(fragment.clone())),
        ordinary_state,
        GpuPipelineConfiguration::default(),
    )
    .expect_err("selected view_index must reject an ordinary render pipeline");
    assert_eq!(
        error.cause(),
        GpuProgramContractCause::PipelineStageIoMismatch
    );

    let ordinary_program = GpuProgramDescriptor::new(
        ordinary_source,
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
    let multiview_state = GpuRenderPipelineStateDescriptor::new(
        GpuVertexInputStateDescriptor::new([]).unwrap(),
        Some(GpuFragmentOutputStateDescriptor::new([target])),
        GpuPrimitiveStateDescriptor::default(),
        None,
        GpuMultisampleStateDescriptor::default(),
    )
    .unwrap()
    .with_multiview(GpuMultiviewState::new(2).unwrap())
    .unwrap();
    let pipeline = GpuRenderPipelineDescriptor::new(
        ordinary_program,
        GpuRenderEntryPoints::new(vertex, Some(fragment)),
        multiview_state,
        GpuPipelineConfiguration::default(),
    )
    .expect("multiview rendering must not require the selected shader to consume view_index");
    assert!(matches!(
        pipeline.requirements().get(GpuCapabilityFeature::Multiview),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Multiview
        ))
    ));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires the retained Vulkan software adapter"]
fn multiview_native_execution_matches_normalized_adapter_facts() {
    let census = pollster::block_on(GpuContext::request(descriptor(
        GpuBackendFamily::Vulkan,
        false,
        Some(GpuSoftwareFallbackPolicy::Require),
    )))
    .expect("native Conformance must provide the retained Vulkan fallback adapter");
    let outcome = pollster::block_on(run_on_adapter(
        GpuBackendFamily::Vulkan,
        Some(GpuSoftwareFallbackPolicy::Require),
        &census,
    ));
    if outcome.supported {
        assert!(outcome.primary_exercised);
        println!(
            "Multiview Vulkan: EXERCISED (normalized max={})",
            outcome.normalized_max
        );
    } else {
        println!("Multiview Vulkan: UNSUPPORTED");
    }
}
