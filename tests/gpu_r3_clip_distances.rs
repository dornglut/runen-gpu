use runen_gpu::*;

#[path = "support/readback_wait.rs"]
mod readback_wait;

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const KEPT_PIXEL: [u8; 4] = [255, 255, 255, 255];
const CLIPPED_PIXEL: [u8; 4] = [0, 0, 0, 255];

const WGSL: &str = r#"
enable clip_distances;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @builtin(clip_distances) distances: array<f32, 1>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(0.0, -1.0),
        vec2<f32>(-0.5, 1.0),
        vec2<f32>(0.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(0.5, 1.0),
    );
    let distance = select(1.0, -1.0, vertex_index >= 3u);
    return VertexOutput(
        vec4<f32>(positions[vertex_index], 0.0, 1.0),
        array<f32, 1>(distance),
    );
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ClipDistancesProofOutcome {
    pub supported: bool,
    pub exercised: bool,
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

fn descriptor(
    backend: GpuBackendFamily,
    require_clip_distances: bool,
    fallback: Option<GpuSoftwareFallbackPolicy>,
) -> GpuContextDescriptor {
    let mut requirements = GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements();
    if require_clip_distances {
        requirements
            .insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::ClipDistances,
            ))
            .unwrap();
    }
    let mut descriptor = GpuContextDescriptor::new(requirements)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
        .with_allowed_backends([backend])
        .with_label(if require_clip_distances {
            "R3 clip-distances proof"
        } else {
            "R3 clip-distances census"
        });
    if let Some(fallback) = fallback {
        descriptor = descriptor.with_fallback_policy(fallback);
    }
    descriptor
}

fn pipeline() -> GpuRenderPipelineDescriptor {
    let [source] = admit_static_wgsl_sources([("proof.r3.clip-distances", 1, WGSL)]).unwrap();
    let vertex = GpuEntryPointName::new("vs_main").unwrap();
    let fragment = GpuEntryPointName::new("fs_main").unwrap();
    let program = GpuProgramDescriptor::new(
        source,
        [vertex.clone(), fragment.clone()],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();
    assert_eq!(
        program
            .requirements()
            .get(GpuCapabilityFeature::ClipDistances),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::ClipDistances
        ))
    );

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
    let target_label = label("R3 clip-distances target");
    let target = scope
        .texture(
            GpuTextureDescriptor::new(
                common("R3 clip-distances target"),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&target_label, GpuTextureDimension::D2, WIDTH, HEIGHT, 1)
                    .unwrap(),
                1,
                1,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(
                    &target_label,
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
                common("R3 clip-distances target view"),
                &target,
                None,
                GpuTextureViewDimension::D2,
                GpuTextureSubresourceRange::whole(&target).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();

    let pipeline = pipeline();
    let bindings = GpuRuntimeBindingSet::new(pipeline.layout().clone(), []).unwrap();
    let draw = GpuRenderDraw::new(
        pipeline,
        bindings,
        [],
        None,
        GpuDrawIntent::direct(
            GpuDrawRange::new(0, 6).unwrap(),
            GpuDrawRange::new(0, 1).unwrap(),
        ),
        GpuViewport::new(0.0, 0.0, WIDTH as f32, HEIGHT as f32, 0.0, 1.0).unwrap(),
        GpuScissorRect::new(0, 0, WIDTH, HEIGHT).unwrap(),
        GpuBlendConstant::new(0.0, 0.0, 0.0, 0.0).unwrap(),
        0,
    )
    .unwrap();

    let attachment = GpuRenderColorAttachment::new(
        view,
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

    let fragment = GpuWorkFragment::build("R3 clip-distances proof", |builder| {
        builder.operation("draw clip-distances proof", render)?;
        builder.operation("read clip-distances target", readback)?;
        Ok(())
    })
    .unwrap();
    (
        GpuPreparedWorkGraph::prepare(label("R3 clip-distances proof graph"), [fragment]).unwrap(),
        readback_id,
    )
}

fn pixel_at(bytes: &GpuReadbackBytes, x: u32, y: u32) -> [u8; 4] {
    let offset = usize::try_from((y * WIDTH + x) * 4).unwrap();
    bytes.as_bytes()[offset..offset + 4].try_into().unwrap()
}

pub(crate) async fn run_case(context: &GpuContext) {
    let (graph, readback_id) = graph();
    assert_eq!(
        graph
            .requirements()
            .get(GpuCapabilityFeature::ClipDistances),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::ClipDistances
        ))
    );
    let prepared = context.prepare_submission(graph).await.unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let bytes = readback_wait::wait_for_readback(
        context,
        &submission,
        readback_id,
        "R3 clip-distances proof",
    )
    .await;
    assert_eq!(bytes.texture_format(), Some(GpuTextureFormat::Rgba8Unorm));
    assert_eq!(pixel_at(&bytes, 2, 4), KEPT_PIXEL);
    assert_eq!(pixel_at(&bytes, 6, 4), CLIPPED_PIXEL);
}

pub(crate) async fn run_on_adapter(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
    census: &GpuContext,
) -> ClipDistancesProofOutcome {
    let supported = census
        .adapter_facts()
        .supported()
        .supports(GpuCapabilityFeature::ClipDistances);

    if !supported {
        let rejected = GpuContext::request(descriptor(backend, true, fallback))
            .await
            .expect_err("unsupported clip distances must reject the feature context");
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
                            && error.feature() == Some(GpuCapabilityFeature::ClipDistances)
                    })
            )
        }));
        return ClipDistancesProofOutcome {
            supported: false,
            exercised: false,
        };
    }

    assert!(
        !census
            .device_facts()
            .is_enabled(GpuCapabilityFeature::ClipDistances),
        "census context must not enable ClipDistances implicitly"
    );
    let (candidate, _) = graph();
    let missing_enablement = census
        .prepare_submission(candidate)
        .await
        .expect_err("supported but non-enabled clip-distances work must reject before realization");
    assert_eq!(
        missing_enablement.kind(),
        GpuSubmissionPreparationErrorKind::CapabilityNotAdmitted
    );

    let context = GpuContext::request(descriptor(backend, true, fallback))
        .await
        .expect("advertised clip distances must admit a feature context");
    assert_eq!(
        context.adapter_facts(),
        census.adapter_facts(),
        "clip-distances proof must remain on the exact census adapter"
    );
    assert!(
        context
            .device_facts()
            .is_enabled(GpuCapabilityFeature::ClipDistances)
    );
    run_case(&context).await;

    ClipDistancesProofOutcome {
        supported: true,
        exercised: true,
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) async fn run_browser_clip_distances(census: &GpuContext) {
    let outcome = run_on_adapter(GpuBackendFamily::BrowserWebGpu, None, census).await;
    assert_eq!(outcome.exercised, outcome.supported);
    if outcome.supported {
        println!("ClipDistances BrowserWebGpu: EXERCISED");
    } else {
        println!("ClipDistances BrowserWebGpu: UNSUPPORTED");
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires retained Vulkan adapter"]
fn clip_distances_native_execution_matches_advertised_capability() {
    let census = pollster::block_on(GpuContext::request(descriptor(
        GpuBackendFamily::Vulkan,
        false,
        Some(GpuSoftwareFallbackPolicy::Require),
    )))
    .expect("retained Vulkan fallback must provide baseline offscreen rendering");
    let outcome = pollster::block_on(run_on_adapter(
        GpuBackendFamily::Vulkan,
        Some(GpuSoftwareFallbackPolicy::Require),
        &census,
    ));
    assert_eq!(outcome.exercised, outcome.supported);
}
