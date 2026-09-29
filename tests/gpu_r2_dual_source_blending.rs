use runen_gpu::*;

#[path = "support/readback_wait.rs"]
mod readback_wait;

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const EXPECTED_PIXEL: [u8; 4] = [0, 255, 0, 0];

const WGSL: &str = r#"
enable dual_source_blending;

struct DualSourceOutput {
    @location(0) @blend_src(0) primary: vec4<f32>,
    @location(0) @blend_src(1) secondary: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(positions[vertex_index], 0.0, 1.0);
}

@fragment
fn fs_main() -> DualSourceOutput {
    var output: DualSourceOutput;
    output.primary = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    output.secondary = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    return output;
}
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DualSourceBlendProofOutcome {
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
    require_dual_source: bool,
    fallback: Option<GpuSoftwareFallbackPolicy>,
) -> GpuContextDescriptor {
    let mut requirements = GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements();
    if require_dual_source {
        requirements
            .insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::DualSourceBlending,
            ))
            .unwrap();
    }
    let mut descriptor = GpuContextDescriptor::new(requirements)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
        .with_allowed_backends([backend])
        .with_label(if require_dual_source {
            "R2 dual-source blending proof"
        } else {
            "R2 dual-source blending census"
        });
    if let Some(fallback) = fallback {
        descriptor = descriptor.with_fallback_policy(fallback);
    }
    descriptor
}

fn blend_state() -> GpuBlendState {
    GpuBlendState::new(
        GpuBlendComponent::new(
            GpuBlendFactor::Zero,
            GpuBlendFactor::Src1,
            GpuBlendOperation::Add,
        )
        .unwrap(),
        GpuBlendComponent::new(
            GpuBlendFactor::Zero,
            GpuBlendFactor::Src1Alpha,
            GpuBlendOperation::Add,
        )
        .unwrap(),
    )
}

fn pipeline() -> GpuRenderPipelineDescriptor {
    let [source] = admit_static_wgsl_sources([("proof.r2.dual-source-blending", 1, WGSL)]).unwrap();
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
        Some(blend_state()),
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
    let pipeline = GpuRenderPipelineDescriptor::new(
        program,
        GpuRenderEntryPoints::new(vertex, Some(fragment)),
        state,
        GpuPipelineConfiguration::default(),
    )
    .unwrap();
    assert_eq!(
        pipeline
            .requirements()
            .get(GpuCapabilityFeature::DualSourceBlending),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::DualSourceBlending
        ))
    );
    pipeline
}

fn graph() -> (GpuPreparedWorkGraph, GpuReadbackId) {
    let mut scope = GpuResourceScope::new();
    let target_label = label("R2 dual-source blending target");
    let target = scope
        .texture(
            GpuTextureDescriptor::new(
                common("R2 dual-source blending target"),
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
                common("R2 dual-source blending target view"),
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
        view,
        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(1.0, 1.0, 1.0, 1.0).unwrap()),
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

    let fragment = GpuWorkFragment::build("R2 dual-source blending proof", |builder| {
        builder.operation("draw dual-source blend proof", render)?;
        builder.operation("read dual-source blend target", readback)?;
        Ok(())
    })
    .unwrap();
    (
        GpuPreparedWorkGraph::prepare(label("R2 dual-source blending proof graph"), [fragment])
            .unwrap(),
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
            .get(GpuCapabilityFeature::DualSourceBlending),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::DualSourceBlending
        ))
    );
    let prepared = context.prepare_submission(graph).await.unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let bytes = readback_wait::wait_for_readback(
        context,
        &submission,
        readback_id,
        "R2 dual-source blending",
    )
    .await;
    assert_eq!(bytes.texture_format(), Some(GpuTextureFormat::Rgba8Unorm));
    assert_eq!(
        pixel_at(&bytes, WIDTH / 2, HEIGHT / 2),
        EXPECTED_PIXEL,
        "secondary source must independently control RGB and alpha"
    );
}

pub(crate) async fn run_on_adapter(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
    census: &GpuContext,
) -> DualSourceBlendProofOutcome {
    let supported = census
        .adapter_facts()
        .supported()
        .supports(GpuCapabilityFeature::DualSourceBlending);

    if !supported {
        let rejected = GpuContext::request(descriptor(backend, true, fallback))
            .await
            .expect_err("unsupported dual-source blending must reject the feature context");
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
                            && error.feature()
                                == Some(GpuCapabilityFeature::DualSourceBlending)
                    })
            )
        }));
        return DualSourceBlendProofOutcome {
            supported: false,
            exercised: false,
        };
    }

    assert!(
        !census
            .device_facts()
            .is_enabled(GpuCapabilityFeature::DualSourceBlending),
        "census context must not enable DualSourceBlending implicitly"
    );
    let (candidate, _) = graph();
    let missing_enablement = census
        .prepare_submission(candidate)
        .await
        .expect_err("supported but non-enabled dual-source work must reject before realization");
    assert_eq!(
        missing_enablement.kind(),
        GpuSubmissionPreparationErrorKind::CapabilityNotAdmitted
    );

    let context = GpuContext::request(descriptor(backend, true, fallback))
        .await
        .expect("advertised dual-source blending must admit a feature context");
    assert_eq!(
        context.adapter_facts(),
        census.adapter_facts(),
        "dual-source proof must remain on the exact census adapter"
    );
    assert!(
        context
            .device_facts()
            .is_enabled(GpuCapabilityFeature::DualSourceBlending)
    );
    run_case(&context).await;

    DualSourceBlendProofOutcome {
        supported: true,
        exercised: true,
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) async fn run_browser_dual_source_blending() -> u32 {
    const OPTIONAL_SUPPORTED: u32 = 1 << 0;
    const OPTIONAL_EXERCISED: u32 = 1 << 1;

    let census = GpuContext::request(descriptor(GpuBackendFamily::BrowserWebGpu, false, None))
        .await
        .expect("actual-browser Conformance must provide baseline offscreen WebGPU");
    let outcome = run_on_adapter(GpuBackendFamily::BrowserWebGpu, None, &census).await;

    let mut mask = 0;
    if outcome.supported {
        mask |= OPTIONAL_SUPPORTED;
    }
    if outcome.exercised {
        mask |= OPTIONAL_EXERCISED;
    }
    if outcome.supported {
        println!("DualSourceBlending BrowserWebGpu: EXERCISED");
    } else {
        println!("DualSourceBlending BrowserWebGpu: UNSUPPORTED");
    }
    mask
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires retained Vulkan adapter"]
fn dual_source_blending_native_execution_matches_advertised_capability() {
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
