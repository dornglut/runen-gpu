use runen_gpu::*;

#[path = "support/readback_wait.rs"]
mod readback_wait;

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const CLEAR_PIXEL: [u8; 4] = [0, 0, 0, 255];
const DRAW_PIXEL: [u8; 4] = [0, 255, 0, 255];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DepthClipProofOutcome {
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

fn pipeline(mode: GpuDepthClipMode) -> GpuRenderPipelineDescriptor {
    let source_text = r#"
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
    output.position = vec4<f32>(positions[vertex_index], -0.5, 1.0);
    return output;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 1.0, 0.0, 1.0);
}
"#;
    let identity = GpuProgramSourceIdentity::new(
        GpuProgramSourceOwnerId::allocate().unwrap(),
        GpuProgramSourceKey::new(match mode {
            GpuDepthClipMode::Clip => "r2.depth_clip.clip",
            GpuDepthClipMode::Unclipped => "r2.depth_clip.unclipped",
        })
        .unwrap(),
        GpuProgramSourceRevision::try_from_raw(1).unwrap(),
    );
    let mut sources = GpuProgramSourceRegistry::new(1, 8 * 1024).unwrap();
    let source = sources
        .admit_wgsl(
            identity,
            source_text,
            GpuProgramSourceProvenance::new("R2 depth clip control proof", None).unwrap(),
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
        None,
        GpuColorWriteMask::ALL,
    )
    .unwrap();
    let primitive = GpuPrimitiveStateDescriptor::new(
        GpuPrimitiveTopology::TriangleList,
        None,
        GpuFrontFace::CounterClockwise,
        GpuCullMode::None,
        mode,
    )
    .unwrap();
    let state = GpuRenderPipelineStateDescriptor::new(
        GpuVertexInputStateDescriptor::new([]).unwrap(),
        Some(GpuFragmentOutputStateDescriptor::new([target])),
        primitive,
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

    match mode {
        GpuDepthClipMode::Clip => assert!(
            pipeline
                .requirements()
                .get(GpuCapabilityFeature::DepthClipControl)
                .is_none(),
            "ordinary clipped rasterization must not request the optional capability"
        ),
        GpuDepthClipMode::Unclipped => assert!(matches!(
            pipeline
                .requirements()
                .get(GpuCapabilityFeature::DepthClipControl),
            Some(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::DepthClipControl
            ))
        )),
    }
    pipeline
}

fn graph(mode: GpuDepthClipMode) -> (GpuPreparedWorkGraph, GpuReadbackId) {
    let mut scope = GpuResourceScope::new();
    let name = match mode {
        GpuDepthClipMode::Clip => "R2 clipped depth",
        GpuDepthClipMode::Unclipped => "R2 unclipped depth",
    };
    let resource_label = label(format!("{name} color"));
    let texture = scope
        .texture(
            GpuTextureDescriptor::new(
                common(format!("{name} color")),
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

    let pipeline = pipeline(mode);
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
        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
        GpuAttachmentStore::Store,
        None,
    )
    .unwrap();
    let render = GpuRenderOperation::new([attachment], None, [GpuRenderPassItem::Draw(draw)], None).unwrap();
    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(
        GpuTextureCopyRegion::new(
            &texture,
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
    let fragment = GpuWorkFragment::build(name, |builder| {
        builder.operation("draw depth clip proof", render)?;
        builder.operation("read depth clip proof", readback)?;
        Ok(())
    })
    .unwrap();
    (
        GpuPreparedWorkGraph::prepare(label(format!("{name} graph")), [fragment]).unwrap(),
        readback_id,
    )
}

fn pixel_at(bytes: &GpuReadbackBytes, x: u32, y: u32) -> [u8; 4] {
    let offset = usize::try_from((y * WIDTH + x) * 4).unwrap();
    bytes.as_bytes()[offset..offset + 4].try_into().unwrap()
}

async fn assert_unclipped_pipeline_rejected_without_device_feature(context: &GpuContext) {
    assert!(
        !context
            .device_facts()
            .is_enabled(GpuCapabilityFeature::DepthClipControl),
        "baseline census context must not enable optional DepthClipControl"
    );

    let pipeline = pipeline(GpuDepthClipMode::Unclipped);
    let program = context
        .realize_program(pipeline.program())
        .await
        .expect("unclipped proof program itself must remain realizable without DepthClipControl");
    let layout = context
        .realize_pipeline_layout(pipeline.layout())
        .await
        .expect("unclipped proof layout itself must remain realizable without DepthClipControl");
    let error = context
        .realize_render_pipeline(&pipeline, &program, &layout)
        .await
        .expect_err(
            "an unclipped pipeline must reject before private creation when the device did not enable DepthClipControl",
        );
    assert_eq!(
        error.category(),
        GpuPipelineRealizationErrorCategory::RequirementNotAdmitted
    );
}

pub(crate) async fn run_case(context: &GpuContext, mode: GpuDepthClipMode) {
    let (graph, readback_id) = graph(mode);
    let prepared = context.prepare_submission(graph).await.unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let bytes = readback_wait::wait_for_readback(
        context,
        &submission,
        readback_id,
        format!("{mode:?} depth clip"),
    )
    .await;
    assert_eq!(bytes.texture_format(), Some(GpuTextureFormat::Rgba8Unorm));
    let observed = pixel_at(&bytes, WIDTH / 2, HEIGHT / 2);
    let expected = match mode {
        GpuDepthClipMode::Clip => CLEAR_PIXEL,
        GpuDepthClipMode::Unclipped => DRAW_PIXEL,
    };
    assert_eq!(
        observed, expected,
        "{mode:?} must distinguish clipping from unclipped rasterization at an interior sample"
    );
}

fn descriptor(
    backend: GpuBackendFamily,
    require_unclipped: bool,
    fallback: Option<GpuSoftwareFallbackPolicy>,
) -> GpuContextDescriptor {
    let mut requirements = GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements();
    if require_unclipped {
        requirements
            .insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::DepthClipControl,
            ))
            .unwrap();
    }
    let mut descriptor = GpuContextDescriptor::new(requirements)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
        .with_allowed_backends([backend])
        .with_label(if require_unclipped {
            "R2 depth clip control optional proof"
        } else {
            "R2 depth clip control census"
        });
    if let Some(fallback) = fallback {
        descriptor = descriptor.with_fallback_policy(fallback);
    }
    descriptor
}

pub(crate) async fn run_on_adapter(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
    census: &GpuAdapterFacts,
) -> DepthClipProofOutcome {
    run_case_for_census_backend(backend, fallback, census).await
}

async fn run_case_for_census_backend(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
    census: &GpuAdapterFacts,
) -> DepthClipProofOutcome {
    let supported = census
        .supported()
        .supports(GpuCapabilityFeature::DepthClipControl);

    if !supported {
        let rejected = GpuContext::request(descriptor(backend, true, fallback))
            .await
            .expect_err(
                "an adapter census without DepthClipControl must reject the optional context",
            );
        assert_eq!(
            rejected.category(),
            GpuContextRequestErrorCategory::NoAdmissibleCandidate
        );
        assert!(
            rejected.candidate_dispositions().iter().any(|disposition| {
                matches!(
                    disposition,
                    GpuCandidateDisposition::Rejected(report)
                        if report
                            .capability_admission_error()
                            .is_some_and(|error| {
                                error.cause() == GpuCapabilityAdmissionCause::RequiredUnavailable
                                    && error.feature()
                                        == Some(GpuCapabilityFeature::DepthClipControl)
                            })
                )
            }),
            "depth-clip rejection must retain typed required-feature-unavailable evidence"
        );
        return DepthClipProofOutcome {
            supported: false,
            exercised: false,
        };
    }

    let context = GpuContext::request(descriptor(backend, true, fallback))
        .await
        .expect("advertised DepthClipControl must admit the optional context");
    assert_eq!(
        context.adapter_facts(),
        census,
        "optional depth-clip proof must stay on the exact census adapter"
    );
    assert!(
        context
            .device_facts()
            .is_enabled(GpuCapabilityFeature::DepthClipControl),
        "admitted optional context must enable DepthClipControl"
    );
    run_case(&context, GpuDepthClipMode::Unclipped).await;
    DepthClipProofOutcome {
        supported: true,
        exercised: true,
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) async fn run_browser_depth_clip_control() -> u32 {
    const CLIP_EXERCISED: u32 = 1 << 0;
    const OPTIONAL_SUPPORTED: u32 = 1 << 1;
    const UNCLIPPED_EXERCISED: u32 = 1 << 2;

    let census_context =
        GpuContext::request(descriptor(GpuBackendFamily::BrowserWebGpu, false, None))
            .await
            .expect("actual-browser Conformance must provide baseline WebGPU rendering");
    assert_eq!(
        census_context.adapter_facts().backend(),
        GpuBackendFamily::BrowserWebGpu
    );
    assert_unclipped_pipeline_rejected_without_device_feature(&census_context).await;
    run_case(&census_context, GpuDepthClipMode::Clip).await;
    let outcome = run_on_adapter(
        GpuBackendFamily::BrowserWebGpu,
        None,
        census_context.adapter_facts(),
    )
    .await;

    let mut mask = CLIP_EXERCISED;
    if outcome.supported {
        mask |= OPTIONAL_SUPPORTED;
    }
    if outcome.exercised {
        mask |= UNCLIPPED_EXERCISED;
    }
    mask
}

#[test]
fn depth_clip_pipeline_requirements_are_mechanical_and_color_only() {
    let clipped = pipeline(GpuDepthClipMode::Clip);
    assert!(
        clipped
            .requirements()
            .get(GpuCapabilityFeature::DepthClipControl)
            .is_none()
    );
    assert!(
        clipped
            .requirements()
            .get(GpuCapabilityFeature::DepthAttachment)
            .is_none(),
        "depth clipping is primitive raster state, not a depth-attachment contract"
    );

    let unclipped = pipeline(GpuDepthClipMode::Unclipped);
    assert!(matches!(
        unclipped
            .requirements()
            .get(GpuCapabilityFeature::DepthClipControl),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::DepthClipControl
        ))
    ));
    assert!(
        unclipped
            .requirements()
            .get(GpuCapabilityFeature::DepthAttachment)
            .is_none()
    );
}

#[test]
fn unclipped_rejects_conflicting_disabled_depth_clip_requirement() {
    let base = pipeline(GpuDepthClipMode::Unclipped);
    let mut requirements = GpuCapabilityRequirements::new();
    requirements
        .insert(GpuCapabilityRequirement::Disabled(
            GpuCapabilityFeature::DepthClipControl,
        ))
        .unwrap();

    let error = GpuRenderPipelineDescriptor::new(
        base.program().clone(),
        base.entry_points().clone(),
        base.state().clone(),
        GpuPipelineConfiguration::new(None, Some(requirements)),
    )
    .expect_err("derived unclipped requirement must conflict with caller-disabled capability");
    assert_eq!(
        error.cause(),
        GpuProgramContractCause::PipelineDescriptorInvalid
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires the retained Vulkan software adapter"]
fn depth_clip_control_native_execution_is_backend_proven_when_advertised() {
    let census_context = pollster::block_on(GpuContext::request(descriptor(
        GpuBackendFamily::Vulkan,
        false,
        Some(GpuSoftwareFallbackPolicy::Require),
    )))
    .expect("native Conformance must provide the retained Vulkan fallback adapter");
    assert_eq!(
        census_context.adapter_facts().backend(),
        GpuBackendFamily::Vulkan
    );
    assert_eq!(
        census_context.adapter_facts().fallback(),
        GpuFallbackStatus::ConfirmedFallback
    );
    pollster::block_on(assert_unclipped_pipeline_rejected_without_device_feature(
        &census_context,
    ));
    pollster::block_on(run_case(&census_context, GpuDepthClipMode::Clip));
    let outcome = pollster::block_on(run_on_adapter(
        GpuBackendFamily::Vulkan,
        Some(GpuSoftwareFallbackPolicy::Require),
        census_context.adapter_facts(),
    ));
    if outcome.supported {
        assert!(outcome.exercised);
        println!("DepthClipControl Vulkan: EXERCISED");
    } else {
        println!("DepthClipControl Vulkan: UNSUPPORTED");
    }
}
