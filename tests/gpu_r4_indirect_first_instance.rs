use bytemuck::{Pod, Zeroable};
use runen_gpu::*;

#[path = "support/readback_wait.rs"]
mod readback_wait;

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const CLEAR_PIXEL: [u8; 4] = [0, 0, 0, 255];
const ZERO_PIXEL: [u8; 4] = [255, 0, 0, 255];
const NONZERO_PIXEL: [u8; 4] = [0, 255, 0, 255];

const COMPUTE_ZERO_WGSL: &str = r#"
struct DrawIndirectArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0)
var<storage, read_write> draw_args: DrawIndirectArgs;

@compute @workgroup_size(1)
fn cs_main() {
    draw_args.vertex_count = 3u;
    draw_args.instance_count = 1u;
    draw_args.first_vertex = 0u;
    draw_args.first_instance = 0u;
}
"#;

const COMPUTE_NONZERO_WGSL: &str = r#"
struct DrawIndirectArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0)
var<storage, read_write> draw_args: DrawIndirectArgs;

@compute @workgroup_size(1)
fn cs_main() {
    draw_args.vertex_count = 3u;
    draw_args.instance_count = 1u;
    draw_args.first_vertex = 0u;
    draw_args.first_instance = 3u;
}
"#;

const RENDER_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-0.75, -0.75),
        vec2<f32>(0.75, -0.75),
        vec2<f32>(0.0, 0.75),
    );
    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    output.color = select(
        vec4<f32>(1.0, 0.0, 0.0, 1.0),
        vec4<f32>(0.0, 1.0, 0.0, 1.0),
        instance_index == 3u,
    );
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IndirectFirstInstanceProofOutcome {
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
    require_nonzero: bool,
    fallback: Option<GpuSoftwareFallbackPolicy>,
) -> GpuContextDescriptor {
    let mut requirements = GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements();
    for feature in [
        GpuCapabilityFeature::Compute,
        GpuCapabilityFeature::IndirectExecution,
    ] {
        requirements
            .insert(GpuCapabilityRequirement::Required(feature))
            .unwrap();
    }
    if require_nonzero {
        requirements
            .insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::IndirectFirstInstance,
            ))
            .unwrap();
    }
    let mut descriptor = GpuContextDescriptor::new(requirements)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
        .with_allowed_backends([backend])
        .with_label(if require_nonzero {
            "R4 indirect first-instance optional proof"
        } else {
            "R4 indirect first-instance census"
        });
    if let Some(fallback) = fallback {
        descriptor = descriptor.with_fallback_policy(fallback);
    }
    descriptor
}

fn pipelines(
    mode: GpuIndirectFirstInstanceMode,
) -> (GpuComputePipelineDescriptor, GpuRenderPipelineDescriptor) {
    let compute_wgsl = match mode {
        GpuIndirectFirstInstanceMode::ZeroOnly => COMPUTE_ZERO_WGSL,
        GpuIndirectFirstInstanceMode::MayBeNonZero => COMPUTE_NONZERO_WGSL,
    };
    let [compute_source, render_source] = admit_static_wgsl_sources([
        ("proof.r4-indirect-first-instance.compute", 1, compute_wgsl),
        ("proof.r4-indirect-first-instance.render", 1, RENDER_WGSL),
    ])
    .unwrap();

    let compute = GpuComputePipelineDescriptor::ordinary(compute_source, "cs_main").unwrap();

    let vertex = GpuEntryPointName::new("vs_main").unwrap();
    let fragment = GpuEntryPointName::new("fs_main").unwrap();
    let render_program = GpuProgramDescriptor::new(
        render_source,
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
    let render = GpuRenderPipelineDescriptor::new(
        render_program,
        GpuRenderEntryPoints::new(vertex, Some(fragment)),
        state,
        GpuPipelineConfiguration::default(),
    )
    .unwrap();
    (compute, render)
}

fn graph(mode: GpuIndirectFirstInstanceMode) -> (GpuPreparedWorkGraph, GpuReadbackId) {
    let mut scope = GpuResourceScope::new();
    let prepared = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
        "R4 indirect first-instance args",
        &[DrawIndirectArgs::zeroed()],
    )
    .unwrap();
    let args_label = label("R4 indirect first-instance args");
    let args = scope
        .buffer(
            GpuBufferDescriptor::new(
                common("R4 indirect first-instance args"),
                prepared.layout().byte_len(),
                GpuBufferUsages::new(
                    &args_label,
                    [
                        GpuBufferUsage::Storage,
                        GpuBufferUsage::Indirect,
                        GpuBufferUsage::CopyDestination,
                    ],
                )
                .unwrap(),
                GpuBufferInitialization::Prepared(prepared),
            )
            .unwrap(),
        )
        .unwrap();

    let target_label = label("R4 indirect first-instance target");
    let target = scope
        .texture(
            GpuTextureDescriptor::new(
                common("R4 indirect first-instance target"),
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
                common("R4 indirect first-instance target view"),
                &target,
                None,
                GpuTextureViewDimension::D2,
                GpuTextureSubresourceRange::whole(&target).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();

    let (compute_pipeline, render_pipeline) = pipelines(mode);
    let compute_bindings = compute_pipeline
        .runtime_bindings([GpuRuntimeBindingValue::whole_buffer(0, 0, &args)])
        .unwrap();
    let compute = GpuComputeOperation::new(
        compute_pipeline,
        compute_bindings,
        GpuDispatchIntent::direct(GpuDispatchSize::new(1, 1, 1)),
    )
    .unwrap();

    let render_bindings = GpuRuntimeBindingSet::new(render_pipeline.layout().clone(), []).unwrap();
    let draw = GpuRenderDraw::new(
        render_pipeline,
        render_bindings,
        [],
        None,
        GpuDrawIntent::indirect(&args, GpuBufferRange::whole(&args).unwrap(), false, mode).unwrap(),
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

    let fragment = GpuWorkFragment::build("R4 indirect first-instance proof", |builder| {
        builder.operation("write indirect first-instance args", compute)?;
        builder.operation("consume indirect first-instance args", render)?;
        builder.operation("read indirect first-instance target", readback)?;
        Ok(())
    })
    .unwrap();
    (
        GpuPreparedWorkGraph::prepare(label("R4 indirect first-instance proof graph"), [fragment])
            .unwrap(),
        readback_id,
    )
}

fn pixel_at(bytes: &GpuReadbackBytes, x: u32, y: u32) -> [u8; 4] {
    let offset = usize::try_from((y * WIDTH + x) * 4).unwrap();
    bytes.as_bytes()[offset..offset + 4].try_into().unwrap()
}

pub(crate) async fn run_case(context: &GpuContext, mode: GpuIndirectFirstInstanceMode) {
    let (graph, readback_id) = graph(mode);
    let expected_requirement = match mode {
        GpuIndirectFirstInstanceMode::ZeroOnly => {
            GpuCapabilityRequirement::Disabled(GpuCapabilityFeature::IndirectFirstInstance)
        }
        GpuIndirectFirstInstanceMode::MayBeNonZero => {
            GpuCapabilityRequirement::Required(GpuCapabilityFeature::IndirectFirstInstance)
        }
    };
    assert_eq!(
        graph
            .requirements()
            .get(GpuCapabilityFeature::IndirectFirstInstance),
        Some(expected_requirement)
    );
    let prepared = context.prepare_submission(graph).await.unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let bytes = readback_wait::wait_for_readback(
        context,
        &submission,
        readback_id,
        format!("R4 indirect first-instance {mode:?}"),
    )
    .await;
    assert_eq!(bytes.texture_format(), Some(GpuTextureFormat::Rgba8Unorm));
    assert_eq!(pixel_at(&bytes, 0, 0), CLEAR_PIXEL);
    let expected = match mode {
        GpuIndirectFirstInstanceMode::ZeroOnly => ZERO_PIXEL,
        GpuIndirectFirstInstanceMode::MayBeNonZero => NONZERO_PIXEL,
    };
    assert_eq!(
        pixel_at(&bytes, WIDTH / 2, HEIGHT / 2),
        expected,
        "{mode:?} must make the consumed indirect first_instance observable"
    );
}

pub(crate) async fn run_on_adapter(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
    census: &GpuContext,
) -> IndirectFirstInstanceProofOutcome {
    let supported = census
        .adapter_facts()
        .supported()
        .supports(GpuCapabilityFeature::IndirectFirstInstance);
    if !supported {
        let rejected = GpuContext::request(descriptor(backend, true, fallback))
            .await
            .expect_err("unsupported indirect first-instance must reject the feature context");
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
                                == Some(GpuCapabilityFeature::IndirectFirstInstance)
                    })
            )
        }));
        return IndirectFirstInstanceProofOutcome {
            supported: false,
            exercised: false,
        };
    }

    assert!(
        !census
            .device_facts()
            .is_enabled(GpuCapabilityFeature::IndirectFirstInstance),
        "census context must not enable IndirectFirstInstance implicitly"
    );
    let (nonzero_on_census, _) = graph(GpuIndirectFirstInstanceMode::MayBeNonZero);
    let missing_enablement = census
        .prepare_submission(nonzero_on_census)
        .await
        .expect_err(
            "MayBeNonZero must reject during normalized admission when the supported feature was not enabled",
        );
    assert_eq!(
        missing_enablement.kind(),
        GpuSubmissionPreparationErrorKind::CapabilityNotAdmitted
    );

    let context = GpuContext::request(descriptor(backend, true, fallback))
        .await
        .expect("advertised indirect first-instance must admit a feature context");
    assert_eq!(
        context.adapter_facts(),
        census.adapter_facts(),
        "indirect first-instance proof must remain on the exact census adapter"
    );
    assert!(
        context
            .device_facts()
            .is_enabled(GpuCapabilityFeature::IndirectFirstInstance)
    );

    let (zero_graph, _) = graph(GpuIndirectFirstInstanceMode::ZeroOnly);
    let zero_error = context.prepare_submission(zero_graph).await.expect_err(
        "ZeroOnly must reject during normalized admission when IndirectFirstInstance is enabled",
    );
    assert_eq!(
        zero_error.kind(),
        GpuSubmissionPreparationErrorKind::CapabilityNotAdmitted
    );

    run_case(&context, GpuIndirectFirstInstanceMode::MayBeNonZero).await;
    IndirectFirstInstanceProofOutcome {
        supported: true,
        exercised: true,
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) async fn run_browser_indirect_first_instance() -> u32 {
    const ZERO_EXERCISED: u32 = 1 << 0;
    const OPTIONAL_SUPPORTED: u32 = 1 << 1;
    const NONZERO_EXERCISED: u32 = 1 << 2;

    let census = GpuContext::request(descriptor(GpuBackendFamily::BrowserWebGpu, false, None))
        .await
        .expect("actual-browser Conformance must provide baseline indirect execution");
    run_case(&census, GpuIndirectFirstInstanceMode::ZeroOnly).await;
    let outcome = run_on_adapter(
        GpuBackendFamily::BrowserWebGpu,
        None,
        &census,
    )
    .await;

    let mut mask = ZERO_EXERCISED;
    if outcome.supported {
        mask |= OPTIONAL_SUPPORTED;
    }
    if outcome.exercised {
        mask |= NONZERO_EXERCISED;
    }
    if outcome.supported {
        println!("IndirectFirstInstance BrowserWebGpu: EXERCISED");
    } else {
        println!("IndirectFirstInstance BrowserWebGpu: UNSUPPORTED");
    }
    mask
}

#[test]
fn indirect_first_instance_requirements_are_exact_and_conflicting_modes_reject() {
    let (zero, _) = graph(GpuIndirectFirstInstanceMode::ZeroOnly);
    let (nonzero, _) = graph(GpuIndirectFirstInstanceMode::MayBeNonZero);
    assert_eq!(
        zero.requirements()
            .get(GpuCapabilityFeature::IndirectFirstInstance),
        Some(GpuCapabilityRequirement::Disabled(
            GpuCapabilityFeature::IndirectFirstInstance
        ))
    );
    assert_eq!(
        nonzero
            .requirements()
            .get(GpuCapabilityFeature::IndirectFirstInstance),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::IndirectFirstInstance
        ))
    );
    assert!(zero.requirements().merge(nonzero.requirements()).is_err());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires the retained Vulkan software adapter"]
fn indirect_first_instance_native_execution_is_backend_proven_when_advertised() {
    let census = pollster::block_on(GpuContext::request(descriptor(
        GpuBackendFamily::Vulkan,
        false,
        Some(GpuSoftwareFallbackPolicy::Require),
    )))
    .expect("native Conformance must provide the retained Vulkan fallback adapter");
    pollster::block_on(run_case(&census, GpuIndirectFirstInstanceMode::ZeroOnly));
    let outcome = pollster::block_on(run_on_adapter(
        GpuBackendFamily::Vulkan,
        Some(GpuSoftwareFallbackPolicy::Require),
        &census,
    ));
    if outcome.supported {
        assert!(outcome.exercised);
        println!("IndirectFirstInstance Vulkan: EXERCISED");
    } else {
        println!("IndirectFirstInstance Vulkan: UNSUPPORTED");
    }
}
