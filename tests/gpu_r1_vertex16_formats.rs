use runen_gpu::*;

#[path = "support/readback_wait.rs"]
mod readback_wait;

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const CLEAR_PIXEL: [u8; 4] = [0, 0, 0, 255];
const DRAW_PIXEL: [u8; 4] = [0, 255, 0, 255];

#[derive(Clone, Copy)]
struct Vertex16Case {
    name: &'static str,
    format: GpuVertexFormat,
    offset: u64,
    stride: u64,
    bytes: [u8; 8],
    wgsl_type: &'static str,
    signal_expression: &'static str,
}

const CASES: [Vertex16Case; 15] = [
    Vertex16Case {
        name: "uint16",
        format: GpuVertexFormat::Uint16,
        offset: 2,
        stride: 4,
        bytes: [0, 0, 1, 0, 0, 0, 0, 0],
        wgsl_type: "u32",
        signal_expression: "f32(value)",
    },
    Vertex16Case {
        name: "uint16x2",
        format: GpuVertexFormat::Uint16x2,
        offset: 0,
        stride: 4,
        bytes: [1, 0, 0, 0, 0, 0, 0, 0],
        wgsl_type: "vec2<u32>",
        signal_expression: "f32(value.x)",
    },
    Vertex16Case {
        name: "uint16x4",
        format: GpuVertexFormat::Uint16x4,
        offset: 0,
        stride: 8,
        bytes: [1, 0, 0, 0, 0, 0, 0, 0],
        wgsl_type: "vec4<u32>",
        signal_expression: "f32(value.x)",
    },
    Vertex16Case {
        name: "sint16",
        format: GpuVertexFormat::Sint16,
        offset: 2,
        stride: 4,
        bytes: [0, 0, 1, 0, 0, 0, 0, 0],
        wgsl_type: "i32",
        signal_expression: "f32(value)",
    },
    Vertex16Case {
        name: "sint16x2",
        format: GpuVertexFormat::Sint16x2,
        offset: 0,
        stride: 4,
        bytes: [1, 0, 0, 0, 0, 0, 0, 0],
        wgsl_type: "vec2<i32>",
        signal_expression: "f32(value.x)",
    },
    Vertex16Case {
        name: "sint16x4",
        format: GpuVertexFormat::Sint16x4,
        offset: 0,
        stride: 8,
        bytes: [1, 0, 0, 0, 0, 0, 0, 0],
        wgsl_type: "vec4<i32>",
        signal_expression: "f32(value.x)",
    },
    Vertex16Case {
        name: "unorm16",
        format: GpuVertexFormat::Unorm16,
        offset: 2,
        stride: 4,
        bytes: [0, 0, 255, 255, 0, 0, 0, 0],
        wgsl_type: "f32",
        signal_expression: "value",
    },
    Vertex16Case {
        name: "unorm16x2",
        format: GpuVertexFormat::Unorm16x2,
        offset: 0,
        stride: 4,
        bytes: [255, 255, 0, 0, 0, 0, 0, 0],
        wgsl_type: "vec2<f32>",
        signal_expression: "value.x",
    },
    Vertex16Case {
        name: "unorm16x4",
        format: GpuVertexFormat::Unorm16x4,
        offset: 0,
        stride: 8,
        bytes: [255, 255, 0, 0, 0, 0, 0, 0],
        wgsl_type: "vec4<f32>",
        signal_expression: "value.x",
    },
    Vertex16Case {
        name: "snorm16",
        format: GpuVertexFormat::Snorm16,
        offset: 2,
        stride: 4,
        bytes: [0, 0, 255, 127, 0, 0, 0, 0],
        wgsl_type: "f32",
        signal_expression: "value",
    },
    Vertex16Case {
        name: "snorm16x2",
        format: GpuVertexFormat::Snorm16x2,
        offset: 0,
        stride: 4,
        bytes: [255, 127, 0, 0, 0, 0, 0, 0],
        wgsl_type: "vec2<f32>",
        signal_expression: "value.x",
    },
    Vertex16Case {
        name: "snorm16x4",
        format: GpuVertexFormat::Snorm16x4,
        offset: 0,
        stride: 8,
        bytes: [255, 127, 0, 0, 0, 0, 0, 0],
        wgsl_type: "vec4<f32>",
        signal_expression: "value.x",
    },
    Vertex16Case {
        name: "float16",
        format: GpuVertexFormat::Float16,
        offset: 2,
        stride: 4,
        bytes: [0, 0, 0, 60, 0, 0, 0, 0],
        wgsl_type: "f32",
        signal_expression: "value",
    },
    Vertex16Case {
        name: "float16x2",
        format: GpuVertexFormat::Float16x2,
        offset: 0,
        stride: 4,
        bytes: [0, 60, 0, 0, 0, 0, 0, 0],
        wgsl_type: "vec2<f32>",
        signal_expression: "value.x",
    },
    Vertex16Case {
        name: "float16x4",
        format: GpuVertexFormat::Float16x4,
        offset: 0,
        stride: 8,
        bytes: [0, 60, 0, 0, 0, 0, 0, 0],
        wgsl_type: "vec4<f32>",
        signal_expression: "value.x",
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

fn shader_source(case: Vertex16Case) -> String {
    format!(
        r#"
struct VertexOutput {{
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) value: {}
) -> VertexOutput {{
    var position = vec2<f32>(0.0, 0.75);
    if vertex_index == 0u {{
        position = vec2<f32>(-0.75, -0.75);
    }} else if vertex_index == 1u {{
        position = vec2<f32>(0.75, -0.75);
    }}
    let signal = {};
    var output: VertexOutput;
    output.position = vec4<f32>(position, 0.0, 1.0);
    output.color = vec4<f32>(0.0, select(0.0, 1.0, signal > 0.0), 0.0, 1.0);
    return output;
}}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {{
    return input.color;
}}
"#,
        case.wgsl_type, case.signal_expression
    )
}

fn pipeline(case: Vertex16Case) -> GpuRenderPipelineDescriptor {
    let source_text = shader_source(case);
    let identity = GpuProgramSourceIdentity::new(
        GpuProgramSourceOwnerId::allocate().unwrap(),
        GpuProgramSourceKey::new(format!("r1.vertex16.{}", case.name)).unwrap(),
        GpuProgramSourceRevision::try_from_raw(1).unwrap(),
    );
    let mut sources = GpuProgramSourceRegistry::new(2, 16 * 1024).unwrap();
    let source = sources
        .admit_wgsl(
            identity,
            &source_text,
            GpuProgramSourceProvenance::new(format!("R1 vertex16 {} proof", case.name), None)
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
        case.stride,
        GpuVertexStepMode::Vertex,
        [GpuVertexAttribute::new(0, case.offset, case.format)],
    )
    .unwrap();
    let target = GpuColorTargetStateDescriptor::new(
        GpuTextureFormat::Rgba8Unorm,
        None,
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

fn vertex_buffer(scope: &mut GpuResourceScope, case: Vertex16Case) -> GpuBufferHandle {
    let mut bytes = Vec::with_capacity(usize::try_from(case.stride * 3).unwrap());
    for _ in 0..3 {
        bytes.extend_from_slice(&case.bytes[..usize::try_from(case.stride).unwrap()]);
    }
    let prepared =
        PreparedGpuData::<TransferData>::ordinary_pod_transfer(case.name, bytes.as_slice())
            .unwrap();
    let name = format!("{} compact vertex buffer", case.name);
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
    case: Vertex16Case,
) -> (GpuTextureHandle, GpuTextureViewHandle) {
    let name = format!("{} compact vertex target", case.name);
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

fn graph(case: Vertex16Case) -> (GpuPreparedWorkGraph, GpuReadbackId) {
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
    let graph_name = format!("R1 vertex16 {}", case.name);
    let fragment = GpuWorkFragment::build(&graph_name, |builder| {
        builder.operation("draw compact vertex format", render)?;
        builder.operation("read compact vertex target", readback)?;
        Ok(())
    })
    .unwrap();
    (
        GpuPreparedWorkGraph::prepare(label(format!("{graph_name} graph")), [fragment]).unwrap(),
        readback_id,
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
        let bytes = readback_wait::wait_for_readback(
            context,
            &submission,
            readback_id,
            format!("{} vertex", case.name),
        )
        .await;
        assert_eq!(bytes.texture_format(), Some(GpuTextureFormat::Rgba8Unorm));
        assert_eq!(pixel_at(&bytes, WIDTH / 2, HEIGHT / 2), DRAW_PIXEL);
        assert_eq!(pixel_at(&bytes, 0, 0), CLEAR_PIXEL);
        println!(
            "{:?} vertex: EXERCISED (offset={}, size={}, alignment={}, compact-offset draw + exact readback)",
            case.format,
            case.offset,
            case.format.size_bytes(),
            case.format.attribute_alignment_bytes()
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
            .with_label("R1 compact 16-bit vertex proof");
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
pub(crate) async fn run_browser_vertex16() -> u32 {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
            .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
            .with_label("R1 browser compact 16-bit vertex proof");
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
fn vertex16_case_census_matches_public_semantics() {
    assert_eq!(CASES.len(), 15);
    for case in CASES {
        assert_eq!(
            case.offset % case.format.attribute_alignment_bytes(),
            0,
            "{:?}",
            case.format
        );
        assert!(case.offset + case.format.size_bytes() <= case.stride);
        assert_eq!(case.stride % 4, 0);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires the retained Vulkan software adapter"]
fn vertex16_native_compact_offsets_pipeline_and_draw_are_backend_proven() {
    let context = native_context();
    let mask = pollster::block_on(run_suite(&context));
    assert_eq!(mask, (1_u32 << CASES.len()) - 1);
}
