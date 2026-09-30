use runen_gpu::*;

#[path = "support/readback_wait.rs"]
mod readback_wait;

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const SENTINEL_PIXEL: [u8; 4] = [0, 0, 255, 255];
const VIEW_ZERO_PIXEL: [u8; 4] = [255, 0, 0, 255];
const VIEW_ONE_PIXEL: [u8; 4] = [0, 255, 0, 255];
const SHARED_VIEW_PIXEL: [u8; 4] = [0, 255, 255, 255];
const CLEAR_ONLY_PIXEL: [u8; 4] = [255, 0, 255, 255];

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

const VERTEX_VIEW_INDEX_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(view_index) view_index: u32,
) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var output: VertexOutput;
    let view_offset = f32(view_index) * 0.0;
    output.position = vec4<f32>(positions[vertex_index] + vec2<f32>(view_offset), 0.0, 1.0);
    return output;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 1.0, 1.0, 1.0);
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
    pub shader_without_view_index_exercised: bool,
    pub clear_only_exercised: bool,
}

impl MultiviewProofOutcome {
    pub(crate) const fn fully_exercised(self) -> bool {
        self.primary_exercised
            && self.shader_without_view_index_exercised
            && self.clear_only_exercised
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

fn multiview_pipeline(
    source_key: &'static str,
    source_text: &'static str,
) -> GpuRenderPipelineDescriptor {
    multiview_pipeline_with_count(source_key, source_text, 2)
}

fn multiview_pipeline_with_count(
    source_key: &'static str,
    source_text: &'static str,
    view_count: u32,
) -> GpuRenderPipelineDescriptor {
    multiview_pipeline_with_count_and_samples(source_key, source_text, view_count, 1)
}

fn multiview_pipeline_with_count_and_samples(
    source_key: &'static str,
    source_text: &'static str,
    view_count: u32,
    sample_count: u32,
) -> GpuRenderPipelineDescriptor {
    multiview_pipeline_with_count_samples_and_depth(
        source_key,
        source_text,
        view_count,
        sample_count,
        false,
    )
}

fn multiview_pipeline_with_count_samples_and_depth(
    source_key: &'static str,
    source_text: &'static str,
    view_count: u32,
    sample_count: u32,
    depth: bool,
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
    let depth_stencil = depth
        .then(|| {
            GpuDepthStencilStateDescriptor::new(
                GpuTextureFormat::Depth32Float,
                Some(GpuDepthStateDescriptor::new(
                    true,
                    GpuCompareFunction::LessEqual,
                )),
                None,
                GpuDepthBiasState::default(),
            )
        })
        .transpose()
        .unwrap();
    let state = GpuRenderPipelineStateDescriptor::new(
        GpuVertexInputStateDescriptor::new([]).unwrap(),
        Some(GpuFragmentOutputStateDescriptor::new([target])),
        GpuPrimitiveStateDescriptor::default(),
        depth_stencil,
        GpuMultisampleStateDescriptor::new(sample_count, (1_u64 << sample_count) - 1, false)
            .unwrap(),
    )
    .unwrap()
    .with_multiview(GpuMultiviewState::new(view_count).unwrap())
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
    texture_view_with_aspect(
        scope,
        texture,
        name,
        dimension,
        base_layer,
        layer_count,
        GpuTextureAspect::Color,
    )
}

fn texture_view_with_aspect(
    scope: &mut GpuResourceScope,
    texture: &GpuTextureHandle,
    name: &str,
    dimension: GpuTextureViewDimension,
    base_layer: u32,
    layer_count: u32,
    aspect: GpuTextureAspect,
) -> GpuTextureViewHandle {
    let range = GpuTextureSubresourceRange::new(
        texture.descriptor().common().label(),
        0,
        1,
        base_layer,
        layer_count,
        aspect,
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

    let has_dependency = |before, after| {
        graph.dependencies().iter().any(|dependency| {
            dependency.before().local_node() == before && dependency.after().local_node() == after
        })
    };
    assert!(
        !has_dependency(1, 2),
        "disjoint parent layer zero must not gain a false dependency on the multiview writer"
    );
    assert!(
        has_dependency(1, 3) && !has_dependency(2, 3),
        "layer-zero readback must depend only on the disjoint sentinel writer"
    );
    assert!(
        !has_dependency(1, 4) && has_dependency(2, 4),
        "first selected multiview layer readback must depend only on the multiview writer"
    );
    assert!(
        !has_dependency(1, 5) && has_dependency(2, 5),
        "second selected multiview layer readback must depend only on the multiview writer"
    );
    (graph, ids)
}

fn layered_multisample_graph(include_depth: bool) -> (GpuPreparedWorkGraph, [GpuReadbackId; 3]) {
    let mut scope = GpuResourceScope::new();
    let source_label = label("R4 layered MSAA source");
    let source = scope
        .texture(
            GpuTextureDescriptor::new(
                common("R4 layered MSAA source"),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&source_label, GpuTextureDimension::D2, WIDTH, HEIGHT, 3)
                    .unwrap(),
                1,
                4,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(&source_label, [GpuTextureUsage::ColorAttachment]).unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let destination_label = label("R4 layered MSAA destination");
    let destination = scope
        .texture(
            GpuTextureDescriptor::new(
                common("R4 layered MSAA destination"),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(
                    &destination_label,
                    GpuTextureDimension::D2,
                    WIDTH,
                    HEIGHT,
                    4,
                )
                .unwrap(),
                1,
                1,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(
                    &destination_label,
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
        &destination,
        "R4 layered MSAA sentinel",
        GpuTextureViewDimension::D2,
        0,
        1,
    );
    let source_view = texture_view(
        &mut scope,
        &source,
        "R4 layered MSAA source layers 1 2",
        GpuTextureViewDimension::D2Array,
        1,
        2,
    );
    let destination_view = texture_view(
        &mut scope,
        &destination,
        "R4 layered MSAA destination layers 1 2",
        GpuTextureViewDimension::D2Array,
        1,
        2,
    );
    let depth_attachment = if include_depth {
        let depth_label = label("R4 layered MSAA depth");
        let depth = scope
            .texture(
                GpuTextureDescriptor::new(
                    common("R4 layered MSAA depth"),
                    GpuTextureDimension::D2,
                    GpuTextureExtent::new(&depth_label, GpuTextureDimension::D2, WIDTH, HEIGHT, 3)
                        .unwrap(),
                    1,
                    4,
                    GpuTextureFormat::Depth32Float,
                    GpuTextureUsages::new(&depth_label, [GpuTextureUsage::DepthStencilAttachment])
                        .unwrap(),
                    GpuTextureInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();
        let depth_view = texture_view_with_aspect(
            &mut scope,
            &depth,
            "R4 layered MSAA depth layers 1 2",
            GpuTextureViewDimension::D2Array,
            1,
            2,
            GpuTextureAspect::DepthOnly,
        );
        Some(
            GpuRenderDepthStencilAttachment::new(
                depth_view,
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
            .unwrap(),
        )
    } else {
        None
    };
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
    let pipeline = multiview_pipeline_with_count_samples_and_depth(
        "proof.r4.layered-msaa.view-index",
        VIEW_INDEX_WGSL,
        2,
        4,
        include_depth,
    );
    let layered = GpuRenderOperation::new(
        [GpuRenderColorAttachment::new(
            source_view,
            GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
            GpuAttachmentStore::Store,
            Some(GpuMultisampleResolveTarget::new(destination_view).unwrap()),
        )
        .unwrap()],
        depth_attachment,
        [render_draw(pipeline)],
        None,
    )
    .unwrap();
    assert_eq!(layered.signature().sample_count(), 4);
    assert_eq!(layered.signature().multiview().unwrap().view_count(), 2);
    let ids = [
        GpuReadbackId::allocate().unwrap(),
        GpuReadbackId::allocate().unwrap(),
        GpuReadbackId::allocate().unwrap(),
    ];
    let readbacks = ids.into_iter().enumerate().map(|(layer, id)| {
        let region = GpuTextureCopyRegion::new(
            &destination,
            0,
            GpuTextureOrigin::new(0, 0, u32::try_from(layer).unwrap()),
            GpuTextureAspect::Color,
            GpuCopyExtent::new(WIDTH, HEIGHT, 1).unwrap(),
        )
        .unwrap();
        GpuReadbackOperation::new(region.into(), id).unwrap()
    });
    let fragment = GpuWorkFragment::build("R4 layered MSAA resolve oracle", |builder| {
        builder.operation("initialize disjoint resolve sentinel", sentinel)?;
        builder.operation("render and resolve layers one two", layered)?;
        for (layer, readback) in readbacks.enumerate() {
            builder.operation(format!("read resolved layer {layer}"), readback)?;
        }
        Ok(())
    })
    .unwrap();
    let graph =
        GpuPreparedWorkGraph::prepare(label("R4 layered MSAA resolve graph"), [fragment]).unwrap();
    for feature in [
        GpuCapabilityFeature::MultisampleArray,
        GpuCapabilityFeature::Multiview,
    ] {
        assert_eq!(
            graph.requirements().get(feature),
            Some(GpuCapabilityRequirement::Required(feature))
        );
    }
    let has_dependency = |before, after| {
        graph.dependencies().iter().any(|dependency| {
            dependency.before().local_node() == before && dependency.after().local_node() == after
        })
    };
    assert!(!has_dependency(1, 2));
    assert!(has_dependency(1, 3) && !has_dependency(2, 3));
    assert!(!has_dependency(1, 4) && has_dependency(2, 4));
    assert!(!has_dependency(1, 5) && has_dependency(2, 5));
    (graph, ids)
}

#[test]
fn layered_multisample_graph_tracks_selected_layers_and_independent_capabilities() {
    let (graph, _) = layered_multisample_graph(false);
    assert_eq!(graph.nodes().len(), 5);

    let mut scope = GpuResourceScope::new();
    let resource_label = label("R4 one-layer view into multisample array");
    let texture = scope
        .texture(
            GpuTextureDescriptor::new(
                common("R4 one-layer view into multisample array"),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, WIDTH, HEIGHT, 3)
                    .unwrap(),
                1,
                4,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(&resource_label, [GpuTextureUsage::ColorAttachment]).unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let ordinary_view = texture_view(
        &mut scope,
        &texture,
        "R4 one-layer multisample view",
        GpuTextureViewDimension::D2,
        1,
        1,
    );
    let render = GpuRenderOperation::new(
        [GpuRenderColorAttachment::new(
            ordinary_view,
            GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
            GpuAttachmentStore::Store,
            None,
        )
        .unwrap()],
        None,
        std::iter::empty::<GpuRenderDraw>(),
        None,
    )
    .unwrap();
    assert_eq!(render.signature().sample_count(), 4);
    assert_eq!(render.signature().multiview(), None);
    let fragment = GpuWorkFragment::build("R4 one-layer multisample pass", |builder| {
        builder.operation("render one array layer", render)?;
        Ok(())
    })
    .unwrap();
    let graph =
        GpuPreparedWorkGraph::prepare(label("R4 one-layer multisample graph"), [fragment]).unwrap();
    assert_eq!(
        graph
            .requirements()
            .get(GpuCapabilityFeature::MultisampleArray),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::MultisampleArray
        ))
    );
    assert_eq!(
        graph.requirements().get(GpuCapabilityFeature::Multiview),
        None
    );
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

async fn run_layered_multisample_oracle(context: &GpuContext, include_depth: bool) {
    let (graph, ids) = layered_multisample_graph(include_depth);
    let prepared = context.prepare_submission(graph).await.unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    for (layer, (id, expected)) in ids
        .into_iter()
        .zip([SENTINEL_PIXEL, VIEW_ZERO_PIXEL, VIEW_ONE_PIXEL])
        .enumerate()
    {
        let bytes = readback_wait::wait_for_readback(
            context,
            &submission,
            id,
            format!("R4 layered MSAA resolved layer {layer}"),
        )
        .await;
        assert_eq!(pixel_at(&bytes, WIDTH / 2, HEIGHT / 2), expected);
    }
}

fn two_layer_graph(
    name: &str,
    source: Option<(&'static str, &'static str)>,
    clear: GpuColorClearValue,
) -> (GpuPreparedWorkGraph, [GpuReadbackId; 2]) {
    let mut scope = GpuResourceScope::new();
    let texture_label = label(format!("{name} texture"));
    let texture = scope
        .texture(
            GpuTextureDescriptor::new(
                common(format!("{name} texture")),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&texture_label, GpuTextureDimension::D2, WIDTH, HEIGHT, 2)
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
    let view = texture_view(
        &mut scope,
        &texture,
        &format!("{name} view"),
        GpuTextureViewDimension::D2Array,
        0,
        2,
    );
    let draws = source
        .map(|(key, text)| render_draw(multiview_pipeline(key, text)))
        .into_iter();
    let render = GpuRenderOperation::new(
        [GpuRenderColorAttachment::new(
            view,
            GpuColorAttachmentLoad::Clear(clear),
            GpuAttachmentStore::Store,
            None,
        )
        .unwrap()],
        None,
        draws,
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

    let fragment = GpuWorkFragment::build(name, |builder| {
        builder.operation(format!("{name} render"), render)?;
        for (layer, readback) in readbacks.enumerate() {
            builder.operation(format!("{name} read layer {layer}"), readback)?;
        }
        Ok(())
    })
    .unwrap();
    let graph = GpuPreparedWorkGraph::prepare(label(format!("{name} graph")), [fragment]).unwrap();
    assert!(matches!(
        graph.requirements().get(GpuCapabilityFeature::Multiview),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Multiview
        ))
    ));
    (graph, ids)
}

async fn run_two_layer_oracle(
    context: &GpuContext,
    name: &str,
    graph: GpuPreparedWorkGraph,
    ids: [GpuReadbackId; 2],
    expected: [u8; 4],
) {
    let prepared = context.prepare_submission(graph).await.unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    for (layer, id) in ids.into_iter().enumerate() {
        let bytes = readback_wait::wait_for_readback(
            context,
            &submission,
            id,
            format!("{name} layer {layer}"),
        )
        .await;
        assert_eq!(
            pixel_at(&bytes, WIDTH / 2, HEIGHT / 2),
            expected,
            "{name} must produce the exact expected pixel on layer {layer}"
        );
    }
}

async fn run_shader_without_view_index_oracle(context: &GpuContext) {
    let (graph, ids) = two_layer_graph(
        "R4 multiview shader-without-view-index oracle",
        Some(("proof.r4.multiview.no-view-index", NO_VIEW_INDEX_WGSL)),
        GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap(),
    );
    run_two_layer_oracle(
        context,
        "R4 multiview shader-without-view-index",
        graph,
        ids,
        SHARED_VIEW_PIXEL,
    )
    .await;
}

async fn run_clear_only_oracle(context: &GpuContext) {
    let (graph, ids) = two_layer_graph(
        "R4 multiview clear-only oracle",
        None,
        GpuColorClearValue::new(1.0, 0.0, 1.0, 1.0).unwrap(),
    );
    run_two_layer_oracle(
        context,
        "R4 multiview clear-only",
        graph,
        ids,
        CLEAR_ONLY_PIXEL,
    )
    .await;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LayeredMultisampleProofOutcome {
    pub multisample_array_supported: bool,
    pub multiview_supported: bool,
    pub depth_stencil_supported: bool,
    pub depth_stencil_exercised: bool,
    pub exercised: bool,
}

fn layered_multisample_descriptor(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
    include_depth: bool,
) -> GpuContextDescriptor {
    let mut requirements = GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements();
    for feature in [
        GpuCapabilityFeature::MultisampleArray,
        GpuCapabilityFeature::Multiview,
    ] {
        requirements
            .insert(GpuCapabilityRequirement::Required(feature))
            .unwrap();
    }
    if include_depth {
        requirements
            .insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::DepthAttachment,
            ))
            .unwrap();
    }
    let mut descriptor = GpuContextDescriptor::new(requirements)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
        .require_limit(GpuLimitKind::MaxMultiviewViewCount, 2)
        .with_allowed_backends([backend])
        .with_label("R4 layered MSAA retained proof");
    if include_depth {
        descriptor = descriptor
            .require_format_role(GpuTextureFormat::Depth32Float, GpuFormatRole::DepthStencil);
    }
    if let Some(fallback) = fallback {
        descriptor = descriptor.with_fallback_policy(fallback);
    }
    descriptor
}

pub(crate) async fn run_layered_multisample_on_adapter(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
    census: &GpuContext,
) -> LayeredMultisampleProofOutcome {
    let supported = census.adapter_facts().supported();
    let multisample_array_supported = supported.supports(GpuCapabilityFeature::MultisampleArray);
    let multiview_supported = supported.supports(GpuCapabilityFeature::Multiview);
    let depth_stencil_supported = supported
        .format(GpuTextureFormat::Depth32Float)
        .is_some_and(|facts| facts.depth_stencil);
    assert!(
        !census
            .device_facts()
            .is_enabled(GpuCapabilityFeature::MultisampleArray),
        "census context must not enable optional MultisampleArray implicitly"
    );
    if !multisample_array_supported {
        let rejected = GpuContext::request(layered_multisample_descriptor(
            backend,
            fallback,
            depth_stencil_supported,
        ))
        .await
        .expect_err("unsupported MultisampleArray must reject a required context");
        assert_eq!(
            rejected.category(),
            GpuContextRequestErrorCategory::NoAdmissibleCandidate
        );
        assert!(rejected.candidate_dispositions().iter().any(|disposition| {
            matches!(disposition, GpuCandidateDisposition::Rejected(report)
                if report.capability_admission_error().is_some_and(|error|
                    error.cause() == GpuCapabilityAdmissionCause::RequiredUnavailable
                        && error.feature() == Some(GpuCapabilityFeature::MultisampleArray)))
        }));
        return LayeredMultisampleProofOutcome {
            multisample_array_supported,
            multiview_supported,
            depth_stencil_supported,
            depth_stencil_exercised: false,
            exercised: false,
        };
    }
    let (graph, _) = layered_multisample_graph(false);
    let error = census
        .prepare_submission(graph)
        .await
        .expect_err("unadmitted layered MSAA work must reject before realization");
    assert_eq!(
        error.kind(),
        GpuSubmissionPreparationErrorKind::CapabilityNotAdmitted
    );
    if !multiview_supported {
        return LayeredMultisampleProofOutcome {
            multisample_array_supported,
            multiview_supported,
            depth_stencil_supported,
            depth_stencil_exercised: false,
            exercised: false,
        };
    }
    let context = GpuContext::request(layered_multisample_descriptor(
        backend,
        fallback,
        depth_stencil_supported,
    ))
    .await
    .expect("advertised layered MSAA capability must admit the retained proof context");
    assert_eq!(context.adapter_facts(), census.adapter_facts());
    assert!(
        context
            .device_facts()
            .is_enabled(GpuCapabilityFeature::MultisampleArray)
    );
    assert!(
        context
            .device_facts()
            .is_enabled(GpuCapabilityFeature::Multiview)
    );
    run_layered_multisample_oracle(&context, depth_stencil_supported).await;
    LayeredMultisampleProofOutcome {
        multisample_array_supported,
        multiview_supported,
        depth_stencil_supported,
        depth_stencil_exercised: depth_stencil_supported,
        exercised: true,
    }
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
    let normalized_max = census
        .adapter_facts()
        .adapter_limits()
        .values()
        .max_multiview_view_count();

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

        let limit_rejected = GpuContext::request(
            descriptor(backend, false, fallback)
                .require_limit(GpuLimitKind::MaxMultiviewViewCount, 2),
        )
        .await
        .expect_err("unsupported Multiview must reject a positive multiview limit request");
        assert_eq!(
            limit_rejected.category(),
            GpuContextRequestErrorCategory::NoAdmissibleCandidate
        );
        assert!(
            limit_rejected
                .candidate_dispositions()
                .iter()
                .any(|disposition| {
                    matches!(
                        disposition,
                        GpuCandidateDisposition::Rejected(report)
                            if report.category()
                                == GpuContextRequestErrorCategory::LimitBelowRequiredMinimum
                                && report.limit_rejection()
                                    == Some((GpuLimitKind::MaxMultiviewViewCount, 2, 0))
                    )
                })
        );
        return MultiviewProofOutcome {
            supported: false,
            normalized_max,
            primary_exercised: false,
            shader_without_view_index_exercised: false,
            clear_only_exercised: false,
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

    let four_view_pipeline = multiview_pipeline_with_count(
        "proof.r4.multiview.four-view-realization-limit",
        NO_VIEW_INDEX_WGSL,
        4,
    );
    let four_view_program = context
        .realize_program(four_view_pipeline.program())
        .await
        .unwrap();
    let four_view_layout = context
        .realize_pipeline_layout(four_view_pipeline.layout())
        .await
        .unwrap();
    let realization_error = context
        .realize_render_pipeline(&four_view_pipeline, &four_view_program, &four_view_layout)
        .await
        .expect_err("four-view pipeline must reject against an admitted two-view workload budget");
    assert_eq!(
        realization_error.category(),
        GpuPipelineRealizationErrorCategory::FormatOrAlignmentNotAdmitted
    );

    run_primary_oracle(&context).await;
    run_shader_without_view_index_oracle(&context).await;
    run_clear_only_oracle(&context).await;
    MultiviewProofOutcome {
        supported: true,
        normalized_max,
        primary_exercised: true,
        shader_without_view_index_exercised: true,
        clear_only_exercised: true,
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
    let layered =
        run_layered_multisample_on_adapter(GpuBackendFamily::BrowserWebGpu, None, &census).await;
    assert!(
        !layered.multisample_array_supported && !layered.exercised,
        "pinned BrowserWebGpu must not advertise native-only MultisampleArray"
    );
}

#[test]
fn layered_attachment_boundaries_and_initialization_are_structural() {
    let mut scope = GpuResourceScope::new();
    let resource_label = label("R4 multiview attachment boundaries");
    let texture = scope
        .texture(
            GpuTextureDescriptor::new(
                common("R4 multiview attachment boundaries"),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, WIDTH, HEIGHT, 2)
                    .unwrap(),
                1,
                1,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(&resource_label, [GpuTextureUsage::ColorAttachment]).unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();

    let layered = texture_view(
        &mut scope,
        &texture,
        "R4 multiview boundary layered",
        GpuTextureViewDimension::D2Array,
        0,
        2,
    );
    let discard = GpuRenderColorAttachment::new(
        layered.clone(),
        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
        GpuAttachmentStore::Discard,
        None,
    )
    .expect_err("layered multiview Discard must reject before private realization");
    assert_eq!(discard.cause(), GpuWorkOperationCause::InvalidAttachment);

    let ordinary = texture_view(
        &mut scope,
        &texture,
        "R4 ordinary one-layer discard",
        GpuTextureViewDimension::D2,
        0,
        1,
    );
    GpuRenderColorAttachment::new(
        ordinary,
        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
        GpuAttachmentStore::Discard,
        None,
    )
    .expect("ordinary one-layer Discard must remain valid");

    let one_layer_array = texture_view(
        &mut scope,
        &texture,
        "R4 D2Array one-layer rejection",
        GpuTextureViewDimension::D2Array,
        0,
        1,
    );
    let one_layer = GpuRenderColorAttachment::new(
        one_layer_array,
        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
        GpuAttachmentStore::Store,
        None,
    )
    .expect_err("one-layer D2Array attachment remains outside the first multiview contract");
    assert_eq!(one_layer.cause(), GpuWorkOperationCause::InvalidAttachment);

    let initialized_layer = texture_view(
        &mut scope,
        &texture,
        "R4 partially initialized layer zero",
        GpuTextureViewDimension::D2,
        0,
        1,
    );
    let initialize_one = GpuRenderOperation::new(
        [GpuRenderColorAttachment::new(
            initialized_layer,
            GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.25, 0.5, 0.75, 1.0).unwrap()),
            GpuAttachmentStore::Store,
            None,
        )
        .unwrap()],
        None,
        std::iter::empty::<GpuRenderDraw>(),
        None,
    )
    .unwrap();
    let load = GpuRenderOperation::new(
        [GpuRenderColorAttachment::new(
            layered,
            GpuColorAttachmentLoad::Load,
            GpuAttachmentStore::Store,
            None,
        )
        .unwrap()],
        None,
        [render_draw(multiview_pipeline(
            "proof.r4.multiview.partially-initialized-load",
            NO_VIEW_INDEX_WGSL,
        ))],
        None,
    )
    .unwrap();
    let fragment = GpuWorkFragment::build("R4 multiview partially initialized load", |builder| {
        builder.operation("initialize only selected layer zero", initialize_one)?;
        builder.operation("load both selected multiview layers", load)?;
        Ok(())
    })
    .unwrap();
    let error = GpuPreparedWorkGraph::prepare(
        label("R4 multiview partially initialized load graph"),
        [fragment],
    )
    .expect_err(
        "Load + Store must reject when even one selected multiview layer remains uninitialized",
    );
    assert_eq!(error.cause(), GpuWorkGraphCause::ReadBeforeInitialization);

    let multisampled_label = label("R4 multisampled D2Array admission");
    let multisampled = scope
        .texture(
            GpuTextureDescriptor::new(
                common("R4 multisampled D2Array admission"),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(
                    &multisampled_label,
                    GpuTextureDimension::D2,
                    WIDTH,
                    HEIGHT,
                    2,
                )
                .unwrap(),
                1,
                4,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(&multisampled_label, [GpuTextureUsage::ColorAttachment])
                    .unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .expect("multisampled D2 arrays have a normalized descriptor contract"),
        )
        .unwrap();
    let multisampled_view = texture_view(
        &mut scope,
        &multisampled,
        "R4 multisampled D2Array attachment",
        GpuTextureViewDimension::D2Array,
        0,
        2,
    );
    GpuRenderColorAttachment::new(
        multisampled_view,
        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
        GpuAttachmentStore::Store,
        None,
    )
    .expect("layered multisample attachment uses the canonical checked view");

    let transient_label = label("R4 transient D2Array rejection");
    let transient_error = GpuTextureDescriptor::new(
        common("R4 transient D2Array rejection"),
        GpuTextureDimension::D2,
        GpuTextureExtent::new(&transient_label, GpuTextureDimension::D2, WIDTH, HEIGHT, 2).unwrap(),
        1,
        1,
        GpuTextureFormat::Rgba8Unorm,
        GpuTextureUsages::new(
            &transient_label,
            [
                GpuTextureUsage::ColorAttachment,
                GpuTextureUsage::TransientAttachment,
            ],
        )
        .unwrap(),
        GpuTextureInitialization::Uninitialized,
    )
    .expect_err("layered transient attachments remain outside the accepted transient slice");
    assert_eq!(
        transient_error.cause(),
        GpuResourceDescriptorCause::InvalidExtent
    );

    let depth_label = label("R4 multiview depth discard");
    let depth = scope
        .texture(
            GpuTextureDescriptor::new(
                common("R4 multiview depth discard"),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&depth_label, GpuTextureDimension::D2, WIDTH, HEIGHT, 2)
                    .unwrap(),
                1,
                1,
                GpuTextureFormat::Depth32Float,
                GpuTextureUsages::new(&depth_label, [GpuTextureUsage::DepthStencilAttachment])
                    .unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let depth_range = GpuTextureSubresourceRange::new(
        depth.descriptor().common().label(),
        0,
        1,
        0,
        2,
        GpuTextureAspect::DepthOnly,
    )
    .unwrap();
    let depth_view = scope
        .texture_view(
            GpuTextureViewDescriptor::new(
                common("R4 multiview depth discard view"),
                &depth,
                None,
                GpuTextureViewDimension::D2Array,
                depth_range,
            )
            .unwrap(),
        )
        .unwrap();
    let depth_discard = GpuRenderDepthStencilAttachment::new(
        depth_view,
        Some(
            GpuDepthAttachmentState::new(
                GpuDepthStencilAccess::ReadWrite,
                GpuDepthAttachmentLoad::Clear(GpuDepthClearValue::new(0.5).unwrap()),
                GpuAttachmentStore::Discard,
            )
            .unwrap(),
        ),
        None,
    )
    .expect_err("writable layered depth Discard must reject before private realization");
    assert_eq!(
        depth_discard.cause(),
        GpuWorkOperationCause::InvalidAttachment
    );
}

#[test]
fn vertex_view_index_is_a_supported_multiview_input() {
    let pipeline = multiview_pipeline(
        "proof.r4.multiview.vertex-view-index",
        VERTEX_VIEW_INDEX_WGSL,
    );
    assert!(matches!(
        pipeline.requirements().get(GpuCapabilityFeature::Multiview),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Multiview
        ))
    ));
    assert_eq!(
        pipeline.state().multiview(),
        Some(GpuMultiviewState::new(2).unwrap())
    );
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

#[test]
fn pass_and_pipeline_multiview_state_must_match_exactly() {
    let mut scope = GpuResourceScope::new();
    let texture_label = label("R4 multiview parity texture");
    let texture = scope
        .texture(
            GpuTextureDescriptor::new(
                common("R4 multiview parity texture"),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&texture_label, GpuTextureDimension::D2, WIDTH, HEIGHT, 4)
                    .unwrap(),
                1,
                1,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(&texture_label, [GpuTextureUsage::ColorAttachment]).unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();

    let ordinary_view = texture_view(
        &mut scope,
        &texture,
        "R4 ordinary parity view",
        GpuTextureViewDimension::D2,
        0,
        1,
    );
    let two_view = texture_view(
        &mut scope,
        &texture,
        "R4 two-view parity view",
        GpuTextureViewDimension::D2Array,
        0,
        2,
    );
    let four_view = texture_view(
        &mut scope,
        &texture,
        "R4 four-view parity view",
        GpuTextureViewDimension::D2Array,
        0,
        4,
    );

    let attachment = |view| {
        GpuRenderColorAttachment::new(
            view,
            GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
            GpuAttachmentStore::Store,
            None,
        )
        .unwrap()
    };

    let two_pipeline =
        multiview_pipeline_with_count("proof.r4.multiview.parity-two", NO_VIEW_INDEX_WGSL, 2);
    let four_pipeline =
        multiview_pipeline_with_count("proof.r4.multiview.parity-four", NO_VIEW_INDEX_WGSL, 4);

    let ordinary_with_multiview = GpuRenderOperation::new(
        [attachment(ordinary_view)],
        None,
        [render_draw(two_pipeline.clone())],
        None,
    )
    .expect_err("ordinary pass must reject a multiview pipeline");
    assert_eq!(
        ordinary_with_multiview.cause(),
        GpuWorkOperationCause::InvalidDraw
    );

    let mismatched_count = GpuRenderOperation::new(
        [attachment(four_view)],
        None,
        [render_draw(two_pipeline)],
        None,
    )
    .expect_err("four-view pass must reject a two-view pipeline");
    assert_eq!(mismatched_count.cause(), GpuWorkOperationCause::InvalidDraw);

    let matched = GpuRenderOperation::new(
        [attachment(two_view)],
        None,
        [render_draw(multiview_pipeline_with_count(
            "proof.r4.multiview.parity-matched",
            NO_VIEW_INDEX_WGSL,
            2,
        ))],
        None,
    )
    .expect("equal pass/pipeline multiview state must remain valid");
    assert_eq!(
        matched.signature().multiview(),
        Some(GpuMultiviewState::new(2).unwrap())
    );

    // Keep the four-view descriptor alive in this structural test so both checked
    // pipeline cardinalities are independently constructed.
    assert_eq!(
        four_pipeline.state().multiview(),
        Some(GpuMultiviewState::new(4).unwrap())
    );
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
        assert!(outcome.fully_exercised());
        println!(
            "Multiview Vulkan: EXERCISED (normalized max={})",
            outcome.normalized_max
        );
    } else {
        println!("Multiview Vulkan: UNSUPPORTED");
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires the retained Vulkan software adapter"]
fn layered_multisample_native_execution_matches_normalized_adapter_facts() {
    let census = pollster::block_on(GpuContext::request(descriptor(
        GpuBackendFamily::Vulkan,
        false,
        Some(GpuSoftwareFallbackPolicy::Require),
    )))
    .expect("native Conformance must provide the retained Vulkan fallback adapter");
    let outcome = pollster::block_on(run_layered_multisample_on_adapter(
        GpuBackendFamily::Vulkan,
        Some(GpuSoftwareFallbackPolicy::Require),
        &census,
    ));
    if outcome.multisample_array_supported && outcome.multiview_supported {
        assert!(outcome.exercised);
        if outcome.depth_stencil_supported {
            assert!(outcome.depth_stencil_exercised);
        }
        println!(
            "Layered MSAA Vulkan: EXERCISED (depth={})",
            outcome.depth_stencil_exercised
        );
    } else {
        println!(
            "Layered MSAA Vulkan: UNSUPPORTED (array={}, multiview={})",
            outcome.multisample_array_supported, outcome.multiview_supported
        );
    }
}
