use runen_gpu::*;

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const EXPECTED: [u8; 4] = [255, 0, 255, 255];

const DEPTH_WGSL: &str = r#"
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
    output.position = vec4<f32>(positions[vertex_index], 0.5, 1.0);
    return output;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
}
"#;

fn label(value: impl AsRef<str>) -> GpuResourceLabel {
    GpuResourceLabel::new(value.as_ref()).unwrap()
}

fn provenance(value: impl AsRef<str>) -> GpuResourceProvenance {
    let value = value.as_ref();
    GpuResourceProvenance::new(label(value), None, None)
}

fn common(value: impl AsRef<str>, lifetime: GpuResourceLifetime) -> GpuResourceCommon {
    let value = value.as_ref();
    GpuResourceCommon::owned(
        label(value),
        lifetime,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        provenance(value),
    )
    .unwrap()
}

fn transient_target(
    scope: &mut GpuResourceScope,
    name: &str,
    sample_count: u32,
    lifetime: GpuResourceLifetime,
) -> (GpuTextureHandle, GpuTextureViewHandle) {
    let resource_label = label(name);
    let texture = scope
        .texture(
            GpuTextureDescriptor::new(
                common(name, lifetime),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, WIDTH, HEIGHT, 1)
                    .unwrap(),
                1,
                sample_count,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(
                    &resource_label,
                    [
                        GpuTextureUsage::ColorAttachment,
                        GpuTextureUsage::TransientAttachment,
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
            GpuTextureViewDescriptor::ordinary_full_owned(format!("{name} view"), &texture)
                .unwrap(),
        )
        .unwrap();
    (texture, view)
}

fn ordinary_resolve_target(
    scope: &mut GpuResourceScope,
) -> (GpuTextureHandle, GpuTextureViewHandle) {
    let name = "transient resolve target";
    let resource_label = label(name);
    let texture = scope
        .texture(
            GpuTextureDescriptor::new(
                common(name, GpuResourceLifetime::Transient),
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
            GpuTextureViewDescriptor::ordinary_full_owned(
                "transient resolve target view",
                &texture,
            )
            .unwrap(),
        )
        .unwrap();
    (texture, view)
}

fn clear() -> GpuColorAttachmentLoad {
    GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(1.0, 0.0, 1.0, 1.0).unwrap())
}

pub(crate) fn descriptor(backend: GpuBackendFamily) -> GpuContextDescriptor {
    let mut requirements = GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::DepthAttachment,
        ))
        .unwrap();
    GpuContextDescriptor::new(requirements)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
        .require_format_role(GpuTextureFormat::Depth16Unorm, GpuFormatRole::DepthStencil)
        .with_allowed_backends([backend])
        .with_label("transient attachment retained proof")
}

pub(crate) fn stencil_descriptor(backend: GpuBackendFamily) -> GpuContextDescriptor {
    let mut requirements = GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::DepthAttachment,
        ))
        .unwrap();
    GpuContextDescriptor::new(requirements)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
        .require_format_role(GpuTextureFormat::Stencil8, GpuFormatRole::DepthStencil)
        .with_allowed_backends([backend])
        .with_label("transient Stencil8 retained proof")
}

pub(crate) fn stencil_supported(context: &GpuContext) -> bool {
    context
        .adapter_facts()
        .supported()
        .format(GpuTextureFormat::Stencil8)
        .is_some_and(|facts| facts.depth_stencil)
}

pub(crate) fn graph() -> (GpuPreparedWorkGraph, GpuReadbackId) {
    let mut scope = GpuResourceScope::new();
    let (_single_texture, single_view) = transient_target(
        &mut scope,
        "single-sample transient attachment",
        1,
        GpuResourceLifetime::Retained,
    );
    let (_msaa_texture, msaa_view) = transient_target(
        &mut scope,
        "multisample transient attachment",
        4,
        GpuResourceLifetime::Transient,
    );
    let (resolve_texture, resolve_view) = ordinary_resolve_target(&mut scope);

    let single = GpuRenderOperation::new(
        [
            GpuRenderColorAttachment::new(single_view, clear(), GpuAttachmentStore::Discard, None)
                .unwrap(),
        ],
        None,
        [],
        None,
    )
    .unwrap();

    let resolve = GpuMultisampleResolveTarget::new(resolve_view).unwrap();
    let multisample = GpuRenderOperation::new(
        [GpuRenderColorAttachment::new(
            msaa_view,
            clear(),
            GpuAttachmentStore::Discard,
            Some(resolve),
        )
        .unwrap()],
        None,
        [],
        None,
    )
    .unwrap();

    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(
        GpuTextureCopyRegion::new(
            &resolve_texture,
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

    let fragment = GpuWorkFragment::build("transient attachment retained proof", |builder| {
        builder.operation("single-sample transient clear discard", single)?;
        builder.operation("multisample transient resolve", multisample)?;
        builder.operation("read ordinary resolve target", readback)?;
        Ok(())
    })
    .unwrap();

    (
        GpuPreparedWorkGraph::prepare(
            label("transient attachment retained proof graph"),
            [fragment],
        )
        .unwrap(),
        readback_id,
    )
}

fn ordinary_color_target(
    scope: &mut GpuResourceScope,
    name: &str,
) -> (GpuTextureHandle, GpuTextureViewHandle) {
    let resource_label = label(name);
    let texture = scope
        .texture(
            GpuTextureDescriptor::new(
                common(name, GpuResourceLifetime::Transient),
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
            GpuTextureViewDescriptor::ordinary_full_owned(format!("{name} view"), &texture).unwrap(),
        )
        .unwrap();
    (texture, view)
}

fn transient_depth_target(scope: &mut GpuResourceScope) -> GpuTextureViewHandle {
    let name = "transient depth attachment";
    let resource_label = label(name);
    let texture = scope
        .texture(
            GpuTextureDescriptor::new(
                common(name, GpuResourceLifetime::Retained),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, WIDTH, HEIGHT, 1)
                    .unwrap(),
                1,
                1,
                GpuTextureFormat::Depth16Unorm,
                GpuTextureUsages::new(
                    &resource_label,
                    [
                        GpuTextureUsage::DepthStencilAttachment,
                        GpuTextureUsage::TransientAttachment,
                    ],
                )
                .unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    scope
        .texture_view(
            GpuTextureViewDescriptor::ordinary_full_owned("transient depth attachment view", &texture)
                .unwrap(),
        )
        .unwrap()
}

fn depth_pipeline() -> GpuRenderPipelineDescriptor {
    let identity = GpuProgramSourceIdentity::new(
        GpuProgramSourceOwnerId::allocate().unwrap(),
        GpuProgramSourceKey::new("proof.transient.depth").unwrap(),
        GpuProgramSourceRevision::try_from_raw(1).unwrap(),
    );
    let mut sources = GpuProgramSourceRegistry::new(2, 4096).unwrap();
    let source = sources
        .admit_wgsl(
            identity,
            DEPTH_WGSL,
            GpuProgramSourceProvenance::new("transient depth retained proof", None).unwrap(),
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
        Some(GpuDepthStateDescriptor::new(
            true,
            GpuCompareFunction::Less,
        )),
        None,
        GpuDepthBiasState::default(),
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

fn depth_draw() -> GpuRenderDraw {
    let pipeline = depth_pipeline();
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

pub(crate) fn depth_graph() -> (GpuPreparedWorkGraph, GpuReadbackId) {
    let mut scope = GpuResourceScope::new();
    let (color, color_view) = ordinary_color_target(&mut scope, "transient depth observable color");
    let depth_view = transient_depth_target(&mut scope);

    let color_attachment = GpuRenderColorAttachment::new(
        color_view,
        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
        GpuAttachmentStore::Store,
        None,
    )
    .unwrap();
    let depth_attachment = GpuRenderDepthStencilAttachment::new(
        depth_view,
        Some(
            GpuDepthAttachmentState::new(
                GpuDepthStencilAccess::ReadWrite,
                GpuDepthAttachmentLoad::Clear(GpuDepthClearValue::new(0.75).unwrap()),
                GpuAttachmentStore::Discard,
            )
            .unwrap(),
        ),
        None,
    )
    .unwrap();
    let render = GpuRenderOperation::new(
        [color_attachment],
        Some(depth_attachment),
        [depth_draw()],
        None,
    )
    .unwrap();

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
    let fragment = GpuWorkFragment::build("transient depth retained proof", |builder| {
        builder.operation("draw through transient depth", render)?;
        builder.operation("read transient depth observable color", readback)?;
        Ok(())
    })
    .unwrap();
    (
        GpuPreparedWorkGraph::prepare(label("transient depth retained proof graph"), [fragment])
            .unwrap(),
        readback_id,
    )
}

pub(crate) fn stencil_graph() -> (GpuPreparedWorkGraph, GpuReadbackId) {
    let mut scope = GpuResourceScope::new();
    let stencil_name = "transient stencil attachment";
    let stencil_label = label(stencil_name);
    let stencil = scope
        .texture(
            GpuTextureDescriptor::new(
                common(stencil_name, GpuResourceLifetime::Retained),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&stencil_label, GpuTextureDimension::D2, WIDTH, HEIGHT, 1)
                    .unwrap(),
                1,
                1,
                GpuTextureFormat::Stencil8,
                GpuTextureUsages::new(
                    &stencil_label,
                    [
                        GpuTextureUsage::DepthStencilAttachment,
                        GpuTextureUsage::TransientAttachment,
                    ],
                )
                .unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let stencil_view = scope
        .texture_view(
            GpuTextureViewDescriptor::ordinary_full_owned(
                "transient stencil attachment view",
                &stencil,
            )
            .unwrap(),
        )
        .unwrap();
    let stencil_attachment = GpuRenderDepthStencilAttachment::new(
        stencil_view,
        None,
        Some(
            GpuStencilAttachmentState::new(
                GpuDepthStencilAccess::ReadWrite,
                GpuStencilAttachmentLoad::Clear(GpuStencilClearValue::new(7).unwrap()),
                GpuAttachmentStore::Discard,
            )
            .unwrap(),
        ),
    )
    .unwrap();
    let stencil_render =
        GpuRenderOperation::new([], Some(stencil_attachment), [], None).unwrap();

    let (color, color_view) = ordinary_color_target(&mut scope, "transient stencil terminal color");
    let terminal_render = GpuRenderOperation::new(
        [GpuRenderColorAttachment::new(
            color_view,
            clear(),
            GpuAttachmentStore::Store,
            None,
        )
        .unwrap()],
        None,
        [],
        None,
    )
    .unwrap();
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

    let fragment = GpuWorkFragment::build("transient stencil retained proof", |builder| {
        builder.operation("clear discard transient stencil", stencil_render)?;
        builder.operation("clear terminal color", terminal_render)?;
        builder.operation("read transient stencil terminal color", readback)?;
        Ok(())
    })
    .unwrap();
    (
        GpuPreparedWorkGraph::prepare(label("transient stencil retained proof graph"), [fragment])
            .unwrap(),
        readback_id,
    )
}

pub(crate) fn assert_depth_color(bytes: &GpuReadbackBytes) {
    assert_resolved(bytes);
}

pub(crate) fn assert_stencil_terminal(bytes: &GpuReadbackBytes) {
    assert_resolved(bytes);
}

pub(crate) fn assert_resolved(bytes: &GpuReadbackBytes) {
    assert_eq!(bytes.texture_format(), Some(GpuTextureFormat::Rgba8Unorm));
    assert_eq!(
        bytes.as_bytes().len(),
        usize::try_from(WIDTH * HEIGHT * 4).unwrap()
    );
    for pixel in bytes.as_bytes().chunks_exact(4) {
        assert_eq!(pixel, EXPECTED);
    }
}
