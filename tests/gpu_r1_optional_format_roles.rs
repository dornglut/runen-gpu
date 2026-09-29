use runen_gpu::*;

#[path = "support/readback_wait.rs"]
mod readback_wait;

const FILTER_WGSL: &str = r#"
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var<storage, read_write> output: array<f32, 4>;
@compute @workgroup_size(1)
fn cs_main() {
    let value = textureSampleLevel(source, linear_sampler, vec2<f32>(0.5, 0.5), 0.0);
    output[0] = value.r;
    output[1] = value.g;
    output[2] = value.b;
    output[3] = value.a;
}
"#;

const BGRA_WGSL: &str = r#"
@group(0) @binding(0) var output_image: texture_storage_2d<bgra8unorm, write>;
@compute @workgroup_size(1)
fn cs_main() {
    textureStore(output_image, vec2<i32>(0, 0), vec4<f32>(1.0, 0.0, 0.0, 1.0));
}
"#;

const BLEND_WGSL: &str = r#"
struct VertexOutput { @builtin(position) position: vec4<f32> };
@vertex fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var output: VertexOutput;
    output.position = vec4<f32>(positions[index], 0.0, 1.0);
    return output;
}
@fragment fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.25, 0.125, 0.5, 0.25);
}
"#;

fn label(name: impl AsRef<str>) -> GpuResourceLabel {
    GpuResourceLabel::new(name.as_ref()).unwrap()
}

fn common(name: impl AsRef<str>) -> GpuResourceCommon {
    let name = name.as_ref();
    GpuResourceCommon::owned(
        label(name),
        GpuResourceLifetime::Transient,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        GpuResourceProvenance::new(label(name), None, None),
    )
    .unwrap()
}

fn compute_pipeline(
    key: &str,
    wgsl: &str,
    refinements: impl IntoIterator<Item = GpuBindingLayoutRefinement>,
) -> GpuComputePipelineDescriptor {
    let identity = GpuProgramSourceIdentity::new(
        GpuProgramSourceOwnerId::allocate().unwrap(),
        GpuProgramSourceKey::new(key).unwrap(),
        GpuProgramSourceRevision::try_from_raw(1).unwrap(),
    );
    let mut registry = GpuProgramSourceRegistry::new(2, 16 * 1024).unwrap();
    let source = registry
        .admit_wgsl(
            identity,
            wgsl,
            GpuProgramSourceProvenance::new(key, None).unwrap(),
        )
        .unwrap();
    let entry = GpuEntryPointName::new("cs_main").unwrap();
    let program = GpuProgramDescriptor::new(source, [entry.clone()], refinements).unwrap();
    GpuComputePipelineDescriptor::new(program, entry, GpuPipelineConfiguration::default()).unwrap()
}

fn texture_view(
    scope: &mut GpuResourceScope,
    texture: &GpuTextureHandle,
    name: &str,
) -> GpuTextureViewHandle {
    scope
        .texture_view(
            GpuTextureViewDescriptor::new(
                common(name),
                texture,
                None,
                GpuTextureViewDimension::D2,
                GpuTextureSubresourceRange::whole(texture).unwrap(),
            )
            .unwrap(),
        )
        .unwrap()
}

fn texture_binding(index: u64, view: &GpuTextureViewHandle) -> GpuRuntimeBindingValue {
    GpuRuntimeBindingValue::new(
        GpuBindingKey::try_new(0, index).unwrap(),
        [GpuRuntimeBindingResource::TextureView(
            GpuRuntimeTextureViewBinding::new(view.clone()),
        )],
    )
    .unwrap()
}

fn graph(
    name: &str,
    operation: impl Into<GpuWorkOperation>,
    readback: GpuReadbackOperation,
) -> GpuPreparedWorkGraph {
    let fragment = GpuWorkFragment::build(name, |builder| {
        builder.operation("exercise optional format role", operation)?;
        builder.operation("read optional format result", readback)?;
        Ok(())
    })
    .unwrap();
    GpuPreparedWorkGraph::prepare(label(name), [fragment]).unwrap()
}

fn texture_readback(
    texture: &GpuTextureHandle,
    width: u32,
) -> (GpuReadbackOperation, GpuReadbackId) {
    let id = GpuReadbackId::allocate().unwrap();
    let region = GpuTextureCopyRegion::new(
        texture,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::Color,
        GpuCopyExtent::new(width, 1, 1).unwrap(),
    )
    .unwrap();
    (GpuReadbackOperation::new(region.into(), id).unwrap(), id)
}

async fn execute(
    context: &GpuContext,
    graph: GpuPreparedWorkGraph,
    id: GpuReadbackId,
    name: &str,
) -> GpuReadbackBytes {
    let prepared = context.prepare_submission(graph).await.unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    readback_wait::wait_for_readback(context, &submission, id, name).await
}

async fn filter_oracle(context: &GpuContext) {
    let mut scope = GpuResourceScope::new();
    let name = "optional float32 filter input";
    let extent = GpuTextureExtent::new(&label(name), GpuTextureDimension::D2, 2, 1, 1).unwrap();
    let pixels: [f32; 8] = [0.0, 0.25, 0.5, 1.0, 1.0, 0.75, 0.5, 0.0];
    let prepared = GpuPreparedTextureData::new(
        &label(name),
        PreparedGpuData::<TransferData>::from_pod_transfer(
            name,
            &pixels,
            GpuResourceProvenance::new(label(name), None, None),
        )
        .unwrap(),
        GpuTextureFormat::Rgba32Float,
        extent,
        32,
        0,
    )
    .unwrap();
    let texture = scope
        .texture(
            GpuTextureDescriptor::new(
                common(name),
                GpuTextureDimension::D2,
                extent,
                1,
                1,
                GpuTextureFormat::Rgba32Float,
                GpuTextureUsages::new(
                    &label(name),
                    [GpuTextureUsage::Sampled, GpuTextureUsage::CopyDestination],
                )
                .unwrap(),
                GpuTextureInitialization::Prepared(prepared),
            )
            .unwrap(),
        )
        .unwrap();
    let view = texture_view(&mut scope, &texture, "optional float32 filter view");
    let sampler = scope
        .sampler(
            GpuSamplerDescriptor::new(
                common("optional float32 linear sampler"),
                GpuAddressMode::ClampToEdge,
                GpuAddressMode::ClampToEdge,
                GpuAddressMode::ClampToEdge,
                GpuSamplerFilterState::new(
                    GpuFilterMode::Linear,
                    GpuFilterMode::Linear,
                    GpuFilterMode::Linear,
                    1,
                )
                .unwrap(),
                0.0,
                16.0,
                None,
            )
            .unwrap(),
        )
        .unwrap();
    let zero = [0.0_f32; 4];
    let data = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
        "optional float32 filter output",
        &zero,
    )
    .unwrap();
    let output = scope
        .buffer(
            GpuBufferDescriptor::ordinary_owned(
                "optional float32 filter output",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                data.layout().byte_len(),
                [
                    GpuBufferUsage::Storage,
                    GpuBufferUsage::CopyDestination,
                    GpuBufferUsage::CopySource,
                ],
                GpuBufferInitialization::Prepared(data),
            )
            .unwrap(),
        )
        .unwrap();
    let pipeline = compute_pipeline(
        "optional.float32.filter",
        FILTER_WGSL,
        [
            GpuBindingLayoutRefinement::new(GpuBindingKey::try_new(0, 0).unwrap())
                .with_texture_sample_class(GpuTextureSampleClass::FloatFilterable),
            GpuBindingLayoutRefinement::new(GpuBindingKey::try_new(0, 1).unwrap())
                .with_sampler_class(GpuSamplerClass::Filtering),
        ],
    );
    let bindings = GpuRuntimeBindingSet::new(
        pipeline.layout().clone(),
        [
            texture_binding(0, &view),
            GpuRuntimeBindingValue::new(
                GpuBindingKey::try_new(0, 1).unwrap(),
                [GpuRuntimeBindingResource::Sampler(sampler)],
            )
            .unwrap(),
            GpuRuntimeBindingValue::whole_buffer(0, 2, &output),
        ],
    )
    .unwrap();
    let compute = GpuComputeOperation::new(
        pipeline,
        bindings,
        GpuDispatchIntent::direct(GpuDispatchSize::new(1, 1, 1)),
    )
    .unwrap();
    let id = GpuReadbackId::allocate().unwrap();
    let region = GpuBufferRegion::new(&output, GpuBufferRange::whole(&output).unwrap()).unwrap();
    let readback = GpuReadbackOperation::new(region.into(), id).unwrap();
    let bytes = execute(
        context,
        graph("optional float32 filtering", compute, readback),
        id,
        "optional float32 filtering",
    )
    .await;
    let actual: Vec<f32> = bytes
        .as_bytes()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    assert_eq!(
        actual,
        [0.5, 0.5, 0.5, 0.5],
        "linear float32 sampling must interpolate both texels"
    );
}

async fn bgra_oracle(context: &GpuContext) {
    let mut scope = GpuResourceScope::new();
    let name = "optional BGRA8 storage target";
    let texture = scope
        .texture(
            GpuTextureDescriptor::new(
                common(name),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&label(name), GpuTextureDimension::D2, 1, 1, 1).unwrap(),
                1,
                1,
                GpuTextureFormat::Bgra8Unorm,
                GpuTextureUsages::new(
                    &label(name),
                    [GpuTextureUsage::StorageWrite, GpuTextureUsage::CopySource],
                )
                .unwrap(),
                GpuTextureInitialization::Zeroed,
            )
            .unwrap(),
        )
        .unwrap();
    let view = texture_view(&mut scope, &texture, "optional BGRA8 storage view");
    let pipeline = compute_pipeline("optional.bgra8.storage", BGRA_WGSL, []);
    let bindings =
        GpuRuntimeBindingSet::new(pipeline.layout().clone(), [texture_binding(0, &view)]).unwrap();
    let compute = GpuComputeOperation::new(
        pipeline,
        bindings,
        GpuDispatchIntent::direct(GpuDispatchSize::new(1, 1, 1)),
    )
    .unwrap();
    let (readback, id) = texture_readback(&texture, 1);
    let bytes = execute(
        context,
        graph("optional BGRA8 storage", compute, readback),
        id,
        "optional BGRA8 storage",
    )
    .await;
    assert_eq!(bytes.texture_format(), Some(GpuTextureFormat::Bgra8Unorm));
    assert_eq!(
        bytes.as_bytes(),
        &[0, 0, 255, 255],
        "BGRA storage write must preserve channel order"
    );
}

async fn blend_oracle(context: &GpuContext) {
    let mut scope = GpuResourceScope::new();
    let name = "optional float32 blend target";
    let texture = scope
        .texture(
            GpuTextureDescriptor::new(
                common(name),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&label(name), GpuTextureDimension::D2, 1, 1, 1).unwrap(),
                1,
                1,
                GpuTextureFormat::Rgba32Float,
                GpuTextureUsages::new(
                    &label(name),
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
    let view = texture_view(&mut scope, &texture, "optional float32 blend view");
    let identity = GpuProgramSourceIdentity::new(
        GpuProgramSourceOwnerId::allocate().unwrap(),
        GpuProgramSourceKey::new("optional.float32.blend").unwrap(),
        GpuProgramSourceRevision::try_from_raw(1).unwrap(),
    );
    let mut registry = GpuProgramSourceRegistry::new(2, 16 * 1024).unwrap();
    let source = registry
        .admit_wgsl(
            identity,
            BLEND_WGSL,
            GpuProgramSourceProvenance::new("optional float32 blend", None).unwrap(),
        )
        .unwrap();
    let vertex = GpuEntryPointName::new("vs_main").unwrap();
    let fragment = GpuEntryPointName::new("fs_main").unwrap();
    let program =
        GpuProgramDescriptor::new(source, [vertex.clone(), fragment.clone()], []).unwrap();
    let component = GpuBlendComponent::new(
        GpuBlendFactor::One,
        GpuBlendFactor::One,
        GpuBlendOperation::Add,
    )
    .unwrap();
    let target = GpuColorTargetStateDescriptor::new(
        GpuTextureFormat::Rgba32Float,
        Some(GpuBlendState::new(component, component)),
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
        GpuViewport::new(0.0, 0.0, 1.0, 1.0, 0.0, 1.0).unwrap(),
        GpuScissorRect::new(0, 0, 1, 1).unwrap(),
        GpuBlendConstant::new(0.0, 0.0, 0.0, 0.0).unwrap(),
        0,
    )
    .unwrap();
    let attachment = GpuRenderColorAttachment::new(
        view,
        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.25, 0.375, 0.0, 0.25).unwrap()),
        GpuAttachmentStore::Store,
        None,
    )
    .unwrap();
    let render = GpuRenderOperation::new([attachment], None, [draw], None).unwrap();
    let (readback, id) = texture_readback(&texture, 1);
    let bytes = execute(
        context,
        graph("optional float32 blending", render, readback),
        id,
        "optional float32 blending",
    )
    .await;
    assert_eq!(bytes.texture_format(), Some(GpuTextureFormat::Rgba32Float));
    let actual: Vec<f32> = bytes
        .as_bytes()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    assert_eq!(
        actual,
        [0.5, 0.5, 0.5, 0.5],
        "additive float32 blend must combine clear and fragment values"
    );
}

/// Bits: Rgba32Float filtering, Rgba32Float blending, BGRA8 storage write.
/// Each supported role executes; each absent role remains explicitly UNSUPPORTED.
pub(crate) async fn run_on_adapter(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
    expected_adapter: Option<&GpuAdapterFacts>,
) -> u32 {
    let mut baseline =
        GpuContextDescriptor::new(GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
            .with_allowed_backends([backend])
            .with_label("optional format role baseline");
    if let Some(fallback) = fallback {
        baseline = baseline.with_fallback_policy(fallback);
    }
    let baseline_context = GpuContext::request(baseline.clone())
        .await
        .expect("optional role proof requires baseline context");
    if let Some(expected) = expected_adapter {
        assert_eq!(baseline_context.adapter_facts(), expected);
    }
    let facts = baseline_context.adapter_facts();
    let cases = [
        (
            GpuTextureFormat::Rgba32Float,
            GpuFormatRole::Filterable,
            "float32 filtering",
        ),
        (
            GpuTextureFormat::Rgba32Float,
            GpuFormatRole::Blendable,
            "float32 blending",
        ),
        (
            GpuTextureFormat::Bgra8Unorm,
            GpuFormatRole::StorageWrite,
            "BGRA8 storage write",
        ),
    ];
    let mut mask = 0;
    for (index, (format, role, name)) in cases.into_iter().enumerate() {
        let format_facts = facts.supported().format(format).unwrap();
        let supported = match role {
            GpuFormatRole::Filterable => format_facts.filterable,
            GpuFormatRole::Blendable => format_facts.blendable,
            GpuFormatRole::StorageWrite => format_facts.storage_write,
            _ => unreachable!(),
        };
        if !supported {
            println!("{name}: UNSUPPORTED (normalized adapter format role absent)");
            continue;
        }
        match role {
            GpuFormatRole::Filterable => assert!(
                format_facts.sampled,
                "advertised float32 filtering lacks the sampled proof prerequisite"
            ),
            GpuFormatRole::StorageWrite => assert!(
                format_facts.copy_source,
                "advertised BGRA8 storage write lacks the readback proof prerequisite"
            ),
            GpuFormatRole::Blendable => assert!(
                format_facts.color_attachment,
                "advertised float32 blending lacks the color-attachment prerequisite"
            ),
            _ => {}
        }
        let mut requirements = GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements();
        if role != GpuFormatRole::Blendable {
            requirements
                .insert(GpuCapabilityRequirement::Required(
                    GpuCapabilityFeature::Compute,
                ))
                .unwrap();
        }
        if role == GpuFormatRole::StorageWrite {
            requirements
                .insert(GpuCapabilityRequirement::Required(
                    GpuCapabilityFeature::StorageTexture,
                ))
                .unwrap();
        }
        let mut descriptor = GpuContextDescriptor::new(requirements)
            .with_allowed_backends([backend])
            .with_label("optional format role execution")
            .require_format_role(format, role);
        if let Some(fallback) = fallback {
            descriptor = descriptor.with_fallback_policy(fallback);
        }
        match role {
            GpuFormatRole::Filterable => {
                descriptor = descriptor.require_format_role(format, GpuFormatRole::Sampled)
            }
            GpuFormatRole::Blendable => {
                descriptor = descriptor.require_format_role(format, GpuFormatRole::ColorAttachment)
            }
            GpuFormatRole::StorageWrite => {
                descriptor = descriptor.require_format_role(format, GpuFormatRole::CopySource)
            }
            _ => unreachable!(),
        }
        let context = GpuContext::request(descriptor)
            .await
            .expect("advertised optional format role must admit a device");
        assert_eq!(
            context.adapter_facts(),
            facts,
            "optional role proof must stay on the qualified adapter"
        );
        match role {
            GpuFormatRole::Filterable => filter_oracle(&context).await,
            GpuFormatRole::Blendable => blend_oracle(&context).await,
            GpuFormatRole::StorageWrite => bgra_oracle(&context).await,
            _ => unreachable!(),
        }
        println!("{name}: EXERCISED (admitted role and exact public API readback)");
        mask |= 1 << index;
    }
    mask
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn optional_role_oracle_shaders_admit_without_a_device() {
    let _filter = compute_pipeline(
        "optional.float32.filter.syntax",
        FILTER_WGSL,
        [
            GpuBindingLayoutRefinement::new(GpuBindingKey::try_new(0, 0).unwrap())
                .with_texture_sample_class(GpuTextureSampleClass::FloatFilterable),
            GpuBindingLayoutRefinement::new(GpuBindingKey::try_new(0, 1).unwrap())
                .with_sampler_class(GpuSamplerClass::Filtering),
        ],
    );
    let _bgra = compute_pipeline("optional.bgra8.storage.syntax", BGRA_WGSL, []);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires a retained Vulkan software adapter"]
fn optional_format_roles_vulkan_execution() {
    pollster::block_on(run_on_adapter(
        GpuBackendFamily::Vulkan,
        Some(GpuSoftwareFallbackPolicy::Require),
        None,
    ));
}
