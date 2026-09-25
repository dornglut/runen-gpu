use runen_gpu::*;

#[path = "../support/readback_wait.rs"]
mod readback_wait;

const UNUSED_STORAGE_ARRAY_WGSL: &str = r#"
enable wgpu_binding_array;

struct Value {
    value: u32,
}

@group(0) @binding(0)
var<storage, read> unused_inputs: binding_array<Value, 2>;

@compute @workgroup_size(1)
fn cs_main() {}
"#;

const STORAGE_ARRAY_WGSL: &str = r#"
enable wgpu_binding_array;

struct Value {
    value: u32,
}

@group(0) @binding(0)
var<storage, read> inputs: binding_array<Value, 2>;

@group(0) @binding(1)
var<storage, read_write> output: Value;

@compute @workgroup_size(1)
fn cs_main() {
    output.value = inputs[0].value + inputs[1].value;
}
"#;

const SAMPLED_TEXTURE_ARRAY_WGSL: &str = r#"
enable wgpu_binding_array;

struct Value {
    value: u32,
}

@group(0) @binding(0)
var inputs: binding_array<texture_2d<u32>, 2>;

@group(0) @binding(1)
var<storage, read_write> output: Value;

@compute @workgroup_size(1)
fn cs_main() {
    let left = textureLoad(inputs[0], vec2<i32>(0, 0), 0).x;
    let right = textureLoad(inputs[1], vec2<i32>(0, 0), 0).x;
    output.value = left + right;
}
"#;

const SAMPLER_ARRAY_WGSL: &str = r#"
enable wgpu_binding_array;

struct Value {
    value: u32,
}

@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var input_samplers: binding_array<sampler, 2>;

@group(0) @binding(2)
var<storage, read_write> output: Value;

@compute @workgroup_size(1)
fn cs_main() {
    let first = textureSampleLevel(input_texture, input_samplers[0], vec2<f32>(0.5, 0.5), 0.0).r;
    let second = textureSampleLevel(input_texture, input_samplers[1], vec2<f32>(0.5, 0.5), 0.0).r;
    output.value = u32(round((first + second) * 255.0));
}
"#;

const STORAGE_TEXTURE_ARRAY_WGSL: &str = r#"
enable wgpu_binding_array;

@group(0) @binding(0)
var outputs: binding_array<texture_storage_2d<r32uint, write>, 2>;

@compute @workgroup_size(1)
fn cs_main() {
    textureStore(outputs[0], vec2<i32>(0, 0), vec4<u32>(17u, 0u, 0u, 0u));
    textureStore(outputs[1], vec2<i32>(0, 0), vec4<u32>(25u, 0u, 0u, 0u));
}
"#;

fn label(value: impl AsRef<str>) -> GpuResourceLabel {
    GpuResourceLabel::new(value.as_ref()).unwrap()
}

fn provenance(value: impl AsRef<str>) -> GpuResourceProvenance {
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

fn prepared_texture(
    resources: &mut GpuResourceScope,
    name: &str,
    format: GpuTextureFormat,
    bytes: &[u8],
    usages: impl IntoIterator<Item = GpuTextureUsage>,
) -> (GpuTextureHandle, GpuTextureViewHandle) {
    let texture_label = label(name);
    let extent = GpuTextureExtent::new(&texture_label, GpuTextureDimension::D2, 1, 1, 1).unwrap();
    let prepared = GpuPreparedTextureData::new(
        &texture_label,
        PreparedGpuData::<TransferData>::from_pod_transfer(name, bytes, provenance(name)).unwrap(),
        format,
        extent,
        u32::try_from(bytes.len()).unwrap(),
        0,
    )
    .unwrap();
    let mut usages = usages.into_iter().collect::<Vec<_>>();
    if !usages.contains(&GpuTextureUsage::CopyDestination) {
        usages.push(GpuTextureUsage::CopyDestination);
    }
    let texture = resources
        .texture(
            GpuTextureDescriptor::new(
                common(name),
                GpuTextureDimension::D2,
                extent,
                1,
                1,
                format,
                GpuTextureUsages::new(&texture_label, usages).unwrap(),
                GpuTextureInitialization::Prepared(prepared),
            )
            .unwrap(),
        )
        .unwrap();
    let view = resources
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

fn uninitialized_storage_texture(
    resources: &mut GpuResourceScope,
    name: &str,
) -> (GpuTextureHandle, GpuTextureViewHandle) {
    let texture_label = label(name);
    let extent = GpuTextureExtent::new(&texture_label, GpuTextureDimension::D2, 1, 1, 1).unwrap();
    let texture = resources
        .texture(
            GpuTextureDescriptor::new(
                common(name),
                GpuTextureDimension::D2,
                extent,
                1,
                1,
                GpuTextureFormat::R32Uint,
                GpuTextureUsages::new(
                    &texture_label,
                    [GpuTextureUsage::StorageWrite, GpuTextureUsage::CopySource],
                )
                .unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let view = resources
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

fn filtering_sampler(resources: &mut GpuResourceScope, name: &str) -> GpuSamplerHandle {
    resources
        .sampler(
            GpuSamplerDescriptor::new(
                common(name),
                GpuAddressMode::ClampToEdge,
                GpuAddressMode::ClampToEdge,
                GpuAddressMode::ClampToEdge,
                GpuSamplerFilterState::new(
                    GpuFilterMode::Nearest,
                    GpuFilterMode::Nearest,
                    GpuFilterMode::Nearest,
                    1,
                )
                .unwrap(),
                0.0,
                16.0,
                None,
            )
            .unwrap(),
        )
        .unwrap()
}

fn prepared_u32_buffer(
    resources: &mut GpuResourceScope,
    name: &str,
    value: u32,
    readback: bool,
) -> GpuBufferHandle {
    let data = PreparedGpuData::<TransferData>::ordinary_pod_transfer(name, &[value]).unwrap();
    let mut usages = vec![GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination];
    if readback {
        usages.push(GpuBufferUsage::CopySource);
    }
    resources
        .buffer(
            GpuBufferDescriptor::ordinary_owned(
                name,
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                data.layout().byte_len(),
                usages,
                GpuBufferInitialization::Prepared(data),
            )
            .unwrap(),
        )
        .unwrap()
}

fn unused_storage_array_program() -> GpuProgramDescriptor {
    let [source] = admit_static_wgsl_sources([(
        "proof.fixed-binding-array.unused-storage",
        1,
        UNUSED_STORAGE_ARRAY_WGSL,
    )])
    .unwrap();
    let entry = GpuEntryPointName::new("cs_main").unwrap();
    let program = GpuProgramDescriptor::new(
        source,
        [entry],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();
    assert_eq!(program.interface().bindings().count(), 0);
    program
}

fn storage_buffer_pipeline() -> GpuComputePipelineDescriptor {
    let [source] =
        admit_static_wgsl_sources([("proof.fixed-binding-array.storage", 1, STORAGE_ARRAY_WGSL)])
            .unwrap();
    let entry = GpuEntryPointName::new("cs_main").unwrap();
    let program = GpuProgramDescriptor::new(
        source,
        [entry.clone()],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();
    assert!(matches!(
        program
            .requirements()
            .get(GpuCapabilityFeature::BufferBindingArray),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::BufferBindingArray
        ))
    ));
    assert!(matches!(
        program
            .requirements()
            .get(GpuCapabilityFeature::StorageResourceBindingArray),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::StorageResourceBindingArray
        ))
    ));
    GpuComputePipelineDescriptor::new(program, entry, GpuPipelineConfiguration::default()).unwrap()
}

fn compute_pipeline(
    key: &str,
    wgsl: &str,
    refinements: impl IntoIterator<Item = GpuBindingLayoutRefinement>,
) -> GpuComputePipelineDescriptor {
    let [source] = admit_static_wgsl_sources([(key, 1, wgsl)]).unwrap();
    let entry = GpuEntryPointName::new("cs_main").unwrap();
    let program = GpuProgramDescriptor::new(source, [entry.clone()], refinements).unwrap();
    GpuComputePipelineDescriptor::new(program, entry, GpuPipelineConfiguration::default()).unwrap()
}

fn sampled_texture_array_pipeline() -> GpuComputePipelineDescriptor {
    compute_pipeline(
        "proof.fixed-binding-array.sampled-texture",
        SAMPLED_TEXTURE_ARRAY_WGSL,
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
}

fn sampler_array_pipeline() -> GpuComputePipelineDescriptor {
    compute_pipeline(
        "proof.fixed-binding-array.sampler",
        SAMPLER_ARRAY_WGSL,
        [
            GpuBindingLayoutRefinement::new(GpuBindingKey::try_new(0, 0).unwrap())
                .with_texture_sample_class(GpuTextureSampleClass::FloatFilterable),
            GpuBindingLayoutRefinement::new(GpuBindingKey::try_new(0, 1).unwrap())
                .with_sampler_class(GpuSamplerClass::Filtering),
        ],
    )
}

fn storage_texture_array_pipeline() -> GpuComputePipelineDescriptor {
    compute_pipeline(
        "proof.fixed-binding-array.storage-texture",
        STORAGE_TEXTURE_ARRAY_WGSL,
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
}

fn array_binding(inputs: [&GpuBufferHandle; 2]) -> GpuRuntimeBindingValue {
    GpuRuntimeBindingValue::new(
        GpuBindingKey::try_new(0, 0).unwrap(),
        inputs.into_iter().map(|buffer| {
            GpuRuntimeBindingResource::Buffer(GpuRuntimeBufferBinding::whole(buffer))
        }),
    )
    .unwrap()
}

fn texture_array_binding(
    binding: u32,
    views: [&GpuTextureViewHandle; 2],
) -> GpuRuntimeBindingValue {
    GpuRuntimeBindingValue::new(
        GpuBindingKey::try_new(0, u64::from(binding)).unwrap(),
        views.into_iter().map(|view| {
            GpuRuntimeBindingResource::TextureView(GpuRuntimeTextureViewBinding::new(view.clone()))
        }),
    )
    .unwrap()
}

fn texture_binding(binding: u32, view: &GpuTextureViewHandle) -> GpuRuntimeBindingValue {
    GpuRuntimeBindingValue::new(
        GpuBindingKey::try_new(0, u64::from(binding)).unwrap(),
        [GpuRuntimeBindingResource::TextureView(
            GpuRuntimeTextureViewBinding::new(view.clone()),
        )],
    )
    .unwrap()
}

fn sampler_array_binding(binding: u32, samplers: [&GpuSamplerHandle; 2]) -> GpuRuntimeBindingValue {
    GpuRuntimeBindingValue::new(
        GpuBindingKey::try_new(0, u64::from(binding)).unwrap(),
        samplers
            .into_iter()
            .map(|sampler| GpuRuntimeBindingResource::Sampler(sampler.clone())),
    )
    .unwrap()
}

fn storage_buffer_proof_graph() -> (
    GpuPreparedWorkGraph,
    GpuReadbackId,
    GpuPipelineLayoutDescriptor,
) {
    let mut resources = GpuResourceScope::new();
    let left = prepared_u32_buffer(&mut resources, "fixed-array left", 17, false);
    let right = prepared_u32_buffer(&mut resources, "fixed-array right", 25, false);
    let output = prepared_u32_buffer(&mut resources, "fixed-array output", 0, true);

    let pipeline = storage_buffer_pipeline();
    let layout = pipeline.layout().clone();
    let bindings = GpuRuntimeBindingSet::new(
        layout.clone(),
        [
            array_binding([&left, &right]),
            GpuRuntimeBindingValue::whole_buffer(0, 1, &output),
        ],
    )
    .unwrap();
    let compute = GpuComputeOperation::new(
        pipeline,
        bindings,
        GpuDispatchIntent::direct(GpuDispatchSize::new(1, 1, 1)),
    )
    .unwrap();
    let output_region =
        GpuBufferRegion::new(&output, GpuBufferRange::whole(&output).unwrap()).unwrap();
    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(output_region.into(), readback_id).unwrap();
    let fragment = GpuWorkFragment::build("fixed binding-array storage proof", |work| {
        work.operation("sum fixed storage-buffer array", compute)?;
        work.operation("read fixed-array output", readback)?;
        Ok(())
    })
    .unwrap();
    (
        GpuPreparedWorkGraph::prepare(label("fixed binding-array storage graph"), [fragment])
            .unwrap(),
        readback_id,
        layout,
    )
}

fn sampled_texture_proof_graph() -> (GpuPreparedWorkGraph, GpuReadbackId) {
    let mut resources = GpuResourceScope::new();
    let (_left_texture, left_view) = prepared_texture(
        &mut resources,
        "fixed-array sampled left",
        GpuTextureFormat::R32Uint,
        &17_u32.to_le_bytes(),
        [GpuTextureUsage::Sampled],
    );
    let (_right_texture, right_view) = prepared_texture(
        &mut resources,
        "fixed-array sampled right",
        GpuTextureFormat::R32Uint,
        &25_u32.to_le_bytes(),
        [GpuTextureUsage::Sampled],
    );
    let output = prepared_u32_buffer(&mut resources, "fixed-array sampled output", 0, true);

    let pipeline = sampled_texture_array_pipeline();
    let bindings = GpuRuntimeBindingSet::new(
        pipeline.layout().clone(),
        [
            texture_array_binding(0, [&left_view, &right_view]),
            GpuRuntimeBindingValue::whole_buffer(0, 1, &output),
        ],
    )
    .unwrap();
    let compute = GpuComputeOperation::new(
        pipeline,
        bindings,
        GpuDispatchIntent::direct(GpuDispatchSize::new(1, 1, 1)),
    )
    .unwrap();
    let output_region =
        GpuBufferRegion::new(&output, GpuBufferRange::whole(&output).unwrap()).unwrap();
    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(output_region.into(), readback_id).unwrap();
    let fragment = GpuWorkFragment::build("fixed sampled-texture array proof", |work| {
        work.operation("sum fixed sampled-texture array", compute)?;
        work.operation("read sampled-array output", readback)?;
        Ok(())
    })
    .unwrap();
    (
        GpuPreparedWorkGraph::prepare(label("fixed sampled-texture array graph"), [fragment])
            .unwrap(),
        readback_id,
    )
}

fn sampler_proof_graph() -> (GpuPreparedWorkGraph, GpuReadbackId) {
    let mut resources = GpuResourceScope::new();
    let (_texture, view) = prepared_texture(
        &mut resources,
        "fixed-array sampler texture",
        GpuTextureFormat::Rgba8Unorm,
        &[17_u8, 0, 0, 255],
        [GpuTextureUsage::Sampled],
    );
    let first_sampler = filtering_sampler(&mut resources, "fixed-array sampler first");
    let second_sampler = filtering_sampler(&mut resources, "fixed-array sampler second");
    let output = prepared_u32_buffer(&mut resources, "fixed-array sampler output", 0, true);

    let pipeline = sampler_array_pipeline();
    let bindings = GpuRuntimeBindingSet::new(
        pipeline.layout().clone(),
        [
            texture_binding(0, &view),
            sampler_array_binding(1, [&first_sampler, &second_sampler]),
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
    let output_region =
        GpuBufferRegion::new(&output, GpuBufferRange::whole(&output).unwrap()).unwrap();
    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(output_region.into(), readback_id).unwrap();
    let fragment = GpuWorkFragment::build("fixed sampler array proof", |work| {
        work.operation("sample through fixed sampler array", compute)?;
        work.operation("read sampler-array output", readback)?;
        Ok(())
    })
    .unwrap();
    (
        GpuPreparedWorkGraph::prepare(label("fixed sampler array graph"), [fragment]).unwrap(),
        readback_id,
    )
}

fn storage_texture_proof_graph() -> (GpuPreparedWorkGraph, [GpuReadbackId; 2]) {
    let mut resources = GpuResourceScope::new();
    let (first_texture, first_view) =
        uninitialized_storage_texture(&mut resources, "fixed-array storage texture first");
    let (second_texture, second_view) =
        uninitialized_storage_texture(&mut resources, "fixed-array storage texture second");

    let pipeline = storage_texture_array_pipeline();
    let bindings = GpuRuntimeBindingSet::new(
        pipeline.layout().clone(),
        [texture_array_binding(0, [&first_view, &second_view])],
    )
    .unwrap();
    let compute = GpuComputeOperation::new(
        pipeline,
        bindings,
        GpuDispatchIntent::direct(GpuDispatchSize::new(1, 1, 1)),
    )
    .unwrap();

    let first_region = GpuTextureCopyRegion::new(
        &first_texture,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::Color,
        GpuCopyExtent::new(1, 1, 1).unwrap(),
    )
    .unwrap();
    let second_region = GpuTextureCopyRegion::new(
        &second_texture,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::Color,
        GpuCopyExtent::new(1, 1, 1).unwrap(),
    )
    .unwrap();
    let first_readback = GpuReadbackId::allocate().unwrap();
    let second_readback = GpuReadbackId::allocate().unwrap();
    let fragment = GpuWorkFragment::build("fixed storage-texture array proof", |work| {
        work.operation("write fixed storage-texture array", compute)?;
        work.operation(
            "read first storage-array texture",
            GpuReadbackOperation::new(first_region.into(), first_readback)?,
        )?;
        work.operation(
            "read second storage-array texture",
            GpuReadbackOperation::new(second_region.into(), second_readback)?,
        )?;
        Ok(())
    })
    .unwrap();
    (
        GpuPreparedWorkGraph::prepare(label("fixed storage-texture array graph"), [fragment])
            .unwrap(),
        [first_readback, second_readback],
    )
}

fn context_descriptor(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
    requirements: GpuCapabilityRequirements,
) -> GpuContextDescriptor {
    let mut descriptor = GpuContextDescriptor::new(requirements)
        .with_allowed_backends([backend])
        .with_label("fixed binding-array execution proof");
    if let Some(fallback) = fallback {
        descriptor = descriptor.with_fallback_policy(fallback);
    }
    descriptor
}

fn context_descriptor_with_roles(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
    requirements: GpuCapabilityRequirements,
    roles: impl IntoIterator<Item = (GpuTextureFormat, GpuFormatRole)>,
) -> GpuContextDescriptor {
    let mut descriptor = context_descriptor(backend, fallback, requirements);
    for (format, role) in roles {
        descriptor = descriptor.require_format_role(format, role);
    }
    descriptor
}

async fn execute_u32_buffer_graph(
    context: &GpuContext,
    graph: GpuPreparedWorkGraph,
    readback_id: GpuReadbackId,
    label: &str,
) -> u32 {
    let prepared = context.prepare_submission(graph).await.unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let bytes = readback_wait::wait_for_readback(context, &submission, readback_id, label).await;
    assert_eq!(bytes.as_bytes().len(), 4);
    u32::from_le_bytes(bytes.as_bytes().try_into().unwrap())
}

pub(crate) async fn run_storage_buffer_array_proof(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
) -> bool {
    let baseline = GpuContext::request(context_descriptor(
        backend,
        fallback,
        GpuCapabilityProfile::ComputeBaseline.requirements(),
    ))
    .await
    .expect("retained fixed-array proof requires the requested baseline context");

    if !baseline
        .adapter_facts()
        .supported()
        .supports(GpuCapabilityFeature::BufferBindingArray)
        || !baseline
            .adapter_facts()
            .supported()
            .supports(GpuCapabilityFeature::StorageResourceBindingArray)
    {
        println!("Fixed binding arrays: UNSUPPORTED (storage-buffer array capability absent)");
        return false;
    }

    let unused_program = unused_storage_array_program();
    let error = baseline
        .realize_program(&unused_program)
        .await
        .expect_err("baseline context must reject unused module-global array requirements");
    assert_eq!(
        error.category(),
        GpuProgramBindingRealizationErrorCategory::RequirementNotAdmitted
    );
    let unused_context = GpuContext::request(context_descriptor(
        backend,
        fallback,
        unused_program.requirements().clone(),
    ))
    .await
    .expect("unused module-global array requirements must admit the capability-bearing context");
    unused_context
        .realize_program(&unused_program)
        .await
        .expect(
            "whole-module shader realization must succeed after admitting its exact requirements",
        );

    let (graph, readback_id, layout) = storage_buffer_proof_graph();
    assert!(
        baseline.prepare_submission(graph.clone()).await.is_err(),
        "a baseline context must reject graph-derived fixed-array capabilities"
    );

    let capped = GpuContext::request(
        context_descriptor(backend, fallback, graph.requirements().clone())
            .permit_limit(GpuLimitKind::MaxBindingArrayElementsPerShaderStage, 1),
    )
    .await
    .expect("capability-bearing adapter must admit an explicit one-element workload cap");
    let group = layout.group(0).unwrap();
    let error = capped
        .realize_bind_group_layout(group)
        .await
        .expect_err("a two-element public layout must reject against a one-element workload cap");
    assert_eq!(
        error.category(),
        GpuProgramBindingRealizationErrorCategory::LayoutDescriptorInvalid
    );

    let context = GpuContext::request(context_descriptor(
        backend,
        fallback,
        graph.requirements().clone(),
    ))
    .await
    .expect("graph-derived fixed-array requirements must admit the capability-bearing context");
    assert!(
        context
            .device_facts()
            .workload_budget()
            .limits()
            .max_binding_array_elements_per_shader_stage()
            >= 2
    );

    assert_eq!(
        execute_u32_buffer_graph(
            &context,
            graph,
            readback_id,
            "fixed binding-array storage execution",
        )
        .await,
        42
    );
    println!("Fixed binding arrays: EXERCISED (storage-buffer array + exact readback)");
    true
}

pub(crate) async fn run_sampled_texture_array_proof(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
) -> bool {
    let baseline = GpuContext::request(context_descriptor(
        backend,
        fallback,
        GpuCapabilityProfile::ComputeBaseline.requirements(),
    ))
    .await
    .expect("retained fixed-array proof requires the requested baseline context");
    if !baseline
        .adapter_facts()
        .supported()
        .supports(GpuCapabilityFeature::TextureBindingArray)
        || baseline
            .adapter_facts()
            .supported()
            .format(GpuTextureFormat::R32Uint)
            .is_none_or(|facts| !facts.sampled || !facts.copy_destination)
    {
        println!(
            "Fixed binding arrays: UNSUPPORTED (sampled-texture array capability/format role absent)"
        );
        return false;
    }

    let (graph, readback_id) = sampled_texture_proof_graph();
    let context = GpuContext::request(context_descriptor_with_roles(
        backend,
        fallback,
        graph.requirements().clone(),
        [
            (GpuTextureFormat::R32Uint, GpuFormatRole::Sampled),
            (GpuTextureFormat::R32Uint, GpuFormatRole::CopyDestination),
        ],
    ))
    .await
    .expect("sampled-texture fixed-array requirements must admit the capability-bearing context");
    assert_eq!(
        execute_u32_buffer_graph(
            &context,
            graph,
            readback_id,
            "fixed binding-array sampled-texture execution",
        )
        .await,
        42
    );
    println!("Fixed binding arrays: EXERCISED (sampled-texture array + exact readback)");
    true
}

pub(crate) async fn run_sampler_array_proof(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
) -> bool {
    let baseline = GpuContext::request(context_descriptor(
        backend,
        fallback,
        GpuCapabilityProfile::ComputeBaseline.requirements(),
    ))
    .await
    .expect("retained fixed-array proof requires the requested baseline context");
    if !baseline
        .adapter_facts()
        .supported()
        .supports(GpuCapabilityFeature::TextureBindingArray)
        || baseline
            .adapter_facts()
            .supported()
            .format(GpuTextureFormat::Rgba8Unorm)
            .is_none_or(|facts| !facts.sampled || !facts.filterable || !facts.copy_destination)
    {
        println!("Fixed binding arrays: UNSUPPORTED (sampler array capability/format role absent)");
        return false;
    }

    let (graph, readback_id) = sampler_proof_graph();
    let context = GpuContext::request(context_descriptor_with_roles(
        backend,
        fallback,
        graph.requirements().clone(),
        [
            (GpuTextureFormat::Rgba8Unorm, GpuFormatRole::Sampled),
            (GpuTextureFormat::Rgba8Unorm, GpuFormatRole::Filterable),
            (GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopyDestination),
        ],
    ))
    .await
    .expect("sampler fixed-array requirements must admit the capability-bearing context");
    assert_eq!(
        execute_u32_buffer_graph(
            &context,
            graph,
            readback_id,
            "fixed binding-array sampler execution",
        )
        .await,
        34
    );
    println!("Fixed binding arrays: EXERCISED (sampler array + exact readback)");
    true
}

pub(crate) async fn run_storage_texture_array_proof(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
) -> bool {
    let baseline = GpuContext::request(context_descriptor(
        backend,
        fallback,
        GpuCapabilityProfile::ComputeBaseline.requirements(),
    ))
    .await
    .expect("retained fixed-array proof requires the requested baseline context");
    if !baseline
        .adapter_facts()
        .supported()
        .supports(GpuCapabilityFeature::TextureBindingArray)
        || !baseline
            .adapter_facts()
            .supported()
            .supports(GpuCapabilityFeature::StorageResourceBindingArray)
        || baseline
            .adapter_facts()
            .supported()
            .format(GpuTextureFormat::R32Uint)
            .is_none_or(|facts| !facts.storage_write || !facts.copy_source)
    {
        println!(
            "Fixed binding arrays: UNSUPPORTED (storage-texture array capability/format role absent)"
        );
        return false;
    }

    let (graph, readback_ids) = storage_texture_proof_graph();
    let context = GpuContext::request(context_descriptor_with_roles(
        backend,
        fallback,
        graph.requirements().clone(),
        [
            (GpuTextureFormat::R32Uint, GpuFormatRole::StorageWrite),
            (GpuTextureFormat::R32Uint, GpuFormatRole::CopySource),
        ],
    ))
    .await
    .expect("storage-texture fixed-array requirements must admit the capability-bearing context");
    let prepared = context.prepare_submission(graph).await.unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let first = readback_wait::wait_for_readback(
        &context,
        &submission,
        readback_ids[0],
        "fixed storage-texture array first readback",
    )
    .await;
    let second = readback_wait::wait_for_readback(
        &context,
        &submission,
        readback_ids[1],
        "fixed storage-texture array second readback",
    )
    .await;
    assert_eq!(first.as_bytes().len(), 4);
    assert_eq!(second.as_bytes().len(), 4);
    assert_eq!(u32::from_le_bytes(first.as_bytes().try_into().unwrap()), 17);
    assert_eq!(
        u32::from_le_bytes(second.as_bytes().try_into().unwrap()),
        25
    );
    println!("Fixed binding arrays: EXERCISED (storage-texture array + exact readback)");
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FixedBindingArrayProof {
    pub(crate) storage_buffer: bool,
    pub(crate) sampled_texture: bool,
    pub(crate) sampler: bool,
    pub(crate) storage_texture: bool,
}

pub(crate) async fn run_suite(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
) -> FixedBindingArrayProof {
    FixedBindingArrayProof {
        storage_buffer: run_storage_buffer_array_proof(backend, fallback).await,
        sampled_texture: run_sampled_texture_array_proof(backend, fallback).await,
        sampler: run_sampler_array_proof(backend, fallback).await,
        storage_texture: run_storage_texture_array_proof(backend, fallback).await,
    }
}
