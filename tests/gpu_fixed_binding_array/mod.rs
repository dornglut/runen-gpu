use runen_gpu::*;

#[path = "../support/readback_wait.rs"]
pub(crate) mod readback_wait;

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

const UNUSED_SAMPLED_TEXTURE_ARRAY_WGSL: &str = r#"
enable wgpu_binding_array;

@group(0) @binding(0)
var unused_textures: binding_array<texture_2d<u32>, 2>;

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
    output.value = inputs[0].value * 1000u + inputs[1].value;
}
"#;

const UNIFORM_BUFFER_ARRAY_WGSL: &str = r#"
enable wgpu_binding_array;

struct UniformValue {
    value: vec4<u32>,
}

struct OutputValue {
    value: u32,
}

@group(0) @binding(0)
var<uniform> inputs: binding_array<UniformValue, 2>;

@group(0) @binding(1)
var<storage, read_write> output: OutputValue;

@compute @workgroup_size(1)
fn cs_main() {
    output.value = inputs[0].value.x * 1000u + inputs[1].value.x;
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
    output.value = left * 1000u + right;
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
    let clamped = textureSampleLevel(input_texture, input_samplers[0], vec2<f32>(1.25, 0.5), 0.0).r;
    let repeated = textureSampleLevel(input_texture, input_samplers[1], vec2<f32>(1.25, 0.5), 0.0).r;
    let clamped_byte = u32(round(clamped * 255.0));
    let repeated_byte = u32(round(repeated * 255.0));
    output.value = clamped_byte * 1000u + repeated_byte;
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

pub(crate) fn label(value: impl AsRef<str>) -> GpuResourceLabel {
    GpuResourceLabel::new(value.as_ref()).unwrap()
}

pub(crate) fn provenance(value: impl AsRef<str>) -> GpuResourceProvenance {
    GpuResourceProvenance::new(label(value), None, None)
}

pub(crate) fn common(value: impl AsRef<str>) -> GpuResourceCommon {
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

pub(crate) fn prepared_texture(
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

pub(crate) fn sampler_distinguishing_texture(
    resources: &mut GpuResourceScope,
) -> (GpuTextureHandle, GpuTextureViewHandle) {
    let name = "fixed-array sampler texture";
    let texture_label = label(name);
    let extent = GpuTextureExtent::new(&texture_label, GpuTextureDimension::D2, 2, 1, 1).unwrap();
    let pixels = [17_u8, 0, 0, 255, 101, 0, 0, 255];
    let prepared = GpuPreparedTextureData::new(
        &texture_label,
        PreparedGpuData::<TransferData>::from_pod_transfer(name, &pixels, provenance(name))
            .unwrap(),
        GpuTextureFormat::Rgba8Unorm,
        extent,
        8,
        0,
    )
    .unwrap();
    let texture = resources
        .texture(
            GpuTextureDescriptor::new(
                common(name),
                GpuTextureDimension::D2,
                extent,
                1,
                1,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(
                    &texture_label,
                    [GpuTextureUsage::Sampled, GpuTextureUsage::CopyDestination],
                )
                .unwrap(),
                GpuTextureInitialization::Prepared(prepared),
            )
            .unwrap(),
        )
        .unwrap();
    let view = resources
        .texture_view(
            GpuTextureViewDescriptor::new(
                common("fixed-array sampler texture view"),
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

pub(crate) fn zeroed_storage_texture(
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
                GpuTextureInitialization::Zeroed,
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

pub(crate) fn filtering_sampler(
    resources: &mut GpuResourceScope,
    name: &str,
    address_u: GpuAddressMode,
) -> GpuSamplerHandle {
    resources
        .sampler(
            GpuSamplerDescriptor::new(
                common(name),
                address_u,
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
        .unwrap()
}

pub(crate) fn prepared_u32_buffer(
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

fn prepared_uniform_buffer(
    resources: &mut GpuResourceScope,
    name: &str,
    value: u32,
) -> GpuBufferHandle {
    let data =
        PreparedGpuData::<TransferData>::ordinary_pod_transfer(name, &[value, 0_u32, 0_u32, 0_u32])
            .unwrap();
    resources
        .buffer(
            GpuBufferDescriptor::ordinary_owned(
                name,
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                data.layout().byte_len(),
                [GpuBufferUsage::Uniform, GpuBufferUsage::CopyDestination],
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

fn unused_sampled_texture_array_program() -> GpuProgramDescriptor {
    let [source] = admit_static_wgsl_sources([(
        "proof.fixed-binding-array.unused-sampled-texture",
        1,
        UNUSED_SAMPLED_TEXTURE_ARRAY_WGSL,
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

fn uniform_buffer_array_pipeline() -> GpuComputePipelineDescriptor {
    compute_pipeline(
        "proof.fixed-binding-array.uniform-buffer",
        UNIFORM_BUFFER_ARRAY_WGSL,
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
}

pub(crate) fn compute_pipeline(
    key: &'static str,
    wgsl: &'static str,
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

pub(crate) fn array_binding(inputs: [&GpuBufferHandle; 2]) -> GpuRuntimeBindingValue {
    GpuRuntimeBindingValue::new(
        GpuBindingKey::try_new(0, 0).unwrap(),
        inputs.into_iter().map(|buffer| {
            GpuRuntimeBindingResource::Buffer(GpuRuntimeBufferBinding::whole(buffer))
        }),
    )
    .unwrap()
}

fn uniform_array_binding(inputs: [&GpuBufferHandle; 2]) -> GpuRuntimeBindingValue {
    GpuRuntimeBindingValue::new(
        GpuBindingKey::try_new(0, 0).unwrap(),
        inputs.into_iter().map(|buffer| {
            GpuRuntimeBindingResource::Buffer(GpuRuntimeBufferBinding::whole(buffer))
        }),
    )
    .unwrap()
}

pub(crate) fn texture_array_binding(
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

pub(crate) fn texture_binding(binding: u32, view: &GpuTextureViewHandle) -> GpuRuntimeBindingValue {
    GpuRuntimeBindingValue::new(
        GpuBindingKey::try_new(0, u64::from(binding)).unwrap(),
        [GpuRuntimeBindingResource::TextureView(
            GpuRuntimeTextureViewBinding::new(view.clone()),
        )],
    )
    .unwrap()
}

pub(crate) fn sampler_array_binding(
    binding: u32,
    samplers: [&GpuSamplerHandle; 2],
) -> GpuRuntimeBindingValue {
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

fn uniform_buffer_proof_graph() -> (GpuPreparedWorkGraph, GpuReadbackId) {
    let mut resources = GpuResourceScope::new();
    let left = prepared_uniform_buffer(&mut resources, "fixed-array uniform left", 17);
    let right = prepared_uniform_buffer(&mut resources, "fixed-array uniform right", 25);
    let output = prepared_u32_buffer(&mut resources, "fixed-array uniform output", 0, true);

    let pipeline = uniform_buffer_array_pipeline();
    let bindings = GpuRuntimeBindingSet::new(
        pipeline.layout().clone(),
        [
            uniform_array_binding([&left, &right]),
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
    let fragment = GpuWorkFragment::build("fixed uniform-buffer array proof", |work| {
        work.operation("sum fixed uniform-buffer array", compute)?;
        work.operation("read uniform-array output", readback)?;
        Ok(())
    })
    .unwrap();
    (
        GpuPreparedWorkGraph::prepare(label("fixed uniform-buffer array graph"), [fragment])
            .unwrap(),
        readback_id,
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
    let (_texture, view) = sampler_distinguishing_texture(&mut resources);
    let first_sampler = filtering_sampler(
        &mut resources,
        "fixed-array sampler clamp-to-edge",
        GpuAddressMode::ClampToEdge,
    );
    let second_sampler = filtering_sampler(
        &mut resources,
        "fixed-array sampler repeat",
        GpuAddressMode::Repeat,
    );
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
        zeroed_storage_texture(&mut resources, "fixed-array storage texture first");
    let (second_texture, second_view) =
        zeroed_storage_texture(&mut resources, "fixed-array storage texture second");

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
    let first_readback_operation =
        GpuReadbackOperation::new(first_region.into(), first_readback).unwrap();
    let second_readback_operation =
        GpuReadbackOperation::new(second_region.into(), second_readback).unwrap();
    let fragment = GpuWorkFragment::build("fixed storage-texture array proof", |work| {
        work.operation("write fixed storage-texture array", compute)?;
        work.operation("read first storage-array texture", first_readback_operation)?;
        work.operation(
            "read second storage-array texture",
            second_readback_operation,
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

pub(crate) fn context_descriptor(
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

pub(crate) fn context_descriptor_with_roles(
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

pub(crate) async fn request_proof_context(
    descriptor: GpuContextDescriptor,
    expected_adapter: Option<&GpuAdapterFacts>,
    expectation: &str,
) -> GpuContext {
    let context = GpuContext::request(descriptor).await.expect(expectation);
    if let Some(expected_adapter) = expected_adapter {
        assert_eq!(
            context.adapter_facts(),
            expected_adapter,
            "retained fixed-array proof must stay on the qualified adapter"
        );
    }
    context
}

fn assert_ordinary_workload_array_limits_zero(context: &GpuContext) {
    let limits = context.device_facts().workload_budget().limits();
    assert_eq!(
        limits.max_binding_array_elements_per_shader_stage(),
        0,
        "ordinary workloads must not inflate the general fixed-array budget"
    );
    assert_eq!(
        limits.max_binding_array_sampler_elements_per_shader_stage(),
        0,
        "ordinary workloads must not inflate the sampler fixed-array budget"
    );
}

fn assert_typed_array_budget_rejection(
    error: &GpuSubmissionPreparationError,
    expected_label: &str,
) {
    assert_eq!(
        error.kind(),
        GpuSubmissionPreparationErrorKind::WorkNotAdmitted
    );
    let Some(GpuWorkNotAdmittedSource::ProgramContract(source)) = error.work_not_admitted_source()
    else {
        panic!("fixed-array budget rejection must retain its typed program-contract source");
    };
    assert_eq!(source.label(), expected_label);
}

pub(crate) async fn execute_u32_buffer_graph(
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
    expected_adapter: Option<&GpuAdapterFacts>,
) -> bool {
    let baseline = request_proof_context(
        context_descriptor(
            backend,
            fallback,
            GpuCapabilityProfile::ComputeBaseline.requirements(),
        ),
        expected_adapter,
        "retained fixed-array proof requires the requested baseline context",
    )
    .await;
    assert_ordinary_workload_array_limits_zero(&baseline);

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
    let unused_context = request_proof_context(
        context_descriptor(backend, fallback, unused_program.requirements().clone()),
        expected_adapter,
        "unused module-global array requirements must admit the capability-bearing context",
    )
    .await;
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

    let capped = request_proof_context(
        context_descriptor(backend, fallback, graph.requirements().clone())
            .permit_limit(GpuLimitKind::MaxBindingArrayElementsPerShaderStage, 1),
        expected_adapter,
        "capability-bearing adapter must admit an explicit one-element workload cap",
    )
    .await;
    let group = layout.group(0).unwrap();
    let error = capped
        .realize_bind_group_layout(group)
        .await
        .expect_err("a two-element public layout must reject against a one-element workload cap");
    assert_eq!(
        error.category(),
        GpuProgramBindingRealizationErrorCategory::LayoutDescriptorInvalid
    );

    let one_element_group = |group: u32| {
        let declaration = GpuBindingDeclaration::new(
            GpuBindingKey::try_new(u64::from(group), 0).unwrap(),
            GpuShaderStages::one(GpuShaderStage::Compute),
            GpuBindingKind::storage_buffer(GpuStorageBufferAccess::ReadOnly, false, None),
            core::num::NonZeroU32::new(1),
            format!("fixed-array aggregate group {group}"),
            GpuBindingProvenance::new("fixed-binding-array-public-aggregate-proof", None).unwrap(),
        )
        .unwrap();
        GpuBindGroupLayoutDescriptor::new(group, [declaration]).unwrap()
    };
    let first_group = one_element_group(0);
    let second_group = one_element_group(1);
    capped
        .realize_bind_group_layout(&first_group)
        .await
        .expect("each one-element group must fit the one-element per-stage budget");
    capped
        .realize_bind_group_layout(&second_group)
        .await
        .expect("each one-element group must fit the one-element per-stage budget");
    let aggregate_layout = GpuPipelineLayoutDescriptor::new([first_group, second_group]).unwrap();
    let error = capped
        .realize_pipeline_layout(&aggregate_layout)
        .await
        .expect_err(
            "two individually admissible groups must reject when their same-stage array demand sums above the pipeline budget",
        );
    assert_eq!(
        error.category(),
        GpuProgramBindingRealizationErrorCategory::LayoutDescriptorInvalid
    );

    let error = capped
        .prepare_submission(graph.clone())
        .await
        .expect_err("over-budget selected work must reject before private realization");
    assert_typed_array_budget_rejection(&error, "binding-array elements");

    let context = request_proof_context(
        context_descriptor(backend, fallback, graph.requirements().clone()),
        expected_adapter,
        "graph-derived fixed-array requirements must admit the capability-bearing context",
    )
    .await;
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
        17_025
    );
    println!("Fixed binding arrays: EXERCISED (storage-buffer array + exact readback)");
    true
}

pub(crate) async fn run_uniform_buffer_array_proof(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
    expected_adapter: Option<&GpuAdapterFacts>,
) -> bool {
    let baseline = request_proof_context(
        context_descriptor(
            backend,
            fallback,
            GpuCapabilityProfile::ComputeBaseline.requirements(),
        ),
        expected_adapter,
        "retained fixed-array proof requires the requested baseline context",
    )
    .await;
    if !baseline
        .adapter_facts()
        .supported()
        .supports(GpuCapabilityFeature::BufferBindingArray)
        || !baseline
            .adapter_facts()
            .supported()
            .supports(GpuCapabilityFeature::UniformBufferBindingArray)
    {
        println!("Fixed binding arrays: UNSUPPORTED (uniform-buffer array capability absent)");
        return false;
    }

    let (graph, readback_id) = uniform_buffer_proof_graph();
    let context = request_proof_context(
        context_descriptor(backend, fallback, graph.requirements().clone()),
        expected_adapter,
        "uniform-buffer fixed-array requirements must admit the capability-bearing context",
    )
    .await;
    assert_eq!(
        execute_u32_buffer_graph(
            &context,
            graph,
            readback_id,
            "fixed binding-array uniform-buffer execution",
        )
        .await,
        17_025
    );
    println!("Fixed binding arrays: EXERCISED (uniform-buffer array + exact readback)");
    true
}

pub(crate) async fn run_sampled_texture_array_proof(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
    expected_adapter: Option<&GpuAdapterFacts>,
) -> bool {
    let baseline = request_proof_context(
        context_descriptor(
            backend,
            fallback,
            GpuCapabilityProfile::ComputeBaseline.requirements(),
        ),
        expected_adapter,
        "retained fixed-array proof requires the requested baseline context",
    )
    .await;
    if !baseline
        .adapter_facts()
        .supported()
        .supports(GpuCapabilityFeature::TextureBindingArray)
    {
        println!("Fixed binding arrays: UNSUPPORTED (sampled-texture array capability absent)");
        return false;
    }

    let unused_program = unused_sampled_texture_array_program();
    let error = baseline
        .realize_program(&unused_program)
        .await
        .expect_err("baseline context must reject unused sampled-texture array requirements");
    assert_eq!(
        error.category(),
        GpuProgramBindingRealizationErrorCategory::RequirementNotAdmitted
    );
    let unused_context = request_proof_context(
        context_descriptor(backend, fallback, unused_program.requirements().clone()),
        expected_adapter,
        "unused sampled-texture array requirements must admit the capability-bearing context",
    )
    .await;
    assert!(
        unused_context
            .device_facts()
            .is_enabled(GpuCapabilityFeature::TextureBindingArray),
        "module-global sampled-texture array requirement must be enabled before shader realization"
    );
    unused_context
        .realize_program(&unused_program)
        .await
        .expect("unused sampled-texture fixed array must realize after admitting its requirement");

    let (graph, readback_id) = sampled_texture_proof_graph();
    let capped = request_proof_context(
        context_descriptor(backend, fallback, graph.requirements().clone())
            .permit_limit(GpuLimitKind::MaxBindingArrayElementsPerShaderStage, 1),
        expected_adapter,
        "sampled-texture array capability must admit the focused one-element workload cap",
    )
    .await;
    let error = capped
        .prepare_submission(graph.clone())
        .await
        .expect_err("sampled fixed array must reject above the admitted array-element budget");
    assert_typed_array_budget_rejection(&error, "binding-array elements");

    if baseline
        .adapter_facts()
        .supported()
        .format(GpuTextureFormat::R32Uint)
        .is_none_or(|facts| !facts.sampled || !facts.copy_destination)
    {
        println!(
            "Fixed binding arrays: UNSUPPORTED (sampled-texture execution format role absent)"
        );
        return false;
    }
    let context = request_proof_context(
        context_descriptor_with_roles(
            backend,
            fallback,
            graph.requirements().clone(),
            [
                (GpuTextureFormat::R32Uint, GpuFormatRole::Sampled),
                (GpuTextureFormat::R32Uint, GpuFormatRole::CopyDestination),
            ],
        ),
        expected_adapter,
        "sampled-texture fixed-array requirements must admit the capability-bearing context",
    )
    .await;
    assert_eq!(
        execute_u32_buffer_graph(
            &context,
            graph,
            readback_id,
            "fixed binding-array sampled-texture execution",
        )
        .await,
        17_025
    );
    println!("Fixed binding arrays: EXERCISED (sampled-texture array + exact readback)");
    true
}

pub(crate) async fn run_sampler_array_proof(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
    expected_adapter: Option<&GpuAdapterFacts>,
) -> bool {
    let baseline = request_proof_context(
        context_descriptor(
            backend,
            fallback,
            GpuCapabilityProfile::ComputeBaseline.requirements(),
        ),
        expected_adapter,
        "retained fixed-array proof requires the requested baseline context",
    )
    .await;
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
    let capped = request_proof_context(
        context_descriptor(backend, fallback, graph.requirements().clone()).permit_limit(
            GpuLimitKind::MaxBindingArraySamplerElementsPerShaderStage,
            1,
        ),
        expected_adapter,
        "sampler fixed-array capability must admit the focused one-element sampler cap",
    )
    .await;
    let error = capped
        .prepare_submission(graph.clone())
        .await
        .expect_err("sampler fixed array must reject above the admitted sampler-array budget");
    assert_typed_array_budget_rejection(&error, "binding-array sampler elements");

    let context = request_proof_context(
        context_descriptor_with_roles(
            backend,
            fallback,
            graph.requirements().clone(),
            [
                (GpuTextureFormat::Rgba8Unorm, GpuFormatRole::Sampled),
                (GpuTextureFormat::Rgba8Unorm, GpuFormatRole::Filterable),
                (GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopyDestination),
            ],
        ),
        expected_adapter,
        "sampler fixed-array requirements must admit the capability-bearing context",
    )
    .await;
    assert_eq!(
        execute_u32_buffer_graph(
            &context,
            graph,
            readback_id,
            "fixed binding-array sampler execution",
        )
        .await,
        101_017
    );
    println!("Fixed binding arrays: EXERCISED (sampler array + exact readback)");
    true
}

pub(crate) async fn run_storage_texture_array_proof(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
    expected_adapter: Option<&GpuAdapterFacts>,
) -> bool {
    let baseline = request_proof_context(
        context_descriptor(
            backend,
            fallback,
            GpuCapabilityProfile::ComputeBaseline.requirements(),
        ),
        expected_adapter,
        "retained fixed-array proof requires the requested baseline context",
    )
    .await;
    if !baseline
        .adapter_facts()
        .supported()
        .supports(GpuCapabilityFeature::TextureBindingArray)
        || !baseline
            .adapter_facts()
            .supported()
            .supports(GpuCapabilityFeature::StorageResourceBindingArray)
        || !baseline
            .adapter_facts()
            .supported()
            .supports(GpuCapabilityFeature::StorageTexture)
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
    let context = request_proof_context(
        context_descriptor_with_roles(
            backend,
            fallback,
            graph.requirements().clone(),
            [
                (GpuTextureFormat::R32Uint, GpuFormatRole::StorageWrite),
                (GpuTextureFormat::R32Uint, GpuFormatRole::CopySource),
            ],
        ),
        expected_adapter,
        "storage-texture fixed-array requirements must admit the capability-bearing context",
    )
    .await;
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

#[allow(dead_code)] // Shared proof module is compiled into binaries that use different retained entry points.
pub(crate) fn assert_contract_suite_authors_all_resource_families() {
    let unused = unused_storage_array_program();
    assert_eq!(unused.interface().bindings().count(), 0);
    assert!(matches!(
        unused
            .requirements()
            .get(GpuCapabilityFeature::BufferBindingArray),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::BufferBindingArray
        ))
    ));
    assert!(matches!(
        unused
            .requirements()
            .get(GpuCapabilityFeature::StorageResourceBindingArray),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::StorageResourceBindingArray
        ))
    ));

    let unused_sampled = unused_sampled_texture_array_program();
    assert_eq!(unused_sampled.interface().bindings().count(), 0);
    assert!(matches!(
        unused_sampled
            .requirements()
            .get(GpuCapabilityFeature::TextureBindingArray),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::TextureBindingArray
        ))
    ));

    let (storage_graph, _, _) = storage_buffer_proof_graph();
    assert!(matches!(
        storage_graph
            .requirements()
            .get(GpuCapabilityFeature::BufferBindingArray),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::BufferBindingArray
        ))
    ));
    assert!(matches!(
        storage_graph
            .requirements()
            .get(GpuCapabilityFeature::StorageResourceBindingArray),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::StorageResourceBindingArray
        ))
    ));

    let (uniform_graph, _) = uniform_buffer_proof_graph();
    for feature in [
        GpuCapabilityFeature::BufferBindingArray,
        GpuCapabilityFeature::UniformBufferBindingArray,
    ] {
        assert!(matches!(
            uniform_graph.requirements().get(feature),
            Some(GpuCapabilityRequirement::Required(required)) if required == feature
        ));
    }

    let (sampled_graph, _) = sampled_texture_proof_graph();
    assert!(matches!(
        sampled_graph
            .requirements()
            .get(GpuCapabilityFeature::TextureBindingArray),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::TextureBindingArray
        ))
    ));

    let (sampler_graph, _) = sampler_proof_graph();
    assert!(matches!(
        sampler_graph
            .requirements()
            .get(GpuCapabilityFeature::TextureBindingArray),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::TextureBindingArray
        ))
    ));

    let (storage_texture_graph, _) = storage_texture_proof_graph();
    for feature in [
        GpuCapabilityFeature::TextureBindingArray,
        GpuCapabilityFeature::StorageResourceBindingArray,
        GpuCapabilityFeature::StorageTexture,
    ] {
        assert!(matches!(
            storage_texture_graph.requirements().get(feature),
            Some(GpuCapabilityRequirement::Required(required)) if required == feature
        ));
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) async fn prove_browser_webgpu_unsupported_contract() {
    let baseline = GpuContext::request(context_descriptor(
        GpuBackendFamily::BrowserWebGpu,
        None,
        GpuCapabilityProfile::ComputeBaseline.requirements(),
    ))
    .await
    .expect("actual-browser fixed-array proof requires the baseline WebGPU context");

    for feature in [
        GpuCapabilityFeature::TextureBindingArray,
        GpuCapabilityFeature::BufferBindingArray,
        GpuCapabilityFeature::StorageResourceBindingArray,
        GpuCapabilityFeature::UniformBufferBindingArray,
    ] {
        assert!(
            !baseline.adapter_facts().supported().supports(feature),
            "browser WebGPU must not advertise native fixed-array feature {feature:?}"
        );
        assert!(
            !baseline.device_facts().is_enabled(feature),
            "browser WebGPU must not enable native fixed-array feature {feature:?}"
        );
    }

    let adapter_limits = baseline.adapter_facts().adapter_limits().values();
    let device_limits = baseline.device_facts().device_limits().values();
    let workload_limits = baseline.device_facts().workload_budget().limits();
    for (scope, limits) in [
        ("adapter", adapter_limits),
        ("device", device_limits),
        ("workload", workload_limits),
    ] {
        assert_eq!(
            limits.max_binding_array_elements_per_shader_stage(),
            0,
            "{scope} general fixed-array element limit must remain zero on browser WebGPU"
        );
        assert_eq!(
            limits.max_binding_array_sampler_elements_per_shader_stage(),
            0,
            "{scope} sampler fixed-array element limit must remain zero on browser WebGPU"
        );
    }

    let mut requirements = GpuCapabilityProfile::ComputeBaseline.requirements();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::TextureBindingArray,
        ))
        .unwrap();
    let rejected = GpuContext::request(context_descriptor(
        GpuBackendFamily::BrowserWebGpu,
        None,
        requirements,
    ))
    .await
    .expect_err("browser WebGPU fixed-array requirements must reject during context admission");
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
                                    == Some(GpuCapabilityFeature::TextureBindingArray)
                        })
            )
        }),
        "browser fixed-array rejection must retain typed required-feature-unavailable evidence"
    );

    println!(
        "Fixed binding arrays: UNSUPPORTED (browser WebGPU native features absent; normalized array limits zero)"
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FixedBindingArrayProof {
    pub(crate) storage_buffer: bool,
    pub(crate) uniform_buffer: bool,
    pub(crate) sampled_texture: bool,
    pub(crate) sampler: bool,
    pub(crate) storage_texture: bool,
}

async fn run_suite_inner(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
    expected_adapter: Option<&GpuAdapterFacts>,
) -> FixedBindingArrayProof {
    FixedBindingArrayProof {
        storage_buffer: run_storage_buffer_array_proof(backend, fallback, expected_adapter).await,
        uniform_buffer: run_uniform_buffer_array_proof(backend, fallback, expected_adapter).await,
        sampled_texture: run_sampled_texture_array_proof(backend, fallback, expected_adapter).await,
        sampler: run_sampler_array_proof(backend, fallback, expected_adapter).await,
        storage_texture: run_storage_texture_array_proof(backend, fallback, expected_adapter).await,
    }
}

#[allow(dead_code)] // Shared proof module is compiled into binaries that use different retained entry points.
pub(crate) async fn run_suite(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
) -> FixedBindingArrayProof {
    let anchor = GpuContext::request(context_descriptor(
        backend,
        fallback,
        GpuCapabilityProfile::ComputeBaseline.requirements(),
    ))
    .await
    .expect("retained fixed-array suite requires one anchor adapter");

    // Pinned WGPU 30.0.1 explicitly skips llvmpipe for affected buffer binding-array GPU
    // cases because of driver crashes. This retained Lavapipe target also loses the device when
    // the fixed storage-buffer array pipeline is prepared. Keep that exception in qualification
    // policy only: RunenGPU capability normalization still follows actual feature/limit facts,
    // texture-family array proofs remain live here, and real Vulkan adapters run the full suite.
    let retained_lavapipe = backend == GpuBackendFamily::Vulkan
        && anchor
            .adapter_facts()
            .diagnostic_name()
            .is_some_and(|name| {
                let name = name.to_ascii_lowercase();
                name.contains("llvmpipe") || name.contains("lavapipe")
            });
    if retained_lavapipe {
        println!(
            "Fixed binding arrays: UNQUALIFIED (storage-buffer array on retained Lavapipe/llvmpipe driver)"
        );
        let expected_adapter = Some(anchor.adapter_facts());
        return FixedBindingArrayProof {
            storage_buffer: false,
            uniform_buffer: run_uniform_buffer_array_proof(backend, fallback, expected_adapter)
                .await,
            sampled_texture: run_sampled_texture_array_proof(backend, fallback, expected_adapter)
                .await,
            sampler: run_sampler_array_proof(backend, fallback, expected_adapter).await,
            storage_texture: run_storage_texture_array_proof(backend, fallback, expected_adapter)
                .await,
        };
    }

    run_suite_on_adapter(backend, fallback, anchor.adapter_facts()).await
}

pub(crate) async fn run_suite_on_adapter(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
    expected_adapter: &GpuAdapterFacts,
) -> FixedBindingArrayProof {
    run_suite_inner(backend, fallback, Some(expected_adapter)).await
}
