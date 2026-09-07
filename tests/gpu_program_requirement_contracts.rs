use runen_gpu::{
    GpuAdmittedProgramSource, GpuBindingKey, GpuBindingLayoutRefinement, GpuBlendMode,
    GpuCapabilityFeature, GpuCapabilityRequirement, GpuCapabilityRequirements,
    GpuColorTargetStateDescriptor, GpuColorWriteMask, GpuComputePipelineDescriptor,
    GpuEntryPointName, GpuFragmentOutputStateDescriptor, GpuMultisampleStateDescriptor,
    GpuPipelineConfiguration, GpuPrimitiveStateDescriptor, GpuProgramDescriptor,
    GpuProgramSourceIdentity, GpuProgramSourceKey, GpuProgramSourceOwnerId,
    GpuProgramSourceProvenance, GpuProgramSourceRegistry, GpuProgramSourceRevision,
    GpuRenderEntryPoints, GpuRenderPipelineDescriptor, GpuRenderPipelineStateDescriptor,
    GpuSamplerClass, GpuTextureFormat, GpuTextureSampleClass, GpuVertexInputStateDescriptor,
};

const FIXED_ARRAY_WGSL: &str = r#"
enable wgpu_binding_array;

struct UniformValue {
    value: vec4<f32>,
}

struct StorageValue {
    value: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> uniform_values: binding_array<UniformValue, 2>;

@group(0) @binding(1)
var<storage, read> storage_values: binding_array<StorageValue, 2>;

@group(0) @binding(2)
var sampled_textures: binding_array<texture_2d<f32>, 2>;

@group(0) @binding(3)
var storage_textures: binding_array<texture_storage_2d<rgba8unorm, write>, 2>;

@group(0) @binding(4)
var sampling_samplers: binding_array<sampler, 2>;

fn resource_value() -> vec4<f32> {
    let sampled = textureSampleLevel(
        sampled_textures[0],
        sampling_samplers[0],
        vec2<f32>(0.5, 0.5),
        0.0,
    );
    return uniform_values[0].value + storage_values[0].value + sampled;
}

@compute @workgroup_size(1)
fn compute_main() {
    textureStore(storage_textures[0], vec2<i32>(0, 0), resource_value());
}

@vertex
fn vertex_main() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fragment_main() -> @location(0) vec4<f32> {
    let value = resource_value();
    textureStore(storage_textures[0], vec2<i32>(0, 0), value);
    return value;
}
"#;

fn admitted_source(key: &str) -> (GpuProgramSourceRegistry, GpuAdmittedProgramSource) {
    let identity = GpuProgramSourceIdentity::new(
        GpuProgramSourceOwnerId::allocate().expect("test source owner should allocate"),
        GpuProgramSourceKey::new(key).expect("test source key should be valid"),
        GpuProgramSourceRevision::try_from_raw(1).expect("test source revision should be nonzero"),
    );
    let mut registry =
        GpuProgramSourceRegistry::new(4, 4096).expect("test source registry should construct");
    let source = registry
        .admit_wgsl(
            identity,
            FIXED_ARRAY_WGSL,
            GpuProgramSourceProvenance::new("gpu-program-requirement-test", None)
                .expect("test source provenance should be valid"),
        )
        .expect("test source should admit");
    (registry, source)
}

fn entry_point(name: &str) -> GpuEntryPointName {
    GpuEntryPointName::new(name).expect("test entry-point name should be valid")
}

fn binding_key(binding: u64) -> GpuBindingKey {
    GpuBindingKey::try_new(0, binding).expect("test binding key should fit u32")
}

fn filtering_refinements() -> [GpuBindingLayoutRefinement; 2] {
    [
        GpuBindingLayoutRefinement::new(binding_key(2))
            .with_texture_sample_class(GpuTextureSampleClass::FloatFilterable),
        GpuBindingLayoutRefinement::new(binding_key(4))
            .with_sampler_class(GpuSamplerClass::Filtering),
    ]
}

fn assert_required(requirements: &GpuCapabilityRequirements, feature: GpuCapabilityFeature) {
    assert_eq!(
        requirements.get(feature),
        Some(GpuCapabilityRequirement::Required(feature))
    );
}

fn assert_fixed_array_requirements(requirements: &GpuCapabilityRequirements) {
    for feature in [
        GpuCapabilityFeature::StorageTexture,
        GpuCapabilityFeature::TextureBindingArray,
        GpuCapabilityFeature::BufferBindingArray,
        GpuCapabilityFeature::StorageResourceBindingArray,
        GpuCapabilityFeature::UniformBufferBindingArray,
    ] {
        assert_required(requirements, feature);
    }
}

#[test]
fn compute_pipeline_inherits_program_interface_requirements() {
    let (_registry, source) = admitted_source("compute.program-requirements");
    let program = GpuProgramDescriptor::new(
        source,
        [entry_point("compute_main")],
        filtering_refinements(),
    )
    .expect("compute program requirements should derive from canonical WGSL");
    assert_fixed_array_requirements(program.requirements());

    let pipeline = GpuComputePipelineDescriptor::new(
        program,
        entry_point("compute_main"),
        GpuPipelineConfiguration::default(),
    )
    .unwrap();
    assert_fixed_array_requirements(pipeline.requirements());
}

#[test]
fn render_pipeline_inherits_program_interface_requirements() {
    let (_registry, source) = admitted_source("render.program-requirements");
    let program = GpuProgramDescriptor::new(
        source,
        [entry_point("vertex_main"), entry_point("fragment_main")],
        filtering_refinements(),
    )
    .expect("render program requirements should derive from canonical WGSL");
    assert_fixed_array_requirements(program.requirements());

    let color_target = GpuColorTargetStateDescriptor::new(
        GpuTextureFormat::Rgba8Unorm,
        GpuBlendMode::Replace,
        GpuColorWriteMask::ALL,
    )
    .unwrap();
    let state = GpuRenderPipelineStateDescriptor::new(
        GpuVertexInputStateDescriptor::new([]).unwrap(),
        Some(GpuFragmentOutputStateDescriptor::new([color_target])),
        GpuPrimitiveStateDescriptor::default(),
        None,
        GpuMultisampleStateDescriptor::default(),
    )
    .unwrap();
    let pipeline = GpuRenderPipelineDescriptor::new(
        program,
        GpuRenderEntryPoints::new(
            entry_point("vertex_main"),
            Some(entry_point("fragment_main")),
        ),
        state,
        GpuPipelineConfiguration::default(),
    )
    .unwrap();
    assert_fixed_array_requirements(pipeline.requirements());
}
