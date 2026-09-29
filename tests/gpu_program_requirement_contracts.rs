use runen_gpu::{
    GpuAdmittedProgramSource, GpuBindingKey, GpuBindingLayoutRefinement, GpuBlendComponent,
    GpuBlendFactor, GpuBlendOperation, GpuBlendState, GpuCapabilityFeature,
    GpuCapabilityRequirement, GpuCapabilityRequirements, GpuColorTargetStateDescriptor,
    GpuColorWriteMask, GpuComputePipelineDescriptor, GpuEntryPointName,
    GpuFragmentOutputStateDescriptor, GpuMultisampleStateDescriptor, GpuPipelineConfiguration,
    GpuPrimitiveStateDescriptor, GpuProgramContractCause, GpuProgramDescriptor,
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
    admitted_source_from(key, FIXED_ARRAY_WGSL)
}

fn admitted_source_from(
    key: &str,
    wgsl: &str,
) -> (GpuProgramSourceRegistry, GpuAdmittedProgramSource) {
    let identity = GpuProgramSourceIdentity::new(
        GpuProgramSourceOwnerId::allocate().expect("test source owner should allocate"),
        GpuProgramSourceKey::new(key).expect("test source key should be valid"),
        GpuProgramSourceRevision::try_from_raw(1).expect("test source revision should be nonzero"),
    );
    let mut registry =
        GpuProgramSourceRegistry::new(4, 16 * 1024).expect("test source registry should construct");
    let source = registry
        .admit_wgsl(
            identity,
            wgsl,
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
        None,
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

const STORAGE_NON_UNIFORM_WGSL: &str = r#"
enable wgpu_binding_array;

struct StorageValue {
    value: u32,
}

struct Output {
    value: u32,
}

@group(0) @binding(0)
var<storage, read> storage_values: binding_array<StorageValue, 2>;

@group(0) @binding(1)
var<storage, read_write> output: Output;

@compute @workgroup_size(2)
fn compute_main(@builtin(local_invocation_index) index: u32) {
    output.value = storage_values[index].value;
}
"#;

const STORAGE_DYNAMIC_UNIFORM_WGSL: &str = r#"
enable wgpu_binding_array;

struct StorageValue {
    value: u32,
}

struct Selector {
    value: u32,
}

struct Output {
    value: u32,
}

@group(0) @binding(0)
var<storage, read> storage_values: binding_array<StorageValue, 2>;

@group(0) @binding(1)
var<uniform> selector: Selector;

@group(0) @binding(2)
var<storage, read_write> output: Output;

@compute @workgroup_size(1)
fn compute_main() {
    output.value = storage_values[selector.value].value;
}
"#;

const UNUSED_STORAGE_NON_UNIFORM_WGSL: &str = r#"
enable wgpu_binding_array;

struct StorageValue {
    value: u32,
}

@group(0) @binding(0)
var<storage, read> storage_values: binding_array<StorageValue, 2>;

fn unused_non_uniform(index: u32) -> u32 {
    return storage_values[index].value;
}

@compute @workgroup_size(1)
fn compute_main() {}
"#;

const TEXTURE_NON_UNIFORM_WGSL: &str = r#"
enable wgpu_binding_array;

struct Output {
    value: vec4<f32>,
}

@group(0) @binding(0)
var sampled_textures: binding_array<texture_2d<f32>, 2>;

@group(0) @binding(1)
var sampling_samplers: binding_array<sampler, 2>;

@group(0) @binding(2)
var<storage, read_write> output: Output;

@compute @workgroup_size(2)
fn compute_main(@builtin(local_invocation_index) index: u32) {
    output.value = textureSampleLevel(
        sampled_textures[index],
        sampling_samplers[index],
        vec2<f32>(0.5, 0.5),
        0.0,
    );
}
"#;

const STORAGE_TEXTURE_NON_UNIFORM_WGSL: &str = r#"
enable wgpu_binding_array;

@group(0) @binding(0)
var storage_textures: binding_array<texture_storage_2d<rgba8unorm, write>, 2>;

@compute @workgroup_size(2)
fn compute_main(@builtin(local_invocation_index) index: u32) {
    textureStore(
        storage_textures[index],
        vec2<i32>(0, 0),
        vec4<f32>(f32(index), 0.0, 0.0, 1.0),
    );
}
"#;

const UNIFORM_BUFFER_NON_UNIFORM_WGSL: &str = r#"
enable wgpu_binding_array;

struct UniformValue {
    value: u32,
}

struct Output {
    value: u32,
}

@group(0) @binding(0)
var<uniform> uniform_values: binding_array<UniformValue, 2>;

@group(0) @binding(1)
var<storage, read_write> output: Output;

@compute @workgroup_size(2)
fn compute_main(@builtin(local_invocation_index) index: u32) {
    output.value = uniform_values[index].value;
}
"#;

#[test]
fn storage_buffer_non_uniform_indexing_derives_the_resource_family_capability() {
    let (_registry, source) = admitted_source_from("storage.non-uniform", STORAGE_NON_UNIFORM_WGSL);
    let program = GpuProgramDescriptor::new(
        source,
        [entry_point("compute_main")],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();

    assert_required(
        program.requirements(),
        GpuCapabilityFeature::BufferBindingArray,
    );
    assert_required(
        program.requirements(),
        GpuCapabilityFeature::StorageResourceBindingArray,
    );
    assert_required(
        program.requirements(),
        GpuCapabilityFeature::StorageBufferBindingArrayNonUniformIndexing,
    );
    assert_eq!(
        program
            .requirements()
            .get(GpuCapabilityFeature::TextureBindingArrayNonUniformIndexing),
        None
    );
}

#[test]
fn dynamic_uniform_binding_array_indexing_derives_no_non_uniform_capability() {
    let (_registry, source) =
        admitted_source_from("storage.dynamic-uniform", STORAGE_DYNAMIC_UNIFORM_WGSL);
    let program = GpuProgramDescriptor::new(
        source,
        [entry_point("compute_main")],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();

    assert_required(
        program.requirements(),
        GpuCapabilityFeature::BufferBindingArray,
    );
    assert_required(
        program.requirements(),
        GpuCapabilityFeature::StorageResourceBindingArray,
    );
    assert_eq!(
        program
            .requirements()
            .get(GpuCapabilityFeature::StorageBufferBindingArrayNonUniformIndexing),
        None
    );
}

#[test]
fn unused_non_uniform_binding_array_access_is_whole_module_authority_not_selected_interface() {
    let (_registry, source) = admitted_source_from(
        "storage.unused-non-uniform",
        UNUSED_STORAGE_NON_UNIFORM_WGSL,
    );
    let program = GpuProgramDescriptor::new(
        source,
        [entry_point("compute_main")],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();

    assert_eq!(program.interface().bindings().count(), 0);
    assert_required(
        program.requirements(),
        GpuCapabilityFeature::BufferBindingArray,
    );
    assert_required(
        program.requirements(),
        GpuCapabilityFeature::StorageResourceBindingArray,
    );
    assert_required(
        program.requirements(),
        GpuCapabilityFeature::StorageBufferBindingArrayNonUniformIndexing,
    );
}

#[test]
fn texture_and_sampler_non_uniform_indexing_share_one_normalized_capability() {
    let (_registry, source) = admitted_source_from("texture.non-uniform", TEXTURE_NON_UNIFORM_WGSL);
    let refinements = [
        GpuBindingLayoutRefinement::new(binding_key(0))
            .with_texture_sample_class(GpuTextureSampleClass::FloatFilterable),
        GpuBindingLayoutRefinement::new(binding_key(1))
            .with_sampler_class(GpuSamplerClass::Filtering),
    ];
    let program =
        GpuProgramDescriptor::new(source, [entry_point("compute_main")], refinements).unwrap();

    assert_required(
        program.requirements(),
        GpuCapabilityFeature::TextureBindingArray,
    );
    assert_required(
        program.requirements(),
        GpuCapabilityFeature::TextureBindingArrayNonUniformIndexing,
    );
}

#[test]
fn storage_texture_non_uniform_indexing_derives_its_independent_capability() {
    let (_registry, source) = admitted_source_from(
        "storage-texture.non-uniform",
        STORAGE_TEXTURE_NON_UNIFORM_WGSL,
    );
    let program = GpuProgramDescriptor::new(
        source,
        [entry_point("compute_main")],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();

    assert_required(
        program.requirements(),
        GpuCapabilityFeature::TextureBindingArray,
    );
    assert_required(
        program.requirements(),
        GpuCapabilityFeature::StorageResourceBindingArray,
    );
    assert_required(
        program.requirements(),
        GpuCapabilityFeature::StorageTextureBindingArrayNonUniformIndexing,
    );
}

#[test]
fn uniform_buffer_non_uniform_indexing_is_explicitly_deferred() {
    let (_registry, source) = admitted_source_from(
        "uniform-buffer.non-uniform",
        UNIFORM_BUFFER_NON_UNIFORM_WGSL,
    );
    let error = GpuProgramDescriptor::new(
        source,
        [entry_point("compute_main")],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .expect_err("uniform-buffer non-uniform indexing is intentionally not normalized");

    assert_eq!(
        error.cause(),
        GpuProgramContractCause::ProgramCapabilityUnsupported
    );
    assert!(
        error
            .detail()
            .is_some_and(|detail| detail.contains("uniform-buffer binding-array indexing"))
    );
}

const DUAL_SOURCE_WGSL: &str = r#"
enable dual_source_blending;

struct DualSourceOutput {
    @location(0) @blend_src(0) primary: vec4<f32>,
    @location(0) @blend_src(1) secondary: vec4<f32>,
}

@vertex
fn dual_vs() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn dual_fs() -> DualSourceOutput {
    var output: DualSourceOutput;
    output.primary = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    output.secondary = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    return output;
}
"#;

const DUAL_SOURCE_PRIMARY_ONLY_WGSL: &str = r#"
enable dual_source_blending;

struct DualSourceOutput {
    @location(0) @blend_src(0) primary: vec4<f32>,
}

@fragment
fn dual_fs() -> DualSourceOutput {
    var output: DualSourceOutput;
    output.primary = vec4<f32>(1.0);
    return output;
}
"#;

const DUAL_SOURCE_ENABLE_ONLY_WGSL: &str = r#"
enable dual_source_blending;

@compute @workgroup_size(1)
fn compute_main() {}
"#;

const DUAL_SOURCE_WITHOUT_ENABLE_WGSL: &str = r#"
struct DualSourceOutput {
    @location(0) @blend_src(0) primary: vec4<f32>,
    @location(0) @blend_src(1) secondary: vec4<f32>,
}

@fragment
fn dual_fs() -> DualSourceOutput {
    var output: DualSourceOutput;
    output.primary = vec4<f32>(1.0);
    output.secondary = vec4<f32>(0.0);
    return output;
}
"#;

const DUAL_SOURCE_UNUSED_WGSL: &str = r#"
enable dual_source_blending;

struct DualSourceOutput {
    @location(0) @blend_src(0) primary: vec4<f32>,
    @location(0) @blend_src(1) secondary: vec4<f32>,
}

@fragment
fn unused_dual_fs() -> DualSourceOutput {
    var output: DualSourceOutput;
    output.primary = vec4<f32>(1.0);
    output.secondary = vec4<f32>(0.0);
    return output;
}

@compute @workgroup_size(1)
fn compute_main() {}
"#;

const F16_DUAL_SOURCE_WGSL: &str = r#"
enable f16;
enable dual_source_blending;

struct DualSourceOutput {
    @location(0) @blend_src(0) primary: vec4<f32>,
    @location(0) @blend_src(1) secondary: vec4<f32>,
}

@fragment
fn dual_fs() -> DualSourceOutput {
    let half_value = f16(1.0);
    var output: DualSourceOutput;
    output.primary = vec4<f32>(f32(half_value), 0.0, 0.0, 1.0);
    output.secondary = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    return output;
}
"#;

const ORDINARY_RENDER_WGSL: &str = r#"
@vertex
fn ordinary_vs() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn ordinary_fs() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
"#;

fn dual_source_blend() -> GpuBlendState {
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

fn render_state_with_blend(blend: Option<GpuBlendState>) -> GpuRenderPipelineStateDescriptor {
    let target = GpuColorTargetStateDescriptor::new(
        GpuTextureFormat::Rgba8Unorm,
        blend,
        GpuColorWriteMask::ALL,
    )
    .unwrap();
    GpuRenderPipelineStateDescriptor::new(
        GpuVertexInputStateDescriptor::new([]).unwrap(),
        Some(GpuFragmentOutputStateDescriptor::new([target])),
        GpuPrimitiveStateDescriptor::default(),
        None,
        GpuMultisampleStateDescriptor::default(),
    )
    .unwrap()
}

#[test]
fn canonical_dual_source_output_requires_the_complete_primary_secondary_pair() {
    let (_registry, source) =
        admitted_source_from("dual-source.primary-only", DUAL_SOURCE_PRIMARY_ONLY_WGSL);
    let error = GpuProgramDescriptor::new(
        source,
        [entry_point("dual_fs")],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .expect_err("a primary-only dual-source fragment output must fail canonical WGSL validation");

    assert_eq!(error.cause(), GpuProgramContractCause::CanonicalWgslInvalid);
    assert!(
        error
            .detail()
            .is_some_and(|detail| detail.contains("canonical WGSL validation failed"))
    );
}

#[test]
fn dual_source_enable_directive_alone_derives_the_whole_module_requirement() {
    let (_registry, source) =
        admitted_source_from("dual-source.enable-only", DUAL_SOURCE_ENABLE_ONLY_WGSL);
    let program = GpuProgramDescriptor::new(
        source,
        [entry_point("compute_main")],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .expect("the accepted dual-source parse profile must admit the enable directive");

    assert_required(
        program.requirements(),
        GpuCapabilityFeature::DualSourceBlending,
    );
    assert_eq!(
        program.requirements().get(GpuCapabilityFeature::ShaderF16),
        None
    );
}

#[test]
fn dual_source_extension_is_whole_module_requirement_even_when_selected_entry_point_is_unrelated() {
    let (_registry, source) =
        admitted_source_from("dual-source.unused-entry-point", DUAL_SOURCE_UNUSED_WGSL);
    let program = GpuProgramDescriptor::new(
        source,
        [entry_point("compute_main")],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .expect("dual-source extension should admit through its normalized parse profile");

    assert_required(
        program.requirements(),
        GpuCapabilityFeature::DualSourceBlending,
    );
    assert_eq!(
        program.requirements().get(GpuCapabilityFeature::ShaderF16),
        None
    );
}

#[test]
fn combined_f16_and_dual_source_profile_derives_both_requirements() {
    let (_registry, source) =
        admitted_source_from("dual-source.f16-combined", F16_DUAL_SOURCE_WGSL);
    let program = GpuProgramDescriptor::new(
        source,
        [entry_point("dual_fs")],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .expect("combined optional WGSL profile should admit");

    assert_required(program.requirements(), GpuCapabilityFeature::ShaderF16);
    assert_required(
        program.requirements(),
        GpuCapabilityFeature::DualSourceBlending,
    );
}

#[test]
fn dual_source_attributes_require_the_canonical_enable_extension() {
    let (_registry, source) = admitted_source_from(
        "dual-source.missing-enable",
        DUAL_SOURCE_WITHOUT_ENABLE_WGSL,
    );
    let error = GpuProgramDescriptor::new(
        source,
        [entry_point("dual_fs")],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .expect_err("blend_src must require enable dual_source_blending");

    assert_eq!(error.cause(), GpuProgramContractCause::CanonicalWgslInvalid);
}

#[test]
fn malformed_wgsl_remains_canonical_wgsl_invalid_across_optional_profiles() {
    let (_registry, source) = admitted_source_from(
        "dual-source.invalid-wgsl",
        "enable dual_source_blending; @compute @workgroup_size(1) fn broken(",
    );
    let error = GpuProgramDescriptor::new(
        source,
        [entry_point("broken")],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .expect_err("syntax-invalid WGSL must not become valid under a broader capability profile");

    assert_eq!(error.cause(), GpuProgramContractCause::CanonicalWgslInvalid);
}

#[test]
fn dual_source_pipeline_derives_capability_and_requires_exact_shader_parity() {
    let (_registry, dual_source) = admitted_source_from("dual-source.pipeline", DUAL_SOURCE_WGSL);
    let dual_program = GpuProgramDescriptor::new(
        dual_source,
        [entry_point("dual_vs"), entry_point("dual_fs")],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();

    let pipeline = GpuRenderPipelineDescriptor::new(
        dual_program,
        GpuRenderEntryPoints::new(entry_point("dual_vs"), Some(entry_point("dual_fs"))),
        render_state_with_blend(Some(dual_source_blend())),
        GpuPipelineConfiguration::default(),
    )
    .expect("exact dual-source shader and blend state should agree");
    assert_required(
        pipeline.requirements(),
        GpuCapabilityFeature::DualSourceBlending,
    );

    let (_registry, ordinary_source) =
        admitted_source_from("dual-source.ordinary-shader", ORDINARY_RENDER_WGSL);
    let ordinary_program = GpuProgramDescriptor::new(
        ordinary_source,
        [entry_point("ordinary_vs"), entry_point("ordinary_fs")],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();
    let ordinary_error = GpuRenderPipelineDescriptor::new(
        ordinary_program,
        GpuRenderEntryPoints::new(entry_point("ordinary_vs"), Some(entry_point("ordinary_fs"))),
        render_state_with_blend(Some(dual_source_blend())),
        GpuPipelineConfiguration::default(),
    )
    .expect_err("Src1 blend state with ordinary fragment output must reject");
    assert_eq!(
        ordinary_error.cause(),
        GpuProgramContractCause::PipelineStageIoMismatch
    );

    let (_registry, dual_source) =
        admitted_source_from("dual-source.unconsumed-secondary", DUAL_SOURCE_WGSL);
    let dual_program = GpuProgramDescriptor::new(
        dual_source,
        [entry_point("dual_vs"), entry_point("dual_fs")],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();
    let dual_error = GpuRenderPipelineDescriptor::new(
        dual_program,
        GpuRenderEntryPoints::new(entry_point("dual_vs"), Some(entry_point("dual_fs"))),
        render_state_with_blend(None),
        GpuPipelineConfiguration::default(),
    )
    .expect_err(
        "dual-source output without Src1 blend consumption must reject under strict parity",
    );
    assert_eq!(
        dual_error.cause(),
        GpuProgramContractCause::PipelineStageIoMismatch
    );
}

#[test]
fn dual_source_blending_rejects_multiple_color_targets_before_realization() {
    let target = || {
        GpuColorTargetStateDescriptor::new(
            GpuTextureFormat::Rgba8Unorm,
            Some(dual_source_blend()),
            GpuColorWriteMask::ALL,
        )
        .unwrap()
    };
    let error = GpuRenderPipelineStateDescriptor::new(
        GpuVertexInputStateDescriptor::new([]).unwrap(),
        Some(GpuFragmentOutputStateDescriptor::new([target(), target()])),
        GpuPrimitiveStateDescriptor::default(),
        None,
        GpuMultisampleStateDescriptor::default(),
    )
    .expect_err("dual-source MRT must reject structurally");
    assert_eq!(
        error.cause(),
        GpuProgramContractCause::RenderPipelineStateInvalid
    );
    assert_eq!(error.label(), "dual_source_color_target_count=2");
}
