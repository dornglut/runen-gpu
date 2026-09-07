use runen_gpu::{
    GpuAdmittedProgramSource, GpuBindingKey, GpuBindingLayoutRefinement, GpuCapabilityFeature,
    GpuCapabilityRequirement, GpuEntryPointName, GpuProgramContractCause, GpuProgramDescriptor,
    GpuProgramSourceIdentity, GpuProgramSourceKey, GpuProgramSourceOwnerId,
    GpuProgramSourceProvenance, GpuProgramSourceRegistry, GpuProgramSourceRevision,
    GpuSamplerClass, GpuShaderStage, GpuShaderStages, GpuStorageBufferAccess,
    GpuTextureSampleClass,
};
use std::num::NonZeroU64;

const PROGRAM_WGSL: &str = r#"
@group(0) @binding(0)
var<storage, read> input_values: array<u32>;

@group(0) @binding(1)
var<storage, read_write> output_values: array<u32>;

@group(0) @binding(2)
var<storage, read> unused_values: array<u32>;

@compute @workgroup_size(1)
fn copy_values() {
    output_values[0] = input_values[0];
}

@compute @workgroup_size(1)
fn reduce_values() {
    output_values[0] = input_values[0] + 1u;
}
"#;

const FIXED_BUFFER_WGSL: &str = r#"
struct Values {
    first: vec4<u32>,
    second: vec4<u32>,
}

@group(0) @binding(0)
var<storage, read> values: Values;

@compute @workgroup_size(1)
fn fixed_values() {
    let observed = values.second.x;
}
"#;

const SAMPLED_TEXTURE_WGSL: &str = r#"
@group(0) @binding(0)
var sampled_texture: texture_2d<f32>;

@group(0) @binding(1)
var sampled_sampler: sampler;

@fragment
fn sample_color() -> @location(0) vec4<f32> {
    return textureSample(sampled_texture, sampled_sampler, vec2<f32>(0.5, 0.5));
}
"#;

const BINDING_ARRAY_WGSL: &str = r#"
enable wgpu_binding_array;

@group(0) @binding(0)
var sampled_textures: binding_array<texture_2d<u32>, 3>;

@compute @workgroup_size(1)
fn inspect_textures() {
    let dimensions = textureDimensions(sampled_textures[0]);
}
"#;

const VISIBILITY_WGSL: &str = r#"
@group(0) @binding(0)
var<storage, read> values: array<u32>;

@compute @workgroup_size(1)
fn compute_values() {
    let observed = values[0];
}

@fragment
fn fragment_color() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#;

fn admitted_source_from(source_text: &str) -> (GpuProgramSourceRegistry, GpuAdmittedProgramSource) {
    let owner = GpuProgramSourceOwnerId::allocate().expect("source owner should allocate");
    let identity = GpuProgramSourceIdentity::new(
        owner,
        GpuProgramSourceKey::new("compute.program-descriptor")
            .expect("test source key should be valid"),
        GpuProgramSourceRevision::try_from_raw(1).expect("test source revision should be nonzero"),
    );
    let mut registry = GpuProgramSourceRegistry::new(4, 16 * 1024)
        .expect("bounded source registry should construct");
    let source = registry
        .admit_wgsl(
            identity,
            source_text,
            GpuProgramSourceProvenance::new("gpu-program-descriptor-test", None)
                .expect("test source provenance should be valid"),
        )
        .expect("source registry admission owns identity/content bounds, not WGSL semantics");
    (registry, source)
}

fn admitted_source() -> (GpuProgramSourceRegistry, GpuAdmittedProgramSource) {
    admitted_source_from(PROGRAM_WGSL)
}

fn entry(name: &str) -> GpuEntryPointName {
    GpuEntryPointName::new(name).expect("test entry-point name should be valid")
}

fn key(binding: u64) -> GpuBindingKey {
    GpuBindingKey::try_new(0, binding).expect("test binding key should fit u32")
}

#[test]
fn admitted_program_derives_entries_interface_and_static_visibility() {
    let (mut registry, source) = admitted_source();
    let program = GpuProgramDescriptor::new(
        source,
        [entry("reduce_values"), entry("copy_values")],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .expect("canonical WGSL should derive the selected program contract");

    assert_eq!(
        program
            .entry_points()
            .map(|entry_point| (entry_point.name().as_str(), entry_point.stage()))
            .collect::<Vec<_>>(),
        [
            ("copy_values", GpuShaderStage::Compute),
            ("reduce_values", GpuShaderStage::Compute),
        ]
    );
    assert!(
        program
            .entry_point(GpuShaderStage::Compute, &entry("copy_values"))
            .is_some()
    );

    let bindings = program.interface().bindings().collect::<Vec<_>>();
    assert_eq!(
        bindings.len(),
        2,
        "unused bound globals are not program-interface members"
    );
    assert_eq!(bindings[0].key(), key(0));
    assert_eq!(bindings[1].key(), key(1));
    assert_eq!(
        bindings[0].kind().storage_buffer_access(),
        Some(GpuStorageBufferAccess::ReadOnly)
    );
    assert_eq!(
        bindings[1].kind().storage_buffer_access(),
        Some(GpuStorageBufferAccess::ReadWrite)
    );
    assert_eq!(
        bindings[0].visibility(),
        GpuShaderStages::one(GpuShaderStage::Compute)
    );
    assert!(program.interface().binding(key(2)).is_none());
    assert!(program.is_same_record(&program.clone()));
    assert_eq!(registry.collect_unretained(), 0);

    drop(program);
    assert_eq!(registry.collect_unretained(), 1);
}

#[test]
fn fixed_binding_array_cardinality_and_capability_are_compiler_derived() {
    let (_registry, source) = admitted_source_from(BINDING_ARRAY_WGSL);
    let program = GpuProgramDescriptor::new(
        source,
        [entry("inspect_textures")],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .expect("fixed binding array should derive from canonical WGSL");
    let binding = program.interface().binding(key(0)).unwrap();

    assert_eq!(binding.array_count().map(|count| count.get()), Some(3));
    assert_eq!(
        binding.kind().texture_sample_class(),
        Some(GpuTextureSampleClass::Uint)
    );
    assert_eq!(
        binding.visibility(),
        GpuShaderStages::one(GpuShaderStage::Compute)
    );
    assert!(matches!(
        program
            .requirements()
            .get(GpuCapabilityFeature::TextureBindingArray),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::TextureBindingArray
        ))
    ));
}

#[test]
fn admitted_program_rejects_duplicate_selected_entry_names() {
    let (_registry, source) = admitted_source();
    let duplicate = entry("copy_values");

    let error = GpuProgramDescriptor::new(
        source,
        [duplicate.clone(), duplicate],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .expect_err("duplicate selected names must be rejected before backend realization");

    assert_eq!(error.cause(), GpuProgramContractCause::DuplicateEntryPoint);
}

#[test]
fn admitted_program_rejects_missing_selected_entry() {
    let (_registry, source) = admitted_source();

    let error = GpuProgramDescriptor::new(
        source,
        [entry("missing")],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .expect_err("missing selected entries must reject during program admission");

    assert_eq!(error.cause(), GpuProgramContractCause::EntryPointMissing);
}

#[test]
fn admitted_program_rejects_malformed_canonical_wgsl() {
    let (_registry, source) = admitted_source_from("@compute fn broken(");

    let error = GpuProgramDescriptor::new(
        source,
        [entry("broken")],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .expect_err("WGSL syntax must be validated before backend realization");

    assert_eq!(error.cause(), GpuProgramContractCause::CanonicalWgslInvalid);
}

#[test]
fn refinement_cannot_resurrect_an_unused_shader_binding() {
    let (_registry, source) = admitted_source();
    let refinement = GpuBindingLayoutRefinement::new(key(2)).with_dynamic_offset(true);

    let error = GpuProgramDescriptor::new(source, [entry("copy_values")], [refinement])
        .expect_err("refinements must target effective selected-program bindings only");

    assert_eq!(
        error.cause(),
        GpuProgramContractCause::BindingRefinementInvalid
    );
}

#[test]
fn unknown_refinement_key_rejects() {
    let (_registry, source) = admitted_source();
    let refinement = GpuBindingLayoutRefinement::new(key(99));

    let error = GpuProgramDescriptor::new(source, [entry("copy_values")], [refinement])
        .expect_err("refinements must target a compiler-derived effective binding");

    assert_eq!(
        error.cause(),
        GpuProgramContractCause::BindingRefinementInvalid
    );
}

#[test]
fn buffer_refinement_changes_only_host_layout_policy() {
    let (_registry, source) = admitted_source();
    let refinement = GpuBindingLayoutRefinement::new(key(0)).with_dynamic_offset(true);

    let program = GpuProgramDescriptor::new(source, [entry("copy_values")], [refinement])
        .expect("dynamic offset is valid host policy for a storage buffer");
    let binding = program.interface().binding(key(0)).unwrap();

    assert!(binding.kind().uses_dynamic_offset());
    assert_eq!(
        binding.kind().storage_buffer_access(),
        Some(GpuStorageBufferAccess::ReadOnly)
    );
    assert_eq!(
        binding.visibility(),
        GpuShaderStages::one(GpuShaderStage::Compute)
    );
}

#[test]
fn compiler_and_host_buffer_minimums_remain_distinct_and_weaker_host_policy_rejects() {
    let (_registry, source) = admitted_source_from(FIXED_BUFFER_WGSL);
    let program = GpuProgramDescriptor::new(
        source,
        [entry("fixed_values")],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .expect("fixed-layout storage buffer should expose a compiler-required minimum");
    let binding = program.interface().binding(key(0)).unwrap();
    let compiler_minimum = binding
        .compiler_required_minimum_size()
        .expect("fixed-layout storage buffer must have a compiler-required minimum");
    assert!(compiler_minimum.get() > 1);
    assert_eq!(binding.kind().minimum_buffer_size(), None);

    let (_registry, source) = admitted_source_from(FIXED_BUFFER_WGSL);
    let weaker_host_minimum = NonZeroU64::new(compiler_minimum.get() - 1).unwrap();
    let error = GpuProgramDescriptor::new(
        source,
        [entry("fixed_values")],
        [GpuBindingLayoutRefinement::new(key(0)).with_host_minimum_size(weaker_host_minimum)],
    )
    .expect_err("host layout minimum cannot be weaker than the compiler requirement");
    assert_eq!(
        error.cause(),
        GpuProgramContractCause::BindingRefinementInvalid
    );

    let (_registry, source) = admitted_source_from(FIXED_BUFFER_WGSL);
    let stronger_host_minimum = NonZeroU64::new(compiler_minimum.get() + 16).unwrap();
    let program = GpuProgramDescriptor::new(
        source,
        [entry("fixed_values")],
        [GpuBindingLayoutRefinement::new(key(0)).with_host_minimum_size(stronger_host_minimum)],
    )
    .expect("stronger host layout minimum should remain independent policy");
    let binding = program.interface().binding(key(0)).unwrap();
    assert_eq!(
        binding.compiler_required_minimum_size(),
        Some(compiler_minimum)
    );
    assert_eq!(
        binding.kind().minimum_buffer_size(),
        Some(stronger_host_minimum)
    );
}

#[test]
fn ambiguous_float_texture_and_sampler_require_explicit_layout_policy() {
    let (_registry, source) = admitted_source_from(SAMPLED_TEXTURE_WGSL);
    let error = GpuProgramDescriptor::new(
        source,
        [entry("sample_color")],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .expect_err("WGSL float texture and ordinary sampler do not decide host filtering policy");
    assert_eq!(
        error.cause(),
        GpuProgramContractCause::BindingRefinementInvalid
    );

    let (_registry, source) = admitted_source_from(SAMPLED_TEXTURE_WGSL);
    let program = GpuProgramDescriptor::new(
        source,
        [entry("sample_color")],
        [
            GpuBindingLayoutRefinement::new(key(0))
                .with_texture_sample_class(GpuTextureSampleClass::FloatFilterable),
            GpuBindingLayoutRefinement::new(key(1)).with_sampler_class(GpuSamplerClass::Filtering),
        ],
    )
    .expect("explicit texture and sampler layout policy should complete the effective interface");
    assert_eq!(
        program
            .interface()
            .binding(key(0))
            .unwrap()
            .kind()
            .texture_sample_class(),
        Some(GpuTextureSampleClass::FloatFilterable)
    );
    assert_eq!(
        program
            .interface()
            .binding(key(1))
            .unwrap()
            .kind()
            .sampler_class(),
        Some(GpuSamplerClass::Filtering)
    );
}

#[test]
fn buffer_only_refinements_reject_on_non_buffer_bindings() {
    let (_registry, source) = admitted_source_from(SAMPLED_TEXTURE_WGSL);
    let error = GpuProgramDescriptor::new(
        source,
        [entry("sample_color")],
        [
            GpuBindingLayoutRefinement::new(key(0))
                .with_dynamic_offset(true)
                .with_texture_sample_class(GpuTextureSampleClass::FloatFilterable),
            GpuBindingLayoutRefinement::new(key(1)).with_sampler_class(GpuSamplerClass::Filtering),
        ],
    )
    .expect_err("dynamic offsets apply only to compiler-derived buffer bindings");
    assert_eq!(
        error.cause(),
        GpuProgramContractCause::BindingRefinementInvalid
    );
}

#[test]
fn visibility_refinement_can_widen_only_within_selected_program_stages() {
    let selected_visibility =
        GpuShaderStages::new([GpuShaderStage::Compute, GpuShaderStage::Fragment]).unwrap();
    let (_registry, source) = admitted_source_from(VISIBILITY_WGSL);
    let program = GpuProgramDescriptor::new(
        source,
        [entry("compute_values"), entry("fragment_color")],
        [GpuBindingLayoutRefinement::new(key(0)).with_visibility(selected_visibility)],
    )
    .expect("visibility may widen from observed compute use to another selected program stage");
    assert_eq!(
        program.interface().binding(key(0)).unwrap().visibility(),
        selected_visibility
    );

    let (_registry, source) = admitted_source_from(VISIBILITY_WGSL);
    let error = GpuProgramDescriptor::new(
        source,
        [entry("compute_values"), entry("fragment_color")],
        [GpuBindingLayoutRefinement::new(key(0))
            .with_visibility(GpuShaderStages::one(GpuShaderStage::Fragment))],
    )
    .expect_err("visibility refinement cannot omit the compiler-observed compute use");
    assert_eq!(
        error.cause(),
        GpuProgramContractCause::BindingRefinementInvalid
    );
}

#[test]
fn visibility_refinement_cannot_invent_an_unselected_stage() {
    let (_registry, source) = admitted_source();
    let visibility = GpuShaderStages::new([GpuShaderStage::Compute, GpuShaderStage::Fragment])
        .expect("test visibility should be nonempty");
    let refinement = GpuBindingLayoutRefinement::new(key(0)).with_visibility(visibility);

    let error = GpuProgramDescriptor::new(source, [entry("copy_values")], [refinement])
        .expect_err("visibility may widen only within stages selected by this program");

    assert_eq!(
        error.cause(),
        GpuProgramContractCause::BindingRefinementInvalid
    );
}
