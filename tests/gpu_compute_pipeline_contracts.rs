use runen_gpu::{
    GpuAdmittedProgramSource, GpuBindingLayoutRefinement, GpuCapabilityFeature,
    GpuCapabilityRequirement, GpuCapabilityRequirements, GpuComputePipelineDescriptor,
    GpuEntryPointName, GpuPipelineConfiguration, GpuPipelineLayoutDescriptor,
    GpuProgramContractCause, GpuProgramDescriptor, GpuProgramSourceIdentity, GpuProgramSourceKey,
    GpuProgramSourceOwnerId, GpuProgramSourceProvenance, GpuProgramSourceRegistry,
    GpuProgramSourceRevision, GpuSpecializationDeclaration, GpuSpecializationKey,
    GpuSpecializationSchema, GpuSpecializationValue, GpuSpecializationValueSet,
    GpuSpecializationValueType,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const COMPUTE_WGSL: &str = "@compute @workgroup_size(1) fn main() {}";
const VERTEX_WGSL: &str = r#"
@vertex
fn vertex_main() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
"#;

fn source(key: &str, wgsl: &str) -> (GpuProgramSourceRegistry, GpuAdmittedProgramSource) {
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
            wgsl,
            GpuProgramSourceProvenance::new("gpu-compute-pipeline-test", None)
                .expect("test provenance should be valid"),
        )
        .expect("test source should admit");
    (registry, source)
}

fn program(source_key: &str, wgsl: &str, name: &str) -> GpuProgramDescriptor {
    let (_registry, source) = source(source_key, wgsl);
    GpuProgramDescriptor::new(
        source,
        [GpuEntryPointName::new(name).expect("test entry-point name should be valid")],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .expect("test program should derive from canonical WGSL")
}

fn compute_program() -> GpuProgramDescriptor {
    program("compute.pipeline", COMPUTE_WGSL, "main")
}

fn specialization(requirements: GpuCapabilityRequirements) -> GpuSpecializationValueSet {
    let declaration = GpuSpecializationDeclaration::new(
        GpuSpecializationKey::new("iterations").expect("test specialization key should be valid"),
        GpuSpecializationValueType::U32,
        Some(GpuSpecializationValue::U32(1)),
        requirements,
    )
    .expect("test specialization declaration should be valid");
    let schema = GpuSpecializationSchema::new([declaration])
        .expect("test specialization schema should be valid");
    GpuSpecializationValueSet::new(schema, [])
        .expect("test specialization values should use the default")
}

fn hash_of(value: &impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn compute_pipeline_derives_layout_and_binds_specialization_requirements() {
    let program = compute_program();
    let expected_layout = GpuPipelineLayoutDescriptor::from_interface(program.interface())
        .expect("test pipeline layout should derive");
    let mut specialization_requirements = GpuCapabilityRequirements::new();
    specialization_requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::TimestampQuery,
        ))
        .unwrap();
    let specialization = specialization(specialization_requirements);
    let configuration = GpuPipelineConfiguration::new(Some(specialization.clone()), None);

    let descriptor = GpuComputePipelineDescriptor::new(
        program.clone(),
        GpuEntryPointName::new("main").unwrap(),
        configuration.clone(),
    )
    .expect("valid compute pipeline descriptor should construct");
    let equivalent = GpuComputePipelineDescriptor::new(
        program,
        GpuEntryPointName::new("main").unwrap(),
        configuration,
    )
    .unwrap();

    assert_eq!(descriptor, equivalent);
    assert_eq!(hash_of(&descriptor), hash_of(&equivalent));
    assert!(descriptor.is_same_record(&descriptor.clone()));
    assert_eq!(descriptor.layout(), &expected_layout);
    assert_eq!(descriptor.specialization(), &specialization);
    assert!(matches!(
        descriptor.requirements().get(GpuCapabilityFeature::Compute),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Compute
        ))
    ));
    assert!(matches!(
        descriptor
            .requirements()
            .get(GpuCapabilityFeature::TimestampQuery),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::TimestampQuery
        ))
    ));
}

#[test]
fn compute_pipeline_default_configuration_has_no_caller_ceremony() {
    let program = compute_program();
    let expected_layout = GpuPipelineLayoutDescriptor::from_interface(program.interface())
        .expect("test pipeline layout should derive");

    let descriptor = GpuComputePipelineDescriptor::new(
        program,
        GpuEntryPointName::new("main").unwrap(),
        GpuPipelineConfiguration::default(),
    )
    .expect("ordinary compute pipeline descriptor should construct");

    assert_eq!(descriptor.layout(), &expected_layout);
    assert_eq!(descriptor.specialization().schema().declarations().len(), 0);
    assert_eq!(descriptor.specialization().entries().len(), 0);
}

#[test]
fn compute_pipeline_rejects_a_non_compute_entry_point() {
    let program = program("vertex.pipeline", VERTEX_WGSL, "vertex_main");
    let error = GpuComputePipelineDescriptor::new(
        program,
        GpuEntryPointName::new("vertex_main").unwrap(),
        GpuPipelineConfiguration::default(),
    )
    .expect_err("compute pipelines must select a declared compute entry point");

    assert_eq!(
        error.cause(),
        GpuProgramContractCause::PipelineDescriptorInvalid
    );
}

#[test]
fn compute_pipeline_rejects_conflicting_additional_requirements() {
    let program = compute_program();
    let mut requirements = GpuCapabilityRequirements::new();
    requirements
        .insert(GpuCapabilityRequirement::Disabled(
            GpuCapabilityFeature::Compute,
        ))
        .unwrap();

    let error = GpuComputePipelineDescriptor::new(
        program,
        GpuEntryPointName::new("main").unwrap(),
        GpuPipelineConfiguration::new(None, Some(requirements)),
    )
    .expect_err("compute cannot be disabled for a compute pipeline");

    assert_eq!(
        error.cause(),
        GpuProgramContractCause::PipelineDescriptorInvalid
    );
}
