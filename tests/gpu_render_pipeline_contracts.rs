use runen_gpu::{
    GpuAdmittedProgramSource, GpuBindingLayoutRefinement, GpuBlendMode, GpuCapabilityFeature,
    GpuCapabilityRequirement, GpuCapabilityRequirements, GpuColorTargetStateDescriptor,
    GpuColorWriteMask, GpuCompareFunction, GpuDepthStencilStateDescriptor, GpuEntryPointName,
    GpuFragmentOutputStateDescriptor, GpuMultisampleStateDescriptor, GpuPipelineConfiguration,
    GpuPipelineLayoutDescriptor, GpuPrimitiveStateDescriptor, GpuProgramContractCause,
    GpuProgramDescriptor, GpuProgramSourceIdentity, GpuProgramSourceKey, GpuProgramSourceOwnerId,
    GpuProgramSourceProvenance, GpuProgramSourceRegistry, GpuProgramSourceRevision,
    GpuRenderEntryPoints, GpuRenderPipelineDescriptor, GpuRenderPipelineStateDescriptor,
    GpuShaderIoScalarClass, GpuTextureFormat, GpuVertexInputStateDescriptor,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const RENDER_WGSL: &str = "@vertex fn vertex_main() -> @builtin(position) vec4f { return vec4f(); }\n@fragment fn fragment_main() -> @location(0) vec4f { return vec4f(); }";

fn source_from(wgsl: &str) -> (GpuProgramSourceRegistry, GpuAdmittedProgramSource) {
    let identity = GpuProgramSourceIdentity::new(
        GpuProgramSourceOwnerId::allocate().expect("test source owner should allocate"),
        GpuProgramSourceKey::new("render.pipeline").expect("test source key should be valid"),
        GpuProgramSourceRevision::try_from_raw(1).expect("test source revision should be nonzero"),
    );
    let mut registry =
        GpuProgramSourceRegistry::new(4, 4096).expect("test source registry should construct");
    let source = registry
        .admit_wgsl(
            identity,
            wgsl,
            GpuProgramSourceProvenance::new("gpu-render-pipeline-test", None)
                .expect("test provenance should be valid"),
        )
        .expect("test source should admit");
    (registry, source)
}

fn program_from(wgsl: &str) -> GpuProgramDescriptor {
    let (_registry, source) = source_from(wgsl);
    GpuProgramDescriptor::new(
        source,
        [entry_point("fragment_main"), entry_point("vertex_main")],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .expect("test render program should derive from canonical WGSL")
}

fn program() -> GpuProgramDescriptor {
    program_from(RENDER_WGSL)
}

fn entry_point(value: &str) -> GpuEntryPointName {
    GpuEntryPointName::new(value).expect("test entry-point name should be valid")
}

fn color_target() -> GpuColorTargetStateDescriptor {
    GpuColorTargetStateDescriptor::new(
        GpuTextureFormat::Rgba8UnormSrgb,
        GpuBlendMode::Replace,
        GpuColorWriteMask::ALL,
    )
    .expect("test color target should be valid")
}

fn depth_state() -> GpuDepthStencilStateDescriptor {
    GpuDepthStencilStateDescriptor::new(
        GpuTextureFormat::Depth32Float,
        true,
        GpuCompareFunction::LessEqual,
    )
    .expect("test depth state should be valid")
}

fn color_and_depth_state() -> GpuRenderPipelineStateDescriptor {
    GpuRenderPipelineStateDescriptor::new(
        GpuVertexInputStateDescriptor::new([]).unwrap(),
        Some(GpuFragmentOutputStateDescriptor::new([color_target()])),
        GpuPrimitiveStateDescriptor::default(),
        Some(depth_state()),
        GpuMultisampleStateDescriptor::default(),
    )
    .expect("test render state should be valid")
}

fn depth_only_state() -> GpuRenderPipelineStateDescriptor {
    GpuRenderPipelineStateDescriptor::new(
        GpuVertexInputStateDescriptor::new([]).unwrap(),
        None,
        GpuPrimitiveStateDescriptor::default(),
        Some(depth_state()),
        GpuMultisampleStateDescriptor::default(),
    )
    .expect("test depth-only state should be valid")
}

fn hash_of(value: &impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn render_pipeline_derives_layout_and_binds_additional_requirements() {
    let program = program();
    let expected_layout = GpuPipelineLayoutDescriptor::from_interface(program.interface()).unwrap();
    let entry_points = GpuRenderEntryPoints::new(
        entry_point("vertex_main"),
        Some(entry_point("fragment_main")),
    );
    let state = color_and_depth_state();

    let mut requirements_a = GpuCapabilityRequirements::new();
    requirements_a
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::TimestampQuery,
        ))
        .unwrap();
    requirements_a
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Copy,
        ))
        .unwrap();
    let mut requirements_b = GpuCapabilityRequirements::new();
    requirements_b
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Copy,
        ))
        .unwrap();
    requirements_b
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::TimestampQuery,
        ))
        .unwrap();

    let descriptor = GpuRenderPipelineDescriptor::new(
        program.clone(),
        entry_points.clone(),
        state.clone(),
        GpuPipelineConfiguration::new(None, Some(requirements_a)),
    )
    .expect("valid render descriptor should construct");
    let equivalent = GpuRenderPipelineDescriptor::new(
        program,
        entry_points,
        state,
        GpuPipelineConfiguration::new(None, Some(requirements_b)),
    )
    .unwrap();

    assert_eq!(descriptor, equivalent);
    assert_eq!(hash_of(&descriptor), hash_of(&equivalent));
    assert!(descriptor.is_same_record(&descriptor.clone()));
    assert_eq!(descriptor.layout(), &expected_layout);
    assert_eq!(descriptor.specialization().entries().len(), 0);
    assert!(matches!(
        descriptor
            .requirements()
            .get(GpuCapabilityFeature::RenderPipeline),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::RenderPipeline
        ))
    ));
    assert!(matches!(
        descriptor
            .requirements()
            .get(GpuCapabilityFeature::DepthAttachment),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::DepthAttachment
        ))
    ));

    let vertex = descriptor.expected_vertex_input_signature().unwrap();
    assert_eq!(vertex.entry_point().as_str(), "vertex_main");
    assert_eq!(vertex.locations().len(), 0);

    let fragment = descriptor
        .expected_fragment_output_signature()
        .unwrap()
        .expect("fragment expectation should be present");
    assert_eq!(fragment.entry_point().as_str(), "fragment_main");
    let location = fragment.locations().next().unwrap();
    assert_eq!(location.location(), 0);
    assert_eq!(
        location.value_type().scalar_class(),
        GpuShaderIoScalarClass::Float
    );
    assert_eq!(location.value_type().vector_width().get(), 4);
}

#[test]
fn render_pipeline_rejects_wrong_stage_entry_points() {
    let program = program();

    let wrong_vertex = GpuRenderPipelineDescriptor::new(
        program.clone(),
        GpuRenderEntryPoints::new(
            entry_point("fragment_main"),
            Some(entry_point("fragment_main")),
        ),
        color_and_depth_state(),
        GpuPipelineConfiguration::default(),
    )
    .expect_err("fragment entry points cannot be selected for the vertex stage");
    assert_eq!(
        wrong_vertex.cause(),
        GpuProgramContractCause::PipelineDescriptorInvalid
    );

    let wrong_fragment = GpuRenderPipelineDescriptor::new(
        program,
        GpuRenderEntryPoints::new(entry_point("vertex_main"), Some(entry_point("vertex_main"))),
        color_and_depth_state(),
        GpuPipelineConfiguration::default(),
    )
    .expect_err("vertex entry points cannot be selected for the fragment stage");
    assert_eq!(
        wrong_fragment.cause(),
        GpuProgramContractCause::PipelineDescriptorInvalid
    );
}

#[test]
fn render_pipeline_rejects_compiler_observed_stage_io_mismatch_before_backend() {
    let program = program_from(
        "struct VertexIn { @location(0) position: vec3f }\n@vertex fn vertex_main(input: VertexIn) -> @builtin(position) vec4f { return vec4f(input.position, 1.0); }\n@fragment fn fragment_main() -> @location(0) vec4f { return vec4f(); }",
    );

    let error = GpuRenderPipelineDescriptor::new(
        program,
        GpuRenderEntryPoints::new(
            entry_point("vertex_main"),
            Some(entry_point("fragment_main")),
        ),
        color_and_depth_state(),
        GpuPipelineConfiguration::default(),
    )
    .expect_err("shader vertex IO must be checked before any backend realization exists");

    assert_eq!(
        error.cause(),
        GpuProgramContractCause::PipelineStageIoMismatch
    );
}

#[test]
fn render_pipeline_requires_fragment_selection_and_state_parity() {
    let program = program();

    let missing_selection = GpuRenderPipelineDescriptor::new(
        program.clone(),
        GpuRenderEntryPoints::new(entry_point("vertex_main"), None),
        color_and_depth_state(),
        GpuPipelineConfiguration::default(),
    )
    .expect_err("fragment output state requires a selected fragment entry point");
    assert_eq!(
        missing_selection.cause(),
        GpuProgramContractCause::PipelineDescriptorInvalid
    );

    let unexpected_selection = GpuRenderPipelineDescriptor::new(
        program,
        GpuRenderEntryPoints::new(
            entry_point("vertex_main"),
            Some(entry_point("fragment_main")),
        ),
        depth_only_state(),
        GpuPipelineConfiguration::default(),
    )
    .expect_err("a selected fragment entry point requires fragment output state");
    assert_eq!(
        unexpected_selection.cause(),
        GpuProgramContractCause::PipelineDescriptorInvalid
    );
}

#[test]
fn render_pipeline_rejects_conflicting_additional_requirements() {
    let program = program();
    let entry_points = GpuRenderEntryPoints::new(
        entry_point("vertex_main"),
        Some(entry_point("fragment_main")),
    );
    let mut requirements = GpuCapabilityRequirements::new();
    requirements
        .insert(GpuCapabilityRequirement::Disabled(
            GpuCapabilityFeature::DepthAttachment,
        ))
        .unwrap();
    let requirement_error = GpuRenderPipelineDescriptor::new(
        program,
        entry_points,
        color_and_depth_state(),
        GpuPipelineConfiguration::new(None, Some(requirements)),
    )
    .expect_err("depth attachment cannot be disabled for depth state");
    assert_eq!(
        requirement_error.cause(),
        GpuProgramContractCause::PipelineDescriptorInvalid
    );
}

#[test]
fn vertex_only_depth_pipeline_exposes_no_fragment_expectation() {
    let program = program();
    let descriptor = GpuRenderPipelineDescriptor::new(
        program,
        GpuRenderEntryPoints::new(entry_point("vertex_main"), None),
        depth_only_state(),
        GpuPipelineConfiguration::default(),
    )
    .expect("vertex-only depth pipeline should construct");

    assert!(
        descriptor
            .expected_fragment_output_signature()
            .unwrap()
            .is_none()
    );
}
