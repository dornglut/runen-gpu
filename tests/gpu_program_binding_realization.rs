use runen_gpu::{
    GpuBindGroupLayoutDescriptor, GpuBindingDeclaration, GpuBindingKey, GpuBindingKind,
    GpuBindingLayoutRefinement, GpuBindingProvenance, GpuBufferDescriptor, GpuBufferInitialization,
    GpuBufferUsage, GpuBufferUsages, GpuCapabilityProfile, GpuContext, GpuContextDescriptor,
    GpuContextRequestError, GpuContextRequestErrorCategory, GpuEntryPointName, GpuMemoryIntent,
    GpuPipelineLayoutDescriptor, GpuProgramBindingRealizationErrorCategory,
    GpuProgramBindingRealizationPolicy, GpuProgramContractCause, GpuProgramDescriptor,
    GpuProgramSourceIdentity, GpuProgramSourceKey, GpuProgramSourceOwnerId,
    GpuProgramSourceProvenance, GpuProgramSourceRegistry, GpuProgramSourceRevision,
    GpuRealizationPolicies, GpuReconstruction, GpuResourceCommon, GpuResourceLabel,
    GpuResourceLifetime, GpuResourceProvenance, GpuRuntimeBindingResource, GpuRuntimeBindingValue,
    GpuRuntimeBufferBinding, GpuShaderStage, GpuShaderStages, GpuStorageBufferAccess,
    GpuWorkResourceIdAllocator,
};
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Mutex;

// Adapter-backed WGPU tests create separate devices and exercise scoped error publication.
// Serialize this binary's cases so their environment setup cannot contend with the behavior each
// case is meant to verify; the explicit single-flight test below still creates concurrent callers.
static GPU_REALIZATION_TEST_LOCK: Mutex<()> = Mutex::new(());

const COMPUTE_WGSL: &str = r#"
@group(0) @binding(0)
var<storage, read_write> values: array<u32>;

@compute @workgroup_size(1)
fn cs_main() {
    values[0] = values[0] + 1u;
}
"#;

const RENDER_WGSL: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(f32(vertex_index), 0.0, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.25, 0.5, 0.75, 1.0);
}
"#;

const INVALID_WGSL: &str = "@compute @workgroup_size(1) fn cs_main( {";

fn admitted_context(
    max_program_binding_records: usize,
) -> Result<GpuContext, GpuContextRequestError> {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::ComputeBaseline.requirements())
            .with_label("G4C2 program and binding realization test");
    pollster::block_on(GpuContext::request_with_realization_policies(
        descriptor,
        GpuRealizationPolicies::new(
            Default::default(),
            GpuProgramBindingRealizationPolicy::new(
                NonZeroUsize::new(max_program_binding_records)
                    .expect("G4C2 test record limit must be nonzero"),
            ),
        ),
    ))
}

fn context_or_skip(max_program_binding_records: usize) -> Option<GpuContext> {
    match admitted_context(max_program_binding_records) {
        Ok(context) => Some(context),
        Err(error)
            if matches!(
                error.category(),
                GpuContextRequestErrorCategory::NoAdapterAvailable
                    | GpuContextRequestErrorCategory::NoAdmissibleCandidate
                    | GpuContextRequestErrorCategory::MandatoryFeatureMissing
            ) =>
        {
            eprintln!("G4C2 representative environment unavailable: {error}");
            None
        }
        Err(error) => panic!("unexpected G4C2 context admission failure: {error}"),
    }
}

fn owned_common(label: &str) -> GpuResourceCommon {
    let label = GpuResourceLabel::new(label).expect("test resource label should be valid");
    GpuResourceCommon::owned(
        label.clone(),
        GpuResourceLifetime::Retained,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        GpuResourceProvenance::new(label, None, None),
    )
    .expect("test resource common descriptor should be valid")
}

fn storage_buffer(label: &str) -> runen_gpu::GpuBufferHandle {
    let common = owned_common(label);
    let usages = GpuBufferUsages::new(common.label(), [GpuBufferUsage::Storage])
        .expect("test storage buffer usage should be valid");
    let descriptor =
        GpuBufferDescriptor::new(common, 64, usages, GpuBufferInitialization::Uninitialized)
            .expect("test storage buffer descriptor should be valid");
    let mut resources = GpuWorkResourceIdAllocator::new();
    resources
        .allocate_buffer_handle(descriptor)
        .expect("test storage buffer handle should allocate")
}

fn storage_binding() -> GpuBindingDeclaration {
    GpuBindingDeclaration::new(
        GpuBindingKey::try_new(0, 0).expect("test binding key should fit"),
        GpuShaderStages::one(GpuShaderStage::Compute),
        GpuBindingKind::storage_buffer(GpuStorageBufferAccess::ReadWrite, false, None),
        None,
        "values",
        GpuBindingProvenance::new("gpu-program-binding-realization-test", None)
            .expect("test binding provenance should be valid"),
    )
    .expect("test storage binding should be valid")
}

fn admitted_source(key: &str, canonical_wgsl: &str) -> runen_gpu::GpuAdmittedProgramSource {
    let owner = GpuProgramSourceOwnerId::allocate().expect("test source owner should allocate");
    let identity = GpuProgramSourceIdentity::new(
        owner,
        GpuProgramSourceKey::new(key).expect("test source key should be valid"),
        GpuProgramSourceRevision::try_from_raw(1).expect("test source revision should be valid"),
    );
    let mut sources =
        GpuProgramSourceRegistry::new(4, 16 * 1024).expect("test source registry should construct");
    sources
        .admit_wgsl(
            identity,
            canonical_wgsl,
            GpuProgramSourceProvenance::new("gpu-program-binding-realization-test", None)
                .expect("test source provenance should be valid"),
        )
        .expect("test source should admit")
}

fn compute_program(key: &str, canonical_wgsl: &str) -> GpuProgramDescriptor {
    let source = admitted_source(key, canonical_wgsl);
    let entry_point =
        GpuEntryPointName::new("cs_main").expect("compute entry point should be valid");
    GpuProgramDescriptor::new(
        source,
        [entry_point],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .expect("test compute program descriptor should be valid")
}

fn render_program() -> GpuProgramDescriptor {
    GpuProgramDescriptor::new(
        admitted_source("g4c2.test.render", RENDER_WGSL),
        [
            GpuEntryPointName::new("vs_main").expect("vertex entry point should be valid"),
            GpuEntryPointName::new("fs_main").expect("fragment entry point should be valid"),
        ],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .expect("test render program descriptor should be valid")
}

fn binding_value(buffer: runen_gpu::GpuBufferHandle) -> GpuRuntimeBindingValue {
    GpuRuntimeBindingValue::new(
        GpuBindingKey::try_new(0, 0).expect("test binding key should fit"),
        [GpuRuntimeBindingResource::Buffer(
            GpuRuntimeBufferBinding::new(
                buffer,
                0,
                NonZeroU64::new(64).expect("test buffer range should be nonzero"),
                None,
            ),
        )],
    )
    .expect("test runtime binding value should be valid")
}

#[test]
fn representative_compute_render_layout_and_bind_group_realization_reuse_records() {
    let _serialization = GPU_REALIZATION_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(context) = context_or_skip(16) else {
        return;
    };
    assert_eq!(
        context
            .program_binding_realization_policy()
            .max_records()
            .get(),
        16
    );

    let compute = compute_program("g4c2.test.compute", COMPUTE_WGSL);
    let render = render_program();

    let realized_compute = pollster::block_on(context.realize_program(&compute))
        .expect("representative compute WGSL should realize");
    let repeated_compute = pollster::block_on(context.realize_program(&compute))
        .expect("identical compute program should reuse its realization");
    assert!(realized_compute.is_same_record(&repeated_compute));

    let realized_render = pollster::block_on(context.realize_program(&render))
        .expect("representative render WGSL should realize without G4C3 pipeline checks");
    assert_eq!(realized_compute.affinity(), realized_render.affinity());
    assert_eq!(
        realized_compute.affinity(),
        context.affinity(),
        "realized programs remain context and generation bound"
    );

    let layout = GpuBindGroupLayoutDescriptor::new(0, [storage_binding()])
        .expect("test bind-group layout should be valid");
    let realized_layout = pollster::block_on(context.realize_bind_group_layout(&layout))
        .expect("typed bind-group layout should realize");
    let repeated_layout = pollster::block_on(context.realize_bind_group_layout(&layout))
        .expect("identical bind-group layout should reuse its realization");
    assert!(realized_layout.is_same_record(&repeated_layout));

    let pipeline_layout = GpuPipelineLayoutDescriptor::new([layout.clone()])
        .expect("test pipeline layout should be valid");
    let realized_pipeline_layout =
        pollster::block_on(context.realize_pipeline_layout(&pipeline_layout))
            .expect("typed pipeline layout should realize through the same G4C2 layout authority");

    let buffer = storage_buffer("g4c2 representative values");
    let _realized_buffer = context
        .realize_buffer(&buffer)
        .expect("G4C1 buffer prerequisite should realize");
    let realized_bind_group = pollster::block_on(
        context.realize_bind_group(&realized_layout, [binding_value(buffer.clone())]),
    )
    .expect("typed runtime binding should realize one bind group");
    let repeated_bind_group =
        pollster::block_on(context.realize_bind_group(&realized_layout, [binding_value(buffer)]))
            .expect("identical typed runtime binding should reuse its bind group");
    assert!(realized_bind_group.is_same_record(&repeated_bind_group));
    assert_eq!(realized_bind_group.layout_descriptor(), &layout);
    assert_eq!(realized_pipeline_layout.descriptor(), &pipeline_layout);

    let stats = context.program_binding_realization_stats();
    assert_eq!(stats.programs(), 2);
    assert_eq!(stats.bind_group_layouts(), 1);
    assert_eq!(stats.pipeline_layouts(), 1);
    assert_eq!(stats.bind_groups(), 1);
    assert_eq!(stats.in_flight_records(), 0);
    assert_eq!(stats.retained_records(), 5);
}

#[test]
fn equal_concurrent_g4c2_requests_singleflight_and_contexts_remain_isolated() {
    let _serialization = GPU_REALIZATION_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(context) = context_or_skip(16) else {
        return;
    };
    let program = compute_program("g4c2.test.concurrent", COMPUTE_WGSL);
    let realized = std::thread::scope(|scope| {
        (0..4)
            .map(|_| {
                scope.spawn(|| {
                    pollster::block_on(context.realize_program(&program))
                        .expect("equal concurrent program realization should succeed")
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|worker| worker.join().expect("single-flight worker must not panic"))
            .collect::<Vec<_>>()
    });
    assert!(
        realized
            .iter()
            .all(|candidate| realized[0].is_same_record(candidate)),
        "equal concurrent misses must publish one authoritative program record"
    );

    let layout = GpuBindGroupLayoutDescriptor::new(0, [storage_binding()])
        .expect("test bind-group layout should be valid");
    let realized_layouts = std::thread::scope(|scope| {
        (0..4)
            .map(|_| {
                scope.spawn(|| {
                    pollster::block_on(context.realize_bind_group_layout(&layout))
                        .expect("equal concurrent layout realization should succeed")
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|worker| worker.join().expect("layout worker must not panic"))
            .collect::<Vec<_>>()
    });
    assert!(
        realized_layouts
            .iter()
            .all(|candidate| realized_layouts[0].is_same_record(candidate)),
        "equal concurrent layout misses must publish one authoritative layout record"
    );

    let pipeline_layout = GpuPipelineLayoutDescriptor::new([layout.clone()])
        .expect("test pipeline layout should be valid");
    let realized_pipeline_layouts = std::thread::scope(|scope| {
        (0..4)
            .map(|_| {
                scope.spawn(|| {
                    pollster::block_on(context.realize_pipeline_layout(&pipeline_layout))
                        .expect("equal concurrent pipeline-layout realization should succeed")
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|worker| {
                worker
                    .join()
                    .expect("pipeline-layout worker must not panic")
            })
            .collect::<Vec<_>>()
    });
    assert!(
        realized_pipeline_layouts
            .iter()
            .all(|candidate| realized_pipeline_layouts[0].is_same_record(candidate)),
        "equal concurrent pipeline-layout misses must publish one authoritative layout record"
    );

    let buffer = storage_buffer("g4c2 concurrent binding values");
    let _realized_buffer = context
        .realize_buffer(&buffer)
        .expect("G4C1 buffer prerequisite should realize");
    let realized_layout = realized_layouts[0].clone();
    let realized_bind_groups = std::thread::scope(|scope| {
        (0..4)
            .map(|_| {
                scope.spawn(|| {
                    pollster::block_on(
                        context
                            .realize_bind_group(&realized_layout, [binding_value(buffer.clone())]),
                    )
                    .expect("equal concurrent bind-group realization should succeed")
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|worker| worker.join().expect("bind-group worker must not panic"))
            .collect::<Vec<_>>()
    });
    assert!(
        realized_bind_groups
            .iter()
            .all(|candidate| realized_bind_groups[0].is_same_record(candidate)),
        "equal concurrent bind-group misses must publish one authoritative group record"
    );

    let second_context = admitted_context(16)
        .expect("a second context should remain admissible after the first succeeded");
    let second_program = pollster::block_on(second_context.realize_program(&program))
        .expect("the same admitted program may realize independently in another context");
    assert!(!realized[0].is_same_record(&second_program));
    assert_ne!(realized[0].affinity(), second_program.affinity());

    let foreign_layout = pollster::block_on(second_context.realize_bind_group_layout(&layout))
        .expect("second-context layout should realize");
    let foreign = pollster::block_on(context.realize_bind_group(
        &foreign_layout,
        std::iter::empty::<GpuRuntimeBindingValue>(),
    ))
    .expect_err(
        "a bind-group layout from another context must be rejected before binding creation",
    );
    assert_eq!(
        foreign.category(),
        GpuProgramBindingRealizationErrorCategory::ForeignContext
    );
}

#[test]
fn program_admission_failures_publish_nothing_and_live_handles_bound_registry_reclamation() {
    let _serialization = GPU_REALIZATION_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(context) = context_or_skip(4) else {
        return;
    };

    let malformed_error = GpuProgramDescriptor::new(
        admitted_source("g4c2.test.malformed", INVALID_WGSL),
        [GpuEntryPointName::new("cs_main").expect("compute entry point should be valid")],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .expect_err("malformed canonical WGSL must reject at program admission");
    assert_eq!(
        malformed_error.cause(),
        GpuProgramContractCause::CanonicalWgslInvalid
    );
    assert_eq!(
        context
            .program_binding_realization_stats()
            .retained_records(),
        0,
        "program-admission failure must never publish a backend realization record"
    );

    let program = compute_program("g4c2.test.capacity", COMPUTE_WGSL);
    let realized_program = pollster::block_on(context.realize_program(&program))
        .expect("program should occupy one G4C2 record");
    let layout = GpuBindGroupLayoutDescriptor::new(0, [storage_binding()])
        .expect("test bind-group layout should be valid");
    let realized_layout = pollster::block_on(context.realize_bind_group_layout(&layout))
        .expect("layout should occupy one G4C2 record");
    let pipeline_layout =
        GpuPipelineLayoutDescriptor::new([layout]).expect("test pipeline layout should be valid");
    let realized_pipeline_layout =
        pollster::block_on(context.realize_pipeline_layout(&pipeline_layout))
            .expect("pipeline layout should occupy one G4C2 record");
    let buffer = storage_buffer("g4c2 capacity values");
    let _realized_buffer = context
        .realize_buffer(&buffer)
        .expect("G4C1 buffer prerequisite should realize");
    let realized_bind_group =
        pollster::block_on(context.realize_bind_group(&realized_layout, [binding_value(buffer)]))
            .expect("bind group should occupy one G4C2 record");
    assert_eq!(
        context
            .program_binding_realization_stats()
            .retained_records(),
        4
    );

    let replacement = compute_program("g4c2.test.capacity.replacement", COMPUTE_WGSL);
    let capacity_error = pollster::block_on(context.realize_program(&replacement))
        .expect_err("live opaque handles must prevent registry-only reclamation");
    assert_eq!(
        capacity_error.category(),
        GpuProgramBindingRealizationErrorCategory::RegistryCapacityExceeded
    );

    drop(realized_bind_group);
    drop(realized_pipeline_layout);
    drop(realized_layout);
    drop(realized_program);
    let replacement = pollster::block_on(context.realize_program(&replacement))
        .expect("lookup-only records should be reclaimed before a later capacity failure");
    assert_eq!(replacement.affinity(), context.affinity());
    let stats = context.program_binding_realization_stats();
    assert_eq!(stats.retained_records(), 1);
    assert_eq!(stats.programs(), 1);
    assert_eq!(stats.bind_group_layouts(), 0);
    assert_eq!(stats.pipeline_layouts(), 0);
    assert_eq!(stats.bind_groups(), 0);
}
