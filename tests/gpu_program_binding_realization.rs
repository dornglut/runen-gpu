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
    GpuResourceLifetime, GpuResourceProvenance, GpuResourceRealizationErrorCategory,
    GpuResourceRealizationPolicy, GpuRuntimeBindingResource, GpuRuntimeBindingValue,
    GpuRuntimeBufferBinding, GpuShaderStage, GpuShaderStages, GpuStorageBufferAccess,
    GpuWorkResourceIdAllocator,
};
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Mutex;

#[path = "support/readback_wait.rs"]
mod readback_wait;

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

fn cross_registry_context_or_skip(
    resource_limit: usize,
    binding_limit: usize,
) -> Option<GpuContext> {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::ComputeBaseline.requirements())
            .with_label("G4C1/G4C2 independent-pressure investigation");
    cross_registry_context_with_descriptor_or_skip(descriptor, resource_limit, binding_limit)
}

fn cross_registry_context_with_descriptor_or_skip(
    descriptor: GpuContextDescriptor,
    resource_limit: usize,
    binding_limit: usize,
) -> Option<GpuContext> {
    let context = match pollster::block_on(GpuContext::request_with_realization_policies(
        descriptor,
        GpuRealizationPolicies::new(
            GpuResourceRealizationPolicy::new(NonZeroUsize::new(resource_limit).unwrap()),
            GpuProgramBindingRealizationPolicy::new(NonZeroUsize::new(binding_limit).unwrap()),
        ),
    )) {
        Ok(context) => context,
        Err(error)
            if matches!(
                error.category(),
                GpuContextRequestErrorCategory::NoAdapterAvailable
                    | GpuContextRequestErrorCategory::NoAdmissibleCandidate
                    | GpuContextRequestErrorCategory::MandatoryFeatureMissing
            ) =>
        {
            if std::env::var_os("RUNEN_GPU_REQUIRE_GPU").is_some() {
                panic!("GPU-required cross-registry experiment could not run: {error}");
            }
            eprintln!("G4C1/G4C2 cross-registry experiment unavailable: {error}");
            return None;
        }
        Err(error) => panic!("unexpected investigation context admission failure: {error}"),
    };
    eprintln!(
        "CROSS_REGISTRY_ADAPTER backend={:?} name={:?} software={:?}",
        context.adapter_facts().backend(),
        context.adapter_facts().diagnostic_name(),
        context.adapter_facts().software(),
    );
    Some(context)
}

// Regression #176: resource pressure reclaims abandoned bind groups below G4C2 capacity.
#[test]
fn g4c1_pressure_reclaims_lookup_only_groups_below_g4c2_capacity() {
    let _serialization = GPU_REALIZATION_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(context) = cross_registry_context_or_skip(4, 6) else {
        return;
    };
    let layout =
        GpuBindGroupLayoutDescriptor::new(0, [storage_binding()]).expect("valid storage layout");
    let realized_layout =
        pollster::block_on(context.realize_bind_group_layout(&layout)).expect("layout realization");

    // The four buffers are distinct transient logical objects, not reused/aliased IDs.
    // After each bind-group realization the caller releases *both* physical handles.
    // No G5 submission has been prepared or accepted.
    let mut allocator = GpuWorkResourceIdAllocator::new();
    for frame in 0..4 {
        let name = format!("cross-registry-transient-{frame}");
        let label = GpuResourceLabel::new(&name).expect("valid label");
        let common = GpuResourceCommon::owned(
            label.clone(),
            GpuResourceLifetime::Transient,
            GpuMemoryIntent::Device,
            GpuReconstruction::SourceBacked,
            GpuResourceProvenance::new(label, None, None),
        )
        .expect("transient resource common");
        let usages =
            GpuBufferUsages::new(common.label(), [GpuBufferUsage::Storage]).expect("storage usage");
        let buffer = allocator
            .allocate_buffer_handle(
                GpuBufferDescriptor::new(
                    common,
                    64,
                    usages,
                    GpuBufferInitialization::Uninitialized,
                )
                .expect("transient buffer descriptor"),
            )
            .expect("unique logical buffer ID");
        let physical = context.realize_buffer(&buffer).expect("G4C1 realization");
        let group = pollster::block_on(
            context.realize_bind_group(&realized_layout, [binding_value(buffer)]),
        )
        .expect("G4C2 binding realization");
        drop(group);
        drop(physical);
    }
    let g4c1 = context.resource_realization_stats();
    let g4c2 = context.program_binding_realization_stats();
    assert_eq!(g4c1.retained_records(), 4);
    assert_eq!(g4c1.buffers(), 4);
    assert_eq!(g4c2.bind_groups(), 4);
    assert_eq!(g4c2.bind_group_layouts(), 1);
    assert_eq!(g4c2.retained_records(), 5);
    let g4c2_limit = context
        .program_binding_realization_policy()
        .max_records()
        .get();
    assert!(g4c2.retained_records() < g4c2_limit);
    assert_eq!(g4c2.in_flight_records(), 0);
    assert_eq!(context.execution_stats().in_flight_submissions(), 0);

    // G4C2 is below its own cap. G4C1 pressure must release only registry-owned
    // dependent groups and then admit the resource without a G4C2 realization.
    let candidate = storage_buffer("cross-registry-candidate");
    let recovered = context
        .realize_buffer(&candidate)
        .expect("G4C1 pressure must reclaim abandoned G4C2 dependencies");
    assert_eq!(context.program_binding_realization_stats().bind_groups(), 0);
    assert_eq!(
        context
            .program_binding_realization_stats()
            .bind_group_layouts(),
        1
    );
    assert_eq!(recovered.affinity(), context.affinity());
    let after = context.resource_realization_stats();
    assert_eq!(after.retained_records(), 1);
    assert_eq!(after.buffers(), 1);
    eprintln!(
        "CROSS_REGISTRY_PROOF g4c1_before={} g4c2_bind_groups_before={} g4c1_after={} g4c2_bind_groups_after={}",
        g4c1.retained_records(),
        g4c2.bind_groups(),
        after.retained_records(),
        context.program_binding_realization_stats().bind_groups(),
    );
}

// The retained group is the only external physical owner: the direct buffer handle is dropped.
// This control must keep rejecting G4C1 pressure until that group itself is released.
#[test]
fn caller_retained_bind_group_protects_g4c1_dependency_during_registry_pressure() {
    let _serialization = GPU_REALIZATION_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(context) = cross_registry_context_or_skip(1, 2) else {
        return;
    };
    let layout =
        GpuBindGroupLayoutDescriptor::new(0, [storage_binding()]).expect("valid storage layout");
    let realized_layout =
        pollster::block_on(context.realize_bind_group_layout(&layout)).expect("layout realization");
    let buffer = storage_buffer("caller-retained group dependency");
    let physical = context.realize_buffer(&buffer).expect("G4C1 realization");
    let group = pollster::block_on(
        context.realize_bind_group(&realized_layout, [binding_value(buffer.clone())]),
    )
    .expect("G4C2 binding realization");
    drop(physical);
    assert_eq!(context.resource_realization_stats().buffers(), 1);
    assert_eq!(
        context
            .program_binding_realization_stats()
            .retained_records(),
        2
    );
    assert_eq!(context.execution_stats().in_flight_submissions(), 0);

    let replacement_layout =
        GpuBindGroupLayoutDescriptor::new(1, std::iter::empty::<GpuBindingDeclaration>())
            .expect("independent layout");
    let rejected = pollster::block_on(context.realize_bind_group_layout(&replacement_layout))
        .expect_err("G4C2 pressure must preserve the caller-held group and its layout");
    assert_eq!(
        rejected.category(),
        GpuProgramBindingRealizationErrorCategory::RegistryCapacityExceeded
    );
    let candidate = storage_buffer("caller-retained group capacity candidate");
    let rejected = context
        .realize_buffer(&candidate)
        .expect_err("the group alone must protect its G4C1 dependency");
    assert_eq!(
        rejected.category(),
        GpuResourceRealizationErrorCategory::RegistryCapacityExceeded
    );
    let repeated_group =
        pollster::block_on(context.realize_bind_group(&realized_layout, [binding_value(buffer)]))
            .expect("the live group remains usable after both pressure attempts");
    assert!(group.is_same_record(&repeated_group));
    assert_eq!(context.program_binding_realization_stats().bind_groups(), 1);
    assert_eq!(context.resource_realization_stats().buffers(), 1);

    drop(repeated_group);
    drop(group);
    let recovered = context
        .realize_buffer(&candidate)
        .expect("G4C1 pressure reclaims dependencies only after the caller releases the group");
    assert_eq!(context.program_binding_realization_stats().bind_groups(), 0);
    assert_eq!(recovered.affinity(), context.affinity());
    assert_eq!(context.resource_realization_stats().retained_records(), 1);
    eprintln!(
        "CROSS_REGISTRY_LIVE_GROUP_PROOF protected_buffers=1 released_groups=0 recovered_buffers=1"
    );
}

#[test]
fn prepared_and_accepted_compute_keep_dependencies_until_terminal_readback() {
    use runen_gpu::*;
    let _serialization = GPU_REALIZATION_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(context) = cross_registry_context_or_skip(1, 16) else {
        return;
    };
    let mut scope = GpuResourceScope::new();
    let data = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
        "pressure compute initial values",
        &[1_u32; 16],
    )
    .unwrap();
    let buffer = scope
        .buffer(
            GpuBufferDescriptor::ordinary_owned(
                "pressure compute values",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                64,
                [
                    GpuBufferUsage::Storage,
                    GpuBufferUsage::CopySource,
                    GpuBufferUsage::CopyDestination,
                ],
                GpuBufferInitialization::Prepared(data),
            )
            .unwrap(),
        )
        .unwrap();
    let pipeline = GpuComputePipelineDescriptor::new(
        compute_program("pressure.compute", COMPUTE_WGSL),
        GpuEntryPointName::new("cs_main").unwrap(),
        GpuPipelineConfiguration::default(),
    )
    .unwrap();
    let bindings =
        GpuRuntimeBindingSet::new(pipeline.layout().clone(), [binding_value(buffer.clone())])
            .unwrap();
    let compute = GpuComputeOperation::new(
        pipeline,
        bindings,
        GpuDispatchIntent::direct(GpuDispatchSize::new(1, 1, 1)),
    )
    .unwrap();
    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(
        GpuBufferRegion::new(&buffer, GpuBufferRange::whole(&buffer).unwrap())
            .unwrap()
            .into(),
        readback_id,
    )
    .unwrap();
    let fragment = GpuWorkFragment::build("pressure compute", |builder| {
        builder.compute("increment under pressure", compute)?;
        builder.operation("read increment", readback)?;
        Ok(())
    })
    .unwrap();
    let graph = GpuPreparedWorkGraph::prepare(
        GpuResourceLabel::new("pressure compute graph").unwrap(),
        [fragment],
    )
    .unwrap();
    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    assert_eq!(context.program_binding_realization_stats().bind_groups(), 1);
    let candidate = storage_buffer("pressure while G5 owns compute resources");
    let rejected = context
        .realize_buffer(&candidate)
        .expect_err("prepared work must retain its bind-group dependency");
    assert_eq!(
        rejected.category(),
        GpuResourceRealizationErrorCategory::RegistryCapacityExceeded
    );

    let submission = context.submit_prepared(prepared).unwrap();
    // GPU completion may race this observation. A successful collection is legal
    // only after the actual G5 terminal callback has released the physical plan.
    let status_before_pressure = submission.status();
    match context.realize_buffer(&candidate) {
        Ok(realized) => {
            assert_eq!(submission.status(), GpuSubmissionStatus::Completed);
            drop(realized);
        }
        Err(error) => assert_eq!(
            error.category(),
            GpuResourceRealizationErrorCategory::RegistryCapacityExceeded
        ),
    }
    eprintln!("CROSS_REGISTRY_G5_ACCEPTED_OBSERVATION status={status_before_pressure:?}");
    let bytes = pollster::block_on(readback_wait::wait_for_readback(
        &context,
        &submission,
        readback_id,
        "pressure protected compute",
    ));
    let mut expected = [1_u32; 16];
    expected[0] = 2;
    assert_eq!(bytes.as_bytes(), bytemuck::cast_slice(&expected));
    assert_eq!(submission.status(), GpuSubmissionStatus::Completed);
    assert_eq!(context.execution_stats().in_flight_submissions(), 0);
    let recovered = context.realize_buffer(&candidate).expect(
        "terminal work must release dependencies despite retaining its submission/readback handles",
    );
    assert_eq!(recovered.affinity(), context.affinity());
    assert_eq!(context.resource_realization_stats().retained_records(), 1);
    assert_eq!(context.program_binding_realization_stats().bind_groups(), 0);
    eprintln!(
        "CROSS_REGISTRY_G5_PROOF prepared_protected=true readback_exact=true terminal_reclaimed=true"
    );
}

#[test]
fn repeated_transient_binding_realization_remains_within_unchanged_limits() {
    let _serialization = GPU_REALIZATION_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(context) = cross_registry_context_or_skip(4, 6) else {
        return;
    };
    let layout = GpuBindGroupLayoutDescriptor::new(0, [storage_binding()]).unwrap();
    let realized_layout = pollster::block_on(context.realize_bind_group_layout(&layout)).unwrap();
    for ordinal in 0..1024 {
        let buffer = storage_buffer(&format!("bounded realization {ordinal}"));
        let group = pollster::block_on(
            context.realize_bind_group(&realized_layout, [binding_value(buffer)]),
        )
        .expect("abandoned dependencies must not accumulate into permanent refusal");
        drop(group);
        assert!(context.resource_realization_stats().retained_records() <= 4);
        assert!(
            context
                .program_binding_realization_stats()
                .retained_records()
                <= 6
        );
    }
    assert_eq!(context.execution_stats().in_flight_submissions(), 0);
    eprintln!("CROSS_REGISTRY_REPETITION_PROOF iterations=1024 resource_limit=4 binding_limit=6");
}

#[test]
fn concurrent_binding_realizations_remain_valid_during_resource_pressure() {
    let _serialization = GPU_REALIZATION_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(context) = cross_registry_context_or_skip(8, 16) else {
        return;
    };
    let layout = GpuBindGroupLayoutDescriptor::new(0, [storage_binding()]).unwrap();
    let realized_layout = pollster::block_on(context.realize_bind_group_layout(&layout)).unwrap();
    std::thread::scope(|scope| {
        let workers = (0..4)
            .map(|worker| {
                let context = &context;
                let layout = &realized_layout;
                scope.spawn(move || {
                    for ordinal in 0..128 {
                        let buffer =
                            storage_buffer(&format!("pressure worker {worker} item {ordinal}"));
                        let group = pollster::block_on(
                            context.realize_bind_group(layout, [binding_value(buffer.clone())]),
                        )
                        .expect("live cohort fits unchanged limits");
                        let repeated = pollster::block_on(
                            context.realize_bind_group(layout, [binding_value(buffer)]),
                        )
                        .expect("held group must survive concurrent sweeps");
                        assert!(group.is_same_record(&repeated));
                    }
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker
                .join()
                .expect("pressure worker must finish without deadlock/panic");
        }
    });
    assert!(context.resource_realization_stats().retained_records() <= 8);
    assert!(
        context
            .program_binding_realization_stats()
            .retained_records()
            <= 16
    );
    assert_eq!(
        context
            .program_binding_realization_stats()
            .in_flight_records(),
        0
    );
}

#[test]
fn mixed_binding_cleanup_reclaims_views_before_parent_textures_and_preserves_live_views() {
    use runen_gpu::*;
    let _serialization = GPU_REALIZATION_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for keep_view in [false, true] {
        let descriptor =
            GpuContextDescriptor::new(GpuCapabilityProfile::ComputeBaseline.requirements())
                .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::Sampled);
        let Some(context) = cross_registry_context_with_descriptor_or_skip(descriptor, 4, 16)
        else {
            return;
        };
        let mut scope = GpuResourceScope::new();
        let texture_common = owned_common("pressure sampled texture");
        let extent =
            GpuTextureExtent::new(texture_common.label(), GpuTextureDimension::D2, 1, 1, 1)
                .unwrap();
        let usages =
            GpuTextureUsages::new(texture_common.label(), [GpuTextureUsage::Sampled]).unwrap();
        let texture = scope
            .texture(
                GpuTextureDescriptor::new(
                    texture_common,
                    GpuTextureDimension::D2,
                    extent,
                    1,
                    1,
                    GpuTextureFormat::Rgba8Unorm,
                    usages,
                    GpuTextureInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();
        let view = scope
            .texture_view(
                GpuTextureViewDescriptor::new(
                    owned_common("pressure sampled view"),
                    &texture,
                    None,
                    GpuTextureViewDimension::D2,
                    GpuTextureSubresourceRange::whole(&texture).unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        let sampler = scope
            .sampler(
                GpuSamplerDescriptor::new(
                    owned_common("pressure sampler"),
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
        let buffer = storage_buffer("pressure mixed buffer");
        let source = admitted_source(
            "pressure.mixed",
            r#"
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var<storage, read_write> output: array<f32, 4>;
@compute @workgroup_size(1) fn cs_main() {
    output[0] = textureSampleLevel(source, linear_sampler, vec2<f32>(0.5), 0.0).r;
}"#,
        );
        let program = GpuProgramDescriptor::new(
            source,
            [GpuEntryPointName::new("cs_main").unwrap()],
            [
                GpuBindingLayoutRefinement::new(GpuBindingKey::try_new(0, 0).unwrap())
                    .with_texture_sample_class(GpuTextureSampleClass::FloatFilterable),
                GpuBindingLayoutRefinement::new(GpuBindingKey::try_new(0, 1).unwrap())
                    .with_sampler_class(GpuSamplerClass::Filtering),
            ],
        )
        .unwrap();
        let pipeline = GpuComputePipelineDescriptor::new(
            program,
            GpuEntryPointName::new("cs_main").unwrap(),
            GpuPipelineConfiguration::default(),
        )
        .unwrap();
        let layout = pipeline.layout().groups().next().unwrap();
        let layout = pollster::block_on(context.realize_bind_group_layout(layout)).unwrap();
        let group = pollster::block_on(
            context.realize_bind_group(
                &layout,
                [
                    GpuRuntimeBindingValue::new(
                        GpuBindingKey::try_new(0, 0).unwrap(),
                        [GpuRuntimeBindingResource::TextureView(
                            GpuRuntimeTextureViewBinding::new(view.clone()),
                        )],
                    )
                    .unwrap(),
                    GpuRuntimeBindingValue::new(
                        GpuBindingKey::try_new(0, 1).unwrap(),
                        [GpuRuntimeBindingResource::Sampler(sampler)],
                    )
                    .unwrap(),
                    GpuRuntimeBindingValue::whole_buffer(0, 2, &buffer),
                ],
            ),
        )
        .unwrap();
        let protected_view = if keep_view {
            let parent = context.realize_texture(&texture).unwrap();
            Some(context.realize_texture_view(&view, &parent).unwrap())
        } else {
            None
        };
        assert_eq!(context.resource_realization_stats().retained_records(), 4);
        drop(group);
        let candidate = storage_buffer("mixed cleanup candidate");
        let recovered = context.realize_buffer(&candidate).unwrap();
        let stats = context.resource_realization_stats();
        assert_eq!(stats.textures(), usize::from(keep_view));
        assert_eq!(stats.texture_views(), usize::from(keep_view));
        assert_eq!(stats.samplers(), 0);
        assert_eq!(stats.buffers(), 1);
        assert_eq!(context.program_binding_realization_stats().bind_groups(), 0);
        drop(recovered);
        drop(protected_view);
        // At most four more misses reach pressure regardless of the last cohort size.
        for ordinal in 0..4 {
            drop(
                context
                    .realize_buffer(&storage_buffer(&format!(
                        "released view pressure {ordinal}"
                    )))
                    .unwrap(),
            );
        }
        let after = context.resource_realization_stats();
        assert_eq!(after.textures(), 0);
        assert_eq!(after.texture_views(), 0);
        assert_eq!(after.samplers(), 0);
    }
}
