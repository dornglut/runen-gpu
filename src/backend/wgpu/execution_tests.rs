use super::device_request::{enforce_runengpu_instance_flags, request_with_instance};
use crate::*;
use std::num::{NonZeroU64, NonZeroUsize};
use std::time::{Duration, Instant};
use wgpu::{Backends, Instance, InstanceDescriptor, NoopBackendOptions};

const DYNAMIC_COMPUTE_WGSL: &str = r#"
@group(0) @binding(0)
var<storage, read_write> values: array<u32>;

@compute @workgroup_size(1)
fn cs_main() {
    values[0] = values[0] + 1u;
}
"#;

fn label(value: &str) -> GpuResourceLabel {
    GpuResourceLabel::new(value).unwrap()
}

fn provenance(value: &str) -> GpuResourceProvenance {
    GpuResourceProvenance::new(label(value), None, None)
}

fn common(value: &str) -> GpuResourceCommon {
    GpuResourceCommon::owned(
        label(value),
        GpuResourceLifetime::Transient,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        provenance(value),
    )
    .unwrap()
}

fn buffer(
    allocator: &mut GpuWorkResourceIdAllocator,
    name: &str,
    byte_len: u64,
) -> GpuBufferHandle {
    let resource_label = label(name);
    allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common(name),
                byte_len,
                GpuBufferUsages::new(
                    &resource_label,
                    [GpuBufferUsage::CopySource, GpuBufferUsage::CopyDestination],
                )
                .unwrap(),
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap()
}

fn compute_buffer(
    allocator: &mut GpuWorkResourceIdAllocator,
    name: &str,
    byte_len: u64,
) -> GpuBufferHandle {
    let resource_label = label(name);
    allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common(name),
                byte_len,
                GpuBufferUsages::new(
                    &resource_label,
                    [
                        GpuBufferUsage::Storage,
                        GpuBufferUsage::CopySource,
                        GpuBufferUsage::CopyDestination,
                    ],
                )
                .unwrap(),
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap()
}

fn indirect_buffer(
    allocator: &mut GpuWorkResourceIdAllocator,
    name: &str,
    byte_len: u64,
) -> GpuBufferHandle {
    let resource_label = label(name);
    allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common(name),
                byte_len,
                GpuBufferUsages::new(
                    &resource_label,
                    [GpuBufferUsage::Indirect, GpuBufferUsage::CopyDestination],
                )
                .unwrap(),
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap()
}

fn add_operation(builder: &mut GpuWorkFragmentBuilder, name: &str, operation: GpuWorkOperation) {
    builder
        .add_node(
            label(name),
            operation,
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::Automatic,
            provenance(name),
        )
        .unwrap();
}

fn round_trip_graph(name: &str, values: &[u32]) -> (GpuPreparedWorkGraph, GpuReadbackId) {
    let byte_len = u64::try_from(std::mem::size_of_val(values)).unwrap();
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let source = buffer(&mut allocator, &format!("{name} source"), byte_len);
    let destination = buffer(&mut allocator, &format!("{name} destination"), byte_len);
    let source_region =
        GpuBufferRegion::new(&source, GpuBufferRange::whole(&source).unwrap()).unwrap();
    let destination_region =
        GpuBufferRegion::new(&destination, GpuBufferRange::whole(&destination).unwrap()).unwrap();
    let payload = PreparedGpuData::<TransferData>::from_pod_transfer(
        format!("{name} payload"),
        values,
        provenance(&format!("{name} payload")),
    )
    .unwrap();
    let upload = GpuUploadOperation::new(source_region.clone().into(), payload).unwrap();
    let copy =
        GpuCopyOperation::buffer_to_buffer(source_region, destination_region.clone()).unwrap();
    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(destination_region.into(), readback_id).unwrap();

    let mut builder = GpuWorkFragmentBuilder::new(label(name), provenance(name));
    builder.declare_resource(source.into()).unwrap();
    builder.declare_resource(destination.into()).unwrap();
    add_operation(
        &mut builder,
        &format!("{name} upload"),
        GpuWorkOperation::Upload(upload),
    );
    add_operation(
        &mut builder,
        &format!("{name} copy"),
        GpuWorkOperation::Copy(copy),
    );
    add_operation(
        &mut builder,
        &format!("{name} readback"),
        GpuWorkOperation::Readback(readback),
    );
    (
        GpuPreparedWorkGraph::prepare(label(&format!("{name} graph")), [builder.finish().unwrap()])
            .unwrap(),
        readback_id,
    )
}

fn noop_instance() -> Instance {
    let mut instance_descriptor = InstanceDescriptor::new_without_display_handle();
    instance_descriptor.backends = Backends::NOOP;
    instance_descriptor.backend_options.noop = NoopBackendOptions::enabled();
    Instance::new(enforce_runengpu_instance_flags(instance_descriptor))
}

fn assert_noop_test_context_truth(context: &GpuContext) {
    assert_eq!(
        context.adapter_facts().backend(),
        GpuBackendFamily::UnknownBackend,
        "the deterministic runtime proof must not masquerade WGPU noop as a production backend"
    );
    assert_eq!(
        context.admission_report().candidate().portability(),
        GpuPortabilityClass::Unsupported,
        "the test-only seam must preserve production portability truth"
    );
}

fn noop_context() -> GpuContext {
    let mut requirements = GpuCapabilityRequirements::new();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Copy,
        ))
        .unwrap();
    let descriptor = GpuContextDescriptor::new(requirements)
        .with_allowed_backends([GpuBackendFamily::UnknownBackend])
        .with_label("G5B noop buffer proof");

    let context = pollster::block_on(request_with_instance(
        noop_instance(),
        descriptor,
        None,
        GpuRealizationPolicies::default(),
        GpuExecutionPolicy::default(),
    ))
    .expect("explicitly enabled WGPU noop backend must admit the buffer-only G5B test context");
    assert_noop_test_context_truth(&context);
    context
}

fn noop_compute_context() -> GpuContext {
    noop_compute_context_with_policy(GpuExecutionPolicy::default())
}

fn noop_compute_context_with_policy(policy: GpuExecutionPolicy) -> GpuContext {
    let mut requirements = GpuCapabilityProfile::ComputeBaseline.requirements();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Copy,
        ))
        .unwrap();
    let descriptor = GpuContextDescriptor::new(requirements)
        .with_allowed_backends([GpuBackendFamily::UnknownBackend])
        .with_label("G5B noop direct compute proof");

    let context = pollster::block_on(request_with_instance(
        noop_instance(),
        descriptor,
        None,
        GpuRealizationPolicies::default(),
        policy,
    ))
    .expect("explicitly enabled WGPU noop backend must admit the direct-compute G5B test context");
    assert_noop_test_context_truth(&context);
    context
}

fn noop_indirect_compute_context() -> GpuContext {
    let mut requirements = GpuCapabilityProfile::ComputeBaseline.requirements();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Copy,
        ))
        .unwrap();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::IndirectExecution,
        ))
        .unwrap();
    let descriptor = GpuContextDescriptor::new(requirements)
        .with_allowed_backends([GpuBackendFamily::UnknownBackend])
        .with_label("G5B noop indirect compute proof");

    let context = pollster::block_on(request_with_instance(
        noop_instance(),
        descriptor,
        None,
        GpuRealizationPolicies::default(),
        GpuExecutionPolicy::default(),
    ))
    .expect("explicitly enabled WGPU noop backend must prove indirect-compute admission");
    assert_noop_test_context_truth(&context);
    context
}

fn admitted_compute_source() -> GpuAdmittedProgramSource {
    let owner = GpuProgramSourceOwnerId::allocate().expect("test source owner should allocate");
    let identity = GpuProgramSourceIdentity::new(
        owner,
        GpuProgramSourceKey::new("g5b.noop.dynamic-compute").unwrap(),
        GpuProgramSourceRevision::try_from_raw(1).unwrap(),
    );
    let mut sources = GpuProgramSourceRegistry::new(4, 16 * 1024).unwrap();
    sources
        .admit_wgsl(
            identity,
            DYNAMIC_COMPUTE_WGSL,
            GpuProgramSourceProvenance::new("g5b-noop-direct-compute", None).unwrap(),
        )
        .unwrap()
}

fn dynamic_compute_pipeline() -> GpuComputePipelineDescriptor {
    let entry = GpuEntryPointName::new("cs_main").unwrap();
    let refinement = GpuBindingLayoutRefinement::new(GpuBindingKey::try_new(0, 0).unwrap())
        .with_dynamic_offset(true)
        .with_host_minimum_size(NonZeroU64::new(4).unwrap());
    let program =
        GpuProgramDescriptor::new(admitted_compute_source(), [entry.clone()], [refinement])
            .unwrap();
    GpuComputePipelineDescriptor::new(program, entry, GpuPipelineConfiguration::default()).unwrap()
}

fn dynamic_compute_bindings(
    pipeline: &GpuComputePipelineDescriptor,
    buffer: &GpuBufferHandle,
    dynamic_offset: u64,
) -> GpuRuntimeBindingSet {
    let binding = GpuRuntimeBindingValue::new(
        GpuBindingKey::try_new(0, 0).unwrap(),
        [GpuRuntimeBindingResource::Buffer(
            GpuRuntimeBufferBinding::new(
                buffer.clone(),
                0,
                NonZeroU64::new(4).unwrap(),
                Some(dynamic_offset),
            ),
        )],
    )
    .unwrap();
    GpuRuntimeBindingSet::new(pipeline.layout().clone(), [binding]).unwrap()
}

fn dynamic_compute_operation(
    pipeline: &GpuComputePipelineDescriptor,
    buffer: &GpuBufferHandle,
    dynamic_offset: u64,
    x: u32,
) -> GpuComputeOperation {
    let bindings = dynamic_compute_bindings(pipeline, buffer, dynamic_offset);
    let dispatch = GpuDispatchIntent::direct(GpuDispatchSize::new(x, 1, 1));
    GpuComputeOperation::new(pipeline.clone(), bindings, dispatch).unwrap()
}

fn indirect_compute_operation(
    pipeline: &GpuComputePipelineDescriptor,
    buffer: &GpuBufferHandle,
    dynamic_offset: u64,
    arguments: &GpuBufferHandle,
    indirect_offset: u64,
) -> GpuComputeOperation {
    let bindings = dynamic_compute_bindings(pipeline, buffer, dynamic_offset);
    let dispatch = GpuDispatchIntent::indirect(arguments, indirect_offset).unwrap();
    GpuComputeOperation::new(pipeline.clone(), bindings, dispatch).unwrap()
}

fn greatest_common_divisor(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn checked_least_common_multiple(left: u64, right: u64) -> Option<u64> {
    if left == 0 || right == 0 {
        return None;
    }
    left.checked_div(greatest_common_divisor(left, right))?
        .checked_mul(right)
}

fn dynamic_compute_graph(context: &GpuContext) -> (GpuPreparedWorkGraph, GpuReadbackId, Vec<u32>) {
    dynamic_compute_graph_with_first_work(context, 1, 0)
}

fn dynamic_compute_graph_with_first_dispatch(
    context: &GpuContext,
    first_dispatch_x: u32,
) -> (GpuPreparedWorkGraph, GpuReadbackId, Vec<u32>) {
    dynamic_compute_graph_with_first_work(context, first_dispatch_x, 0)
}

fn dynamic_compute_graph_with_first_work(
    context: &GpuContext,
    first_dispatch_x: u32,
    first_dynamic_offset: u64,
) -> (GpuPreparedWorkGraph, GpuReadbackId, Vec<u32>) {
    let storage_alignment = context
        .device_facts()
        .device_limits()
        .alignments()
        .storage_dynamic_offset
        .expect("compute context must publish storage dynamic-offset alignment");
    let copy_alignment = context
        .device_facts()
        .device_limits()
        .alignments()
        .copy_buffer_offset
        .expect("compute context must publish buffer-copy alignment");
    let dynamic_stride = checked_least_common_multiple(storage_alignment, copy_alignment)
        .and_then(|stride| checked_least_common_multiple(stride, 4))
        .expect("test dynamic stride must satisfy storage, copy, and u32 alignment");

    let byte_len = dynamic_stride
        .checked_mul(2)
        .expect("test buffer size must fit u64");
    let value_count = usize::try_from(byte_len / 4).unwrap();
    let second_index = usize::try_from(dynamic_stride / 4).unwrap();
    let mut values = vec![0_u32; value_count];
    values[0] = 10;
    values[second_index] = 20;

    let mut allocator = GpuWorkResourceIdAllocator::new();
    let values_buffer = compute_buffer(&mut allocator, "dynamic compute values", byte_len);
    let whole = GpuBufferRegion::new(
        &values_buffer,
        GpuBufferRange::whole(&values_buffer).unwrap(),
    )
    .unwrap();
    let payload = PreparedGpuData::<TransferData>::from_pod_transfer(
        "dynamic compute payload",
        &values,
        provenance("dynamic compute payload"),
    )
    .unwrap();
    let upload = GpuUploadOperation::new(whole.clone().into(), payload).unwrap();
    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(whole.into(), readback_id).unwrap();

    let pipeline = dynamic_compute_pipeline();
    let first = dynamic_compute_operation(
        &pipeline,
        &values_buffer,
        first_dynamic_offset,
        first_dispatch_x,
    );
    let second = dynamic_compute_operation(&pipeline, &values_buffer, dynamic_stride, 1);
    let zero = dynamic_compute_operation(&pipeline, &values_buffer, 0, 0);

    let name = "noop dynamic compute";
    let mut builder = GpuWorkFragmentBuilder::new(label(name), provenance(name));
    builder.declare_resource(values_buffer.into()).unwrap();
    add_operation(
        &mut builder,
        "dynamic compute upload",
        GpuWorkOperation::Upload(upload),
    );
    let first_id = builder.compute("dynamic compute first", first).unwrap();
    let second_id = builder.compute("dynamic compute second", second).unwrap();
    let zero_id = builder.compute("dynamic compute zero", zero).unwrap();
    builder
        .add_explicit_order(
            GpuExplicitOrder::new(&first_id, &second_id, "prove ordered dynamic-offset reuse")
                .unwrap(),
        )
        .unwrap();
    builder
        .add_explicit_order(
            GpuExplicitOrder::new(
                &second_id,
                &zero_id,
                "prove zero-dispatch execution semantics",
            )
            .unwrap(),
        )
        .unwrap();
    add_operation(
        &mut builder,
        "dynamic compute readback",
        GpuWorkOperation::Readback(readback),
    );

    (
        GpuPreparedWorkGraph::prepare(
            label("noop dynamic compute graph"),
            [builder.finish().unwrap()],
        )
        .unwrap(),
        readback_id,
        values,
    )
}

fn bind_group_limit_violation_graph(context: &GpuContext) -> GpuPreparedWorkGraph {
    let group = context
        .device_facts()
        .device_limits()
        .values()
        .max_bind_groups();
    let source_text = format!(
        r#"
@group({group}) @binding(0)
var<storage, read_write> values: array<u32>;

@compute @workgroup_size(1)
fn cs_main() {{
    values[0] = values[0] + 1u;
}}
"#
    );
    let owner = GpuProgramSourceOwnerId::allocate().expect("test source owner should allocate");
    let identity = GpuProgramSourceIdentity::new(
        owner,
        GpuProgramSourceKey::new("g5b.noop.bind-group-limit-violation").unwrap(),
        GpuProgramSourceRevision::try_from_raw(1).unwrap(),
    );
    let mut sources = GpuProgramSourceRegistry::new(4, 16 * 1024).unwrap();
    let source = sources
        .admit_wgsl(
            identity,
            &source_text,
            GpuProgramSourceProvenance::new("g5b-noop-bind-group-limit", None).unwrap(),
        )
        .unwrap();
    let entry = GpuEntryPointName::new("cs_main").unwrap();
    let key = GpuBindingKey::try_new(u64::from(group), 0).unwrap();
    let refinement =
        GpuBindingLayoutRefinement::new(key).with_host_minimum_size(NonZeroU64::new(4).unwrap());
    let program = GpuProgramDescriptor::new(source, [entry.clone()], [refinement]).unwrap();
    let pipeline =
        GpuComputePipelineDescriptor::new(program, entry, GpuPipelineConfiguration::default())
            .unwrap();

    let mut allocator = GpuWorkResourceIdAllocator::new();
    let values_buffer = compute_buffer(&mut allocator, "bind-group-limit values", 4);
    let whole = GpuBufferRegion::new(
        &values_buffer,
        GpuBufferRange::whole(&values_buffer).unwrap(),
    )
    .unwrap();
    let payload = PreparedGpuData::<TransferData>::from_pod_transfer(
        "bind-group-limit payload",
        &[41_u32],
        provenance("bind-group-limit payload"),
    )
    .unwrap();
    let upload = GpuUploadOperation::new(whole.into(), payload).unwrap();
    let binding = GpuRuntimeBindingValue::new(
        key,
        [GpuRuntimeBindingResource::Buffer(
            GpuRuntimeBufferBinding::new(
                values_buffer.clone(),
                0,
                NonZeroU64::new(4).unwrap(),
                None,
            ),
        )],
    )
    .unwrap();
    let bindings = GpuRuntimeBindingSet::new(pipeline.layout().clone(), [binding]).unwrap();
    let compute = GpuComputeOperation::new(
        pipeline,
        bindings,
        GpuDispatchIntent::direct(GpuDispatchSize::new(1, 1, 1)),
    )
    .unwrap();

    let name = "noop bind-group-limit violation";
    let mut builder = GpuWorkFragmentBuilder::new(label(name), provenance(name));
    builder.declare_resource(values_buffer.into()).unwrap();
    add_operation(
        &mut builder,
        "bind-group-limit upload",
        GpuWorkOperation::Upload(upload),
    );
    builder
        .compute("bind-group-limit compute", compute)
        .unwrap();
    GpuPreparedWorkGraph::prepare(
        label("noop bind-group-limit violation graph"),
        [builder.finish().unwrap()],
    )
    .unwrap()
}

fn indirect_compute_graph() -> (GpuPreparedWorkGraph, GpuReadbackId, [u32; 1]) {
    let values = [41_u32];
    let arguments = [1_u32, 1, 1];
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let values_buffer = compute_buffer(&mut allocator, "indirect compute values", 4);
    let arguments_buffer = indirect_buffer(&mut allocator, "indirect compute arguments", 12);
    let values_region = GpuBufferRegion::new(
        &values_buffer,
        GpuBufferRange::whole(&values_buffer).unwrap(),
    )
    .unwrap();
    let arguments_region = GpuBufferRegion::new(
        &arguments_buffer,
        GpuBufferRange::whole(&arguments_buffer).unwrap(),
    )
    .unwrap();
    let values_upload = GpuUploadOperation::new(
        values_region.clone().into(),
        PreparedGpuData::<TransferData>::from_pod_transfer(
            "indirect compute values payload",
            &values,
            provenance("indirect compute values payload"),
        )
        .unwrap(),
    )
    .unwrap();
    let arguments_upload = GpuUploadOperation::new(
        arguments_region.into(),
        PreparedGpuData::<TransferData>::from_pod_transfer(
            "indirect compute arguments payload",
            &arguments,
            provenance("indirect compute arguments payload"),
        )
        .unwrap(),
    )
    .unwrap();
    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(values_region.into(), readback_id).unwrap();
    let pipeline = dynamic_compute_pipeline();
    let compute = indirect_compute_operation(&pipeline, &values_buffer, 0, &arguments_buffer, 0);

    let name = "noop indirect compute";
    let mut builder = GpuWorkFragmentBuilder::new(label(name), provenance(name));
    builder.declare_resource(values_buffer.into()).unwrap();
    builder.declare_resource(arguments_buffer.into()).unwrap();
    add_operation(
        &mut builder,
        "indirect compute values upload",
        GpuWorkOperation::Upload(values_upload),
    );
    add_operation(
        &mut builder,
        "indirect compute arguments upload",
        GpuWorkOperation::Upload(arguments_upload),
    );
    builder
        .compute("indirect compute dispatch", compute)
        .unwrap();
    add_operation(
        &mut builder,
        "indirect compute readback",
        GpuWorkOperation::Readback(readback),
    );

    (
        GpuPreparedWorkGraph::prepare(
            label("noop indirect compute graph"),
            [builder.finish().unwrap()],
        )
        .unwrap(),
        readback_id,
        values,
    )
}

fn progress_to_readback(
    context: &GpuContext,
    submission: &GpuSubmission,
    readback: &GpuReadback,
) -> GpuReadbackBytes {
    let deadline = Instant::now() + Duration::from_secs(5);
    let bytes = loop {
        context.progress();
        match readback.status() {
            GpuReadbackStatus::Ready(bytes) => break bytes,
            GpuReadbackStatus::Failed(failure) => panic!("G5B readback failed: {failure:?}"),
            GpuReadbackStatus::Pending => {}
        }
        if let GpuSubmissionStatus::Failed(failure) = submission.status() {
            panic!("G5B submission failed before readback: {failure:?}");
        }
        assert!(
            Instant::now() < deadline,
            "G5B readback did not materialize"
        );
        std::thread::yield_now();
    };

    loop {
        context.progress();
        match submission.status() {
            GpuSubmissionStatus::Completed => break,
            GpuSubmissionStatus::Failed(failure) => panic!("G5B submission failed: {failure:?}"),
            GpuSubmissionStatus::Accepted => {}
        }
        assert!(
            Instant::now() < deadline,
            "G5B submission did not terminalize"
        );
        std::thread::yield_now();
    }
    bytes
}

#[test]
fn noop_backend_buffer_round_trip_proves_first_g5b_checkpoint_runtime() {
    let context = noop_context();
    let values = [0x0102_0304_u32, 17, 29, u32::MAX];
    let (graph, readback_id) = round_trip_graph("noop round trip", &values);

    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let readback = submission
        .readback(readback_id)
        .expect("accepted readback must remain observable")
        .clone();
    let bytes = progress_to_readback(&context, &submission, &readback);

    let expected = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    assert_eq!(bytes.as_bytes(), expected.as_slice());
    assert_eq!(bytes.layout().byte_len(), expected.len() as u64);
    assert_eq!(bytes.texture_format(), None);

    let stats = context.execution_stats();
    assert_eq!(stats.prepared_submissions(), 0);
    assert_eq!(stats.in_flight_submissions(), 0);
    assert_eq!(stats.upload_bytes_in_flight(), 0);
    assert_eq!(stats.readback_bytes_in_flight(), 0);
    assert_eq!(stats.pending_readbacks(), 0);
}

#[test]
fn contextual_work_validation_precedes_prepared_capacity_reservation() {
    let defaults = GpuExecutionPolicy::default();
    let policy = GpuExecutionPolicy::new(
        NonZeroUsize::new(1).unwrap(),
        defaults.max_in_flight_submissions(),
        defaults.max_upload_bytes_in_flight(),
        defaults.max_readback_bytes_in_flight(),
        defaults.max_pending_readbacks(),
    );
    let context = noop_compute_context_with_policy(policy);
    let (valid_graph, _, _) = dynamic_compute_graph(&context);
    let held = pollster::block_on(context.prepare_submission(valid_graph))
        .expect("valid work must occupy the sole prepared-capacity slot");
    assert_eq!(context.execution_stats().prepared_submissions(), 1);

    let admitted_max = context
        .device_facts()
        .workload_budget()
        .limits()
        .max_compute_workgroups_per_dimension();
    let invalid_dispatch = admitted_max
        .checked_add(1)
        .expect("noop compute limit must leave room for one invalid dispatch value");
    let (invalid_graph, _, _) =
        dynamic_compute_graph_with_first_dispatch(&context, invalid_dispatch);
    let error = pollster::block_on(context.prepare_submission(invalid_graph))
        .expect_err("device-invalid work must reject before prepared-capacity reservation");
    assert_eq!(
        error.kind(),
        GpuSubmissionPreparationErrorKind::WorkNotAdmitted,
        "contextual work validation must outrank prepared-capacity exhaustion"
    );
    let Some(GpuWorkNotAdmittedSource::Operation(source)) = error.work_not_admitted_source() else {
        panic!("direct-dispatch admission rejection must preserve its typed operation source");
    };
    assert_eq!(
        source.cause(),
        GpuWorkOperationCause::MechanicalCapabilityContradiction
    );
    assert_eq!(
        context.execution_stats().prepared_submissions(),
        1,
        "rejected device-invalid work must not reserve or release the valid prepared slot"
    );

    drop(held);
    assert_eq!(context.execution_stats().prepared_submissions(), 0);
}

#[test]
fn contextual_binding_validation_preserves_typed_program_contract_source() {
    let context = noop_compute_context();
    let invalid_graph = bind_group_limit_violation_graph(&context);
    let error = pollster::block_on(context.prepare_submission(invalid_graph))
        .expect_err("device-invalid runtime binding must reject during contextual validation");
    assert_eq!(
        error.kind(),
        GpuSubmissionPreparationErrorKind::WorkNotAdmitted
    );
    let Some(GpuWorkNotAdmittedSource::ProgramContract(source)) = error.work_not_admitted_source()
    else {
        panic!("runtime-binding admission rejection must preserve its typed program source");
    };
    assert_eq!(
        source.cause(),
        GpuProgramContractCause::RuntimeBindingIncompatible
    );
    assert_eq!(context.execution_stats().prepared_submissions(), 0);
}

#[test]
fn noop_backend_proves_direct_compute_encoding_and_dynamic_bind_group_reuse() {
    let context = noop_compute_context();
    let (graph, readback_id, expected) = dynamic_compute_graph(&context);

    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    assert_eq!(
        context.program_binding_realization_stats().bind_groups(),
        1,
        "per-use dynamic offsets must not split physical bind-group identity"
    );
    let submission = context.submit_prepared(prepared).unwrap();
    let readback = submission
        .readback(readback_id)
        .expect("accepted compute readback must remain observable")
        .clone();
    let bytes = progress_to_readback(&context, &submission, &readback);

    let expected_bytes = expected
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    assert_eq!(
        bytes.as_bytes(),
        expected_bytes.as_slice(),
        "WGPU Noop performs no computation; unchanged bytes prove only the real G5B preparation/encoding/submission/readback path, not shader execution"
    );

    let stats = context.execution_stats();
    assert_eq!(stats.prepared_submissions(), 0);
    assert_eq!(stats.in_flight_submissions(), 0);
    assert_eq!(stats.upload_bytes_in_flight(), 0);
    assert_eq!(stats.readback_bytes_in_flight(), 0);
    assert_eq!(stats.pending_readbacks(), 0);
}

#[test]
fn noop_backend_proves_indirect_compute_preparation_encoding_and_lifecycle() {
    let context = noop_indirect_compute_context();
    let (graph, readback_id, expected) = indirect_compute_graph();

    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    assert_eq!(context.execution_stats().prepared_submissions(), 1);
    let submission = context.submit_prepared(prepared).unwrap();
    let readback = submission
        .readback(readback_id)
        .expect("accepted indirect-compute readback must remain observable")
        .clone();
    let bytes = progress_to_readback(&context, &submission, &readback);

    let expected_bytes = expected
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    assert_eq!(
        bytes.as_bytes(),
        expected_bytes.as_slice(),
        "WGPU Noop performs no computation; unchanged bytes prove indirect preparation/encoding/submission/readback without claiming shader execution"
    );

    let stats = context.execution_stats();
    assert_eq!(stats.prepared_submissions(), 0);
    assert_eq!(stats.in_flight_submissions(), 0);
    assert_eq!(stats.upload_bytes_in_flight(), 0);
    assert_eq!(stats.readback_bytes_in_flight(), 0);
    assert_eq!(stats.pending_readbacks(), 0);
}

#[test]
fn noop_backend_graceful_shutdown_revokes_prepared_and_drains_accepted_work() {
    let context = noop_context();
    assert_eq!(
        context.execution_lifecycle_state(),
        GpuExecutionLifecycleState::Running
    );

    let accepted_values = [3_u32, 5, 8, 13];
    let (accepted_graph, readback_id) =
        round_trip_graph("noop shutdown accepted", &accepted_values);
    let (revoked_graph, _) = round_trip_graph("noop shutdown revoked", &[21_u32, 34, 55, 89]);
    let accepted_prepared = pollster::block_on(context.prepare_submission(accepted_graph)).unwrap();
    let revoked_prepared = pollster::block_on(context.prepare_submission(revoked_graph)).unwrap();
    assert_eq!(context.execution_stats().prepared_submissions(), 2);

    let submission = context.submit_prepared(accepted_prepared).unwrap();
    let readback = submission
        .readback(readback_id)
        .expect("accepted shutdown readback must remain observable")
        .clone();
    assert_eq!(context.execution_stats().prepared_submissions(), 1);

    assert_eq!(
        context.begin_shutdown(),
        GpuExecutionLifecycleState::ShuttingDown
    );
    assert_eq!(
        context.execution_lifecycle_state(),
        GpuExecutionLifecycleState::ShuttingDown
    );
    assert_eq!(context.execution_stats().prepared_submissions(), 0);

    let rejected = context
        .submit_prepared(revoked_prepared)
        .expect_err("shutdown must reject acceptance of a revoked prepared ticket");
    assert_eq!(
        rejected.reason().kind(),
        GpuSubmissionRejectionKind::ExecutionNotRunning
    );
    drop(rejected);

    let (new_graph, _) = round_trip_graph("noop shutdown new preparation", &[1_u32, 2, 3, 4]);
    let error = pollster::block_on(context.prepare_submission(new_graph))
        .expect_err("shutdown must reject new preparation");
    assert_eq!(
        error.kind(),
        GpuSubmissionPreparationErrorKind::ExecutionNotRunning
    );

    let bytes = progress_to_readback(&context, &submission, &readback);
    let expected = accepted_values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    assert_eq!(bytes.as_bytes(), expected.as_slice());
    assert_eq!(
        context.execution_lifecycle_state(),
        GpuExecutionLifecycleState::Closed
    );
    assert_eq!(context.begin_shutdown(), GpuExecutionLifecycleState::Closed);

    let stats = context.execution_stats();
    assert_eq!(stats.prepared_submissions(), 0);
    assert_eq!(stats.in_flight_submissions(), 0);
    assert_eq!(stats.upload_bytes_in_flight(), 0);
    assert_eq!(stats.readback_bytes_in_flight(), 0);
    assert_eq!(stats.pending_readbacks(), 0);
}
