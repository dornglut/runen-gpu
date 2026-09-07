use runen_gpu::*;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

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

fn upload_graph(name: &str, values: &[u32]) -> GpuPreparedWorkGraph {
    let byte_len = u64::try_from(std::mem::size_of_val(values)).unwrap();
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let target = buffer(&mut allocator, &format!("{name} target"), byte_len);
    let target_region =
        GpuBufferRegion::new(&target, GpuBufferRange::whole(&target).unwrap()).unwrap();
    let payload = PreparedGpuData::<TransferData>::from_pod_transfer(
        format!("{name} payload"),
        values,
        provenance(&format!("{name} payload")),
    )
    .unwrap();
    let upload = GpuUploadOperation::new(target_region.into(), payload).unwrap();

    let mut builder = GpuWorkFragmentBuilder::new(label(name), provenance(name));
    builder.declare_resource(target.into()).unwrap();
    add_operation(
        &mut builder,
        &format!("{name} upload"),
        GpuWorkOperation::Upload(upload),
    );
    GpuPreparedWorkGraph::prepare(label(&format!("{name} graph")), [builder.finish().unwrap()])
        .unwrap()
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

fn request_context(policy: GpuExecutionPolicy, test_name: &str) -> Option<GpuContext> {
    let mut requirements = GpuCapabilityRequirements::new();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Copy,
        ))
        .unwrap();
    let descriptor = GpuContextDescriptor::new(requirements).with_label(test_name);
    match pollster::block_on(GpuContext::request_with_policies(
        descriptor,
        GpuRealizationPolicies::default(),
        policy,
    )) {
        Ok(context) => Some(context),
        Err(error) if error.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            eprintln!("G5B native execution environment unavailable: {error}");
            None
        }
        Err(error) => panic!("unexpected G5B context admission failure: {error}"),
    }
}

fn policy(
    max_prepared: usize,
    max_in_flight: usize,
    upload_bytes: u64,
    readback_bytes: u64,
    pending_readbacks: usize,
) -> GpuExecutionPolicy {
    GpuExecutionPolicy::new(
        NonZeroUsize::new(max_prepared).unwrap(),
        NonZeroUsize::new(max_in_flight).unwrap(),
        upload_bytes,
        readback_bytes,
        pending_readbacks,
    )
}

fn drive_submission_to_completion(context: &GpuContext, submission: &GpuSubmission) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        context.progress();
        match submission.status() {
            GpuSubmissionStatus::Completed => return,
            GpuSubmissionStatus::Failed(failure) => {
                panic!("accepted G5B submission failed: {failure:?}")
            }
            GpuSubmissionStatus::Accepted => {}
        }
        assert!(
            Instant::now() < deadline,
            "G5B submission did not terminalize"
        );
        std::thread::yield_now();
    }
}

#[test]
fn headless_upload_copy_readback_round_trip_uses_public_runengpu_lifecycle() {
    let Some(context) =
        request_context(policy(4, 2, 1024, 1024, 4), "G5B public buffer round trip")
    else {
        return;
    };
    let values = [0x0102_0304_u32, 17, 29, u32::MAX];
    let (graph, readback_id) = round_trip_graph("round trip", &values);
    assert_eq!(graph.topological_order().len(), 3);

    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    assert_eq!(prepared.planned_readbacks(), &[readback_id]);
    assert_eq!(context.execution_stats().prepared_submissions(), 1);

    let submission = context.submit_prepared(prepared).unwrap();
    assert!(submission.id().get() != 0);
    assert!(matches!(submission.status(), GpuSubmissionStatus::Accepted));
    let readback = submission
        .readback(readback_id)
        .expect("accepted readback must be observable by its logical identity")
        .clone();

    let deadline = Instant::now() + Duration::from_secs(5);
    let bytes = loop {
        context.progress();
        match readback.status() {
            GpuReadbackStatus::Ready(bytes) => break bytes,
            GpuReadbackStatus::Failed(failure) => {
                panic!("accepted G5B readback failed: {failure:?}")
            }
            GpuReadbackStatus::Pending => {}
        }
        if let GpuSubmissionStatus::Failed(failure) = submission.status() {
            panic!("accepted G5B submission failed before readback: {failure:?}");
        }
        assert!(
            Instant::now() < deadline,
            "G5B readback did not materialize"
        );
        std::thread::yield_now();
    };
    drive_submission_to_completion(&context, &submission);

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
fn prepared_capacity_is_bounded_and_drop_releases_the_record() {
    let Some(context) = request_context(policy(1, 1, 1024, 0, 0), "G5B prepared capacity and drop")
    else {
        return;
    };
    let first =
        pollster::block_on(context.prepare_submission(upload_graph("first", &[1_u32]))).unwrap();
    assert_eq!(context.execution_stats().prepared_submissions(), 1);

    let error = pollster::block_on(context.prepare_submission(upload_graph("second", &[2_u32])))
        .unwrap_err();
    assert_eq!(
        error.kind(),
        GpuSubmissionPreparationErrorKind::PreparedCapacityExceeded
    );
    assert_eq!(context.execution_stats().prepared_submissions(), 1);

    drop(first);
    assert_eq!(context.execution_stats().prepared_submissions(), 0);
    let replacement =
        pollster::block_on(context.prepare_submission(upload_graph("replacement", &[3_u32])))
            .unwrap();
    assert_eq!(context.execution_stats().prepared_submissions(), 1);
    drop(replacement);
    assert_eq!(context.execution_stats().prepared_submissions(), 0);
}

#[test]
fn pre_acceptance_pressure_rejection_preserves_prepared_ownership_for_retry() {
    let Some(context) = request_context(
        policy(2, 1, 1024, 0, 0),
        "G5B pre-acceptance retry ownership",
    ) else {
        return;
    };
    let first =
        pollster::block_on(context.prepare_submission(upload_graph("first", &[1_u32]))).unwrap();
    let second =
        pollster::block_on(context.prepare_submission(upload_graph("second", &[2_u32]))).unwrap();
    let first_submission = context.submit_prepared(first).unwrap();

    let rejected = context
        .submit_prepared(second)
        .expect_err("the occupied in-flight slot must reject before irreversible acceptance");
    assert_eq!(
        rejected.reason().kind(),
        GpuSubmissionRejectionKind::InFlightCapacityExceeded
    );
    let (second, reason) = rejected.into_parts();
    assert_eq!(
        reason.kind(),
        GpuSubmissionRejectionKind::InFlightCapacityExceeded
    );
    assert_eq!(context.execution_stats().prepared_submissions(), 1);

    drive_submission_to_completion(&context, &first_submission);
    let second_submission = context
        .submit_prepared(second)
        .expect("retry must consume the same preserved prepared authority");
    drive_submission_to_completion(&context, &second_submission);
    assert_eq!(context.execution_stats().prepared_submissions(), 0);
    assert_eq!(context.execution_stats().in_flight_submissions(), 0);
}

#[test]
fn graceful_shutdown_revokes_unaccepted_work_and_drains_accepted_readback_once() {
    let Some(context) = request_context(
        policy(4, 2, 4096, 4096, 4),
        "graceful shutdown with pending readback",
    ) else {
        return;
    };
    let values = [11_u32, 23, 47, 89];
    let (accepted_graph, readback_id) = round_trip_graph("shutdown accepted", &values);
    let accepted = pollster::block_on(context.prepare_submission(accepted_graph)).unwrap();
    let unaccepted = pollster::block_on(
        context.prepare_submission(upload_graph("shutdown unaccepted", &[5_u32, 7, 13, 17])),
    )
    .unwrap();
    assert_eq!(context.execution_stats().prepared_submissions(), 2);

    let submission = context.submit_prepared(accepted).unwrap();
    let readback = submission
        .readback(readback_id)
        .expect("accepted shutdown proof must retain its planned readback")
        .clone();
    assert!(matches!(submission.status(), GpuSubmissionStatus::Accepted));
    assert!(matches!(readback.status(), GpuReadbackStatus::Pending));
    let before_shutdown = context.execution_stats();
    assert_eq!(before_shutdown.prepared_submissions(), 1);
    assert_eq!(before_shutdown.in_flight_submissions(), 1);
    assert_eq!(before_shutdown.pending_readbacks(), 1);

    assert_eq!(
        context.begin_shutdown(),
        GpuExecutionLifecycleState::ShuttingDown
    );
    assert_eq!(
        context.begin_shutdown(),
        GpuExecutionLifecycleState::ShuttingDown,
        "graceful shutdown must be idempotent while accepted work remains"
    );
    assert_eq!(
        context.execution_lifecycle_state(),
        GpuExecutionLifecycleState::ShuttingDown
    );
    assert_eq!(context.execution_stats().prepared_submissions(), 0);

    let rejected = context
        .submit_prepared(unaccepted)
        .expect_err("shutdown must revoke unaccepted prepared authority");
    assert_eq!(
        rejected.reason().kind(),
        GpuSubmissionRejectionKind::ExecutionNotRunning
    );
    let preparation_error = pollster::block_on(
        context.prepare_submission(upload_graph("shutdown rejected", &[19_u32])),
    )
    .expect_err("shutdown must reject new preparation");
    assert_eq!(
        preparation_error.kind(),
        GpuSubmissionPreparationErrorKind::ExecutionNotRunning
    );

    let deadline = Instant::now() + Duration::from_secs(5);
    let bytes = loop {
        context.progress();
        match (submission.status(), readback.status()) {
            (GpuSubmissionStatus::Completed, GpuReadbackStatus::Ready(bytes)) => break bytes,
            (GpuSubmissionStatus::Failed(failure), _) => {
                panic!("graceful-shutdown accepted submission failed: {failure:?}")
            }
            (_, GpuReadbackStatus::Failed(failure)) => {
                panic!("graceful-shutdown accepted readback failed: {failure:?}")
            }
            _ => {}
        }
        assert_eq!(
            context.execution_lifecycle_state(),
            GpuExecutionLifecycleState::ShuttingDown,
            "shutdown must remain in progress until accepted submission and readback terminalize"
        );
        assert!(
            Instant::now() < deadline,
            "graceful shutdown did not drain accepted work"
        );
        std::thread::yield_now();
    };

    let expected = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    assert_eq!(bytes.as_bytes(), expected.as_slice());
    assert_eq!(
        context.execution_lifecycle_state(),
        GpuExecutionLifecycleState::Closed
    );
    let drained = context.execution_stats();
    assert_eq!(drained.prepared_submissions(), 0);
    assert_eq!(drained.in_flight_submissions(), 0);
    assert_eq!(drained.upload_bytes_in_flight(), 0);
    assert_eq!(drained.readback_bytes_in_flight(), 0);
    assert_eq!(drained.pending_readbacks(), 0);

    for _ in 0..3 {
        context.progress();
        assert!(matches!(
            submission.status(),
            GpuSubmissionStatus::Completed
        ));
        match readback.status() {
            GpuReadbackStatus::Ready(observed) => {
                assert_eq!(observed.as_bytes(), expected.as_slice())
            }
            other => panic!("terminal readback observation changed after completion: {other:?}"),
        }
        assert_eq!(
            context.execution_lifecycle_state(),
            GpuExecutionLifecycleState::Closed
        );
    }
}

#[test]
fn last_context_drop_terminalizes_detached_accepted_observation_without_waiting() {
    let Some(context) = request_context(
        policy(1, 1, 1024, 0, 0),
        "G5B abrupt context drop observation",
    ) else {
        return;
    };
    let prepared =
        pollster::block_on(context.prepare_submission(upload_graph("drop", &[7_u32]))).unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    assert!(matches!(submission.status(), GpuSubmissionStatus::Accepted));

    drop(context);
    assert!(matches!(
        submission.status(),
        GpuSubmissionStatus::Failed(failure)
            if failure.kind() == GpuSubmissionFailureKind::ContextDropped
    ));
}
