use super::device_request::{enforce_runengpu_instance_flags, request_with_instance};
use crate::*;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};
use wgpu::{Backends, Instance, InstanceDescriptor, NoopBackendOptions};

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

fn upload_graph(name: &str, value: u32) -> GpuPreparedWorkGraph {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let resource_label = label(&format!("{name} buffer"));
    let buffer = allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common(&format!("{name} buffer")),
                4,
                GpuBufferUsages::new(&resource_label, [GpuBufferUsage::CopyDestination]).unwrap(),
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let region = GpuBufferRegion::new(&buffer, GpuBufferRange::whole(&buffer).unwrap()).unwrap();
    let payload = PreparedGpuData::<TransferData>::from_pod_transfer(
        format!("{name} payload"),
        &[value],
        provenance(&format!("{name} payload")),
    )
    .unwrap();
    let upload = GpuUploadOperation::new(region.into(), payload).unwrap();

    let mut builder = GpuWorkFragmentBuilder::new(label(name), provenance(name));
    builder.declare_resource(buffer.into()).unwrap();
    builder
        .add_node(
            label(&format!("{name} upload")),
            GpuWorkOperation::Upload(upload),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::Automatic,
            provenance(&format!("{name} upload")),
        )
        .unwrap();

    GpuPreparedWorkGraph::prepare(label(&format!("{name} graph")), [builder.finish().unwrap()])
        .unwrap()
}

fn noop_instance() -> Instance {
    let mut descriptor = InstanceDescriptor::new_without_display_handle();
    descriptor.backends = Backends::NOOP;
    descriptor.backend_options.noop = NoopBackendOptions::enabled();
    Instance::new(enforce_runengpu_instance_flags(descriptor))
}

fn noop_context_with_policy(policy: GpuExecutionPolicy) -> GpuContext {
    let mut requirements = GpuCapabilityRequirements::new();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Copy,
        ))
        .unwrap();
    let descriptor = GpuContextDescriptor::new(requirements)
        .with_allowed_backends([GpuBackendFamily::UnknownBackend])
        .with_label("G8-D01 in-flight pressure proof");

    let context = pollster::block_on(request_with_instance(
        noop_instance(),
        descriptor,
        None,
        GpuRealizationPolicies::default(),
        policy,
    ))
    .expect("explicitly enabled WGPU noop backend must admit the pressure-proof context");
    assert_eq!(
        context.adapter_facts().backend(),
        GpuBackendFamily::UnknownBackend
    );
    assert_eq!(
        context.admission_report().candidate().portability(),
        GpuPortabilityClass::Unsupported,
        "the deterministic test seam must preserve production portability truth"
    );
    context
}

#[test]
fn in_flight_capacity_rejection_exposes_typed_current_policy_snapshot() {
    let defaults = GpuExecutionPolicy::default();
    let policy = GpuExecutionPolicy::new(
        defaults.max_prepared_submissions(),
        NonZeroUsize::new(1).unwrap(),
        defaults.max_upload_bytes_in_flight(),
        defaults.max_readback_bytes_in_flight(),
        defaults.max_pending_readbacks(),
    );
    let context = noop_context_with_policy(policy);

    let first = pollster::block_on(context.prepare_submission(upload_graph("pressure first", 41)))
        .expect("first upload must prepare");
    let second =
        pollster::block_on(context.prepare_submission(upload_graph("pressure second", 42)))
            .expect("second upload must prepare before live acceptance pressure is evaluated");

    let first_submission = context
        .submit_prepared(first)
        .expect("first submission must occupy the sole in-flight slot");
    assert_eq!(context.execution_stats().in_flight_submissions(), 1);

    let rejected = context
        .submit_prepared(second)
        .expect_err("second submission must reject while the first remains current in-flight work");
    let reason = rejected.reason();
    assert_eq!(
        reason.kind(),
        GpuSubmissionRejectionKind::InFlightCapacityExceeded
    );
    let evidence = reason
        .in_flight_capacity_evidence()
        .expect("in-flight rejection must preserve typed execution pressure evidence");
    assert_eq!(evidence.current_in_flight_submissions(), 1);
    assert_eq!(evidence.policy(), policy);
    assert_eq!(evidence.policy().max_in_flight_submissions().get(), 1);

    drop(rejected);
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        context.progress();
        match first_submission.status() {
            GpuSubmissionStatus::Completed => break,
            GpuSubmissionStatus::Failed(failure) => {
                panic!("accepted pressure-proof submission failed: {failure:?}")
            }
            GpuSubmissionStatus::Accepted => {}
        }
        assert!(
            Instant::now() < deadline,
            "accepted pressure-proof submission did not terminalize"
        );
        std::thread::yield_now();
    }
    assert_eq!(context.execution_stats().in_flight_submissions(), 0);
}
