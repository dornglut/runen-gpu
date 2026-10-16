//! Public RunenGPU cross-frame conformance consumed by ordinary RunenGPU generic execution.
//!
//! This does not duplicate G5 admission or assume that GPU queue ordering grants
//! retained-content continuity. An in-flight read-only participant is conservatively
//! reserved; an unrelated resource may still make progress; a fresh graph is
//! prepared only after its predecessor terminalizes. No rendering window required.
use runen_gpu::{
    GpuBufferDescriptor, GpuBufferHandle, GpuBufferInitialization, GpuBufferRegion, GpuBufferUsage,
    GpuCapabilityProfile, GpuCapabilityRequirements, GpuContext, GpuContextDescriptor,
    GpuContextRequestErrorCategory, GpuExecutionPreference, GpuReadbackId, GpuReadbackOperation,
    GpuReadbackStatus, GpuReconstruction, GpuResourceLabel, GpuResourceLifetime,
    GpuResourceProvenance, GpuRetainedSubmissionRejectionCause, GpuSubmission,
    GpuSubmissionRejectionKind, GpuSubmissionStatus, GpuWorkFragment, GpuWorkFragmentBuilder,
    GpuWorkOperation, GpuWorkResourceIdAllocator, GpuWorkSubmissionError,
};
use std::time::{Duration, Instant};

fn label(value: &str) -> GpuResourceLabel {
    GpuResourceLabel::new(value).expect("valid conformance resource label")
}

fn provenance(value: &str) -> GpuResourceProvenance {
    GpuResourceProvenance::new(label(value), None, None)
}

fn retained_buffer(allocator: &mut GpuWorkResourceIdAllocator, name: &str) -> GpuBufferHandle {
    allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::ordinary_owned(
                name,
                GpuResourceLifetime::Retained,
                GpuReconstruction::SourceBacked,
                64,
                [GpuBufferUsage::CopySource],
                GpuBufferInitialization::Zeroed,
            )
            .expect("valid source-backed zeroed retained buffer"),
        )
        .expect("unique retained resource identity")
}

fn read_only_fragment(buffer: &GpuBufferHandle, name: &str) -> (GpuWorkFragment, GpuReadbackId) {
    let readback_id = GpuReadbackId::allocate().expect("fresh readback identity");
    let operation = GpuReadbackOperation::new(
        GpuBufferRegion::whole(buffer)
            .expect("whole retained buffer")
            .into(),
        readback_id,
    )
    .expect("valid readback operation");
    let mut builder = GpuWorkFragmentBuilder::new(label(name), provenance(name));
    builder
        .declare_resource(buffer.clone().into())
        .expect("declare retained buffer");
    builder
        .add_node(
            label(&format!("{name}.readback")),
            GpuWorkOperation::Readback(operation),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::TransferPreferred,
            provenance(name),
        )
        .expect("read-only G5 work node");
    (
        builder.finish().expect("valid read-only fragment"),
        readback_id,
    )
}

fn gpu_context() -> Option<GpuContext> {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::ComputeBaseline.requirements())
            .with_label("RunenGPU G5 typed rejection native proof");
    match pollster::block_on(GpuContext::request(descriptor)) {
        Ok(context) => Some(context),
        Err(error) if error.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            assert_ne!(
                std::env::var("RUNEN_GPU_G5_REQUIRE_GPU").ok().as_deref(),
                Some("1"),
                "Vulkan CI requires a usable public RunenGPU context"
            );
            None
        }
        Err(error) => panic!("GPU cross-frame proof context failed: {error}"),
    }
}

fn submit(
    context: &GpuContext,
    buffer: &GpuBufferHandle,
    name: &str,
) -> Result<(GpuSubmission, GpuReadbackId), GpuWorkSubmissionError> {
    let (fragment, id) = read_only_fragment(buffer, name);
    let submitted = pollster::block_on(context.submit_work(name, [fragment]))?;
    Ok((submitted, id))
}

fn complete(context: &GpuContext, submission: &GpuSubmission) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        context.progress();
        match submission.status() {
            GpuSubmissionStatus::Completed => return,
            GpuSubmissionStatus::Accepted if Instant::now() < deadline => {
                std::thread::yield_now();
            }
            state => panic!("native G5 submission did not terminally complete: {state:?}"),
        }
    }
}

#[test]
fn retained_read_only_reservation_is_not_an_implicit_ordered_continuation() {
    let Some(context) = gpu_context() else {
        return;
    };
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let shared = retained_buffer(&mut allocator, "g5.typed.shared");
    let unrelated = retained_buffer(&mut allocator, "g5.typed.unrelated");

    let (first, first_readback) =
        submit(&context, &shared, "g5.typed.first").expect("first admission");
    assert_eq!(
        first.status(),
        GpuSubmissionStatus::Accepted,
        "accepted does not imply terminally completed"
    );

    let attempted = submit(&context, &shared, "g5.typed.overlap");
    let rejected_reason = match attempted {
        Err(GpuWorkSubmissionError::SubmissionRejected(reason)) => reason,
        Err(other) => panic!("expected typed G5 rejection, got {other:?}"),
        Ok(_) => panic!("read-only retained overlap must not bypass G5 reservation"),
    };
    assert_eq!(
        rejected_reason.kind(),
        GpuSubmissionRejectionKind::RetainedContinuityChanged,
        "RunenGPU G5 owns conservative retained-resource participation, including reads"
    );
    let evidence = rejected_reason
        .retained_continuity_evidence()
        .expect("G5 error must carry public typed evidence");
    assert_eq!(evidence.resource(), shared.diagnostic_identity());
    assert_eq!(evidence.affinity(), context.affinity());
    assert_eq!(
        evidence.cause(),
        GpuRetainedSubmissionRejectionCause::InFlightParticipation
    );
    assert_eq!(first.status(), GpuSubmissionStatus::Accepted);

    let (independent, independent_readback) = submit(&context, &unrelated, "g5.typed.independent")
        .expect("unrelated retained resource can submit without globally waiting");
    assert_eq!(independent.status(), GpuSubmissionStatus::Accepted);

    complete(&context, &first);
    complete(&context, &independent);
    for (submission, readback) in [
        (&first, first_readback),
        (&independent, independent_readback),
    ] {
        match submission
            .readback(readback)
            .expect("readback exists")
            .status()
        {
            GpuReadbackStatus::Ready(bytes) => {
                assert_eq!(bytes.as_bytes(), &[0_u8; 64]);
            }
            other => panic!("terminally completed retained readback must be ready: {other:?}"),
        }
    }

    // Rebuild from the current RunenGPU context after predecessor completion.
    // A rejected prepared graph must not be cached or replayed across a frame.
    let (fresh, fresh_readback) =
        submit(&context, &shared, "g5.typed.fresh").expect("fresh post-terminal admission");
    complete(&context, &fresh);
    assert!(matches!(
        fresh
            .readback(fresh_readback)
            .expect("fresh readback")
            .status(),
        GpuReadbackStatus::Ready(_)
    ));
}
