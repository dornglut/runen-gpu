use super::*;
use crate::*;
use core::num::NonZeroU64;
use std::collections::BTreeSet;

fn label(value: &str) -> GpuResourceLabel {
    GpuResourceLabel::new(value).unwrap()
}

fn provenance(value: &str) -> GpuResourceProvenance {
    GpuResourceProvenance::new(label(value), None, None)
}

fn affinity() -> GpuContextAffinity {
    GpuContextAffinity::test_value(
        GpuContextId::test_value(NonZeroU64::new(101).unwrap()),
        GpuDeviceGeneration::test_value(NonZeroU64::new(7).unwrap()),
    )
}

fn submission(value: u64) -> GpuSubmissionId {
    GpuSubmissionId::from_nonzero(NonZeroU64::new(value).unwrap())
}

fn retained_buffer() -> GpuBufferHandle {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let resource_label = label("terminal retained buffer");
    let common = GpuResourceCommon::owned(
        resource_label.clone(),
        GpuResourceLifetime::Retained,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        provenance("terminal retained buffer"),
    )
    .unwrap();
    allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common,
                64,
                GpuBufferUsages::new(
                    &resource_label,
                    [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
                )
                .unwrap(),
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap()
}

fn coverage(buffer: &GpuBufferHandle, offset: u64, size: u64) -> GpuInitialCoverage {
    GpuInitialCoverage::buffer(
        buffer,
        [GpuBufferCoverage::dense(
            GpuBufferRange::new(buffer, offset, size).unwrap(),
        )],
    )
    .unwrap()
}

fn transition(
    buffer: &GpuBufferHandle,
    consumed_seed: Option<GpuInitialCoverage>,
    initial: Option<GpuInitialCoverage>,
    final_coverage: Option<GpuInitialCoverage>,
    failure_preserved_coverage: Option<GpuInitialCoverage>,
) -> PreparedRetainedContinuity {
    transition_with_target(
        buffer,
        consumed_seed,
        initial,
        final_coverage,
        failure_preserved_coverage,
        false,
    )
}

fn transition_with_target(
    buffer: &GpuBufferHandle,
    consumed_seed: Option<GpuInitialCoverage>,
    initial: Option<GpuInitialCoverage>,
    final_coverage: Option<GpuInitialCoverage>,
    failure_preserved_coverage: Option<GpuInitialCoverage>,
    reconstruction_target: bool,
) -> PreparedRetainedContinuity {
    let consumed_lifecycle = consumed_seed.is_some();
    PreparedRetainedContinuity {
        resources: [(
            buffer.diagnostic_identity(),
            PreparedRetainedResource {
                resource: GpuResourceRef::Buffer(buffer.clone()),
                consumed_lifecycle,
                consumed_seed,
                initial,
                final_coverage,
                failure_preserved_coverage,
                reconstruction: PreparedReconstructionEvidence {
                    target: reconstruction_target,
                    explicit_write: true,
                    fresh_descriptor_initial_state: false,
                },
            },
        )]
        .into_iter()
        .collect(),
    }
}

fn establish(
    state: &RetainedContinuityState,
    buffer: &GpuBufferHandle,
    initialized: &GpuInitialCoverage,
    completed_write: GpuSubmissionId,
) {
    let prepared = transition(buffer, None, None, Some(initialized.clone()), None);
    state.validate_and_reserve(&prepared).unwrap();
    state.complete(
        completed_write,
        &prepared,
        &BTreeSet::from([buffer.diagnostic_identity()]),
    );
}

#[test]
fn queue_submission_makes_current_opaque_content_unknown_until_completion() {
    let buffer = retained_buffer();
    let full = coverage(&buffer, 0, 32);
    let failure_safe = coverage(&buffer, 0, 16);
    let state = RetainedContinuityState::new(affinity());
    let first_write = submission(1);
    establish(&state, &buffer, &full, first_write);

    let prepared = transition(
        &buffer,
        Some(full.clone()),
        Some(full.clone()),
        Some(full.clone()),
        Some(failure_safe.clone()),
    );
    let writes = BTreeSet::from([buffer.diagnostic_identity()]);
    state.validate_and_reserve(&prepared).unwrap();

    state.mark_may_execute(&prepared, &writes);

    let in_flight = state.snapshot(buffer.diagnostic_identity()).unwrap();
    assert_eq!(
        in_flight.opaque_content(),
        GpuOpaqueContentContinuity::Unknown
    );
    assert_eq!(in_flight.initialized_coverage(), Some(&failure_safe));

    let second_write = submission(2);
    state.complete(second_write, &prepared, &writes);

    let completed = state.snapshot(buffer.diagnostic_identity()).unwrap();
    assert_eq!(completed.initialized_coverage(), Some(&full));
    assert_eq!(
        completed.opaque_content(),
        GpuOpaqueContentContinuity::Established {
            last_completed_write: second_write,
        }
    );
}

#[test]
fn first_queue_submitted_write_is_unknown_without_inventing_initialized_coverage() {
    let buffer = retained_buffer();
    let final_coverage = coverage(&buffer, 0, 16);
    let state = RetainedContinuityState::new(affinity());
    let prepared = transition(&buffer, None, None, Some(final_coverage), None);
    let writes = BTreeSet::from([buffer.diagnostic_identity()]);
    state.validate_and_reserve(&prepared).unwrap();

    state.mark_may_execute(&prepared, &writes);

    let in_flight = state.snapshot(buffer.diagnostic_identity()).unwrap();
    assert_eq!(
        in_flight.opaque_content(),
        GpuOpaqueContentContinuity::Unknown
    );
    assert!(in_flight.initialized_coverage().is_none());

    state.fail_after_acceptance(&prepared, &writes);
    let failed = state.snapshot(buffer.diagnostic_identity()).unwrap();
    assert_eq!(failed.opaque_content(), GpuOpaqueContentContinuity::Unknown);
    assert!(failed.initialized_coverage().is_none());
    assert!(
        state
            .reconstruction_requirement(buffer.diagnostic_identity())
            .is_some()
    );
}

#[test]
fn first_queue_submitted_write_preserves_preexisting_failure_safe_coverage() {
    let buffer = retained_buffer();
    let initialized = coverage(&buffer, 0, 16);
    let state = RetainedContinuityState::new(affinity());
    let prepared = transition(
        &buffer,
        None,
        Some(initialized.clone()),
        Some(initialized.clone()),
        Some(initialized.clone()),
    );
    let writes = BTreeSet::from([buffer.diagnostic_identity()]);
    state.validate_and_reserve(&prepared).unwrap();

    state.mark_may_execute(&prepared, &writes);
    let in_flight = state.snapshot(buffer.diagnostic_identity()).unwrap();
    assert_eq!(
        in_flight.opaque_content(),
        GpuOpaqueContentContinuity::Unknown
    );
    assert_eq!(in_flight.initialized_coverage(), Some(&initialized));

    state.fail_after_acceptance(&prepared, &writes);
    let failed = state.snapshot(buffer.diagnostic_identity()).unwrap();
    assert_eq!(failed.opaque_content(), GpuOpaqueContentContinuity::Unknown);
    assert_eq!(failed.initialized_coverage(), Some(&initialized));
}

#[test]
fn first_queue_submitted_write_establishes_only_after_successful_completion() {
    let buffer = retained_buffer();
    let initialized = coverage(&buffer, 0, 16);
    let state = RetainedContinuityState::new(affinity());
    let prepared = transition(&buffer, None, None, Some(initialized.clone()), None);
    let writes = BTreeSet::from([buffer.diagnostic_identity()]);
    state.validate_and_reserve(&prepared).unwrap();

    state.mark_may_execute(&prepared, &writes);
    let in_flight = state.snapshot(buffer.diagnostic_identity()).unwrap();
    assert_eq!(
        in_flight.opaque_content(),
        GpuOpaqueContentContinuity::Unknown
    );
    assert!(in_flight.initialized_coverage().is_none());

    let completed_write = submission(3);
    state.complete(completed_write, &prepared, &writes);
    let completed = state.snapshot(buffer.diagnostic_identity()).unwrap();
    assert_eq!(completed.initialized_coverage(), Some(&initialized));
    assert_eq!(
        completed.opaque_content(),
        GpuOpaqueContentContinuity::Established {
            last_completed_write: completed_write,
        }
    );
}

#[test]
fn revocation_requires_explicit_reconstruction_before_later_ordinary_work() {
    let buffer = retained_buffer();
    let initialized = coverage(&buffer, 0, 32);
    let state = RetainedContinuityState::new(affinity());
    establish(&state, &buffer, &initialized, submission(4));
    let writes = BTreeSet::from([buffer.diagnostic_identity()]);

    let failed_write = transition(
        &buffer,
        Some(initialized.clone()),
        Some(initialized.clone()),
        Some(initialized.clone()),
        Some(initialized.clone()),
    );
    state.validate_and_reserve(&failed_write).unwrap();
    state.mark_may_execute(&failed_write, &writes);
    state.fail_after_acceptance(&failed_write, &writes);

    let revoked = state.snapshot(buffer.diagnostic_identity()).unwrap();
    assert_eq!(
        revoked.opaque_content(),
        GpuOpaqueContentContinuity::Unknown
    );
    assert_eq!(revoked.initialized_coverage(), Some(&initialized));

    let later_write = transition(
        &buffer,
        Some(initialized.clone()),
        Some(initialized.clone()),
        Some(initialized.clone()),
        Some(initialized.clone()),
    );
    let rejection = state.validate_and_reserve(&later_write).unwrap_err();
    assert_eq!(
        rejection.kind(),
        GpuSubmissionRejectionKind::RetainedContinuityChanged
    );

    let reconstruction = transition_with_target(
        &buffer,
        Some(initialized.clone()),
        Some(initialized.clone()),
        Some(initialized.clone()),
        Some(initialized.clone()),
        true,
    );
    state.validate_and_reserve(&reconstruction).unwrap();
    state.mark_may_execute(&reconstruction, &writes);
    state.complete(submission(5), &reconstruction, &writes);

    let completed = state.snapshot(buffer.diagnostic_identity()).unwrap();
    assert_eq!(
        completed.opaque_content(),
        GpuOpaqueContentContinuity::Established {
            last_completed_write: submission(5),
        }
    );
    assert_eq!(completed.initialized_coverage(), Some(&initialized));
    assert!(
        state
            .reconstruction_requirement(buffer.diagnostic_identity())
            .is_none()
    );
}
