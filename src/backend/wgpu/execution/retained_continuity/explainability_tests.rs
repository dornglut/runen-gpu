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

fn affinity(context: u64, generation: u64) -> GpuContextAffinity {
    GpuContextAffinity::test_value(
        GpuContextId::test_value(NonZeroU64::new(context).unwrap()),
        GpuDeviceGeneration::test_value(NonZeroU64::new(generation).unwrap()),
    )
}

fn submission(value: u64) -> GpuSubmissionId {
    GpuSubmissionId::from_nonzero(NonZeroU64::new(value).unwrap())
}

fn retained_buffer(name: &str) -> GpuBufferHandle {
    let resource_label = label(name);
    let common = GpuResourceCommon::owned(
        resource_label.clone(),
        GpuResourceLifetime::Retained,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        provenance(name),
    )
    .unwrap();
    GpuWorkResourceIdAllocator::new()
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

fn coverage(buffer: &GpuBufferHandle) -> GpuInitialCoverage {
    GpuInitialCoverage::buffer(
        buffer,
        [GpuBufferCoverage::dense(
            GpuBufferRange::new(buffer, 0, 16).unwrap(),
        )],
    )
    .unwrap()
}

fn transition(
    buffer: &GpuBufferHandle,
    consumed_seed: Option<GpuInitialCoverage>,
    initial: Option<GpuInitialCoverage>,
    final_coverage: Option<GpuInitialCoverage>,
    reconstruction: PreparedReconstructionEvidence,
) -> PreparedRetainedContinuity {
    let identity = buffer.diagnostic_identity();
    PreparedRetainedContinuity {
        resources: [(
            identity,
            PreparedRetainedResource {
                resource: GpuResourceRef::Buffer(buffer.clone()),
                consumed_lifecycle: consumed_seed.is_some(),
                consumed_seed,
                initial,
                final_coverage,
                failure_preserved_coverage: Some(coverage(buffer)),
                reconstruction,
            },
        )]
        .into_iter()
        .collect(),
    }
}

#[test]
fn retained_indeterminate_reason_and_reconstruction_outcome_stay_typed_and_distinct() {
    let buffer = retained_buffer("G8-D01 retained explainability");
    let identity = buffer.diagnostic_identity();
    let initialized = coverage(&buffer);
    let state = RetainedContinuityState::new(affinity(41, 7));

    let establish = transition(
        &buffer,
        None,
        None,
        Some(initialized.clone()),
        PreparedReconstructionEvidence {
            explicit_write: true,
            ..Default::default()
        },
    );
    state.validate_and_reserve(&establish).unwrap();
    state.complete(submission(1), &establish, &BTreeSet::from([identity]));

    let write = transition(
        &buffer,
        Some(initialized.clone()),
        Some(initialized.clone()),
        Some(initialized.clone()),
        PreparedReconstructionEvidence {
            explicit_write: true,
            ..Default::default()
        },
    );
    state.validate_and_reserve(&write).unwrap();
    state.mark_may_execute(&write, &BTreeSet::from([identity]));

    let pending = state.snapshot(identity).unwrap();
    assert_eq!(pending.initialized_coverage(), Some(&initialized));
    assert_eq!(
        pending.opaque_content(),
        GpuOpaqueContentContinuity::Unknown
    );
    assert_eq!(
        pending.opaque_content_indeterminate_reason(),
        Some(GpuOpaqueContentIndeterminateReason::AcceptedWritePending)
    );

    state.fail_after_acceptance(&write, &BTreeSet::from([identity]));

    let failed = state.snapshot(identity).unwrap();
    assert_eq!(failed.initialized_coverage(), Some(&initialized));
    assert_eq!(failed.opaque_content(), GpuOpaqueContentContinuity::Unknown);
    assert_eq!(
        failed.opaque_content_indeterminate_reason(),
        Some(GpuOpaqueContentIndeterminateReason::PossibleWriteFailure)
    );

    assert_eq!(
        buffer.descriptor().common().reconstruction(),
        GpuReconstruction::SourceBacked
    );
    let requirement = state.reconstruction_requirement(identity).unwrap();
    assert_eq!(requirement.affinity(), affinity(41, 7));
    assert_eq!(requirement.resource().diagnostic_identity(), identity);

    let reconstruction = transition(
        &buffer,
        Some(initialized.clone()),
        Some(initialized.clone()),
        Some(initialized.clone()),
        PreparedReconstructionEvidence {
            target: true,
            explicit_write: true,
            fresh_descriptor_initial_state: false,
        },
    );
    state.validate_and_reserve(&reconstruction).unwrap();
    state.mark_may_execute(&reconstruction, &BTreeSet::from([identity]));
    let reconstruction_pending = state.snapshot(identity).unwrap();
    assert_eq!(
        reconstruction_pending.opaque_content_indeterminate_reason(),
        Some(GpuOpaqueContentIndeterminateReason::AcceptedWritePending)
    );

    let reconstruction_submission = submission(2);
    state.complete(
        reconstruction_submission,
        &reconstruction,
        &BTreeSet::from([identity]),
    );

    assert!(state.reconstruction_requirement(identity).is_none());
    let reconstructed = state.snapshot(identity).unwrap();
    assert_eq!(reconstructed.affinity(), affinity(41, 7));
    assert_eq!(reconstructed.initialized_coverage(), Some(&initialized));
    assert_eq!(
        reconstructed.opaque_content(),
        GpuOpaqueContentContinuity::Established {
            last_completed_write: reconstruction_submission,
        }
    );
    assert_eq!(reconstructed.opaque_content_indeterminate_reason(), None);
}
