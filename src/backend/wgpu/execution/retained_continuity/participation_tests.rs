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

fn retained_buffer(
    name: &str,
    initialization: GpuBufferInitialization,
    usages: impl IntoIterator<Item = GpuBufferUsage>,
) -> GpuBufferHandle {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let resource_label = label(name);
    let common = GpuResourceCommon::owned(
        resource_label.clone(),
        GpuResourceLifetime::Retained,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        provenance(name),
    )
    .unwrap();
    allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common,
                64,
                GpuBufferUsages::new(&resource_label, usages).unwrap(),
                initialization,
            )
            .unwrap(),
        )
        .unwrap()
}

fn readback_fragment(buffer: &GpuBufferHandle, name: &str) -> GpuWorkFragment {
    let readback = GpuReadbackOperation::new(
        GpuBufferRegion::new(buffer, GpuBufferRange::whole(buffer).unwrap())
            .unwrap()
            .into(),
        GpuReadbackId::allocate().unwrap(),
    )
    .unwrap();
    let mut fragment = GpuWorkFragmentBuilder::new(label(name), provenance(name));
    fragment
        .declare_resource(GpuResourceRef::Buffer(buffer.clone()))
        .unwrap();
    fragment
        .add_node(
            label(&format!("{name} readback")),
            GpuWorkOperation::Readback(readback),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::TransferPreferred,
            provenance(&format!("{name} readback")),
        )
        .unwrap();
    fragment.finish().unwrap()
}

fn affinity() -> GpuContextAffinity {
    GpuContextAffinity::test_value(
        GpuContextId::test_value(NonZeroU64::new(301).unwrap()),
        GpuDeviceGeneration::test_value(NonZeroU64::new(11).unwrap()),
    )
}

fn submission(value: u64) -> GpuSubmissionId {
    GpuSubmissionId::from_nonzero(NonZeroU64::new(value).unwrap())
}

#[test]
fn retained_lifecycle_without_coverage_does_not_resurrect_zeroed_creation_state() {
    let buffer = retained_buffer(
        "retained zeroed",
        GpuBufferInitialization::Zeroed,
        [GpuBufferUsage::CopySource],
    );

    GpuPreparedWorkGraph::prepare(
        label("first zeroed read"),
        [readback_fragment(&buffer, "first zeroed read")],
    )
    .expect("creation-time zeroed state may satisfy the first read");

    let seed = GpuRetainedInitializationSeed::new(GpuResourceRef::Buffer(buffer.clone()), None);
    let error = GpuPreparedWorkGraph::prepare_with_retained_coverage(
        label("later zeroed read"),
        [readback_fragment(&buffer, "later zeroed read")],
        &[seed],
    )
    .unwrap_err();

    assert_eq!(error.cause(), GpuWorkGraphCause::ReadBeforeInitialization);
    assert_eq!(error.resource(), Some(buffer.diagnostic_identity()));
}

#[test]
fn retained_lifecycle_without_coverage_does_not_replay_prepared_initial_content() {
    let data = PreparedGpuData::<TransferData>::from_pod_transfer(
        "retained prepared bytes",
        &[7_u8; 64],
        provenance("retained prepared bytes"),
    )
    .unwrap();
    let buffer = retained_buffer(
        "retained prepared",
        GpuBufferInitialization::Prepared(data),
        [GpuBufferUsage::CopySource, GpuBufferUsage::CopyDestination],
    );

    let first = GpuPreparedWorkGraph::prepare(
        label("first prepared read"),
        [readback_fragment(&buffer, "first prepared read")],
    )
    .expect("first use may select canonical prepared-content materialization");
    assert_eq!(first.initial_content().len(), 1);

    let seed = GpuRetainedInitializationSeed::new(GpuResourceRef::Buffer(buffer.clone()), None);
    let error = GpuPreparedWorkGraph::prepare_with_retained_coverage(
        label("later prepared read"),
        [readback_fragment(&buffer, "later prepared read")],
        &[seed],
    )
    .unwrap_err();

    assert_eq!(error.cause(), GpuWorkGraphCause::ReadBeforeInitialization);
    assert_eq!(error.resource(), Some(buffer.diagnostic_identity()));
}

#[test]
fn unused_retained_declaration_does_not_enter_continuity() {
    let buffer = retained_buffer(
        "unused retained",
        GpuBufferInitialization::Zeroed,
        [GpuBufferUsage::CopySource],
    );
    let mut fragment =
        GpuWorkFragmentBuilder::new(label("unused retained"), provenance("unused retained"));
    fragment
        .declare_resource(GpuResourceRef::Buffer(buffer))
        .unwrap();
    let graph =
        GpuPreparedWorkGraph::prepare(label("unused retained graph"), [fragment.finish().unwrap()])
            .unwrap();

    assert!(PreparedRetainedContinuity::from_graph(&graph).is_empty());
}

#[test]
fn retained_state_seed_preserves_lifecycle_presence_without_initialized_coverage() {
    let buffer = retained_buffer(
        "retained no coverage",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::Storage],
    );
    let identity = buffer.diagnostic_identity();
    let transition = PreparedRetainedContinuity {
        resources: [(
            identity,
            PreparedRetainedResource {
                resource: GpuResourceRef::Buffer(buffer.clone()),
                consumed_lifecycle: false,
                consumed_seed: None,
                initial: None,
                final_coverage: None,
                failure_preserved_coverage: None,
                reconstruction: PreparedReconstructionEvidence::default(),
            },
        )]
        .into_iter()
        .collect(),
    };
    let state = RetainedContinuityState::new(affinity());
    state.validate_and_reserve(&transition).unwrap();
    state.complete(submission(1), &transition, &BTreeSet::from([identity]));

    let seeds = state.coverage_seed();
    assert_eq!(seeds.len(), 1);
    assert_eq!(seeds[0].resource_identity(), identity);
    assert!(seeds[0].initialized_coverage().is_none());
}

#[test]
fn context_free_transition_is_rejected_after_zero_coverage_lifecycle_appears() {
    let buffer = retained_buffer(
        "stale context-free retained state",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::Storage],
    );
    let identity = buffer.diagnostic_identity();
    let unseeded = PreparedRetainedContinuity {
        resources: [(
            identity,
            PreparedRetainedResource {
                resource: GpuResourceRef::Buffer(buffer.clone()),
                consumed_lifecycle: false,
                consumed_seed: None,
                initial: None,
                final_coverage: None,
                failure_preserved_coverage: None,
                reconstruction: PreparedReconstructionEvidence::default(),
            },
        )]
        .into_iter()
        .collect(),
    };
    let state = RetainedContinuityState::new(affinity());
    state.validate_and_reserve(&unseeded).unwrap();
    state.complete(submission(2), &unseeded, &BTreeSet::from([identity]));

    let error = state.validate_and_reserve(&unseeded).unwrap_err();
    assert_eq!(
        error.kind(),
        GpuSubmissionRejectionKind::RetainedContinuityChanged
    );

    let seeded = PreparedRetainedContinuity {
        resources: [(
            identity,
            PreparedRetainedResource {
                resource: GpuResourceRef::Buffer(buffer),
                consumed_lifecycle: true,
                consumed_seed: None,
                initial: None,
                final_coverage: None,
                failure_preserved_coverage: None,
                reconstruction: PreparedReconstructionEvidence::default(),
            },
        )]
        .into_iter()
        .collect(),
    };
    state.validate_and_reserve(&seeded).unwrap();
    state.fail_after_acceptance(&seeded, &BTreeSet::new());
}
