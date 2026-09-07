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
        GpuContextId::test_value(NonZeroU64::new(401).unwrap()),
        GpuDeviceGeneration::test_value(NonZeroU64::new(2).unwrap()),
    )
}

fn submission(value: u64) -> GpuSubmissionId {
    GpuSubmissionId::from_nonzero(NonZeroU64::new(value).unwrap())
}

fn zeroed_retained_buffer() -> GpuBufferHandle {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let resource_label = label("zeroed reconstruction state");
    let common = GpuResourceCommon::owned(
        resource_label.clone(),
        GpuResourceLifetime::Retained,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        provenance("zeroed reconstruction state"),
    )
    .unwrap();
    allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common,
                64,
                GpuBufferUsages::new(
                    &resource_label,
                    [GpuBufferUsage::CopySource, GpuBufferUsage::CopyDestination],
                )
                .unwrap(),
                GpuBufferInitialization::Zeroed,
            )
            .unwrap(),
        )
        .unwrap()
}

fn readback_fragment(buffer: &GpuBufferHandle) -> GpuWorkFragment {
    let range = GpuBufferRange::whole(buffer).unwrap();
    let readback = GpuReadbackOperation::new(
        GpuBufferRegion::new(buffer, range).unwrap().into(),
        GpuReadbackId::allocate().unwrap(),
    )
    .unwrap();
    let mut builder = GpuWorkFragmentBuilder::new(
        label("zeroed read-only reconstruction"),
        provenance("zeroed read-only reconstruction"),
    );
    builder
        .declare_resource(GpuResourceRef::Buffer(buffer.clone()))
        .unwrap();
    builder
        .add_node(
            label("read zeroed state"),
            GpuWorkOperation::Readback(readback),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::TransferPreferred,
            provenance("read zeroed state"),
        )
        .unwrap();
    builder.finish().unwrap()
}

fn clear_fragment(buffer: &GpuBufferHandle) -> GpuWorkFragment {
    let clear = GpuClearOperation::buffer_zero(
        GpuBufferRegion::new(buffer, GpuBufferRange::whole(buffer).unwrap()).unwrap(),
    )
    .unwrap();
    let mut builder = GpuWorkFragmentBuilder::new(
        label("zeroed explicit reconstruction"),
        provenance("zeroed explicit reconstruction"),
    );
    builder
        .declare_resource(GpuResourceRef::Buffer(buffer.clone()))
        .unwrap();
    builder
        .add_node(
            label("clear zeroed state"),
            GpuWorkOperation::Clear(clear),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::TransferPreferred,
            provenance("clear zeroed state"),
        )
        .unwrap();
    builder.finish().unwrap()
}

#[test]
fn zeroed_descriptor_match_does_not_make_read_only_reconstruction_a_content_epoch() {
    let buffer = zeroed_retained_buffer();
    let identity = buffer.diagnostic_identity();
    let state = RetainedContinuityState::new(affinity());
    state.install_reconstruction_obligations([GpuRetainedReconstructionSeed::new(
        GpuResourceRef::Buffer(buffer.clone()),
        true,
    )]);

    let read_only_graph = GpuPreparedWorkGraph::prepare_with_retained_coverage_and_reconstruction(
        label("zeroed read-only reconstruction graph"),
        [readback_fragment(&buffer)],
        &[],
        &[GpuResourceRef::Buffer(buffer.clone())],
    )
    .unwrap();
    let read_only = PreparedRetainedContinuity::from_graph(&read_only_graph);
    let rejection = state.validate_and_reserve(&read_only).unwrap_err();
    assert_eq!(
        rejection.kind(),
        GpuSubmissionRejectionKind::RetainedContinuityChanged
    );
    let requirement = state.reconstruction_requirement(identity).unwrap();
    assert!(requirement.descriptor_initial_state_matches_required_contents());
    assert!(state.snapshot(identity).is_none());

    let clear_graph = GpuPreparedWorkGraph::prepare_with_retained_coverage_and_reconstruction(
        label("zeroed explicit reconstruction graph"),
        [clear_fragment(&buffer)],
        &[],
        &[GpuResourceRef::Buffer(buffer.clone())],
    )
    .unwrap();
    let clear = PreparedRetainedContinuity::from_graph(&clear_graph);
    state.validate_and_reserve(&clear).unwrap();
    let writes = BTreeSet::from([identity]);
    state.mark_may_execute(&clear, &writes);
    state.complete(submission(1), &clear, &writes);

    assert!(state.reconstruction_requirement(identity).is_none());
    let continuity = state.snapshot(identity).unwrap();
    assert!(continuity.initialized_coverage().is_some());
    assert_eq!(
        continuity.opaque_content(),
        GpuOpaqueContentContinuity::Established {
            last_completed_write: submission(1),
        }
    );
}
