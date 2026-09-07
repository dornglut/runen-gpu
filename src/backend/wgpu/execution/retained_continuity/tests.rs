use super::*;
use crate::*;
use core::num::NonZeroU64;
use std::collections::{BTreeMap, BTreeSet};

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

fn buffer_with_lifetime(
    allocator: &mut GpuWorkResourceIdAllocator,
    name: &str,
    lifetime: GpuResourceLifetime,
) -> GpuBufferHandle {
    retained_or_transient_buffer(allocator, name, lifetime, GpuReconstruction::SourceBacked)
}

fn retained_or_transient_buffer(
    allocator: &mut GpuWorkResourceIdAllocator,
    name: &str,
    lifetime: GpuResourceLifetime,
    reconstruction: GpuReconstruction,
) -> GpuBufferHandle {
    let resource_label = label(name);
    let common = GpuResourceCommon::owned(
        resource_label.clone(),
        lifetime,
        GpuMemoryIntent::Device,
        reconstruction,
        provenance(name),
    )
    .unwrap();
    allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common,
                64,
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

fn retained_buffer(allocator: &mut GpuWorkResourceIdAllocator, name: &str) -> GpuBufferHandle {
    buffer_with_lifetime(allocator, name, GpuResourceLifetime::Retained)
}

fn retained_external_buffer(
    allocator: &mut GpuWorkResourceIdAllocator,
    name: &str,
) -> GpuBufferHandle {
    retained_or_transient_buffer(
        allocator,
        name,
        GpuResourceLifetime::Retained,
        GpuReconstruction::ExternallyReconstructed,
    )
}

fn retained_non_reconstructable_buffer(
    allocator: &mut GpuWorkResourceIdAllocator,
    name: &str,
) -> GpuBufferHandle {
    let resource_label = label(name);
    let common = GpuResourceCommon::owned_retained_non_reconstructable(
        resource_label.clone(),
        GpuMemoryIntent::Device,
        provenance(name),
    );
    allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common,
                64,
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

fn surface_acquired_texture(
    allocator: &mut GpuWorkResourceIdAllocator,
    name: &str,
) -> GpuTextureHandle {
    let resource_label = label(name);
    let common = GpuResourceCommon::surface_acquired(resource_label.clone(), provenance(name));
    allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::new(
                common,
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, 8, 8, 1).unwrap(),
                1,
                1,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(&resource_label, [GpuTextureUsage::ColorAttachment]).unwrap(),
                GpuTextureInitialization::Uninitialized,
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

fn prepared_resource(
    buffer: &GpuBufferHandle,
    consumed_seed: Option<GpuInitialCoverage>,
    initial: Option<GpuInitialCoverage>,
    final_coverage: Option<GpuInitialCoverage>,
    failure_preserved_coverage: Option<GpuInitialCoverage>,
) -> (GpuWorkResourceId, PreparedRetainedResource) {
    prepared_resource_with_reconstruction(
        buffer,
        consumed_seed,
        initial,
        final_coverage,
        failure_preserved_coverage,
        PreparedReconstructionEvidence::default(),
    )
}

fn prepared_resource_with_reconstruction(
    buffer: &GpuBufferHandle,
    consumed_seed: Option<GpuInitialCoverage>,
    initial: Option<GpuInitialCoverage>,
    final_coverage: Option<GpuInitialCoverage>,
    failure_preserved_coverage: Option<GpuInitialCoverage>,
    reconstruction: PreparedReconstructionEvidence,
) -> (GpuWorkResourceId, PreparedRetainedResource) {
    let identity = buffer.diagnostic_identity();
    let consumed_lifecycle = consumed_seed.is_some();
    (
        identity,
        PreparedRetainedResource {
            resource: GpuResourceRef::Buffer(buffer.clone()),
            consumed_lifecycle,
            consumed_seed,
            initial,
            final_coverage,
            failure_preserved_coverage,
            reconstruction,
        },
    )
}

fn transition(
    buffer: &GpuBufferHandle,
    consumed_seed: Option<GpuInitialCoverage>,
    initial: Option<GpuInitialCoverage>,
    final_coverage: Option<GpuInitialCoverage>,
    failure_preserved_coverage: Option<GpuInitialCoverage>,
) -> PreparedRetainedContinuity {
    transition_with_reconstruction(
        buffer,
        consumed_seed,
        initial,
        final_coverage,
        failure_preserved_coverage,
        PreparedReconstructionEvidence::default(),
    )
}

fn transition_with_reconstruction(
    buffer: &GpuBufferHandle,
    consumed_seed: Option<GpuInitialCoverage>,
    initial: Option<GpuInitialCoverage>,
    final_coverage: Option<GpuInitialCoverage>,
    failure_preserved_coverage: Option<GpuInitialCoverage>,
    reconstruction: PreparedReconstructionEvidence,
) -> PreparedRetainedContinuity {
    PreparedRetainedContinuity {
        resources: [prepared_resource_with_reconstruction(
            buffer,
            consumed_seed,
            initial,
            final_coverage,
            failure_preserved_coverage,
            reconstruction,
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
    let transition = transition_with_reconstruction(
        buffer,
        None,
        None,
        Some(initialized.clone()),
        None,
        PreparedReconstructionEvidence {
            explicit_write: true,
            ..Default::default()
        },
    );
    state.validate_and_reserve(&transition).unwrap();
    state.complete(
        completed_write,
        &transition,
        &BTreeSet::from([buffer.diagnostic_identity()]),
    );
}

#[test]
fn completion_carries_initialized_coverage_and_read_only_completion_preserves_opaque_epoch() {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let buffer = retained_buffer(&mut allocator, "retained completion");
    let initialized = coverage(&buffer, 0, 16);
    let state = RetainedContinuityState::new(affinity(1, 1));
    let first_write = submission(1);
    establish(&state, &buffer, &initialized, first_write);

    let seeds = state.coverage_seed();
    assert_eq!(seeds.len(), 1);
    assert_eq!(seeds[0].resource_identity(), buffer.diagnostic_identity());
    assert_eq!(seeds[0].initialized_coverage(), Some(&initialized));
    let snapshot = state.snapshot(buffer.diagnostic_identity()).unwrap();
    assert_eq!(snapshot.initialized_coverage(), Some(&initialized));
    assert_eq!(
        snapshot.opaque_content(),
        GpuOpaqueContentContinuity::Established {
            last_completed_write: first_write,
        }
    );

    let read_only = transition(
        &buffer,
        Some(initialized.clone()),
        Some(initialized.clone()),
        Some(initialized.clone()),
        Some(initialized.clone()),
    );
    state.validate_and_reserve(&read_only).unwrap();
    state.complete(submission(2), &read_only, &BTreeSet::new());

    let snapshot = state.snapshot(buffer.diagnostic_identity()).unwrap();
    assert_eq!(snapshot.initialized_coverage(), Some(&initialized));
    assert_eq!(
        snapshot.opaque_content(),
        GpuOpaqueContentContinuity::Established {
            last_completed_write: first_write,
        }
    );
}

#[test]
fn failure_before_any_possible_write_preserves_prior_continuity_and_releases_reservation() {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let buffer = retained_buffer(&mut allocator, "pre-write failure");
    let initialized = coverage(&buffer, 0, 16);
    let state = RetainedContinuityState::new(affinity(2, 1));
    let first_write = submission(3);
    establish(&state, &buffer, &initialized, first_write);

    let prepared = transition(
        &buffer,
        Some(initialized.clone()),
        Some(initialized.clone()),
        Some(initialized.clone()),
        Some(initialized.clone()),
    );
    state.validate_and_reserve(&prepared).unwrap();
    state.fail_after_acceptance(&prepared, &BTreeSet::new());

    let snapshot = state.snapshot(buffer.diagnostic_identity()).unwrap();
    assert_eq!(snapshot.initialized_coverage(), Some(&initialized));
    assert_eq!(
        snapshot.opaque_content(),
        GpuOpaqueContentContinuity::Established {
            last_completed_write: first_write,
        }
    );
    state.validate_and_reserve(&prepared).unwrap();
    state.fail_after_acceptance(&prepared, &BTreeSet::new());
}

#[test]
fn possible_write_failure_revokes_opaque_content_without_erasing_provable_coverage() {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let buffer = retained_buffer(&mut allocator, "possible write failure");
    let initialized = coverage(&buffer, 0, 16);
    let state = RetainedContinuityState::new(affinity(3, 1));
    establish(&state, &buffer, &initialized, submission(4));

    let prepared = transition_with_reconstruction(
        &buffer,
        Some(initialized.clone()),
        Some(initialized.clone()),
        Some(initialized.clone()),
        Some(initialized.clone()),
        PreparedReconstructionEvidence {
            explicit_write: true,
            ..Default::default()
        },
    );
    state.validate_and_reserve(&prepared).unwrap();
    state.fail_after_acceptance(&prepared, &BTreeSet::from([buffer.diagnostic_identity()]));

    let snapshot = state.snapshot(buffer.diagnostic_identity()).unwrap();
    assert_eq!(snapshot.initialized_coverage(), Some(&initialized));
    assert_eq!(
        snapshot.opaque_content(),
        GpuOpaqueContentContinuity::Unknown
    );
    let requirement = state
        .reconstruction_requirement(buffer.diagnostic_identity())
        .unwrap();
    assert!(!requirement.descriptor_initial_state_matches_required_contents());
}

#[test]
fn possible_write_failure_isolated_to_resources_that_may_have_executed() {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let first = retained_buffer(&mut allocator, "affected retained state");
    let second = retained_buffer(&mut allocator, "unaffected retained state");
    let first_coverage = coverage(&first, 0, 16);
    let second_coverage = coverage(&second, 0, 16);
    let state = RetainedContinuityState::new(affinity(4, 1));
    let first_write = submission(5);
    let second_write = submission(6);
    establish(&state, &first, &first_coverage, first_write);
    establish(&state, &second, &second_coverage, second_write);
    let resources = BTreeMap::from([
        prepared_resource_with_reconstruction(
            &first,
            Some(first_coverage.clone()),
            Some(first_coverage.clone()),
            Some(first_coverage.clone()),
            Some(first_coverage.clone()),
            PreparedReconstructionEvidence {
                explicit_write: true,
                ..Default::default()
            },
        ),
        prepared_resource(
            &second,
            Some(second_coverage.clone()),
            Some(second_coverage.clone()),
            Some(second_coverage.clone()),
            Some(second_coverage.clone()),
        ),
    ]);
    let prepared = PreparedRetainedContinuity { resources };
    state.validate_and_reserve(&prepared).unwrap();
    state.fail_after_acceptance(&prepared, &BTreeSet::from([first.diagnostic_identity()]));

    let affected = state.snapshot(first.diagnostic_identity()).unwrap();
    assert_eq!(
        affected.opaque_content(),
        GpuOpaqueContentContinuity::Unknown
    );
    assert_eq!(affected.initialized_coverage(), Some(&first_coverage));
    assert!(
        state
            .reconstruction_requirement(first.diagnostic_identity())
            .is_some()
    );

    let unaffected = state.snapshot(second.diagnostic_identity()).unwrap();
    assert_eq!(unaffected.initialized_coverage(), Some(&second_coverage));
    assert_eq!(
        unaffected.opaque_content(),
        GpuOpaqueContentContinuity::Established {
            last_completed_write: second_write,
        }
    );
    assert!(
        state
            .reconstruction_requirement(second.diagnostic_identity())
            .is_none()
    );
}

#[test]
fn acceptance_rejects_seed_after_completed_coverage_no_longer_contains_it() {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let buffer = retained_buffer(&mut allocator, "stale retained seed");
    let full = coverage(&buffer, 0, 32);
    let half = coverage(&buffer, 0, 16);
    let state = RetainedContinuityState::new(affinity(5, 1));
    establish(&state, &buffer, &full, submission(7));

    let stale = transition(
        &buffer,
        Some(full.clone()),
        Some(full.clone()),
        Some(full.clone()),
        Some(full.clone()),
    );
    let shrink = transition(
        &buffer,
        Some(full.clone()),
        Some(full),
        Some(half.clone()),
        Some(half),
    );
    state.validate_and_reserve(&shrink).unwrap();
    state.complete(submission(8), &shrink, &BTreeSet::new());

    let error = state.validate_and_reserve(&stale).unwrap_err();
    assert_eq!(
        error.kind(),
        GpuSubmissionRejectionKind::RetainedContinuityChanged
    );
}

#[test]
fn retained_transition_reservation_serializes_same_resource_acceptance() {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let buffer = retained_buffer(&mut allocator, "reserved retained state");
    let initialized = coverage(&buffer, 0, 16);
    let state = RetainedContinuityState::new(affinity(6, 1));
    establish(&state, &buffer, &initialized, submission(9));

    let prepared = transition(
        &buffer,
        Some(initialized.clone()),
        Some(initialized.clone()),
        Some(initialized.clone()),
        Some(initialized),
    );
    state.validate_and_reserve(&prepared).unwrap();
    let error = state.validate_and_reserve(&prepared).unwrap_err();
    assert_eq!(
        error.kind(),
        GpuSubmissionRejectionKind::RetainedContinuityChanged
    );
    state.fail_after_acceptance(&prepared, &BTreeSet::new());
}

#[test]
fn device_generation_owner_does_not_inherit_previous_generation_continuity() {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let buffer = retained_buffer(&mut allocator, "generation-local retained state");
    let initialized = coverage(&buffer, 0, 16);
    let old = RetainedContinuityState::new(affinity(7, 1));
    establish(&old, &buffer, &initialized, submission(10));

    let replacement = RetainedContinuityState::new(affinity(7, 2));
    assert!(replacement.coverage_seed().is_empty());
    assert!(replacement.snapshot(buffer.diagnostic_identity()).is_none());

    let old_seeded = transition(
        &buffer,
        Some(initialized.clone()),
        Some(initialized.clone()),
        Some(initialized.clone()),
        Some(initialized),
    );
    let error = replacement.validate_and_reserve(&old_seeded).unwrap_err();
    assert_eq!(
        error.kind(),
        GpuSubmissionRejectionKind::RetainedContinuityChanged
    );
}

#[test]
fn reconstruction_requirement_is_observable_without_fabricating_continuity() {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let buffer = retained_buffer(&mut allocator, "required state");
    let state = RetainedContinuityState::new(affinity(9, 2));
    state.install_reconstruction_obligations([GpuRetainedReconstructionSeed::new(
        GpuResourceRef::Buffer(buffer.clone()),
        true,
    )]);

    assert!(state.snapshot(buffer.diagnostic_identity()).is_none());
    let requirement = state
        .reconstruction_requirement(buffer.diagnostic_identity())
        .unwrap();
    assert_eq!(requirement.affinity(), affinity(9, 2));
    assert!(requirement.descriptor_initial_state_matches_required_contents());

    let ordinary = transition(&buffer, None, None, None, None);
    let error = state.validate_and_reserve(&ordinary).unwrap_err();
    assert_eq!(
        error.kind(),
        GpuSubmissionRejectionKind::RetainedContinuityChanged
    );
}

#[test]
fn source_backed_descriptor_state_reconstructs_only_while_it_still_matches_required_contents() {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let buffer = retained_buffer(&mut allocator, "source-backed state");
    let initialized = coverage(&buffer, 0, 16);
    let reconstruction = PreparedReconstructionEvidence {
        target: true,
        explicit_write: false,
        fresh_descriptor_initial_state: true,
    };

    let valid = RetainedContinuityState::new(affinity(10, 2));
    valid.install_reconstruction_obligations([GpuRetainedReconstructionSeed::new(
        GpuResourceRef::Buffer(buffer.clone()),
        true,
    )]);
    let prepared = transition_with_reconstruction(
        &buffer,
        None,
        Some(initialized.clone()),
        Some(initialized.clone()),
        Some(initialized.clone()),
        reconstruction,
    );
    valid.validate_and_reserve(&prepared).unwrap();
    valid.complete(submission(12), &prepared, &BTreeSet::new());
    assert!(
        valid
            .reconstruction_requirement(buffer.diagnostic_identity())
            .is_none()
    );
    let snapshot = valid.snapshot(buffer.diagnostic_identity()).unwrap();
    assert_eq!(snapshot.initialized_coverage(), Some(&initialized));
    assert_eq!(
        snapshot.opaque_content(),
        GpuOpaqueContentContinuity::Established {
            last_completed_write: submission(12),
        }
    );

    let mutated = RetainedContinuityState::new(affinity(11, 2));
    mutated.install_reconstruction_obligations([GpuRetainedReconstructionSeed::new(
        GpuResourceRef::Buffer(buffer.clone()),
        false,
    )]);
    let error = mutated.validate_and_reserve(&prepared).unwrap_err();
    assert_eq!(
        error.kind(),
        GpuSubmissionRejectionKind::RetainedContinuityChanged
    );
    assert!(
        mutated
            .reconstruction_requirement(buffer.diagnostic_identity())
            .is_some()
    );
}

#[test]
fn explicit_source_backed_replay_reconstructs_mutated_required_state() {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let buffer = retained_buffer(&mut allocator, "replayed state");
    let initialized = coverage(&buffer, 0, 16);
    let state = RetainedContinuityState::new(affinity(12, 2));
    state.install_reconstruction_obligations([GpuRetainedReconstructionSeed::new(
        GpuResourceRef::Buffer(buffer.clone()),
        false,
    )]);
    let replay = transition_with_reconstruction(
        &buffer,
        None,
        None,
        Some(initialized.clone()),
        None,
        PreparedReconstructionEvidence {
            target: true,
            explicit_write: true,
            fresh_descriptor_initial_state: false,
        },
    );
    state.validate_and_reserve(&replay).unwrap();
    state.complete(
        submission(13),
        &replay,
        &BTreeSet::from([buffer.diagnostic_identity()]),
    );

    assert!(
        state
            .reconstruction_requirement(buffer.diagnostic_identity())
            .is_none()
    );
    let snapshot = state.snapshot(buffer.diagnostic_identity()).unwrap();
    assert_eq!(snapshot.initialized_coverage(), Some(&initialized));
    assert_eq!(
        snapshot.opaque_content(),
        GpuOpaqueContentContinuity::Established {
            last_completed_write: submission(13),
        }
    );
    assert!(
        !state
            .reconstruction_seed()
            .into_iter()
            .find(|seed| seed.resource().diagnostic_identity() == buffer.diagnostic_identity())
            .unwrap()
            .descriptor_initial_state_matches_required_contents()
    );
}

#[test]
fn external_reconstruction_requires_an_explicit_canonical_write() {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let buffer = retained_external_buffer(&mut allocator, "external state");
    let initialized = coverage(&buffer, 0, 16);
    let state = RetainedContinuityState::new(affinity(13, 2));
    state.install_reconstruction_obligations([GpuRetainedReconstructionSeed::new(
        GpuResourceRef::Buffer(buffer.clone()),
        true,
    )]);

    let metadata_only = transition_with_reconstruction(
        &buffer,
        None,
        Some(initialized.clone()),
        Some(initialized.clone()),
        Some(initialized.clone()),
        PreparedReconstructionEvidence {
            target: true,
            explicit_write: false,
            fresh_descriptor_initial_state: true,
        },
    );
    assert!(state.validate_and_reserve(&metadata_only).is_err());

    let reimport = transition_with_reconstruction(
        &buffer,
        None,
        None,
        Some(initialized),
        None,
        PreparedReconstructionEvidence {
            target: true,
            explicit_write: true,
            fresh_descriptor_initial_state: false,
        },
    );
    state.validate_and_reserve(&reimport).unwrap();
    state.complete(
        submission(14),
        &reimport,
        &BTreeSet::from([buffer.diagnostic_identity()]),
    );
    assert!(
        state
            .reconstruction_requirement(buffer.diagnostic_identity())
            .is_none()
    );
}

#[test]
fn non_reconstructable_loss_cannot_be_certified_as_reconstructed() {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let buffer = retained_non_reconstructable_buffer(&mut allocator, "lost state");
    let state = RetainedContinuityState::new(affinity(14, 2));
    state.install_reconstruction_obligations([GpuRetainedReconstructionSeed::new(
        GpuResourceRef::Buffer(buffer.clone()),
        false,
    )]);
    let attempted = transition_with_reconstruction(
        &buffer,
        None,
        None,
        None,
        None,
        PreparedReconstructionEvidence {
            target: true,
            explicit_write: true,
            fresh_descriptor_initial_state: false,
        },
    );
    let error = state.validate_and_reserve(&attempted).unwrap_err();
    assert_eq!(
        error.kind(),
        GpuSubmissionRejectionKind::RetainedContinuityChanged
    );
    assert!(
        state
            .reconstruction_requirement(buffer.diagnostic_identity())
            .is_some()
    );
}

#[test]
fn failed_reconstruction_keeps_original_requirement_evidence() {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let buffer = retained_buffer(&mut allocator, "failed reconstruction");
    let initialized = coverage(&buffer, 0, 16);
    let state = RetainedContinuityState::new(affinity(15, 2));
    state.install_reconstruction_obligations([GpuRetainedReconstructionSeed::new(
        GpuResourceRef::Buffer(buffer.clone()),
        true,
    )]);
    let replay = transition_with_reconstruction(
        &buffer,
        None,
        None,
        Some(initialized.clone()),
        Some(initialized.clone()),
        PreparedReconstructionEvidence {
            target: true,
            explicit_write: true,
            fresh_descriptor_initial_state: false,
        },
    );
    state.validate_and_reserve(&replay).unwrap();
    state.fail_after_acceptance(&replay, &BTreeSet::from([buffer.diagnostic_identity()]));

    let requirement = state
        .reconstruction_requirement(buffer.diagnostic_identity())
        .unwrap();
    assert!(requirement.descriptor_initial_state_matches_required_contents());
    let snapshot = state.snapshot(buffer.diagnostic_identity()).unwrap();
    assert_eq!(
        snapshot.opaque_content(),
        GpuOpaqueContentContinuity::Unknown
    );
}

#[test]
fn prepared_transition_excludes_transient_storage() {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let transient = buffer_with_lifetime(
        &mut allocator,
        "transient storage",
        GpuResourceLifetime::Transient,
    );
    let mut fragment = GpuWorkFragmentBuilder::new(
        label("transient fragment"),
        provenance("transient fragment"),
    );
    fragment
        .declare_resource(GpuResourceRef::Buffer(transient))
        .unwrap();
    let graph =
        GpuPreparedWorkGraph::prepare(label("transient graph"), [fragment.finish().unwrap()])
            .unwrap();

    assert!(PreparedRetainedContinuity::from_graph(&graph).is_empty());
}

#[test]
fn prepared_transition_excludes_surface_acquired_storage() {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let surface = surface_acquired_texture(&mut allocator, "surface-acquired storage");
    assert_eq!(
        surface.descriptor().common().ownership(),
        GpuResourceOwnership::SurfaceAcquired
    );
    assert_eq!(
        surface.descriptor().common().lifetime(),
        GpuResourceLifetime::Transient
    );

    let mut fragment = GpuWorkFragmentBuilder::new(
        label("surface-acquired fragment"),
        provenance("surface-acquired fragment"),
    );
    fragment
        .declare_resource(GpuResourceRef::Texture(surface))
        .unwrap();
    let graph = GpuPreparedWorkGraph::prepare(
        label("surface-acquired graph"),
        [fragment.finish().unwrap()],
    )
    .unwrap();

    assert!(PreparedRetainedContinuity::from_graph(&graph).is_empty());
}

#[test]
fn acceptance_rejects_changed_descriptor_for_same_logical_identity() {
    let owner = NonZeroU64::new(91).unwrap();
    let mut original_allocator = GpuWorkResourceIdAllocator::for_owner_scope(owner);
    let mut replacement_allocator = GpuWorkResourceIdAllocator::for_owner_scope(owner);
    let make_buffer = |allocator: &mut GpuWorkResourceIdAllocator, name: &str, size: u64| {
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
                    size,
                    GpuBufferUsages::new(&resource_label, [GpuBufferUsage::Storage]).unwrap(),
                    GpuBufferInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap()
    };
    let original = make_buffer(&mut original_allocator, "original retained descriptor", 64);
    let replacement = make_buffer(
        &mut replacement_allocator,
        "replacement retained descriptor",
        32,
    );
    assert_eq!(
        original.diagnostic_identity(),
        replacement.diagnostic_identity()
    );
    assert_ne!(original.descriptor(), replacement.descriptor());

    let initialized = coverage(&original, 0, 16);
    let state = RetainedContinuityState::new(affinity(8, 1));
    establish(&state, &original, &initialized, submission(11));

    let replacement_transition = transition(&replacement, None, None, None, None);
    let error = state
        .validate_and_reserve(&replacement_transition)
        .unwrap_err();
    assert_eq!(
        error.kind(),
        GpuSubmissionRejectionKind::RetainedContinuityChanged
    );
}
