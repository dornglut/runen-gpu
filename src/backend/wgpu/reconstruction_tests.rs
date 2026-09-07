use super::device_request::{enforce_runengpu_instance_flags, request_with_instance};
use crate::*;
use std::time::{Duration, Instant};
use wgpu::{Backends, Instance, InstanceDescriptor, NoopBackendOptions};

fn label(value: &str) -> GpuResourceLabel {
    GpuResourceLabel::new(value).unwrap()
}

fn provenance(value: &str) -> GpuResourceProvenance {
    GpuResourceProvenance::new(label(value), None, None)
}

fn noop_instance() -> Instance {
    let mut descriptor = InstanceDescriptor::new_without_display_handle();
    descriptor.backends = Backends::NOOP;
    descriptor.backend_options.noop = NoopBackendOptions::enabled();
    Instance::new(enforce_runengpu_instance_flags(descriptor))
}

fn context_descriptor(name: &str) -> GpuContextDescriptor {
    let mut requirements = GpuCapabilityRequirements::new();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Copy,
        ))
        .unwrap();
    GpuContextDescriptor::new(requirements)
        .with_allowed_backends([GpuBackendFamily::UnknownBackend])
        .with_label(name)
}

fn incompatible_noop_descriptor(name: &str) -> GpuContextDescriptor {
    let mut requirements = GpuCapabilityRequirements::new();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Copy,
        ))
        .unwrap();
    GpuContextDescriptor::new(requirements)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label(name)
}

fn noop_context(name: &str) -> GpuContext {
    let context = pollster::block_on(request_with_instance(
        noop_instance(),
        context_descriptor(name),
        None,
        GpuRealizationPolicies::default(),
        GpuExecutionPolicy::default(),
    ))
    .expect("explicit WGPU noop context must be admitted for deterministic reconstruction proof");
    assert_eq!(
        context.adapter_facts().backend(),
        GpuBackendFamily::UnknownBackend,
        "the deterministic reconstruction seam must not masquerade as a production backend"
    );
    context
}

fn transfer_data(name: &str, values: &[u32]) -> PreparedGpuData<TransferData> {
    PreparedGpuData::<TransferData>::from_pod_transfer(
        format!("{name} transfer source"),
        values,
        provenance(&format!("{name} transfer source")),
    )
    .unwrap()
}

fn retained_prepared_buffer(name: &str, values: &[u32]) -> GpuBufferHandle {
    let resource_label = label(name);
    let common = GpuResourceCommon::owned(
        resource_label.clone(),
        GpuResourceLifetime::Retained,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        provenance(name),
    )
    .unwrap();
    let data = transfer_data(name, values);
    let byte_len = u64::try_from(std::mem::size_of_val(values)).unwrap();
    let descriptor = GpuBufferDescriptor::new(
        common,
        byte_len,
        GpuBufferUsages::new(
            &resource_label,
            [GpuBufferUsage::CopySource, GpuBufferUsage::CopyDestination],
        )
        .unwrap(),
        GpuBufferInitialization::Prepared(data),
    )
    .unwrap();
    GpuWorkResourceIdAllocator::new()
        .allocate_buffer_handle(descriptor)
        .unwrap()
}

fn retained_external_buffer(name: &str, byte_len: u64) -> GpuBufferHandle {
    let resource_label = label(name);
    let common = GpuResourceCommon::owned(
        resource_label.clone(),
        GpuResourceLifetime::Retained,
        GpuMemoryIntent::Device,
        GpuReconstruction::ExternallyReconstructed,
        provenance(name),
    )
    .unwrap();
    let descriptor = GpuBufferDescriptor::new(
        common,
        byte_len,
        GpuBufferUsages::new(
            &resource_label,
            [GpuBufferUsage::CopySource, GpuBufferUsage::CopyDestination],
        )
        .unwrap(),
        GpuBufferInitialization::Uninitialized,
    )
    .unwrap();
    GpuWorkResourceIdAllocator::new()
        .allocate_buffer_handle(descriptor)
        .unwrap()
}

fn retained_non_reconstructable_buffer(name: &str, byte_len: u64) -> GpuBufferHandle {
    let resource_label = label(name);
    let common = GpuResourceCommon::owned_retained_non_reconstructable(
        resource_label.clone(),
        GpuMemoryIntent::Device,
        provenance(name),
    );
    let descriptor = GpuBufferDescriptor::new(
        common,
        byte_len,
        GpuBufferUsages::new(
            &resource_label,
            [GpuBufferUsage::CopySource, GpuBufferUsage::CopyDestination],
        )
        .unwrap(),
        GpuBufferInitialization::Uninitialized,
    )
    .unwrap();
    GpuWorkResourceIdAllocator::new()
        .allocate_buffer_handle(descriptor)
        .unwrap()
}

fn add_operation(builder: &mut GpuWorkFragmentBuilder, name: &str, operation: GpuWorkOperation) {
    builder
        .add_node(
            label(name),
            operation,
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::TransferPreferred,
            provenance(name),
        )
        .unwrap();
}

fn readback_fragment(buffer: &GpuBufferHandle, name: &str) -> (GpuWorkFragment, GpuReadbackId) {
    let region = GpuBufferRegion::new(buffer, GpuBufferRange::whole(buffer).unwrap()).unwrap();
    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(region.into(), readback_id).unwrap();
    let mut builder = GpuWorkFragmentBuilder::new(label(name), provenance(name));
    builder
        .declare_resource(GpuResourceRef::Buffer(buffer.clone()))
        .unwrap();
    add_operation(
        &mut builder,
        &format!("{name} readback"),
        GpuWorkOperation::Readback(readback),
    );
    (builder.finish().unwrap(), readback_id)
}

fn upload_and_readback_fragment(
    buffer: &GpuBufferHandle,
    values: &[u32],
    name: &str,
) -> (GpuWorkFragment, GpuReadbackId) {
    let upload_region =
        GpuBufferRegion::new(buffer, GpuBufferRange::whole(buffer).unwrap()).unwrap();
    let readback_region = upload_region.clone();
    let upload =
        GpuUploadOperation::new(upload_region.into(), transfer_data(name, values)).unwrap();
    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(readback_region.into(), readback_id).unwrap();
    let mut builder = GpuWorkFragmentBuilder::new(label(name), provenance(name));
    builder
        .declare_resource(GpuResourceRef::Buffer(buffer.clone()))
        .unwrap();
    add_operation(
        &mut builder,
        &format!("{name} upload"),
        GpuWorkOperation::Upload(upload),
    );
    add_operation(
        &mut builder,
        &format!("{name} readback"),
        GpuWorkOperation::Readback(readback),
    );
    (builder.finish().unwrap(), readback_id)
}

fn zero_and_readback_fragment(
    buffer: &GpuBufferHandle,
    name: &str,
) -> (GpuWorkFragment, GpuReadbackId) {
    let clear_region =
        GpuBufferRegion::new(buffer, GpuBufferRange::whole(buffer).unwrap()).unwrap();
    let readback_region = clear_region.clone();
    let clear = GpuClearOperation::buffer_zero(clear_region).unwrap();
    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(readback_region.into(), readback_id).unwrap();
    let mut builder = GpuWorkFragmentBuilder::new(label(name), provenance(name));
    builder
        .declare_resource(GpuResourceRef::Buffer(buffer.clone()))
        .unwrap();
    add_operation(
        &mut builder,
        &format!("{name} zero"),
        GpuWorkOperation::Clear(clear),
    );
    add_operation(
        &mut builder,
        &format!("{name} readback"),
        GpuWorkOperation::Readback(readback),
    );
    (builder.finish().unwrap(), readback_id)
}

fn submit_and_readback(
    context: &GpuContext,
    graph: GpuPreparedWorkGraph,
    readback_id: GpuReadbackId,
) -> (GpuSubmissionId, GpuReadbackBytes) {
    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let readback = submission
        .readback(readback_id)
        .expect("accepted reconstruction proof readback must remain observable")
        .clone();
    let deadline = Instant::now() + Duration::from_secs(5);
    let bytes = loop {
        context.progress();
        match readback.status() {
            GpuReadbackStatus::Ready(bytes) => break bytes,
            GpuReadbackStatus::Failed(failure) => {
                panic!("reconstruction proof readback failed: {failure:?}")
            }
            GpuReadbackStatus::Pending => {}
        }
        if let GpuSubmissionStatus::Failed(failure) = submission.status() {
            panic!("reconstruction proof submission failed before readback: {failure:?}");
        }
        assert!(
            Instant::now() < deadline,
            "reconstruction proof readback did not materialize"
        );
        std::thread::yield_now();
    };
    loop {
        context.progress();
        match submission.status() {
            GpuSubmissionStatus::Completed => break,
            GpuSubmissionStatus::Failed(failure) => {
                panic!("reconstruction proof submission failed: {failure:?}")
            }
            GpuSubmissionStatus::Accepted => {}
        }
        assert!(
            Instant::now() < deadline,
            "reconstruction proof submission did not terminalize"
        );
        std::thread::yield_now();
    }
    (submission.id(), bytes)
}

fn assert_u32_bytes(bytes: &GpuReadbackBytes, values: &[u32]) {
    let expected = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    assert_eq!(bytes.as_bytes(), expected.as_slice());
}

#[test]
fn descriptor_backed_state_reconstructs_only_through_completed_new_generation_work() {
    let seed = [3_u32, 5, 8, 13];
    let mut context = noop_context("G7B descriptor reconstruction");
    let buffer = retained_prepared_buffer("descriptor-backed retained state", &seed);
    let identity = buffer.diagnostic_identity();

    let (fragment, readback_id) = readback_fragment(&buffer, "initial descriptor state");
    let graph = context
        .prepare_work_graph(label("initial descriptor state graph"), [fragment])
        .unwrap();
    let (initial_submission, bytes) = submit_and_readback(&context, graph, readback_id);
    assert_u32_bytes(&bytes, &seed);
    let initial_continuity = context.retained_resource_continuity(identity).unwrap();
    assert_eq!(initial_continuity.affinity(), context.affinity());
    assert!(initial_continuity.initialized_coverage().is_some());
    assert_eq!(
        initial_continuity.opaque_content(),
        GpuOpaqueContentContinuity::Established {
            last_completed_write: initial_submission,
        }
    );

    let before_replacement = context.affinity();
    let failed = pollster::block_on(
        context.replace_device_generation(incompatible_noop_descriptor("G7B rejected successor")),
    )
    .unwrap_err();
    assert!(matches!(
        failed,
        GpuDeviceGenerationReplacementError::ContextRequest(_)
    ));
    assert_eq!(context.affinity(), before_replacement);
    assert!(context.retained_resource_continuity(identity).is_some());

    let (stale_fragment, _) = readback_fragment(&buffer, "stale prepared state");
    let stale_graph = context
        .prepare_work_graph(label("stale prepared state graph"), [stale_fragment])
        .unwrap();
    let stale_prepared = pollster::block_on(context.prepare_submission(stale_graph)).unwrap();

    pollster::block_on(context.replace_device_generation(context_descriptor(
        "G7B descriptor reconstruction successor",
    )))
    .unwrap();
    assert_eq!(context.id(), before_replacement.context());
    assert_ne!(context.generation(), before_replacement.generation());

    let stale = context.submit_prepared(stale_prepared).unwrap_err();
    assert_eq!(
        stale.reason().kind(),
        GpuSubmissionRejectionKind::StaleDeviceGeneration
    );
    assert!(context.retained_resource_continuity(identity).is_none());
    let requirement = context
        .retained_resource_reconstruction_requirement(identity)
        .expect("lost descriptor-backed state must become an explicit reconstruction requirement");
    assert_eq!(requirement.affinity(), context.affinity());
    assert_eq!(requirement.resource().diagnostic_identity(), identity);
    assert!(requirement.descriptor_initial_state_matches_required_contents());

    let (ordinary_fragment, _) = readback_fragment(&buffer, "ordinary use before reconstruction");
    let ordinary_graph = context
        .prepare_work_graph(
            label("ordinary use before reconstruction graph"),
            [ordinary_fragment],
        )
        .unwrap();
    let ordinary_prepared = pollster::block_on(context.prepare_submission(ordinary_graph)).unwrap();
    let ordinary_rejection = context.submit_prepared(ordinary_prepared).unwrap_err();
    assert_eq!(
        ordinary_rejection.reason().kind(),
        GpuSubmissionRejectionKind::RetainedContinuityChanged
    );
    assert!(context.retained_resource_continuity(identity).is_none());
    assert!(
        context
            .retained_resource_reconstruction_requirement(identity)
            .is_some()
    );

    let (reconstruction_fragment, reconstruction_readback) =
        readback_fragment(&buffer, "descriptor reconstruction");
    let reconstruction_graph = context
        .prepare_reconstruction_work_graph(
            label("descriptor reconstruction graph"),
            [reconstruction_fragment],
            [GpuResourceRef::Buffer(buffer.clone())],
        )
        .unwrap();
    let (reconstruction_submission, bytes) =
        submit_and_readback(&context, reconstruction_graph, reconstruction_readback);
    assert_u32_bytes(&bytes, &seed);
    assert!(
        context
            .retained_resource_reconstruction_requirement(identity)
            .is_none()
    );
    let continuity = context.retained_resource_continuity(identity).unwrap();
    assert_eq!(continuity.affinity(), context.affinity());
    assert!(continuity.initialized_coverage().is_some());
    assert_eq!(
        continuity.opaque_content(),
        GpuOpaqueContentContinuity::Established {
            last_completed_write: reconstruction_submission,
        }
    );
}

#[test]
fn exact_gpu_mutation_makes_original_seed_insufficient_until_replay_completes() {
    let seed = [10_u32];
    let expected_current = [0_u32];
    let mut context = noop_context("G7B deterministic replay");
    let buffer = retained_prepared_buffer("GPU-mutated retained state", &seed);
    let identity = buffer.diagnostic_identity();

    let (fragment, readback_id) = zero_and_readback_fragment(&buffer, "initial mutation");
    let graph = context
        .prepare_work_graph(label("initial mutation graph"), [fragment])
        .unwrap();
    let (_, bytes) = submit_and_readback(&context, graph, readback_id);
    assert_u32_bytes(&bytes, &expected_current);

    pollster::block_on(
        context.replace_device_generation(context_descriptor("G7B deterministic replay successor")),
    )
    .unwrap();
    assert!(context.retained_resource_continuity(identity).is_none());
    let requirement = context
        .retained_resource_reconstruction_requirement(identity)
        .expect("GPU-mutated state must require explicit reconstruction after generation loss");
    assert!(
        !requirement.descriptor_initial_state_matches_required_contents(),
        "the retained original seed must not be reported as the latest GPU-mutated state"
    );

    let (seed_only_fragment, _) = readback_fragment(&buffer, "seed-only reconstruction");
    let seed_only_graph = context
        .prepare_reconstruction_work_graph(
            label("seed-only reconstruction graph"),
            [seed_only_fragment],
            [GpuResourceRef::Buffer(buffer.clone())],
        )
        .unwrap();
    let seed_only_prepared =
        pollster::block_on(context.prepare_submission(seed_only_graph)).unwrap();
    let seed_only_rejection = context.submit_prepared(seed_only_prepared).unwrap_err();
    assert_eq!(
        seed_only_rejection.reason().kind(),
        GpuSubmissionRejectionKind::RetainedContinuityChanged
    );
    assert!(context.retained_resource_continuity(identity).is_none());
    assert!(
        context
            .retained_resource_reconstruction_requirement(identity)
            .is_some()
    );

    let (replay_fragment, replay_readback) =
        zero_and_readback_fragment(&buffer, "deterministic replay");
    let replay_graph = context
        .prepare_reconstruction_work_graph(
            label("deterministic replay graph"),
            [replay_fragment],
            [GpuResourceRef::Buffer(buffer.clone())],
        )
        .unwrap();
    let (replay_submission, bytes) = submit_and_readback(&context, replay_graph, replay_readback);
    assert_u32_bytes(&bytes, &expected_current);
    assert!(
        context
            .retained_resource_reconstruction_requirement(identity)
            .is_none()
    );
    let continuity = context.retained_resource_continuity(identity).unwrap();
    assert!(continuity.initialized_coverage().is_some());
    assert_eq!(
        continuity.opaque_content(),
        GpuOpaqueContentContinuity::Established {
            last_completed_write: replay_submission,
        }
    );

    let (ordinary_fragment, ordinary_readback) =
        readback_fragment(&buffer, "post-replay ordinary read");
    let ordinary_graph = context
        .prepare_work_graph(
            label("post-replay ordinary read graph"),
            [ordinary_fragment],
        )
        .unwrap();
    let (_, bytes) = submit_and_readback(&context, ordinary_graph, ordinary_readback);
    assert_u32_bytes(&bytes, &expected_current);
}

#[test]
fn external_state_requires_designated_canonical_reimport_after_generation_loss() {
    let external = [21_u32];
    let mut context = noop_context("G7B external reimport");
    let buffer = retained_external_buffer(
        "externally reconstructed retained state",
        u64::try_from(std::mem::size_of_val(&external)).unwrap(),
    );
    let identity = buffer.diagnostic_identity();

    let (initial_fragment, initial_readback) =
        upload_and_readback_fragment(&buffer, &external, "initial external import");
    let initial_graph = context
        .prepare_work_graph(label("initial external import graph"), [initial_fragment])
        .unwrap();
    let (_, bytes) = submit_and_readback(&context, initial_graph, initial_readback);
    assert_u32_bytes(&bytes, &external);

    pollster::block_on(
        context.replace_device_generation(context_descriptor("G7B external reimport successor")),
    )
    .unwrap();
    assert!(context.retained_resource_continuity(identity).is_none());
    let requirement = context
        .retained_resource_reconstruction_requirement(identity)
        .expect("externally reconstructed retained state must remain unavailable after loss");
    assert_eq!(
        requirement.resource().common().reconstruction(),
        GpuReconstruction::ExternallyReconstructed
    );

    let (ordinary_fragment, _) =
        upload_and_readback_fragment(&buffer, &external, "undesignated external reimport");
    let ordinary_graph = context
        .prepare_work_graph(
            label("undesignated external reimport graph"),
            [ordinary_fragment],
        )
        .unwrap();
    let ordinary_prepared = pollster::block_on(context.prepare_submission(ordinary_graph)).unwrap();
    let ordinary_rejection = context.submit_prepared(ordinary_prepared).unwrap_err();
    assert_eq!(
        ordinary_rejection.reason().kind(),
        GpuSubmissionRejectionKind::RetainedContinuityChanged
    );
    assert!(
        context
            .retained_resource_reconstruction_requirement(identity)
            .is_some()
    );

    let (reimport_fragment, reimport_readback) =
        upload_and_readback_fragment(&buffer, &external, "designated external reimport");
    let reimport_graph = context
        .prepare_reconstruction_work_graph(
            label("designated external reimport graph"),
            [reimport_fragment],
            [GpuResourceRef::Buffer(buffer.clone())],
        )
        .unwrap();
    let (reimport_submission, bytes) =
        submit_and_readback(&context, reimport_graph, reimport_readback);
    assert_u32_bytes(&bytes, &external);
    assert!(
        context
            .retained_resource_reconstruction_requirement(identity)
            .is_none()
    );
    let continuity = context.retained_resource_continuity(identity).unwrap();
    assert_eq!(
        continuity.opaque_content(),
        GpuOpaqueContentContinuity::Established {
            last_completed_write: reimport_submission,
        }
    );
}

#[test]
fn non_reconstructable_loss_cannot_be_certified_by_same_identity_canonical_write() {
    let state = [34_u32];
    let mut context = noop_context("G7B non-reconstructable loss");
    let buffer = retained_non_reconstructable_buffer(
        "non-reconstructable retained state",
        u64::try_from(std::mem::size_of_val(&state)).unwrap(),
    );
    let identity = buffer.diagnostic_identity();

    let (initial_fragment, initial_readback) =
        upload_and_readback_fragment(&buffer, &state, "initial non-reconstructable state");
    let initial_graph = context
        .prepare_work_graph(
            label("initial non-reconstructable state graph"),
            [initial_fragment],
        )
        .unwrap();
    let (_, bytes) = submit_and_readback(&context, initial_graph, initial_readback);
    assert_u32_bytes(&bytes, &state);

    pollster::block_on(
        context.replace_device_generation(context_descriptor("G7B non-reconstructable successor")),
    )
    .unwrap();
    assert!(context.retained_resource_continuity(identity).is_none());
    let requirement = context
        .retained_resource_reconstruction_requirement(identity)
        .expect("non-reconstructable retained state must remain explicitly lost");
    assert_eq!(
        requirement.resource().common().reconstruction(),
        GpuReconstruction::NonReconstructable
    );

    let (attempt_fragment, _) =
        upload_and_readback_fragment(&buffer, &state, "attempted same-identity reset");
    let attempt_graph = context
        .prepare_reconstruction_work_graph(
            label("attempted same-identity reset graph"),
            [attempt_fragment],
            [GpuResourceRef::Buffer(buffer.clone())],
        )
        .unwrap();
    let attempt_prepared = pollster::block_on(context.prepare_submission(attempt_graph)).unwrap();
    let rejection = context.submit_prepared(attempt_prepared).unwrap_err();
    assert_eq!(
        rejection.reason().kind(),
        GpuSubmissionRejectionKind::RetainedContinuityChanged
    );
    assert!(context.retained_resource_continuity(identity).is_none());
    assert!(
        context
            .retained_resource_reconstruction_requirement(identity)
            .is_some()
    );
}
