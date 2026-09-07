use super::device_request::{enforce_runengpu_instance_flags, request_with_instance};
use crate::*;
use std::num::NonZeroU64;
use wgpu::{Backends, Instance, InstanceDescriptor, NoopBackendOptions};

const BINDING_ONLY_WGSL: &str = r#"
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

fn noop_instance() -> Instance {
    let mut descriptor = InstanceDescriptor::new_without_display_handle();
    descriptor.backends = Backends::NOOP;
    descriptor.backend_options.noop = NoopBackendOptions::enabled();
    Instance::new(enforce_runengpu_instance_flags(descriptor))
}

fn noop_context(requirements: GpuCapabilityRequirements, name: &str) -> GpuContext {
    let descriptor = GpuContextDescriptor::new(requirements)
        .with_allowed_backends([GpuBackendFamily::UnknownBackend])
        .with_label(name);
    let context = pollster::block_on(request_with_instance(
        noop_instance(),
        descriptor,
        None,
        GpuRealizationPolicies::default(),
        GpuExecutionPolicy::default(),
    ))
    .expect("explicit WGPU noop seam must admit deterministic G5R preparation proof");
    assert_eq!(
        context.adapter_facts().backend(),
        GpuBackendFamily::UnknownBackend
    );
    assert_eq!(
        context.admission_report().candidate().portability(),
        GpuPortabilityClass::Unsupported,
        "test-only WGPU noop must not masquerade as a production backend"
    );
    context
}

fn copy_context() -> GpuContext {
    let mut requirements = GpuCapabilityRequirements::new();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Copy,
        ))
        .unwrap();
    noop_context(requirements, "G5R deterministic transfer-alignment proof")
}

fn compute_context() -> GpuContext {
    let mut requirements = GpuCapabilityProfile::ComputeBaseline.requirements();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Copy,
        ))
        .unwrap();
    noop_context(requirements, "G5R deterministic binding-only proof")
}

fn prepared_buffer(
    allocator: &mut GpuWorkResourceIdAllocator,
    name: &str,
    bytes: &[u8],
    usages: impl IntoIterator<Item = GpuBufferUsage>,
) -> GpuBufferHandle {
    let resource_label = label(name);
    allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common(name),
                u64::try_from(bytes.len()).unwrap(),
                GpuBufferUsages::new(&resource_label, usages).unwrap(),
                GpuBufferInitialization::Prepared(
                    PreparedGpuData::<TransferData>::from_pod_transfer(
                        format!("{name} initial bytes"),
                        bytes,
                        provenance(&format!("{name} initial bytes")),
                    )
                    .unwrap(),
                ),
            )
            .unwrap(),
        )
        .unwrap()
}

fn uninitialized_buffer(
    allocator: &mut GpuWorkResourceIdAllocator,
    name: &str,
    size: u64,
    usages: impl IntoIterator<Item = GpuBufferUsage>,
) -> GpuBufferHandle {
    let resource_label = label(name);
    allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common(name),
                size,
                GpuBufferUsages::new(&resource_label, usages).unwrap(),
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap()
}

fn readback_graph(buffer: &GpuBufferHandle, name: &str) -> GpuPreparedWorkGraph {
    let region = GpuBufferRegion::new(buffer, GpuBufferRange::whole(buffer).unwrap()).unwrap();
    let readback =
        GpuReadbackOperation::new(region.into(), GpuReadbackId::allocate().unwrap()).unwrap();
    let mut builder = GpuWorkFragmentBuilder::new(label(name), provenance(name));
    builder.declare_resource(buffer.clone().into()).unwrap();
    builder
        .add_node(
            label("read prepared buffer"),
            GpuWorkOperation::Readback(readback),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::TransferPreferred,
            provenance("read prepared buffer"),
        )
        .unwrap();
    GpuPreparedWorkGraph::prepare(label(&format!("{name} graph")), [builder.finish().unwrap()])
        .unwrap()
}

fn explicit_upload_graph(
    buffer: &GpuBufferHandle,
    bytes: &[u8],
    name: &str,
) -> GpuPreparedWorkGraph {
    let region = GpuBufferRegion::new(buffer, GpuBufferRange::whole(buffer).unwrap()).unwrap();
    let upload = GpuUploadOperation::new(
        region.into(),
        PreparedGpuData::<TransferData>::from_pod_transfer(
            format!("{name} payload"),
            bytes,
            provenance(&format!("{name} payload")),
        )
        .unwrap(),
    )
    .unwrap();
    let mut builder = GpuWorkFragmentBuilder::new(label(name), provenance(name));
    builder.declare_resource(buffer.clone().into()).unwrap();
    builder
        .add_node(
            label("explicit upload"),
            GpuWorkOperation::Upload(upload),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::TransferPreferred,
            provenance("explicit upload"),
        )
        .unwrap();
    GpuPreparedWorkGraph::prepare(label(&format!("{name} graph")), [builder.finish().unwrap()])
        .unwrap()
}

fn binding_pipeline() -> GpuComputePipelineDescriptor {
    let owner = GpuProgramSourceOwnerId::allocate().expect("G5R test source owner should allocate");
    let identity = GpuProgramSourceIdentity::new(
        owner,
        GpuProgramSourceKey::new("g5r.binding-only-prepared").unwrap(),
        GpuProgramSourceRevision::try_from_raw(1).unwrap(),
    );
    let mut sources = GpuProgramSourceRegistry::new(1, 16 * 1024).unwrap();
    let source = sources
        .admit_wgsl(
            identity,
            BINDING_ONLY_WGSL,
            GpuProgramSourceProvenance::new("g5r-binding-only-test", None).unwrap(),
        )
        .unwrap();
    let entry = GpuEntryPointName::new("cs_main").unwrap();
    let refinement = GpuBindingLayoutRefinement::new(GpuBindingKey::try_new(0, 0).unwrap())
        .with_host_minimum_size(NonZeroU64::new(4).unwrap());
    let program = GpuProgramDescriptor::new(source, [entry.clone()], [refinement]).unwrap();
    GpuComputePipelineDescriptor::new(program, entry, GpuPipelineConfiguration::default()).unwrap()
}

fn binding_compute(buffer: &GpuBufferHandle) -> GpuComputeOperation {
    let pipeline = binding_pipeline();
    let binding = GpuRuntimeBindingValue::new(
        GpuBindingKey::try_new(0, 0).unwrap(),
        [GpuRuntimeBindingResource::Buffer(
            GpuRuntimeBufferBinding::new(
                buffer.clone(),
                0,
                NonZeroU64::new(buffer.descriptor().size_bytes()).unwrap(),
                None,
            ),
        )],
    )
    .unwrap();
    let bindings = GpuRuntimeBindingSet::new(pipeline.layout().clone(), [binding]).unwrap();
    let dispatch = GpuDispatchIntent::direct(GpuDispatchSize::new(1, 1, 1));
    GpuComputeOperation::new(pipeline, bindings, dispatch).unwrap()
}

#[test]
fn binding_only_prepared_buffer_is_selected_and_lowered_before_compute_work() {
    let context = compute_context();
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let initial = [41_u32, 7, 9, 11];
    let bytes = initial
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let buffer = prepared_buffer(
        &mut allocator,
        "binding-only prepared buffer",
        &bytes,
        [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
    );
    let compute = binding_compute(&buffer);
    let mut builder = GpuWorkFragmentBuilder::new(
        label("binding-only prepared compute"),
        provenance("binding-only prepared compute"),
    );
    builder.declare_resource(buffer.clone().into()).unwrap();
    builder
        .compute("consume prepared storage binding", compute)
        .unwrap();

    let graph = GpuPreparedWorkGraph::prepare(
        label("binding-only prepared compute graph"),
        [builder.finish().unwrap()],
    )
    .expect("operation-derived binding access must admit the Prepared materialization effect");
    assert_eq!(graph.initial_content().len(), 1);
    assert_eq!(
        graph.initial_content()[0].resource_identity(),
        buffer.diagnostic_identity(),
        "binding-only access must select the exact prepared storage resource"
    );
    let prepared = pollster::block_on(context.prepare_submission(graph))
        .expect("backend preparation must lower binding-only Prepared content before compute work");
    assert_eq!(context.execution_stats().prepared_submissions(), 1);
    assert_eq!(
        context.execution_stats().upload_bytes_in_flight(),
        0,
        "conditional initial-content bytes are not charged until ordered acceptance"
    );
    drop(prepared);
    assert_eq!(context.execution_stats().prepared_submissions(), 0);
}

#[test]
fn prepared_and_explicit_buffer_uploads_share_transfer_alignment_rejection() {
    let context = copy_context();
    let alignment = context
        .device_facts()
        .device_limits()
        .alignments()
        .copy_buffer_offset
        .expect("copy-capable context must publish buffer-copy alignment");
    assert!(alignment > 1, "test requires a nontrivial copy alignment");
    let size = alignment.checked_add(1).unwrap();
    let bytes = vec![0xA5_u8; usize::try_from(size).unwrap()];
    let mut allocator = GpuWorkResourceIdAllocator::new();

    let prepared_buffer = prepared_buffer(
        &mut allocator,
        "unaligned prepared buffer",
        &bytes,
        [GpuBufferUsage::CopySource, GpuBufferUsage::CopyDestination],
    );
    let prepared_error = pollster::block_on(context.prepare_submission(readback_graph(
        &prepared_buffer,
        "unaligned prepared transfer",
    )))
    .expect_err("unencodable Prepared whole-buffer transfer must reject during preparation");

    let explicit_buffer = uninitialized_buffer(
        &mut allocator,
        "unaligned explicit upload buffer",
        size,
        [GpuBufferUsage::CopyDestination],
    );
    let explicit_error = pollster::block_on(context.prepare_submission(explicit_upload_graph(
        &explicit_buffer,
        &bytes,
        "unaligned explicit transfer",
    )))
    .expect_err("equivalent unencodable explicit Upload must reject during preparation");

    assert_eq!(
        prepared_error.kind(),
        GpuSubmissionPreparationErrorKind::TransferAlignmentNotAdmitted
    );
    assert_eq!(explicit_error.kind(), prepared_error.kind());
    assert_eq!(
        explicit_error.detail(),
        prepared_error.detail(),
        "Prepared and explicit buffer transfers must report through the same alignment authority"
    );
}

#[test]
fn prepared_buffer_descriptor_requires_copy_destination_usage() {
    let name = "prepared buffer without copy destination";
    let resource_label = label(name);
    let error = GpuBufferDescriptor::new(
        common(name),
        4,
        GpuBufferUsages::new(&resource_label, [GpuBufferUsage::Storage]).unwrap(),
        GpuBufferInitialization::Prepared(
            PreparedGpuData::<TransferData>::from_pod_transfer(
                format!("{name} initial bytes"),
                &[1_u8, 2, 3, 4],
                provenance(&format!("{name} initial bytes")),
            )
            .unwrap(),
        ),
    )
    .expect_err("Prepared buffers without CopyDestination must reject");
    assert_eq!(
        error.cause(),
        GpuResourceDescriptorCause::InvalidInitialization
    );
}
