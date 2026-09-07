use super::{
    GpuRuntimeBindingDeviceFacts, GpuRuntimeBindingResource, GpuRuntimeBindingSet,
    GpuRuntimeBindingValue, GpuRuntimeBufferBinding, GpuValidatedBindGroupBindings,
};
use crate::{
    GpuBindGroupLayoutDescriptor, GpuBindingDeclaration, GpuBindingKey, GpuBindingKind,
    GpuBindingProvenance, GpuBufferAccessKind, GpuBufferDescriptor, GpuBufferHandle,
    GpuBufferInitialization, GpuBufferUsage, GpuBufferUsages, GpuMemoryIntent,
    GpuPipelineLayoutDescriptor, GpuProgramContractCause, GpuReconstruction, GpuResourceAccess,
    GpuResourceCommon, GpuResourceLabel, GpuResourceLifetime, GpuResourceProvenance,
    GpuShaderStage, GpuShaderStages, GpuStorageBufferAccess, GpuWorkResourceIdAllocator,
};
use core::num::{NonZeroU32, NonZeroU64};

fn storage_buffer(size: u64) -> GpuBufferHandle {
    let label = GpuResourceLabel::new("runtime-binding-storage").unwrap();
    let common = GpuResourceCommon::owned(
        label.clone(),
        GpuResourceLifetime::Transient,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        GpuResourceProvenance::new(label.clone(), None, None),
    )
    .unwrap();
    let usages = GpuBufferUsages::new(&label, [GpuBufferUsage::Storage]).unwrap();
    let descriptor =
        GpuBufferDescriptor::new(common, size, usages, GpuBufferInitialization::Uninitialized)
            .unwrap();
    let mut allocator = GpuWorkResourceIdAllocator::for_owner_scope(NonZeroU64::new(17).unwrap());
    allocator.allocate_buffer_handle(descriptor).unwrap()
}

fn declaration(array_count: Option<NonZeroU32>) -> GpuBindingDeclaration {
    storage_declaration(0, array_count)
}

fn storage_declaration(binding: u32, array_count: Option<NonZeroU32>) -> GpuBindingDeclaration {
    GpuBindingDeclaration::new(
        GpuBindingKey::try_new(0, u64::from(binding)).unwrap(),
        GpuShaderStages::one(GpuShaderStage::Compute),
        GpuBindingKind::storage_buffer(
            GpuStorageBufferAccess::ReadWrite,
            true,
            NonZeroU64::new(16),
        ),
        array_count,
        format!("storage-{binding}"),
        GpuBindingProvenance::new("runtime-binding-test", None).unwrap(),
    )
    .unwrap()
}

fn analyzed_storage_declaration(host_minimum: Option<NonZeroU64>) -> GpuBindingDeclaration {
    GpuBindingDeclaration::from_program_analysis(
        GpuBindingKey::try_new(0, 0).unwrap(),
        GpuShaderStages::one(GpuShaderStage::Compute),
        GpuBindingKind::storage_buffer(GpuStorageBufferAccess::ReadWrite, true, host_minimum),
        None,
        NonZeroU64::new(32),
        "compiler-derived-storage",
        GpuBindingProvenance::new("runtime-binding-test", None).unwrap(),
    )
    .unwrap()
}

fn device_facts() -> GpuRuntimeBindingDeviceFacts {
    device_facts_with_limits(4, 8, 4)
}

fn device_facts_with_limits(
    max_bind_groups: u32,
    max_dynamic_uniform_buffers: u32,
    max_dynamic_storage_buffers: u32,
) -> GpuRuntimeBindingDeviceFacts {
    GpuRuntimeBindingDeviceFacts::new(
        NonZeroU64::new(16),
        NonZeroU64::new(16),
        max_bind_groups,
        max_dynamic_uniform_buffers,
        max_dynamic_storage_buffers,
        [],
    )
}

fn runtime_value(dynamic_offset: u64) -> GpuRuntimeBindingValue {
    runtime_buffer_value(0, storage_buffer(64), 0, 32, dynamic_offset)
}

fn runtime_buffer_value(
    binding: u32,
    buffer: GpuBufferHandle,
    static_offset: u64,
    size: u64,
    dynamic_offset: u64,
) -> GpuRuntimeBindingValue {
    GpuRuntimeBindingValue::new(
        GpuBindingKey::try_new(0, u64::from(binding)).unwrap(),
        [GpuRuntimeBindingResource::Buffer(
            GpuRuntimeBufferBinding::new(
                buffer,
                static_offset,
                NonZeroU64::new(size).unwrap(),
                Some(dynamic_offset),
            ),
        )],
    )
    .unwrap()
}

#[test]
fn whole_buffer_binding_is_exact_explicit_whole_range_without_dynamic_offset() {
    let buffer = storage_buffer(64);
    let whole = GpuRuntimeBufferBinding::whole(&buffer);
    let explicit = GpuRuntimeBufferBinding::new(
        buffer.clone(),
        0,
        NonZeroU64::new(buffer.descriptor().size_bytes()).unwrap(),
        None,
    );

    assert_eq!(whole, explicit);
    assert_eq!(whole.handle(), &buffer);
    assert_eq!(whole.offset(), 0);
    assert_eq!(whole.size().get(), buffer.descriptor().size_bytes());
    assert_eq!(whole.dynamic_offset(), None);
}

#[test]
fn runtime_bindings_validate_logical_usage_range_and_layout_then_admitted_alignment() {
    let layout = GpuBindGroupLayoutDescriptor::new(0, [declaration(None)]).unwrap();
    let validated = GpuValidatedBindGroupBindings::new(layout, [runtime_value(16)])
        .expect("compatible runtime binding should construct logically");
    validated
        .validate_device_facts(&device_facts())
        .expect("compatible runtime binding should satisfy admitted alignment");

    assert_eq!(validated.layout().group(), 0);
    assert!(validated.value(0).is_some());
    assert_eq!(validated.values().len(), 1);
}

#[test]
fn runtime_bindings_enforce_compiler_minimum_and_stronger_host_minimum_without_device_facts() {
    let buffer = storage_buffer(64);
    let compiler_only =
        GpuBindGroupLayoutDescriptor::new(0, [analyzed_storage_declaration(None)]).unwrap();
    let error = GpuValidatedBindGroupBindings::new(
        compiler_only,
        [runtime_buffer_value(0, buffer.clone(), 0, 16, 0)],
    )
    .expect_err("runtime range smaller than compiler-required minimum must reject");
    assert_eq!(
        error.cause(),
        GpuProgramContractCause::RuntimeBindingIncompatible
    );

    let stronger_host = NonZeroU64::new(48).unwrap();
    let host_layout =
        GpuBindGroupLayoutDescriptor::new(0, [analyzed_storage_declaration(Some(stronger_host))])
            .unwrap();
    let error = GpuValidatedBindGroupBindings::new(
        host_layout.clone(),
        [runtime_buffer_value(0, buffer.clone(), 0, 32, 0)],
    )
    .expect_err(
        "runtime range satisfying compiler minimum but not stronger host minimum must reject",
    );
    assert_eq!(
        error.cause(),
        GpuProgramContractCause::RuntimeBindingIncompatible
    );

    GpuValidatedBindGroupBindings::new(host_layout, [runtime_buffer_value(0, buffer, 0, 48, 0)])
        .expect("runtime range satisfying both compiler and host minima should construct");
}

#[test]
fn runtime_bindings_defer_dynamic_offset_alignment_to_explicit_device_validation() {
    let layout = GpuBindGroupLayoutDescriptor::new(0, [declaration(None)]).unwrap();
    let validated = GpuValidatedBindGroupBindings::new(layout, [runtime_value(4)])
        .expect("unaligned dynamic offsets remain valid logical values before device admission");
    let error = validated
        .validate_device_facts(&device_facts())
        .expect_err("unaligned dynamic offsets must reject against admitted device facts");

    assert_eq!(
        error.cause(),
        GpuProgramContractCause::RuntimeBindingIncompatible
    );
}

#[test]
fn runtime_bindings_require_only_the_alignment_fact_the_binding_uses() {
    let empty_layout = GpuPipelineLayoutDescriptor::new([]).unwrap();
    let empty = GpuRuntimeBindingSet::new(empty_layout, [])
        .expect("empty runtime bindings should construct logically");
    let no_alignments = GpuRuntimeBindingDeviceFacts::new(None, None, 4, 8, 4, []);
    empty
        .validate_device_facts(&no_alignments)
        .expect("binding-free work must not require unrelated alignment facts");

    let group = GpuBindGroupLayoutDescriptor::new(0, [declaration(None)]).unwrap();
    let layout = GpuPipelineLayoutDescriptor::new([group]).unwrap();
    let storage = GpuRuntimeBindingSet::new(layout, [runtime_value(16)])
        .expect("storage binding should construct independently of device facts");
    let error = storage
        .validate_device_facts(&no_alignments)
        .expect_err("storage binding must require its admitted storage alignment fact");
    assert_eq!(
        error.cause(),
        GpuProgramContractCause::RuntimeBindingIncompatible
    );
}

#[test]
fn runtime_bindings_reject_missing_values_without_device_facts() {
    let layout = GpuBindGroupLayoutDescriptor::new(0, [declaration(None)]).unwrap();
    let error = GpuValidatedBindGroupBindings::new(layout, [])
        .expect_err("every declaration needs one runtime value");

    assert_eq!(
        error.cause(),
        GpuProgramContractCause::RuntimeBindingIncompatible
    );
}

#[test]
fn runtime_bindings_reject_wrong_fixed_array_cardinality_before_backend_realization() {
    let layout =
        GpuBindGroupLayoutDescriptor::new(0, [declaration(Some(NonZeroU32::new(2).unwrap()))])
            .unwrap();
    let error = GpuValidatedBindGroupBindings::new(layout, [runtime_value(16)])
        .expect_err("fixed binding arrays require exact cardinality");

    assert_eq!(
        error.cause(),
        GpuProgramContractCause::RuntimeBindingIncompatible
    );
}

#[test]
fn runtime_binding_set_is_complete_pipeline_shaped_logical_use() {
    let group = GpuBindGroupLayoutDescriptor::new(0, [declaration(None)]).unwrap();
    let layout = GpuPipelineLayoutDescriptor::new([group]).unwrap();
    let bindings = GpuRuntimeBindingSet::new(layout.clone(), [runtime_value(16)])
        .expect("one complete dynamic storage binding should construct logically");
    bindings
        .validate_device_facts(&device_facts())
        .expect("logical binding set should satisfy admitted facts");

    assert_eq!(bindings.layout(), &layout);
    assert_eq!(bindings.groups().len(), 1);
    assert!(bindings.group(0).is_some());
    assert_eq!(bindings.values().count(), 1);
    assert_eq!(bindings.accesses().len(), 1);
    let GpuResourceAccess::Buffer(access) = &bindings.accesses()[0] else {
        panic!("storage runtime binding must derive one buffer access");
    };
    assert_eq!(access.kind(), GpuBufferAccessKind::StorageReadWrite);
    assert_eq!(access.range().offset(), 16);
    assert_eq!(access.range().size(), 32);
}

#[test]
fn runtime_binding_set_defers_sparse_group_limit_to_explicit_device_validation() {
    let sparse = GpuBindGroupLayoutDescriptor::new(1, []).unwrap();
    let layout = GpuPipelineLayoutDescriptor::new([sparse]).unwrap();
    let bindings = GpuRuntimeBindingSet::new(layout, [])
        .expect("sparse group identity is valid independent of an admitted device");
    let error = bindings
        .validate_device_facts(&device_facts_with_limits(1, 8, 4))
        .expect_err(
            "group one requires two positional slots and must reject against a limit of one",
        );

    assert_eq!(
        error.cause(),
        GpuProgramContractCause::RuntimeBindingIncompatible
    );
    assert_eq!(error.label(), "bind groups");
}

#[test]
fn runtime_binding_set_defers_dynamic_storage_limit_to_explicit_device_validation() {
    let group = GpuBindGroupLayoutDescriptor::new(0, [declaration(None)]).unwrap();
    let layout = GpuPipelineLayoutDescriptor::new([group]).unwrap();
    let bindings = GpuRuntimeBindingSet::new(layout, [runtime_value(16)])
        .expect("dynamic storage declaration is logical before device admission");
    let error = bindings
        .validate_device_facts(&device_facts_with_limits(4, 8, 0))
        .expect_err("zero admitted dynamic-storage capacity must reject the declaration");

    assert_eq!(
        error.cause(),
        GpuProgramContractCause::RuntimeBindingIncompatible
    );
    assert_eq!(error.label(), "dynamic storage buffers");
}

#[test]
fn runtime_binding_set_rejects_overlapping_writable_ranges_without_device_facts() {
    let buffer = storage_buffer(96);
    let group = GpuBindGroupLayoutDescriptor::new(
        0,
        [storage_declaration(0, None), storage_declaration(1, None)],
    )
    .unwrap();
    let layout = GpuPipelineLayoutDescriptor::new([group]).unwrap();
    let error = GpuRuntimeBindingSet::new(
        layout,
        [
            runtime_buffer_value(0, buffer.clone(), 0, 32, 0),
            runtime_buffer_value(1, buffer, 16, 32, 0),
        ],
    )
    .expect_err("one binding use cannot contain overlapping writable effective ranges");

    assert_eq!(
        error.cause(),
        GpuProgramContractCause::RuntimeBindingIncompatible
    );
    assert!(error.label().contains("[0..32)"));
    assert!(error.label().contains("[16..48)"));
}

#[test]
fn runtime_binding_set_accepts_disjoint_writable_ranges_without_device_facts() {
    let buffer = storage_buffer(96);
    let group = GpuBindGroupLayoutDescriptor::new(
        0,
        [storage_declaration(0, None), storage_declaration(1, None)],
    )
    .unwrap();
    let layout = GpuPipelineLayoutDescriptor::new([group]).unwrap();
    let bindings = GpuRuntimeBindingSet::new(
        layout,
        [
            runtime_buffer_value(0, buffer.clone(), 0, 32, 0),
            runtime_buffer_value(1, buffer, 32, 32, 0),
        ],
    )
    .expect("disjoint writable binding ranges are valid");

    assert_eq!(bindings.accesses().len(), 2);
}
