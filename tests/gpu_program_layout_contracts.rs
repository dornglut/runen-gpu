use runen_gpu::{
    GpuBindGroupLayoutDescriptor, GpuBindingDeclaration, GpuBindingKey, GpuBindingKind,
    GpuBindingProvenance, GpuPipelineLayoutDescriptor, GpuProgramContractCause,
    GpuProgramInterfaceDescriptor, GpuShaderStage, GpuShaderStages, GpuStorageBufferAccess,
};

fn binding(group: u64, binding: u64, label: &str) -> GpuBindingDeclaration {
    GpuBindingDeclaration::new(
        GpuBindingKey::try_new(group, binding).expect("test binding key should fit u32"),
        GpuShaderStages::one(GpuShaderStage::Compute),
        GpuBindingKind::storage_buffer(GpuStorageBufferAccess::ReadWrite, false, None),
        None,
        label,
        GpuBindingProvenance::new("gpu-program-layout-test", None)
            .expect("test provenance should be valid"),
    )
    .expect("test binding declaration should be valid")
}

#[test]
fn pipeline_layout_derives_sorted_groups_and_bindings_from_interface() {
    let interface = GpuProgramInterfaceDescriptor::new([
        binding(2, 1, "group-two-binding-one"),
        binding(0, 0, "group-zero-binding-zero"),
        binding(2, 0, "group-two-binding-zero"),
    ])
    .expect("test interface should be valid");

    let layout = GpuPipelineLayoutDescriptor::from_interface(&interface)
        .expect("pipeline layout should derive from the interface");

    assert_eq!(
        layout
            .groups()
            .map(|group| group.group())
            .collect::<Vec<_>>(),
        [0, 2]
    );
    assert_eq!(
        layout
            .group(2)
            .expect("group two should exist")
            .bindings()
            .map(|declaration| declaration.key().binding())
            .collect::<Vec<_>>(),
        [0, 1]
    );
    assert!(
        layout
            .group(2)
            .expect("group two should exist")
            .binding(1)
            .is_some()
    );
}

#[test]
fn bind_group_layout_rejects_a_declaration_from_another_group() {
    let error = GpuBindGroupLayoutDescriptor::new(0, [binding(1, 0, "wrong-group")])
        .expect_err("a bind-group layout must own exactly one group");

    assert_eq!(
        error.cause(),
        GpuProgramContractCause::BindGroupLayoutInvalid
    );
}

#[test]
fn pipeline_layout_rejects_duplicate_group_descriptors() {
    let group = GpuBindGroupLayoutDescriptor::new(0, [binding(0, 0, "storage")])
        .expect("test group layout should be valid");

    let error = GpuPipelineLayoutDescriptor::new([group.clone(), group])
        .expect_err("each group must appear exactly once");

    assert_eq!(
        error.cause(),
        GpuProgramContractCause::DuplicateBindGroupLayout
    );
}
