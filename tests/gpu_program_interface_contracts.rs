use runen_gpu::{
    GpuBindingDeclaration, GpuBindingKey, GpuBindingKind, GpuBindingProvenance, GpuEntryPointName,
    GpuProgramContractCause, GpuProgramInterfaceDescriptor, GpuShaderStage, GpuShaderStages,
    GpuStorageBufferAccess,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroU64;

fn provenance(detail: &str) -> GpuBindingProvenance {
    GpuBindingProvenance::new("gpu-program-interface-test", Some(detail.to_owned()))
        .expect("test provenance should be valid")
}

fn storage_binding(group: u64, binding: u64, label: &str, detail: &str) -> GpuBindingDeclaration {
    GpuBindingDeclaration::new(
        GpuBindingKey::try_new(group, binding).expect("test binding key should fit u32"),
        GpuShaderStages::one(GpuShaderStage::Compute),
        GpuBindingKind::storage_buffer(
            GpuStorageBufferAccess::ReadWrite,
            false,
            NonZeroU64::new(16),
        ),
        None,
        label,
        provenance(detail),
    )
    .expect("test binding declaration should be valid")
}

fn semantic_hash(value: &impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn resource_interface_normalizes_binding_order_and_excludes_diagnostics_from_identity() {
    let left = GpuProgramInterfaceDescriptor::new([
        storage_binding(0, 1, "output", "left output"),
        storage_binding(0, 0, "input", "left input"),
    ])
    .expect("interface should normalize insertion order");
    let right = GpuProgramInterfaceDescriptor::new([
        storage_binding(0, 0, "renamed input", "right input"),
        storage_binding(0, 1, "renamed output", "right output"),
    ])
    .expect("equivalent interface should construct");

    assert_eq!(left, right);
    assert_eq!(semantic_hash(&left), semantic_hash(&right));
    assert_eq!(
        left.bindings()
            .map(|binding| binding.key())
            .collect::<Vec<_>>(),
        [
            GpuBindingKey::try_new(0, 0).unwrap(),
            GpuBindingKey::try_new(0, 1).unwrap(),
        ]
    );
}

#[test]
fn duplicate_binding_keys_are_rejected_before_interface_publication() {
    let error = GpuProgramInterfaceDescriptor::new([
        storage_binding(0, 0, "input", "first"),
        storage_binding(0, 0, "output", "duplicate"),
    ])
    .expect_err("duplicate typed binding keys must be rejected");

    assert_eq!(error.cause(), GpuProgramContractCause::DuplicateBindingKey);
}

#[test]
fn structural_invalidity_is_rejected_by_typed_constructors() {
    assert_eq!(
        GpuBindingKey::try_new(u64::from(u32::MAX) + 1, 0)
            .unwrap_err()
            .cause(),
        GpuProgramContractCause::InvalidBindingKey
    );
    assert_eq!(
        GpuShaderStages::new([]).unwrap_err().cause(),
        GpuProgramContractCause::EmptyStageVisibility
    );
    assert_eq!(
        GpuEntryPointName::new("not valid").unwrap_err().cause(),
        GpuProgramContractCause::InvalidEntryPointName
    );
}
