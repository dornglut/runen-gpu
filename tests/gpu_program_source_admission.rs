use runen_gpu::{
    GpuProgramSourceCause, GpuProgramSourceIdentity, GpuProgramSourceKey, GpuProgramSourceOwnerId,
    GpuProgramSourceProvenance, GpuProgramSourceRegistry, GpuProgramSourceRevision,
};

const COMPUTE_WGSL: &str = r#"
@group(0) @binding(0)
var<storage, read> input_values: array<u32>;

@group(0) @binding(1)
var<storage, read_write> output_values: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output_values[id.x] = input_values[id.x];
}
"#;

fn source_identity(
    owner: GpuProgramSourceOwnerId,
    key: &str,
    revision: u64,
) -> GpuProgramSourceIdentity {
    GpuProgramSourceIdentity::new(
        owner,
        GpuProgramSourceKey::new(key).expect("test source key should be valid"),
        GpuProgramSourceRevision::try_from_raw(revision)
            .expect("test source revision should be nonzero"),
    )
}

fn provenance(detail: &str) -> GpuProgramSourceProvenance {
    GpuProgramSourceProvenance::new("gpu-program-source-admission-test", Some(detail.to_owned()))
        .expect("test provenance should be valid")
}

#[test]
fn public_source_admission_is_idempotent_and_conflict_safe() {
    let owner = GpuProgramSourceOwnerId::allocate().expect("source owner should allocate");
    let identity = source_identity(owner, "compute.copy-values", 1);
    let mut registry = GpuProgramSourceRegistry::new(4, 16 * 1024)
        .expect("bounded source registry should construct");

    let first = registry
        .admit_wgsl(
            identity.clone(),
            COMPUTE_WGSL,
            provenance("initial discovery"),
        )
        .expect("first source admission should succeed");
    let repeated = registry
        .admit_wgsl(identity.clone(), COMPUTE_WGSL, provenance("rediscovery"))
        .expect("identical source admission should be idempotent");

    assert!(first.is_same_record(&repeated));
    assert_eq!(first.canonical_wgsl(), COMPUTE_WGSL);
    assert_eq!(first.identity(), &identity);
    assert_eq!(first.provenance().detail(), Some("initial discovery"));

    let before_conflict = registry.stats();
    let conflict = registry
        .admit_wgsl(
            identity.clone(),
            COMPUTE_WGSL.replace("workgroup_size(64)", "workgroup_size(32)"),
            provenance("invalid revision reuse"),
        )
        .expect_err("different canonical WGSL must require a new revision");

    assert_eq!(
        conflict.cause(),
        GpuProgramSourceCause::SourceRevisionConflict
    );
    assert_eq!(registry.stats(), before_conflict);
    assert!(
        registry
            .get(&identity)
            .expect("accepted source must remain published")
            .is_same_record(&first)
    );
}

#[test]
fn public_source_registry_never_evicts_a_retained_record_for_capacity() {
    let owner = GpuProgramSourceOwnerId::allocate().expect("source owner should allocate");
    let mut registry = GpuProgramSourceRegistry::new(1, COMPUTE_WGSL.len())
        .expect("bounded source registry should construct");
    let retained = registry
        .admit_wgsl(
            source_identity(owner, "compute.first", 1),
            COMPUTE_WGSL,
            provenance("retained source"),
        )
        .expect("first source admission should succeed");

    let error = registry
        .admit_wgsl(
            source_identity(owner, "compute.second", 1),
            COMPUTE_WGSL,
            provenance("capacity pressure"),
        )
        .expect_err("a retained source must not be evicted");

    assert_eq!(
        error.cause(),
        GpuProgramSourceCause::SourceAdmissionCapacityExceeded
    );
    assert_eq!(registry.stats().retained_records(), 1);

    drop(retained);
    assert_eq!(registry.collect_unretained(), 1);
    assert_eq!(registry.stats().retained_records(), 0);
}
