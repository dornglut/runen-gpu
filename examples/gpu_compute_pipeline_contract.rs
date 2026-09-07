use runen_gpu as gpu;
use std::num::NonZeroU64;

const WGSL: &str = r#"
@group(0) @binding(0)
var<storage, read> input_values: array<u32>;

@group(0) @binding(1)
var<storage, read_write> output_values: array<u32>;

@compute @workgroup_size(64)
fn compute_main(@builtin(global_invocation_id) id: vec3<u32>) {
    output_values[id.x] = input_values[id.x];
}
"#;

fn main() {
    let owner = gpu::GpuProgramSourceOwnerId::allocate().expect("source owner should allocate");
    let identity = gpu::GpuProgramSourceIdentity::new(
        owner,
        gpu::GpuProgramSourceKey::new("examples.compute.copy").expect("source key should be valid"),
        gpu::GpuProgramSourceRevision::try_from_raw(1).expect("source revision should be nonzero"),
    );
    let mut registry =
        gpu::GpuProgramSourceRegistry::new(4, 16 * 1024).expect("registry should construct");
    let source = registry
        .admit_wgsl(identity.clone(), WGSL, provenance("initial admission"))
        .expect("source should admit");
    let repeated = registry
        .admit_wgsl(identity, WGSL, provenance("idempotent rediscovery"))
        .expect("identical source should remain idempotent");
    assert!(source.is_same_record(&repeated));

    let host_minimum = NonZeroU64::new(4).expect("host minimum should be nonzero");
    let program = gpu::GpuProgramDescriptor::new(
        source,
        [entry_point("compute_main")],
        [
            gpu::GpuBindingLayoutRefinement::new(binding_key(0))
                .with_host_minimum_size(host_minimum),
            gpu::GpuBindingLayoutRefinement::new(binding_key(1))
                .with_host_minimum_size(host_minimum),
        ],
    )
    .expect("program should derive from canonical WGSL");
    let pipeline = gpu::GpuComputePipelineDescriptor::new(
        program,
        entry_point("compute_main"),
        gpu::GpuPipelineConfiguration::default(),
    )
    .expect("compute pipeline should construct");

    assert!(matches!(
        pipeline
            .requirements()
            .get(gpu::GpuCapabilityFeature::Compute),
        Some(gpu::GpuCapabilityRequirement::Required(
            gpu::GpuCapabilityFeature::Compute
        ))
    ));
}

fn binding_key(binding: u64) -> gpu::GpuBindingKey {
    gpu::GpuBindingKey::try_new(0, binding).expect("binding key should fit u32")
}

fn provenance(detail: &str) -> gpu::GpuProgramSourceProvenance {
    gpu::GpuProgramSourceProvenance::new("gpu-compute-contract-example", Some(detail.to_owned()))
        .expect("source provenance should be valid")
}

fn entry_point(name: &str) -> gpu::GpuEntryPointName {
    gpu::GpuEntryPointName::new(name).expect("entry-point name should be valid")
}
