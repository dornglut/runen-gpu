use runen_gpu as gpu;
use std::num::NonZeroU64;

fn main() {}

/// Compiles the public runtime-binding path for a buffer handle issued by the
/// owning GPU-work authoring layer. Downstream code cannot fabricate handles or
/// their process-local resource identities.
#[allow(dead_code)]
fn validate_uniform_binding(buffer: gpu::GpuBufferHandle) {
    let declaration = gpu::GpuBindingDeclaration::new(
        gpu::GpuBindingKey::try_new(0, 0).expect("binding key should fit u32"),
        gpu::GpuShaderStages::one(gpu::GpuShaderStage::Vertex),
        gpu::GpuBindingKind::uniform_buffer(false, NonZeroU64::new(64)),
        None,
        "view-uniform",
        gpu::GpuBindingProvenance::new("gpu-runtime-binding-example", None)
            .expect("binding provenance should be valid"),
    )
    .expect("uniform declaration should construct");
    let layout = gpu::GpuBindGroupLayoutDescriptor::new(0, [declaration])
        .expect("bind-group layout should construct");
    let value = gpu::GpuRuntimeBindingValue::new(
        gpu::GpuBindingKey::try_new(0, 0).expect("binding key should fit u32"),
        [gpu::GpuRuntimeBindingResource::Buffer(
            gpu::GpuRuntimeBufferBinding::new(
                buffer,
                0,
                NonZeroU64::new(64).expect("uniform range should be nonzero"),
                None,
            ),
        )],
    )
    .expect("runtime binding value should construct");
    let validated = gpu::GpuValidatedBindGroupBindings::new(layout, [value])
        .expect("logical runtime binding structure should satisfy the interface");

    assert_eq!(validated.values().len(), 1);
}
