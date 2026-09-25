#[path = "gpu_fixed_binding_array/mod.rs"]
mod fixed_binding_array;

#[test]
#[ignore = "requires retained Vulkan/Lavapipe execution for the capability-gated fixed-array suite"]
fn fixed_binding_array_native_execution_is_capability_gated_and_backend_proven() {
    let proof = pollster::block_on(fixed_binding_array::run_suite(
        runen_gpu::GpuBackendFamily::Vulkan,
        Some(runen_gpu::GpuSoftwareFallbackPolicy::Require),
    ));
    println!("Fixed binding-array native proof: {proof:?}");
}
