#[path = "gpu_fixed_binding_array/mod.rs"]
mod fixed_binding_array;

#[test]
#[ignore = "requires retained Vulkan/Lavapipe execution when fixed storage-buffer arrays are supported"]
fn fixed_binding_array_native_execution_is_capability_gated_and_backend_proven() {
    let _ = pollster::block_on(fixed_binding_array::run_storage_buffer_array_proof(
        runen_gpu::GpuBackendFamily::Vulkan,
        Some(runen_gpu::GpuSoftwareFallbackPolicy::Require),
    ));
}
