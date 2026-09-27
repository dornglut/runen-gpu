use runen_gpu::{GpuBackendFamily, GpuSoftwareFallbackPolicy};

#[path = "gpu_fixed_binding_array/mod.rs"]
mod retained_fixed_binding_array;
#[path = "gpu_r3_binding_array_non_uniform_indexing/mod.rs"]
mod proof;

#[test]
#[ignore = "requires the retained Vulkan software adapter"]
fn non_uniform_binding_array_native_execution_follows_normalized_adapter_facts() {
    let outcome = pollster::block_on(proof::run_suite(
        GpuBackendFamily::Vulkan,
        Some(GpuSoftwareFallbackPolicy::Require),
    ));
    println!(
        "R3 non-uniform Vulkan qualification: texture_sampler={}, storage_buffer={}, storage_texture={}",
        outcome.texture_sampler, outcome.storage_buffer, outcome.storage_texture
    );
}
