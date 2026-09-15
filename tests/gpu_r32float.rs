use runen_gpu::GpuTextureFormat;

#[test]
fn r32float_is_public_backend_neutral_and_four_bytes_per_texel() {
    let format = GpuTextureFormat::R32Float;

    assert_eq!(format.bytes_per_texel(), 4);
    assert!(!format.is_depth());
    assert!(!format.is_srgb());
}
