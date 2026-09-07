use super::GpuTextureFormat;

/// Backend-neutral storage-format compatibility for texture-to-texture copies.
///
/// Linear and sRGB variants of the same normalized storage format are copy-compatible because the
/// copy preserves raw texel storage rather than performing color-space conversion. Unrelated color
/// formats and every depth/color pairing remain incompatible.
pub fn gpu_texture_formats_copy_compatible(
    source: GpuTextureFormat,
    destination: GpuTextureFormat,
) -> bool {
    source == destination
        || matches!(
            (source, destination),
            (
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureFormat::Rgba8UnormSrgb
            ) | (
                GpuTextureFormat::Rgba8UnormSrgb,
                GpuTextureFormat::Rgba8Unorm
            ) | (
                GpuTextureFormat::Bgra8Unorm,
                GpuTextureFormat::Bgra8UnormSrgb
            ) | (
                GpuTextureFormat::Bgra8UnormSrgb,
                GpuTextureFormat::Bgra8Unorm
            )
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_compatibility_accepts_only_exact_or_srgb_storage_pairs() {
        assert!(gpu_texture_formats_copy_compatible(
            GpuTextureFormat::Rgba8Unorm,
            GpuTextureFormat::Rgba8UnormSrgb
        ));
        assert!(gpu_texture_formats_copy_compatible(
            GpuTextureFormat::Bgra8UnormSrgb,
            GpuTextureFormat::Bgra8Unorm
        ));
        assert!(gpu_texture_formats_copy_compatible(
            GpuTextureFormat::R32Uint,
            GpuTextureFormat::R32Uint
        ));
        assert!(!gpu_texture_formats_copy_compatible(
            GpuTextureFormat::Rgba8Unorm,
            GpuTextureFormat::Bgra8Unorm
        ));
        assert!(!gpu_texture_formats_copy_compatible(
            GpuTextureFormat::Depth32Float,
            GpuTextureFormat::Rgba8Unorm
        ));
    }
}
