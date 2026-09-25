use crate::GpuTextureFormat;
use wgpu::TextureFormat;

macro_rules! define_texture_format_mapping {
    ($($variant:ident),+ $(,)?) => {
        pub(super) const TEXTURE_FORMATS: &[(GpuTextureFormat, TextureFormat)] = &[
            $((GpuTextureFormat::$variant, TextureFormat::$variant),)+
        ];

        pub(super) const fn texture_format(format: GpuTextureFormat) -> TextureFormat {
            match format {
                $(GpuTextureFormat::$variant => TextureFormat::$variant,)+
            }
        }
    };
}

define_texture_format_mapping!(
    R8Unorm,
    R8Snorm,
    R8Uint,
    R8Sint,
    Rg8Unorm,
    Rg8Snorm,
    Rg8Uint,
    Rg8Sint,
    R16Uint,
    R16Sint,
    R16Float,
    Rg16Uint,
    Rg16Sint,
    Rg16Float,
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Rgba8Snorm,
    Rgba8Uint,
    Rgba8Sint,
    Bgra8Unorm,
    Bgra8UnormSrgb,
    Rgb9e5Ufloat,
    Rgb10a2Uint,
    Rgb10a2Unorm,
    Rg11b10Ufloat,
    R32Uint,
    R32Sint,
    R32Float,
    Rg32Uint,
    Rg32Sint,
    Rg32Float,
    Rgba32Uint,
    Rgba32Sint,
    Rgba32Float,
    Rgba16Uint,
    Rgba16Sint,
    Rgba16Float,
    Bc1RgbaUnorm,
    Bc1RgbaUnormSrgb,
    Bc2RgbaUnorm,
    Bc2RgbaUnormSrgb,
    Bc3RgbaUnorm,
    Bc3RgbaUnormSrgb,
    Bc4RUnorm,
    Bc4RSnorm,
    Bc5RgUnorm,
    Bc5RgSnorm,
    Bc6hRgbUfloat,
    Bc6hRgbFloat,
    Bc7RgbaUnorm,
    Bc7RgbaUnormSrgb,
    Stencil8,
    Depth16Unorm,
    Depth24Plus,
    Depth24PlusStencil8,
    Depth32Float,
    Depth32FloatStencil8,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_census_and_mapping_share_one_exhaustive_variant_authority() {
        for &(normalized, native) in TEXTURE_FORMATS {
            assert_eq!(texture_format(normalized), native);
        }
    }
}
