use crate::GpuTextureFormat;
use wgpu::{AstcBlock, AstcChannel, TextureFormat};

macro_rules! native_texture_format {
    ($variant:ident) => { TextureFormat::$variant };
    ($variant:ident => $native:expr) => { $native };
}

macro_rules! define_texture_format_mapping {
    ($($variant:ident $(=> $native:expr)?),+ $(,)?) => {
        pub(super) const TEXTURE_FORMATS: &[(GpuTextureFormat, TextureFormat)] = &[
            $((GpuTextureFormat::$variant, native_texture_format!($variant $(=> $native)?)),)+
        ];
        pub(super) const fn texture_format(format: GpuTextureFormat) -> TextureFormat {
            match format {
                $(GpuTextureFormat::$variant => native_texture_format!($variant $(=> $native)?),)+
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
    Etc2Rgb8Unorm,
    Etc2Rgb8UnormSrgb,
    Etc2Rgb8A1Unorm,
    Etc2Rgb8A1UnormSrgb,
    Etc2Rgba8Unorm,
    Etc2Rgba8UnormSrgb,
    EacR11Unorm,
    EacR11Snorm,
    EacRg11Unorm,
    EacRg11Snorm,
    Astc4x4Unorm => TextureFormat::Astc { block: AstcBlock::B4x4, channel: AstcChannel::Unorm },
    Astc4x4UnormSrgb => TextureFormat::Astc { block: AstcBlock::B4x4, channel: AstcChannel::UnormSrgb },
    Astc5x4Unorm => TextureFormat::Astc { block: AstcBlock::B5x4, channel: AstcChannel::Unorm },
    Astc5x4UnormSrgb => TextureFormat::Astc { block: AstcBlock::B5x4, channel: AstcChannel::UnormSrgb },
    Astc5x5Unorm => TextureFormat::Astc { block: AstcBlock::B5x5, channel: AstcChannel::Unorm },
    Astc5x5UnormSrgb => TextureFormat::Astc { block: AstcBlock::B5x5, channel: AstcChannel::UnormSrgb },
    Astc6x5Unorm => TextureFormat::Astc { block: AstcBlock::B6x5, channel: AstcChannel::Unorm },
    Astc6x5UnormSrgb => TextureFormat::Astc { block: AstcBlock::B6x5, channel: AstcChannel::UnormSrgb },
    Astc6x6Unorm => TextureFormat::Astc { block: AstcBlock::B6x6, channel: AstcChannel::Unorm },
    Astc6x6UnormSrgb => TextureFormat::Astc { block: AstcBlock::B6x6, channel: AstcChannel::UnormSrgb },
    Astc8x5Unorm => TextureFormat::Astc { block: AstcBlock::B8x5, channel: AstcChannel::Unorm },
    Astc8x5UnormSrgb => TextureFormat::Astc { block: AstcBlock::B8x5, channel: AstcChannel::UnormSrgb },
    Astc8x6Unorm => TextureFormat::Astc { block: AstcBlock::B8x6, channel: AstcChannel::Unorm },
    Astc8x6UnormSrgb => TextureFormat::Astc { block: AstcBlock::B8x6, channel: AstcChannel::UnormSrgb },
    Astc8x8Unorm => TextureFormat::Astc { block: AstcBlock::B8x8, channel: AstcChannel::Unorm },
    Astc8x8UnormSrgb => TextureFormat::Astc { block: AstcBlock::B8x8, channel: AstcChannel::UnormSrgb },
    Astc10x5Unorm => TextureFormat::Astc { block: AstcBlock::B10x5, channel: AstcChannel::Unorm },
    Astc10x5UnormSrgb => TextureFormat::Astc { block: AstcBlock::B10x5, channel: AstcChannel::UnormSrgb },
    Astc10x6Unorm => TextureFormat::Astc { block: AstcBlock::B10x6, channel: AstcChannel::Unorm },
    Astc10x6UnormSrgb => TextureFormat::Astc { block: AstcBlock::B10x6, channel: AstcChannel::UnormSrgb },
    Astc10x8Unorm => TextureFormat::Astc { block: AstcBlock::B10x8, channel: AstcChannel::Unorm },
    Astc10x8UnormSrgb => TextureFormat::Astc { block: AstcBlock::B10x8, channel: AstcChannel::UnormSrgb },
    Astc10x10Unorm => TextureFormat::Astc { block: AstcBlock::B10x10, channel: AstcChannel::Unorm },
    Astc10x10UnormSrgb => TextureFormat::Astc { block: AstcBlock::B10x10, channel: AstcChannel::UnormSrgb },
    Astc12x10Unorm => TextureFormat::Astc { block: AstcBlock::B12x10, channel: AstcChannel::Unorm },
    Astc12x10UnormSrgb => TextureFormat::Astc { block: AstcBlock::B12x10, channel: AstcChannel::UnormSrgb },
    Astc12x12Unorm => TextureFormat::Astc { block: AstcBlock::B12x12, channel: AstcChannel::Unorm },
    Astc12x12UnormSrgb => TextureFormat::Astc { block: AstcBlock::B12x12, channel: AstcChannel::UnormSrgb },
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
