use super::super::super::contract_diagnostics::{GpuProgramContractCause, GpuProgramContractError};
use super::super::super::{
    GpuEntryPointName, GpuExpectedFragmentOutputSignature, GpuShaderIoLocation,
    GpuShaderIoScalarClass, GpuShaderIoValueType,
};
use crate::api::texture_format::{self, GpuTextureScalarClass};
use crate::{GpuCompareFunction, GpuTextureFormat};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuBlendMode {
    Replace,
    Alpha,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuColorWriteMask(u8);

impl GpuColorWriteMask {
    const RED_BIT: u8 = 1 << 0;
    const GREEN_BIT: u8 = 1 << 1;
    const BLUE_BIT: u8 = 1 << 2;
    const ALPHA_BIT: u8 = 1 << 3;
    const VALID_BITS: u8 = Self::RED_BIT | Self::GREEN_BIT | Self::BLUE_BIT | Self::ALPHA_BIT;

    pub const NONE: Self = Self(0);
    pub const RED: Self = Self(Self::RED_BIT);
    pub const GREEN: Self = Self(Self::GREEN_BIT);
    pub const BLUE: Self = Self(Self::BLUE_BIT);
    pub const ALPHA: Self = Self(Self::ALPHA_BIT);
    pub const ALL: Self = Self(Self::VALID_BITS);

    pub fn from_bits(bits: u8) -> Result<Self, GpuProgramContractError> {
        if bits & !Self::VALID_BITS != 0 {
            return Err(invalid_attachment_state(
                format!("color_write_mask=0x{bits:02x}"),
                "use only the normalized red, green, blue, and alpha write bits",
            ));
        }
        Ok(Self(bits))
    }

    pub const fn bits(self) -> u8 {
        self.0
    }

    pub const fn contains(self, component: Self) -> bool {
        self.0 & component.0 == component.0
    }
}

/// Typed color-target state independent of raw backend enums.
///
/// Raw WGPU formats cannot enter the generic descriptor:
///
/// ```compile_fail
/// use runen_gpu::{
///     GpuBlendMode, GpuColorTargetStateDescriptor, GpuColorWriteMask,
/// };
///
/// let _target = GpuColorTargetStateDescriptor::new(
///     wgpu::TextureFormat::Rgba8Unorm,
///     GpuBlendMode::Replace,
///     GpuColorWriteMask::ALL,
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuColorTargetStateDescriptor {
    format: GpuTextureFormat,
    blend: GpuBlendMode,
    write_mask: GpuColorWriteMask,
}

impl GpuColorTargetStateDescriptor {
    pub fn new(
        format: GpuTextureFormat,
        blend: GpuBlendMode,
        write_mask: GpuColorWriteMask,
    ) -> Result<Self, GpuProgramContractError> {
        if format.is_depth() || format.is_stencil() {
            return Err(invalid_attachment_state(
                format!("color_format={format:?}"),
                "use a color-attachment format for a color target",
            ));
        }
        if blend == GpuBlendMode::Alpha
            && matches!(
                texture_format::color_scalar_class(format),
                Some(GpuTextureScalarClass::Sint | GpuTextureScalarClass::Uint)
            )
        {
            return Err(invalid_attachment_state(
                format!("color_format={format:?}, blend={blend:?}"),
                "use replacement blending for integer color targets",
            ));
        }
        Ok(Self {
            format,
            blend,
            write_mask,
        })
    }

    pub const fn format(self) -> GpuTextureFormat {
        self.format
    }

    pub const fn blend(self) -> GpuBlendMode {
        self.blend
    }

    pub const fn write_mask(self) -> GpuColorWriteMask {
        self.write_mask
    }

    pub const fn has_blendable_alpha_channel(self) -> bool {
        texture_format::has_alpha(self.format)
    }

    pub fn shader_io_type(self) -> GpuShaderIoValueType {
        let class = match texture_format::color_scalar_class(self.format)
            .expect("validated color targets retain a color scalar class")
        {
            GpuTextureScalarClass::Float => GpuShaderIoScalarClass::Float,
            GpuTextureScalarClass::Sint => GpuShaderIoScalarClass::Sint,
            GpuTextureScalarClass::Uint => GpuShaderIoScalarClass::Uint,
        };
        let width = texture_format::component_count(self.format);
        debug_assert!(!self.format.is_depth() && !self.format.is_stencil());
        GpuShaderIoValueType::try_new(class, width)
            .expect("normalized color targets always map to valid shader IO")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuFragmentOutputStateDescriptor {
    color_targets: Vec<GpuColorTargetStateDescriptor>,
}

impl GpuFragmentOutputStateDescriptor {
    pub fn new(color_targets: impl IntoIterator<Item = GpuColorTargetStateDescriptor>) -> Self {
        Self {
            color_targets: color_targets.into_iter().collect(),
        }
    }

    pub fn color_targets(
        &self,
    ) -> impl ExactSizeIterator<Item = GpuColorTargetStateDescriptor> + '_ {
        self.color_targets.iter().copied()
    }

    pub fn expected_signature(
        &self,
        entry_point: GpuEntryPointName,
    ) -> Result<GpuExpectedFragmentOutputSignature, GpuProgramContractError> {
        let locations = self
            .color_targets
            .iter()
            .copied()
            .enumerate()
            .map(|(index, target)| {
                u32::try_from(index)
                    .map(|location| GpuShaderIoLocation::new(location, target.shader_io_type()))
                    .map_err(|_| {
                        invalid_attachment_state(
                            format!("color_target_count={}", self.color_targets.len()),
                            "reduce the number of color targets to a u32-representable count",
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        GpuExpectedFragmentOutputSignature::new(entry_point, locations)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuDepthStateDescriptor {
    write_enabled: bool,
    compare: GpuCompareFunction,
}

impl GpuDepthStateDescriptor {
    pub const fn new(write_enabled: bool, compare: GpuCompareFunction) -> Self {
        Self {
            write_enabled,
            compare,
        }
    }

    pub const fn write_enabled(self) -> bool {
        self.write_enabled
    }

    pub const fn compare(self) -> GpuCompareFunction {
        self.compare
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuStencilOperation {
    Keep,
    Zero,
    Replace,
    Invert,
    IncrementClamp,
    DecrementClamp,
    IncrementWrap,
    DecrementWrap,
}

impl GpuStencilOperation {
    pub const fn writes(self) -> bool {
        !matches!(self, Self::Keep)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuStencilFaceStateDescriptor {
    compare: GpuCompareFunction,
    fail_op: GpuStencilOperation,
    depth_fail_op: GpuStencilOperation,
    pass_op: GpuStencilOperation,
}

impl GpuStencilFaceStateDescriptor {
    pub const fn new(
        compare: GpuCompareFunction,
        fail_op: GpuStencilOperation,
        depth_fail_op: GpuStencilOperation,
        pass_op: GpuStencilOperation,
    ) -> Self {
        Self {
            compare,
            fail_op,
            depth_fail_op,
            pass_op,
        }
    }

    pub const fn compare(self) -> GpuCompareFunction {
        self.compare
    }

    pub const fn fail_op(self) -> GpuStencilOperation {
        self.fail_op
    }

    pub const fn depth_fail_op(self) -> GpuStencilOperation {
        self.depth_fail_op
    }

    pub const fn pass_op(self) -> GpuStencilOperation {
        self.pass_op
    }

    pub const fn may_write(self) -> bool {
        self.fail_op.writes() || self.depth_fail_op.writes() || self.pass_op.writes()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuStencilStateDescriptor {
    front: GpuStencilFaceStateDescriptor,
    back: GpuStencilFaceStateDescriptor,
    read_mask: u32,
    write_mask: u32,
}

impl GpuStencilStateDescriptor {
    pub const fn new(
        front: GpuStencilFaceStateDescriptor,
        back: GpuStencilFaceStateDescriptor,
        read_mask: u32,
        write_mask: u32,
    ) -> Self {
        Self {
            front,
            back,
            read_mask,
            write_mask,
        }
    }

    pub const fn front(self) -> GpuStencilFaceStateDescriptor {
        self.front
    }

    pub const fn back(self) -> GpuStencilFaceStateDescriptor {
        self.back
    }

    pub const fn read_mask(self) -> u32 {
        self.read_mask
    }

    pub const fn write_mask(self) -> u32 {
        self.write_mask
    }

    pub const fn may_write(self) -> bool {
        self.write_mask != 0 && (self.front.may_write() || self.back.may_write())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuDepthStencilStateDescriptor {
    format: GpuTextureFormat,
    depth: Option<GpuDepthStateDescriptor>,
    stencil: Option<GpuStencilStateDescriptor>,
}

impl GpuDepthStencilStateDescriptor {
    pub fn new(
        format: GpuTextureFormat,
        depth: Option<GpuDepthStateDescriptor>,
        stencil: Option<GpuStencilStateDescriptor>,
    ) -> Result<Self, GpuProgramContractError> {
        if depth.is_none() && stencil.is_none() {
            return Err(invalid_attachment_state(
                format!("depth_stencil_format={format:?}, depth=none, stencil=none"),
                "provide at least one depth or stencil pipeline state",
            ));
        }
        if depth.is_some() && !format.is_depth() {
            return Err(invalid_attachment_state(
                format!("depth_format={format:?}"),
                "use a format with a depth aspect when depth state is present",
            ));
        }
        if stencil.is_some() && !format.is_stencil() {
            return Err(invalid_attachment_state(
                format!("stencil_format={format:?}"),
                "use a format with a stencil aspect when stencil state is present",
            ));
        }
        Ok(Self {
            format,
            depth,
            stencil,
        })
    }

    pub const fn format(self) -> GpuTextureFormat {
        self.format
    }

    pub const fn depth(self) -> Option<GpuDepthStateDescriptor> {
        self.depth
    }

    pub const fn stencil(self) -> Option<GpuStencilStateDescriptor> {
        self.stencil
    }
}

fn invalid_attachment_state(
    label: impl Into<String>,
    correction: &'static str,
) -> GpuProgramContractError {
    GpuProgramContractError::invalid(
        "construct GPU render-attachment state",
        label,
        GpuProgramContractCause::RenderAttachmentStateInvalid,
        correction,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(format: GpuTextureFormat) -> GpuColorTargetStateDescriptor {
        GpuColorTargetStateDescriptor::new(format, GpuBlendMode::Replace, GpuColorWriteMask::ALL)
            .unwrap()
    }

    #[test]
    fn render_output_structure_follows_normalized_texture_format_authority() {
        for (format, class, width, alpha) in [
            (
                GpuTextureFormat::R8Unorm,
                GpuShaderIoScalarClass::Float,
                1,
                false,
            ),
            (
                GpuTextureFormat::Rgba8Unorm,
                GpuShaderIoScalarClass::Float,
                4,
                true,
            ),
            (
                GpuTextureFormat::Rgba8UnormSrgb,
                GpuShaderIoScalarClass::Float,
                4,
                true,
            ),
            (
                GpuTextureFormat::Bgra8Unorm,
                GpuShaderIoScalarClass::Float,
                4,
                true,
            ),
            (
                GpuTextureFormat::Bgra8UnormSrgb,
                GpuShaderIoScalarClass::Float,
                4,
                true,
            ),
            (
                GpuTextureFormat::R32Uint,
                GpuShaderIoScalarClass::Uint,
                1,
                false,
            ),
            (
                GpuTextureFormat::R32Sint,
                GpuShaderIoScalarClass::Sint,
                1,
                false,
            ),
            (
                GpuTextureFormat::R32Float,
                GpuShaderIoScalarClass::Float,
                1,
                false,
            ),
            (
                GpuTextureFormat::Rg32Uint,
                GpuShaderIoScalarClass::Uint,
                2,
                false,
            ),
            (
                GpuTextureFormat::Rg32Sint,
                GpuShaderIoScalarClass::Sint,
                2,
                false,
            ),
            (
                GpuTextureFormat::Rg32Float,
                GpuShaderIoScalarClass::Float,
                2,
                false,
            ),
            (
                GpuTextureFormat::Rgba32Uint,
                GpuShaderIoScalarClass::Uint,
                4,
                true,
            ),
            (
                GpuTextureFormat::Rgba32Sint,
                GpuShaderIoScalarClass::Sint,
                4,
                true,
            ),
            (
                GpuTextureFormat::Rgba32Float,
                GpuShaderIoScalarClass::Float,
                4,
                true,
            ),
        ] {
            let target = target(format);
            assert_eq!(target.shader_io_type().scalar_class(), class);
            assert_eq!(target.shader_io_type().vector_width().get(), width);
            assert_eq!(target.has_blendable_alpha_channel(), alpha);
        }
    }

    #[test]
    fn render_target_role_checks_keep_current_depth_and_integer_behavior() {
        for format in [
            GpuTextureFormat::Depth16Unorm,
            GpuTextureFormat::Depth24Plus,
            GpuTextureFormat::Depth32Float,
        ] {
            assert!(
                GpuColorTargetStateDescriptor::new(
                    format,
                    GpuBlendMode::Replace,
                    GpuColorWriteMask::ALL,
                )
                .is_err()
            );
            assert!(
                GpuDepthStencilStateDescriptor::new(
                    format,
                    Some(GpuDepthStateDescriptor::new(
                        true,
                        GpuCompareFunction::LessEqual
                    )),
                    None,
                )
                .is_ok()
            );
        }
        for format in [
            GpuTextureFormat::R32Uint,
            GpuTextureFormat::R32Sint,
            GpuTextureFormat::Rgba32Uint,
            GpuTextureFormat::Rgba32Sint,
        ] {
            assert!(
                GpuColorTargetStateDescriptor::new(
                    format,
                    GpuBlendMode::Alpha,
                    GpuColorWriteMask::ALL,
                )
                .is_err()
            );
            assert!(
                GpuColorTargetStateDescriptor::new(
                    format,
                    GpuBlendMode::Replace,
                    GpuColorWriteMask::ALL,
                )
                .is_ok()
            );
        }
        assert!(
            GpuColorTargetStateDescriptor::new(
                GpuTextureFormat::Rgba8Unorm,
                GpuBlendMode::Alpha,
                GpuColorWriteMask::ALL,
            )
            .is_ok()
        );

        let keep = GpuStencilFaceStateDescriptor::new(
            GpuCompareFunction::Always,
            GpuStencilOperation::Keep,
            GpuStencilOperation::Keep,
            GpuStencilOperation::Keep,
        );
        let replace = GpuStencilFaceStateDescriptor::new(
            GpuCompareFunction::Equal,
            GpuStencilOperation::Keep,
            GpuStencilOperation::Keep,
            GpuStencilOperation::Replace,
        );
        let stencil = GpuStencilStateDescriptor::new(keep, replace, u32::MAX, 0xff);
        let state =
            GpuDepthStencilStateDescriptor::new(GpuTextureFormat::Stencil8, None, Some(stencil))
                .unwrap();
        assert_eq!(state.stencil(), Some(stencil));
        assert!(stencil.may_write());

        let depth = GpuDepthStateDescriptor::new(true, GpuCompareFunction::LessEqual);
        assert!(
            GpuDepthStencilStateDescriptor::new(
                GpuTextureFormat::Depth24PlusStencil8,
                Some(depth),
                None,
            )
            .is_ok()
        );
        assert!(
            GpuDepthStencilStateDescriptor::new(
                GpuTextureFormat::Depth24PlusStencil8,
                None,
                Some(stencil),
            )
            .is_ok()
        );
        let combined = GpuDepthStencilStateDescriptor::new(
            GpuTextureFormat::Depth24PlusStencil8,
            Some(depth),
            Some(stencil),
        )
        .unwrap();
        assert_eq!(combined.depth(), Some(depth));
        assert_eq!(combined.stencil(), Some(stencil));
        assert!(
            GpuDepthStencilStateDescriptor::new(
                GpuTextureFormat::Stencil8,
                Some(GpuDepthStateDescriptor::new(
                    false,
                    GpuCompareFunction::Always
                )),
                None,
            )
            .is_err()
        );
        assert!(
            GpuDepthStencilStateDescriptor::new(
                GpuTextureFormat::Depth32Float,
                None,
                Some(stencil),
            )
            .is_err()
        );
    }
}
