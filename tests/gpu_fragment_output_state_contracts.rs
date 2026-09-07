use runen_gpu::{
    GpuBlendMode, GpuColorTargetStateDescriptor, GpuColorWriteMask, GpuCompareFunction,
    GpuDepthStencilStateDescriptor, GpuEntryPointName, GpuFragmentOutputStateDescriptor,
    GpuProgramContractCause, GpuShaderIoScalarClass, GpuTextureFormat,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn entry_point(value: &str) -> GpuEntryPointName {
    GpuEntryPointName::new(value).expect("test entry-point name should be valid")
}

fn color_target(
    format: GpuTextureFormat,
    write_mask: GpuColorWriteMask,
) -> GpuColorTargetStateDescriptor {
    GpuColorTargetStateDescriptor::new(format, GpuBlendMode::Replace, write_mask)
        .expect("test color target should be valid")
}

fn hash_of(value: &impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn fragment_output_state_derives_ordered_expected_signature() {
    let state = GpuFragmentOutputStateDescriptor::new([
        color_target(GpuTextureFormat::Rgba8UnormSrgb, GpuColorWriteMask::ALL),
        color_target(GpuTextureFormat::R32Uint, GpuColorWriteMask::RED),
    ]);
    let equivalent = state.clone();

    assert_eq!(state, equivalent);
    assert_eq!(hash_of(&state), hash_of(&equivalent));
    assert_eq!(state.color_targets().len(), 2);
    let signature = state
        .expected_signature(entry_point("fragment_main"))
        .unwrap();
    assert_eq!(signature.entry_point().as_str(), "fragment_main");
    assert_eq!(
        signature
            .locations()
            .map(|location| (
                location.location(),
                location.value_type().scalar_class(),
                location.value_type().vector_width().get(),
            ))
            .collect::<Vec<_>>(),
        [
            (0, GpuShaderIoScalarClass::Float, 4),
            (1, GpuShaderIoScalarClass::Uint, 1),
        ]
    );
}

#[test]
fn fragment_output_state_supports_no_color_outputs() {
    let state = GpuFragmentOutputStateDescriptor::new([]);
    let signature = state
        .expected_signature(entry_point("fragment_depth"))
        .unwrap();

    assert_eq!(state.color_targets().len(), 0);
    assert_eq!(signature.locations().len(), 0);
}

#[test]
fn color_target_state_rejects_depth_and_integer_alpha_blending() {
    let depth = GpuColorTargetStateDescriptor::new(
        GpuTextureFormat::Depth32Float,
        GpuBlendMode::Replace,
        GpuColorWriteMask::ALL,
    )
    .expect_err("depth formats are not color targets");
    assert_eq!(
        depth.cause(),
        GpuProgramContractCause::RenderAttachmentStateInvalid
    );

    let integer_alpha = GpuColorTargetStateDescriptor::new(
        GpuTextureFormat::R32Uint,
        GpuBlendMode::Alpha,
        GpuColorWriteMask::ALL,
    )
    .expect_err("integer color targets cannot use alpha blending");
    assert_eq!(
        integer_alpha.cause(),
        GpuProgramContractCause::RenderAttachmentStateInvalid
    );
}

#[test]
fn color_write_mask_rejects_unknown_bits_and_retains_components() {
    let invalid = GpuColorWriteMask::from_bits(0b1_0000)
        .expect_err("unknown color-write bits must be rejected");
    assert_eq!(
        invalid.cause(),
        GpuProgramContractCause::RenderAttachmentStateInvalid
    );

    let mask = GpuColorWriteMask::from_bits(
        GpuColorWriteMask::RED.bits() | GpuColorWriteMask::ALPHA.bits(),
    )
    .unwrap();
    assert!(mask.contains(GpuColorWriteMask::RED));
    assert!(mask.contains(GpuColorWriteMask::ALPHA));
    assert!(!mask.contains(GpuColorWriteMask::GREEN));
}

#[test]
fn depth_stencil_state_requires_a_depth_format() {
    let state = GpuDepthStencilStateDescriptor::new(
        GpuTextureFormat::Depth32Float,
        true,
        GpuCompareFunction::LessEqual,
    )
    .unwrap();
    assert_eq!(state.format(), GpuTextureFormat::Depth32Float);
    assert!(state.depth_write_enabled());
    assert_eq!(state.depth_compare(), GpuCompareFunction::LessEqual);

    let error = GpuDepthStencilStateDescriptor::new(
        GpuTextureFormat::Rgba8Unorm,
        false,
        GpuCompareFunction::Always,
    )
    .expect_err("color formats are not depth-stencil formats");
    assert_eq!(
        error.cause(),
        GpuProgramContractCause::RenderAttachmentStateInvalid
    );
}
