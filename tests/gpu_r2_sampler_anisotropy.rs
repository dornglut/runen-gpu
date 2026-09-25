use runen_gpu::*;

fn label(value: impl AsRef<str>) -> GpuResourceLabel {
    GpuResourceLabel::new(value.as_ref()).unwrap()
}

fn provenance(value: impl AsRef<str>) -> GpuResourceProvenance {
    let value = value.as_ref();
    GpuResourceProvenance::new(label(value), None, None)
}

fn common(value: impl AsRef<str>) -> GpuResourceCommon {
    let value = value.as_ref();
    GpuResourceCommon::owned(
        label(value),
        GpuResourceLifetime::Transient,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        provenance(value),
    )
    .unwrap()
}

fn linear_anisotropic_state(max_anisotropy: u16) -> GpuSamplerFilterState {
    GpuSamplerFilterState::new(
        GpuFilterMode::Linear,
        GpuFilterMode::Linear,
        GpuFilterMode::Linear,
        max_anisotropy,
    )
    .unwrap()
}

fn sampler_descriptor(max_anisotropy: u16) -> GpuSamplerDescriptor {
    GpuSamplerDescriptor::new(
        common("R2 anisotropic sampler"),
        GpuAddressMode::Repeat,
        GpuAddressMode::Repeat,
        GpuAddressMode::Repeat,
        linear_anisotropic_state(max_anisotropy),
        0.0,
        16.0,
        None,
    )
    .unwrap()
}

fn realize_anisotropic_sampler(context: &GpuContext) {
    let mut scope = GpuResourceScope::new();
    let sampler = scope.sampler(sampler_descriptor(8)).unwrap();
    context.realize_sampler(&sampler).unwrap();
}

#[test]
fn sampler_filter_state_enforces_anisotropy_invariants() {
    let zero = GpuSamplerFilterState::new(
        GpuFilterMode::Linear,
        GpuFilterMode::Linear,
        GpuFilterMode::Linear,
        0,
    )
    .expect_err("anisotropy zero must be rejected");
    assert_eq!(
        zero.cause(),
        GpuResourceDescriptorCause::InvalidSamplerFilterState
    );

    for mag in [GpuFilterMode::Nearest, GpuFilterMode::Linear] {
        for min in [GpuFilterMode::Nearest, GpuFilterMode::Linear] {
            for mipmap in [GpuFilterMode::Nearest, GpuFilterMode::Linear] {
                let state = GpuSamplerFilterState::new(mag, min, mipmap, 1)
                    .expect("anisotropy one preserves every existing filter combination");
                assert_eq!(state.filters(), (mag, min, mipmap));
                assert_eq!(state.max_anisotropy(), 1);
            }
        }
    }

    for (mag, min, mipmap) in [
        (
            GpuFilterMode::Nearest,
            GpuFilterMode::Linear,
            GpuFilterMode::Linear,
        ),
        (
            GpuFilterMode::Linear,
            GpuFilterMode::Nearest,
            GpuFilterMode::Linear,
        ),
        (
            GpuFilterMode::Linear,
            GpuFilterMode::Linear,
            GpuFilterMode::Nearest,
        ),
    ] {
        let error = GpuSamplerFilterState::new(mag, min, mipmap, 8)
            .expect_err("anisotropy above one requires all-linear filtering");
        assert_eq!(
            error.cause(),
            GpuResourceDescriptorCause::InvalidSamplerFilterState
        );
    }

    let unrestricted = linear_anisotropic_state(u16::MAX);
    assert_eq!(
        unrestricted.max_anisotropy(),
        u16::MAX,
        "the public contract must not manufacture a platform-specific anisotropy ceiling"
    );

    let state = linear_anisotropic_state(8);
    assert_eq!(
        state.filters(),
        (
            GpuFilterMode::Linear,
            GpuFilterMode::Linear,
            GpuFilterMode::Linear
        )
    );
    assert_eq!(state.max_anisotropy(), 8);
    assert!(state.is_filtering());

    let descriptor = sampler_descriptor(8);
    assert_eq!(descriptor.filter_state(), state);
}

#[cfg(not(target_arch = "wasm32"))]
fn native_context() -> GpuContext {
    let descriptor = GpuContextDescriptor::new(GpuCapabilityRequirements::new())
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label("R2 sampler anisotropy proof");
    let context = pollster::block_on(GpuContext::request(descriptor))
        .expect("native Conformance must provide the retained Vulkan fallback adapter");
    assert_eq!(context.adapter_facts().backend(), GpuBackendFamily::Vulkan);
    assert_eq!(
        context.adapter_facts().fallback(),
        GpuFallbackStatus::ConfirmedFallback
    );
    context
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires the retained Vulkan software adapter"]
fn sampler_anisotropy_native_realization_is_backend_proven() {
    let context = native_context();
    realize_anisotropic_sampler(&context);
    println!("Sampler anisotropy: EXERCISED (requested max=8 through public sampler realization)");
}

#[cfg(target_arch = "wasm32")]
pub(crate) async fn run_browser_sampler_anisotropy() -> u32 {
    let descriptor = GpuContextDescriptor::new(GpuCapabilityRequirements::new())
        .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
        .with_label("R2 browser sampler anisotropy proof");
    let context = GpuContext::request(descriptor)
        .await
        .expect("actual-browser Conformance must provide WebGPU");
    assert_eq!(
        context.adapter_facts().backend(),
        GpuBackendFamily::BrowserWebGpu
    );
    realize_anisotropic_sampler(&context);
    1
}
