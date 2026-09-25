use runen_gpu::*;
use std::num::NonZeroU64;

#[path = "support/readback_wait.rs"]
mod readback_wait;

const F16_COMPUTE_WGSL: &str = r#"
enable f16;

struct Data {
    values: vec4<f16>,
    result: u32,
}

@group(0) @binding(0)
var<storage, read_write> data: Data;

@compute @workgroup_size(1)
fn cs_main() {
    let value: f16 = data.values.x * data.values.y + data.values.z;
    data.result = u32(value);
}
"#;

const F32_COMPUTE_WGSL: &str = r#"
@compute @workgroup_size(1)
fn cs_main() {}
"#;

const UNUSED_F16_WGSL: &str = r#"
enable f16;

@compute @workgroup_size(1)
fn cs_main() {}
"#;

const F16_WITHOUT_ENABLE_WGSL: &str = r#"
@compute @workgroup_size(1)
fn cs_main() {
    let value = f16(1.0);
}
"#;

const F16_OVERRIDE_WGSL: &str = r#"
enable f16;
override SCALE: f16 = 1.0h;

@compute @workgroup_size(1)
fn cs_main() {
    let value = SCALE;
}
"#;

const F16_VERTEX_IO_WGSL: &str = r#"
enable f16;

@vertex
fn vs_main(@location(0) value: f16) -> @builtin(position) vec4f {
    return vec4f(f32(value), 0.0, 0.0, 1.0);
}
"#;

const F16_FRAGMENT_IO_WGSL: &str = r#"
enable f16;

@fragment
fn fs_main() -> @location(0) f16 {
    return 1.0h;
}
"#;

fn label(value: impl AsRef<str>) -> GpuResourceLabel {
    GpuResourceLabel::new(value.as_ref()).unwrap()
}

fn provenance(value: impl AsRef<str>) -> GpuResourceProvenance {
    GpuResourceProvenance::new(label(value.as_ref()), None, None)
}

fn common(value: &str) -> GpuResourceCommon {
    GpuResourceCommon::owned(
        label(value),
        GpuResourceLifetime::Transient,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        provenance(value),
    )
    .unwrap()
}

fn source(key: &str, wgsl: &str) -> GpuAdmittedProgramSource {
    let owner = GpuProgramSourceOwnerId::allocate().unwrap();
    let identity = GpuProgramSourceIdentity::new(
        owner,
        GpuProgramSourceKey::new(key).unwrap(),
        GpuProgramSourceRevision::try_from_raw(1).unwrap(),
    );
    let mut sources = GpuProgramSourceRegistry::new(4, 16 * 1024).unwrap();
    sources
        .admit_wgsl(
            identity,
            wgsl,
            GpuProgramSourceProvenance::new("r3-shader-f16-proof", None).unwrap(),
        )
        .unwrap()
}

fn program(
    key: &str,
    wgsl: &str,
    entry: &str,
) -> Result<GpuProgramDescriptor, GpuProgramContractError> {
    GpuProgramDescriptor::new(
        source(key, wgsl),
        [GpuEntryPointName::new(entry).unwrap()],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
}

fn has_f16_requirement(program: &GpuProgramDescriptor) -> bool {
    matches!(
        program.requirements().get(GpuCapabilityFeature::ShaderF16),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::ShaderF16
        ))
    )
}

#[test]
fn shader_f16_requirement_is_compiler_derived_and_buffer_layout_is_retained() {
    let f32_program = program("r3.f16.f32", F32_COMPUTE_WGSL, "cs_main").unwrap();
    assert!(!has_f16_requirement(&f32_program));

    let unused = program("r3.f16.unused", UNUSED_F16_WGSL, "cs_main").unwrap();
    assert!(has_f16_requirement(&unused));

    let f16_program = program("r3.f16.compute", F16_COMPUTE_WGSL, "cs_main").unwrap();
    assert!(has_f16_requirement(&f16_program));
    let binding = f16_program.interface().bindings().next().unwrap();
    assert_eq!(binding.key(), GpuBindingKey::try_new(0, 0).unwrap());
    assert_eq!(
        binding.compiler_required_minimum_size(),
        NonZeroU64::new(16),
        "vec4<f16> plus u32 must retain the compiler-derived padded storage minimum"
    );
}

#[test]
fn shader_f16_invalid_and_unrepresentable_forms_fail_closed() {
    let missing_enable = program(
        "r3.f16.missing-enable",
        F16_WITHOUT_ENABLE_WGSL,
        "cs_main",
    )
    .unwrap_err();
    assert_eq!(
        missing_enable.cause(),
        GpuProgramContractCause::CanonicalWgslInvalid
    );

    let override_error = program("r3.f16.override", F16_OVERRIDE_WGSL, "cs_main").unwrap_err();
    assert_eq!(
        override_error.cause(),
        GpuProgramContractCause::SpecializationOverridesUnsupported
    );

    let vertex_error = program("r3.f16.vertex-io", F16_VERTEX_IO_WGSL, "vs_main").unwrap_err();
    assert!(matches!(
        vertex_error.cause(),
        GpuProgramContractCause::StageIoSignatureInvalid
            | GpuProgramContractCause::CanonicalWgslInvalid
    ));

    let fragment_error =
        program("r3.f16.fragment-io", F16_FRAGMENT_IO_WGSL, "fs_main").unwrap_err();
    assert!(matches!(
        fragment_error.cause(),
        GpuProgramContractCause::StageIoSignatureInvalid
            | GpuProgramContractCause::CanonicalWgslInvalid
    ));
}

fn f16_context_descriptor(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
    require_f16: bool,
) -> GpuContextDescriptor {
    let mut requirements = GpuCapabilityProfile::ComputeBaseline.requirements();
    if require_f16 {
        requirements
            .insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::ShaderF16,
            ))
            .unwrap();
    }
    let mut descriptor = GpuContextDescriptor::new(requirements)
        .with_allowed_backends([backend])
        .with_label("R3 ShaderF16 execution proof");
    if let Some(fallback) = fallback {
        descriptor = descriptor.with_fallback_policy(fallback);
    }
    descriptor
}

fn f16_buffer() -> GpuBufferHandle {
    let bytes = [
        0x00_u8, 0x3e, 0x00, 0x40, 0x00, 0x38, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00,
    ];
    let data = PreparedGpuData::<TransferData>::from_pod_transfer(
        "R3 ShaderF16 buffer payload",
        &bytes,
        provenance("R3 ShaderF16 buffer payload"),
    )
    .unwrap();
    let mut resources = GpuResourceScope::new();
    resources
        .buffer(
            GpuBufferDescriptor::new(
                common("R3 ShaderF16 buffer"),
                u64::try_from(bytes.len()).unwrap(),
                GpuBufferUsages::new(
                    &label("R3 ShaderF16 buffer"),
                    [
                        GpuBufferUsage::Storage,
                        GpuBufferUsage::CopySource,
                        GpuBufferUsage::CopyDestination,
                    ],
                )
                .unwrap(),
                GpuBufferInitialization::Prepared(data),
            )
            .unwrap(),
        )
        .unwrap()
}

fn f16_pipeline() -> GpuComputePipelineDescriptor {
    let entry = GpuEntryPointName::new("cs_main").unwrap();
    let program = program("r3.f16.runtime", F16_COMPUTE_WGSL, "cs_main").unwrap();
    GpuComputePipelineDescriptor::new(program, entry, GpuPipelineConfiguration::default()).unwrap()
}

fn f16_graph() -> (GpuPreparedWorkGraph, GpuReadbackId) {
    let buffer = f16_buffer();
    let pipeline = f16_pipeline();
    assert!(has_f16_requirement(pipeline.program()));
    let binding = GpuRuntimeBindingValue::new(
        GpuBindingKey::try_new(0, 0).unwrap(),
        [GpuRuntimeBindingResource::Buffer(
            GpuRuntimeBufferBinding::new(
                buffer.clone(),
                0,
                NonZeroU64::new(16).unwrap(),
                None,
            ),
        )],
    )
    .unwrap();
    let bindings = GpuRuntimeBindingSet::new(pipeline.layout().clone(), [binding]).unwrap();
    let compute = GpuComputeOperation::new(
        pipeline,
        bindings,
        GpuDispatchIntent::direct(GpuDispatchSize::new(1, 1, 1)),
    )
    .unwrap();
    let region = GpuBufferRegion::new(&buffer, GpuBufferRange::whole(&buffer).unwrap()).unwrap();
    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(region.into(), readback_id).unwrap();
    let fragment = GpuWorkFragment::build("R3 ShaderF16 execution", |work| {
        work.operation("execute f16 compute", compute)?;
        work.operation("read f16 result", readback)?;
        Ok(())
    })
    .unwrap();
    (
        GpuPreparedWorkGraph::prepare(label("R3 ShaderF16 graph"), [fragment]).unwrap(),
        readback_id,
    )
}

async fn execute_f16(context: &GpuContext) {
    let (graph, readback_id) = f16_graph();
    let prepared = context.prepare_submission(graph).await.unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let bytes =
        readback_wait::wait_for_readback(context, &submission, readback_id, "R3 ShaderF16").await;
    assert_eq!(bytes.as_bytes().len(), 16);
    assert_eq!(
        u32::from_le_bytes(bytes.as_bytes()[8..12].try_into().unwrap()),
        3,
        "1.5h * 2.0h + 0.5h must execute in f16 and convert deterministically"
    );
    println!("ShaderF16: EXERCISED (buffer-backed f16 arithmetic + exact readback)");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires retained Vulkan/Lavapipe execution"]
fn shader_f16_native_execution_is_backend_proven() {
    let baseline = pollster::block_on(GpuContext::request(f16_context_descriptor(
        GpuBackendFamily::Vulkan,
        Some(GpuSoftwareFallbackPolicy::Require),
        false,
    )))
    .expect("retained Vulkan fallback context must be available");
    assert!(
        baseline
            .adapter_facts()
            .supported()
            .supports(GpuCapabilityFeature::ShaderF16),
        "acceptance requires at least one retained positive ShaderF16 execution"
    );
    let context = pollster::block_on(GpuContext::request(f16_context_descriptor(
        GpuBackendFamily::Vulkan,
        Some(GpuSoftwareFallbackPolicy::Require),
        true,
    )))
    .expect("advertised retained Vulkan ShaderF16 capability must admit a context");
    pollster::block_on(execute_f16(&context));
}

#[cfg(target_arch = "wasm32")]
pub(crate) async fn run_browser_shader_f16() -> u32 {
    const SUPPORTED: u32 = 1 << 0;
    const EXERCISED: u32 = 1 << 1;

    let baseline = GpuContext::request(f16_context_descriptor(
        GpuBackendFamily::BrowserWebGpu,
        None,
        false,
    ))
    .await
    .expect("actual-browser WebGPU baseline must be available");
    if !baseline
        .adapter_facts()
        .supported()
        .supports(GpuCapabilityFeature::ShaderF16)
    {
        println!("ShaderF16: UNSUPPORTED (normalized capability absent)");
        return 0;
    }

    let context = GpuContext::request(f16_context_descriptor(
        GpuBackendFamily::BrowserWebGpu,
        None,
        true,
    ))
    .await
    .expect("advertised browser ShaderF16 capability must admit a context");
    execute_f16(&context).await;
    SUPPORTED | EXERCISED
}
