use runen_gpu as gpu;

const WGSL: &str = r#"
struct ViewUniform {
    transform: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> view: ViewUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
}

@vertex
fn vertex_main(input: VertexInput) -> @builtin(position) vec4<f32> {
    return view.transform * vec4<f32>(input.position, 1.0);
}

@fragment
fn fragment_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
"#;

fn main() {
    let owner = gpu::GpuProgramSourceOwnerId::allocate().expect("source owner should allocate");
    let identity = gpu::GpuProgramSourceIdentity::new(
        owner,
        gpu::GpuProgramSourceKey::new("examples.render.triangle")
            .expect("source key should be valid"),
        gpu::GpuProgramSourceRevision::try_from_raw(1).expect("source revision should be nonzero"),
    );
    let mut registry =
        gpu::GpuProgramSourceRegistry::new(4, 16 * 1024).expect("registry should construct");
    let source = registry
        .admit_wgsl(
            identity,
            WGSL,
            gpu::GpuProgramSourceProvenance::new(
                "gpu-render-contract-example",
                Some("render example".to_owned()),
            )
            .expect("source provenance should be valid"),
        )
        .expect("source should admit");

    let program = gpu::GpuProgramDescriptor::new(
        source,
        [entry_point("vertex_main"), entry_point("fragment_main")],
        std::iter::empty::<gpu::GpuBindingLayoutRefinement>(),
    )
    .expect("program should derive interface and stage IO from canonical WGSL");

    let vertex_layout = gpu::GpuVertexBufferLayoutDescriptor::new(
        0,
        12,
        gpu::GpuVertexStepMode::Vertex,
        [gpu::GpuVertexAttribute::new(
            0,
            0,
            gpu::GpuVertexFormat::Float32x3,
        )],
    )
    .expect("vertex layout should construct");
    let vertex_input = gpu::GpuVertexInputStateDescriptor::new([vertex_layout])
        .expect("vertex input state should construct");
    let color_target = gpu::GpuColorTargetStateDescriptor::new(
        gpu::GpuTextureFormat::Rgba8UnormSrgb,
        gpu::GpuBlendMode::Replace,
        gpu::GpuColorWriteMask::ALL,
    )
    .expect("color target should construct");
    let state = gpu::GpuRenderPipelineStateDescriptor::new(
        vertex_input,
        Some(gpu::GpuFragmentOutputStateDescriptor::new([color_target])),
        gpu::GpuPrimitiveStateDescriptor::default(),
        None,
        gpu::GpuMultisampleStateDescriptor::default(),
    )
    .expect("render state should construct");
    let _pipeline = gpu::GpuRenderPipelineDescriptor::new(
        program,
        gpu::GpuRenderEntryPoints::new(
            entry_point("vertex_main"),
            Some(entry_point("fragment_main")),
        ),
        state,
        gpu::GpuPipelineConfiguration::default(),
    )
    .expect("render pipeline should validate compiler-observed stage IO and construct");
}

fn entry_point(name: &str) -> gpu::GpuEntryPointName {
    gpu::GpuEntryPointName::new(name).expect("entry-point name should be valid")
}
