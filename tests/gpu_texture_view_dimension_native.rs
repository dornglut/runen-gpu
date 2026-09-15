use runen_gpu::*;

const VIEW_DIMENSION_WGSL: &str = r#"
@group(0) @binding(0)
var d2_texture: texture_2d<f32>;

@group(0) @binding(1)
var d2_array_texture: texture_2d_array<f32>;

@group(0) @binding(2)
var cube_texture: texture_cube<f32>;

@group(0) @binding(3)
var cube_array_texture: texture_cube_array<f32>;

var<workgroup> dimension_sink: u32;

@compute @workgroup_size(1)
fn main() {
    dimension_sink = textureDimensions(d2_texture).x
        + textureDimensions(d2_array_texture).x
        + textureDimensions(cube_texture).x
        + textureDimensions(cube_array_texture).x;
}
"#;

fn label(value: &str) -> GpuResourceLabel {
    GpuResourceLabel::new(value).unwrap()
}

fn provenance(value: &str) -> GpuResourceProvenance {
    GpuResourceProvenance::new(label(value), None, None)
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

fn sampled_array_texture(
    allocator: &mut GpuWorkResourceIdAllocator,
    name: &str,
    width: u32,
    height: u32,
    layers: u32,
) -> GpuTextureHandle {
    let texture_label = label(name);
    allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::new(
                common(name),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(
                    &texture_label,
                    GpuTextureDimension::D2,
                    width,
                    height,
                    layers,
                )
                .unwrap(),
                1,
                1,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(&texture_label, [GpuTextureUsage::Sampled]).unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap()
}

fn texture_view(
    allocator: &mut GpuWorkResourceIdAllocator,
    texture: &GpuTextureHandle,
    name: &str,
    dimension: GpuTextureViewDimension,
    base_array_layer: u32,
    array_layer_count: u32,
) -> Result<GpuTextureViewHandle, GpuResourceDescriptorError> {
    let view_label = label(name);
    let subresources = GpuTextureSubresourceRange::new(
        &view_label,
        0,
        1,
        base_array_layer,
        array_layer_count,
        GpuTextureAspect::Color,
    )?;
    let descriptor =
        GpuTextureViewDescriptor::new(common(name), texture, None, dimension, subresources)?;
    Ok(allocator.allocate_texture_view_handle(descriptor).unwrap())
}

fn dimension_pipeline() -> GpuComputePipelineDescriptor {
    let [source] =
        admit_static_wgsl_sources([("texture-view-dimension-authority", 1, VIEW_DIMENSION_WGSL)])
            .unwrap();
    let entry_point = GpuEntryPointName::new("main").unwrap();
    let refinements = (0_u64..4).map(|binding| {
        GpuBindingLayoutRefinement::new(GpuBindingKey::try_new(0, binding).unwrap())
            .with_texture_sample_class(GpuTextureSampleClass::FloatFilterable)
    });
    let program = GpuProgramDescriptor::new(source, [entry_point.clone()], refinements).unwrap();
    GpuComputePipelineDescriptor::new(
        program,
        entry_point,
        GpuPipelineConfiguration::default(),
    )
    .unwrap()
}

fn texture_binding(binding: u32, view: &GpuTextureViewHandle) -> GpuRuntimeBindingValue {
    let key = GpuBindingKey::try_new(0, u64::from(binding)).unwrap();
    GpuRuntimeBindingValue::new(
        key,
        [GpuRuntimeBindingResource::TextureView(
            GpuRuntimeTextureViewBinding::new(view.clone()),
        )],
    )
    .unwrap()
}

fn canonical_views() -> (
    GpuWorkResourceIdAllocator,
    GpuTextureHandle,
    GpuTextureViewHandle,
    GpuTextureViewHandle,
    GpuTextureViewHandle,
    GpuTextureViewHandle,
) {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let texture = sampled_array_texture(&mut allocator, "view-dimension parent", 4, 4, 12);
    let d2 = texture_view(
        &mut allocator,
        &texture,
        "D2 view",
        GpuTextureViewDimension::D2,
        0,
        1,
    )
    .unwrap();
    let d2_array = texture_view(
        &mut allocator,
        &texture,
        "D2Array view",
        GpuTextureViewDimension::D2Array,
        0,
        2,
    )
    .unwrap();
    let cube = texture_view(
        &mut allocator,
        &texture,
        "Cube view",
        GpuTextureViewDimension::Cube,
        0,
        6,
    )
    .unwrap();
    let cube_array = texture_view(
        &mut allocator,
        &texture,
        "CubeArray view",
        GpuTextureViewDimension::CubeArray,
        0,
        12,
    )
    .unwrap();
    (allocator, texture, d2, d2_array, cube, cube_array)
}

#[test]
fn view_dimension_is_single_authority_for_shader_binding_compatibility() {
    let (_allocator, _texture, d2, d2_array, cube, cube_array) = canonical_views();
    assert_eq!(d2.descriptor().dimension(), GpuTextureViewDimension::D2);
    assert_eq!(
        d2_array.descriptor().dimension(),
        GpuTextureViewDimension::D2Array
    );
    assert_eq!(cube.descriptor().dimension(), GpuTextureViewDimension::Cube);
    assert_eq!(
        cube_array.descriptor().dimension(),
        GpuTextureViewDimension::CubeArray
    );

    let pipeline = dimension_pipeline();
    pipeline
        .runtime_bindings([
            texture_binding(0, &d2),
            texture_binding(1, &d2_array),
            texture_binding(2, &cube),
            texture_binding(3, &cube_array),
        ])
        .expect("authoritative view dimensions must match the WGSL-derived layout");

    let cube_mismatch = pipeline
        .runtime_bindings([
            texture_binding(0, &d2),
            texture_binding(1, &d2_array),
            texture_binding(2, &d2_array),
            texture_binding(3, &cube_array),
        ])
        .unwrap_err();
    assert_eq!(
        cube_mismatch.cause(),
        GpuProgramContractCause::RuntimeBindingIncompatible
    );

    let cube_array_mismatch = pipeline
        .runtime_bindings([
            texture_binding(0, &d2),
            texture_binding(1, &d2_array),
            texture_binding(2, &cube),
            texture_binding(3, &d2_array),
        ])
        .unwrap_err();
    assert_eq!(
        cube_array_mismatch.cause(),
        GpuProgramContractCause::RuntimeBindingIncompatible
    );
}

#[test]
fn malformed_cube_layer_ranges_are_rejected_before_realization() {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let texture = sampled_array_texture(&mut allocator, "malformed cube parent", 4, 4, 12);

    let cube_error = texture_view(
        &mut allocator,
        &texture,
        "malformed Cube view",
        GpuTextureViewDimension::Cube,
        0,
        5,
    )
    .unwrap_err();
    assert_eq!(
        cube_error.cause(),
        GpuResourceDescriptorCause::IncompatibleViewDimension
    );

    let cube_array_error = texture_view(
        &mut allocator,
        &texture,
        "malformed CubeArray view",
        GpuTextureViewDimension::CubeArray,
        0,
        7,
    )
    .unwrap_err();
    assert_eq!(
        cube_array_error.cause(),
        GpuResourceDescriptorCause::IncompatibleViewDimension
    );
}

#[test]
#[ignore = "requires a real Vulkan fallback adapter; executed by RunenGPU Native Conformance CI"]
fn d2_d2array_cube_and_cubearray_realize_and_bind_against_wgsl_dimensions() {
    let descriptor = GpuContextDescriptor::new(GpuCapabilityRequirements::new())
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::Sampled)
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label("texture-view dimension native proof");
    let context = pollster::block_on(GpuContext::request(descriptor))
        .expect("native conformance Vulkan fallback must admit sampled RGBA8");

    let (_allocator, texture, d2, d2_array, cube, cube_array) = canonical_views();
    let realized_texture = context
        .realize_texture(&texture)
        .expect("sampled array texture must realize");
    let realized_d2 = context
        .realize_texture_view(&d2, &realized_texture)
        .expect("D2 view must realize as D2");
    let realized_d2_array = context
        .realize_texture_view(&d2_array, &realized_texture)
        .expect("D2Array view must realize as D2Array");
    let realized_cube = context
        .realize_texture_view(&cube, &realized_texture)
        .expect("Cube view must realize as Cube");
    let realized_cube_array = context
        .realize_texture_view(&cube_array, &realized_texture)
        .expect("CubeArray view must realize as CubeArray");

    let pipeline = dimension_pipeline();
    let group_layout = pipeline
        .layout()
        .group(0)
        .expect("WGSL fixture must derive bind group zero");
    let realized_layout = pollster::block_on(context.realize_bind_group_layout(group_layout))
        .expect("WGSL-derived texture-view layout must realize");
    let realized_bind_group = pollster::block_on(context.realize_bind_group(
        &realized_layout,
        [
            texture_binding(0, &d2),
            texture_binding(1, &d2_array),
            texture_binding(2, &cube),
            texture_binding(3, &cube_array),
        ],
    ))
    .expect("all authoritative public texture-view dimensions must bind to their WGPU layout");

    drop(realized_bind_group);
    drop(realized_layout);
    drop(realized_cube_array);
    drop(realized_cube);
    drop(realized_d2_array);
    drop(realized_d2);
    drop(realized_texture);
}
