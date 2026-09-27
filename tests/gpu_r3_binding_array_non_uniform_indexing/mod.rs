use runen_gpu::*;

use crate::retained_fixed_binding_array as fixed;
use fixed::readback_wait;

const STORAGE_BUFFER_WGSL: &str = r#"
enable wgpu_binding_array;

struct Value {
    value: u32,
}

struct Output {
    values: array<u32, 2>,
}

@group(0) @binding(0)
var<storage, read> inputs: binding_array<Value, 2>;

@group(0) @binding(1)
var<storage, read_write> output: Output;

@compute @workgroup_size(2)
fn cs_main(@builtin(local_invocation_index) index: u32) {
    output.values[index] = inputs[index].value;
}
"#;

const TEXTURE_SAMPLER_WGSL: &str = r#"
enable wgpu_binding_array;

struct Output {
    values: array<u32, 2>,
}

@group(0) @binding(0)
var input_textures: binding_array<texture_2d<f32>, 2>;

@group(0) @binding(1)
var input_samplers: binding_array<sampler, 2>;

@group(0) @binding(2)
var<storage, read_write> output: Output;

@compute @workgroup_size(2)
fn cs_main(@builtin(local_invocation_index) index: u32) {
    let sampled = textureSampleLevel(
        input_textures[index],
        input_samplers[index],
        vec2<f32>(1.25, 0.5),
        0.0,
    ).r;
    output.values[index] = u32(round(sampled * 255.0));
}
"#;

const STORAGE_TEXTURE_WGSL: &str = r#"
enable wgpu_binding_array;

@group(0) @binding(0)
var outputs: binding_array<texture_storage_2d<r32uint, write>, 2>;

@compute @workgroup_size(2)
fn cs_main(@builtin(local_invocation_index) index: u32) {
    let value = 17u + index * 84u;
    textureStore(outputs[index], vec2<i32>(0, 0), vec4<u32>(value, 0u, 0u, 0u));
}
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NonUniformBindingArrayProof {
    pub(crate) texture_sampler: bool,
    pub(crate) storage_buffer: bool,
    pub(crate) storage_texture: bool,
}

fn prepared_u32_pair_buffer(
    resources: &mut GpuResourceScope,
    name: &str,
    values: [u32; 2],
    readback: bool,
) -> GpuBufferHandle {
    let data =
        PreparedGpuData::<TransferData>::ordinary_pod_transfer(name, values.as_slice()).unwrap();
    let mut usages = vec![GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination];
    if readback {
        usages.push(GpuBufferUsage::CopySource);
    }
    resources
        .buffer(
            GpuBufferDescriptor::ordinary_owned(
                name,
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                data.layout().byte_len(),
                usages,
                GpuBufferInitialization::Prepared(data),
            )
            .unwrap(),
        )
        .unwrap()
}

fn two_pixel_texture(
    resources: &mut GpuResourceScope,
    name: &str,
    pixels: [u8; 8],
) -> (GpuTextureHandle, GpuTextureViewHandle) {
    let texture_label = fixed::label(name);
    let extent = GpuTextureExtent::new(&texture_label, GpuTextureDimension::D2, 2, 1, 1).unwrap();
    let prepared = GpuPreparedTextureData::new(
        &texture_label,
        PreparedGpuData::<TransferData>::from_pod_transfer(
            name,
            pixels.as_slice(),
            fixed::provenance(name),
        )
        .unwrap(),
        GpuTextureFormat::Rgba8Unorm,
        extent,
        8,
        0,
    )
    .unwrap();
    let texture = resources
        .texture(
            GpuTextureDescriptor::new(
                fixed::common(name),
                GpuTextureDimension::D2,
                extent,
                1,
                1,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(
                    &texture_label,
                    [GpuTextureUsage::Sampled, GpuTextureUsage::CopyDestination],
                )
                .unwrap(),
                GpuTextureInitialization::Prepared(prepared),
            )
            .unwrap(),
        )
        .unwrap();
    let view = resources
        .texture_view(
            GpuTextureViewDescriptor::new(
                fixed::common(format!("{name} view")),
                &texture,
                None,
                GpuTextureViewDimension::D2,
                GpuTextureSubresourceRange::whole(&texture).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    (texture, view)
}

fn assert_pinned_naga_storage_buffer_non_uniformity() {
    let capabilities = naga::valid::Capabilities::default()
        | naga::valid::Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY
        | naga::valid::Capabilities::BUFFER_BINDING_ARRAY
        | naga::valid::Capabilities::STORAGE_TEXTURE_BINDING_ARRAY
        | naga::valid::Capabilities::STORAGE_BUFFER_BINDING_ARRAY
        | naga::valid::Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY_NON_UNIFORM_INDEXING
        | naga::valid::Capabilities::BUFFER_BINDING_ARRAY_NON_UNIFORM_INDEXING
        | naga::valid::Capabilities::STORAGE_TEXTURE_BINDING_ARRAY_NON_UNIFORM_INDEXING
        | naga::valid::Capabilities::STORAGE_BUFFER_BINDING_ARRAY_NON_UNIFORM_INDEXING;
    let mut frontend = naga::front::wgsl::Frontend::new_with_options(naga::front::wgsl::Options {
        parse_doc_comments: false,
        capabilities,
    });
    let module = frontend.parse(STORAGE_BUFFER_WGSL).unwrap();
    let module_info =
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
            .validate(&module)
            .unwrap();
    let function = &module.entry_points[0].function;
    let function_info = module_info.get_entry_point(0);
    let mut observed = Vec::new();

    for (_, expression) in function.expressions.iter() {
        let naga::Expression::Access { base, index } = *expression else {
            continue;
        };
        if function_info[index].uniformity.non_uniform_result.is_none() {
            continue;
        }
        let naga::Expression::GlobalVariable(global_handle) = function.expressions[base] else {
            continue;
        };
        let global = &module.global_variables[global_handle];
        if !matches!(
            module.types[global.ty].inner,
            naga::TypeInner::BindingArray { .. }
        ) {
            continue;
        }
        assert!(
            matches!(
                function_info[base].ty.inner_with(&module.types),
                naga::TypeInner::Pointer { base, .. } if *base == global.ty
            ),
            "pinned Naga exposes non-handle resource globals as pointers to their declared binding-array type"
        );
        observed.push(global.space);
    }

    assert_eq!(
        observed.as_slice(),
        &[naga::AddressSpace::Storage {
            access: naga::StorageAccess::LOAD,
        }],
        "pinned Naga must expose the storage-buffer array access as one non-uniform binding-array access"
    );
}

fn storage_buffer_graph() -> (GpuPreparedWorkGraph, GpuReadbackId) {
    assert_pinned_naga_storage_buffer_non_uniformity();
    let mut resources = GpuResourceScope::new();
    let first = fixed::prepared_u32_buffer(&mut resources, "r3 storage first", 17, false);
    let second = fixed::prepared_u32_buffer(&mut resources, "r3 storage second", 101, false);
    let output =
        prepared_u32_pair_buffer(&mut resources, "r3 storage divergent output", [0, 0], true);

    let pipeline = fixed::compute_pipeline(
        "proof.r3.binding-array-non-uniform.storage-buffer",
        STORAGE_BUFFER_WGSL,
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    );
    assert!(
        matches!(
            pipeline
                .requirements()
                .get(GpuCapabilityFeature::StorageBufferBindingArrayNonUniformIndexing),
            Some(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::StorageBufferBindingArrayNonUniformIndexing
            ))
        ),
        "storage-buffer non-uniform requirement missing from {:?}",
        pipeline.requirements()
    );

    let layout = pipeline.layout().clone();
    let partial = GpuRuntimeBindingValue::new(
        GpuBindingKey::try_new(0, 0).unwrap(),
        [GpuRuntimeBindingResource::Buffer(
            GpuRuntimeBufferBinding::whole(&first),
        )],
    )
    .unwrap();
    assert!(
        GpuRuntimeBindingSet::new(
            layout.clone(),
            [partial, GpuRuntimeBindingValue::whole_buffer(0, 1, &output),],
        )
        .is_err(),
        "non-uniform indexing must retain #99 full fixed-array occupancy"
    );

    let bindings = GpuRuntimeBindingSet::new(
        layout,
        [
            fixed::array_binding([&first, &second]),
            GpuRuntimeBindingValue::whole_buffer(0, 1, &output),
        ],
    )
    .unwrap();
    let compute = GpuComputeOperation::new(
        pipeline,
        bindings,
        GpuDispatchIntent::direct(GpuDispatchSize::new(1, 1, 1)),
    )
    .unwrap();
    let region = GpuBufferRegion::new(&output, GpuBufferRange::whole(&output).unwrap()).unwrap();
    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(region.into(), readback_id).unwrap();
    let fragment = GpuWorkFragment::build("r3 divergent storage-buffer array proof", |work| {
        work.operation("select divergent storage-buffer elements", compute)?;
        work.operation("read divergent storage-buffer output", readback)?;
        Ok(())
    })
    .unwrap();
    (
        GpuPreparedWorkGraph::prepare(
            fixed::label("r3 divergent storage-buffer array graph"),
            [fragment],
        )
        .unwrap(),
        readback_id,
    )
}

fn texture_sampler_graph() -> (GpuPreparedWorkGraph, GpuReadbackId) {
    let mut resources = GpuResourceScope::new();
    let (_first_texture, first_view) = two_pixel_texture(
        &mut resources,
        "r3 texture array first",
        [17, 0, 0, 255, 101, 0, 0, 255],
    );
    let (_second_texture, second_view) = two_pixel_texture(
        &mut resources,
        "r3 texture array second",
        [41, 0, 0, 255, 203, 0, 0, 255],
    );
    let first_sampler = fixed::filtering_sampler(
        &mut resources,
        "r3 sampler array clamp",
        GpuAddressMode::ClampToEdge,
    );
    let second_sampler = fixed::filtering_sampler(
        &mut resources,
        "r3 sampler array repeat",
        GpuAddressMode::Repeat,
    );
    let output = prepared_u32_pair_buffer(
        &mut resources,
        "r3 texture-sampler divergent output",
        [0, 0],
        true,
    );

    let pipeline = fixed::compute_pipeline(
        "proof.r3.binding-array-non-uniform.texture-sampler",
        TEXTURE_SAMPLER_WGSL,
        [
            GpuBindingLayoutRefinement::new(GpuBindingKey::try_new(0, 0).unwrap())
                .with_texture_sample_class(GpuTextureSampleClass::FloatFilterable),
            GpuBindingLayoutRefinement::new(GpuBindingKey::try_new(0, 1).unwrap())
                .with_sampler_class(GpuSamplerClass::Filtering),
        ],
    );
    assert!(matches!(
        pipeline
            .requirements()
            .get(GpuCapabilityFeature::TextureBindingArrayNonUniformIndexing),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::TextureBindingArrayNonUniformIndexing
        ))
    ));

    let bindings = GpuRuntimeBindingSet::new(
        pipeline.layout().clone(),
        [
            fixed::texture_array_binding(0, [&first_view, &second_view]),
            fixed::sampler_array_binding(1, [&first_sampler, &second_sampler]),
            GpuRuntimeBindingValue::whole_buffer(0, 2, &output),
        ],
    )
    .unwrap();
    let compute = GpuComputeOperation::new(
        pipeline,
        bindings,
        GpuDispatchIntent::direct(GpuDispatchSize::new(1, 1, 1)),
    )
    .unwrap();
    let region = GpuBufferRegion::new(&output, GpuBufferRange::whole(&output).unwrap()).unwrap();
    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(region.into(), readback_id).unwrap();
    let fragment = GpuWorkFragment::build("r3 divergent texture-sampler array proof", |work| {
        work.operation("sample divergent texture and sampler elements", compute)?;
        work.operation("read divergent texture-sampler output", readback)?;
        Ok(())
    })
    .unwrap();
    (
        GpuPreparedWorkGraph::prepare(
            fixed::label("r3 divergent texture-sampler array graph"),
            [fragment],
        )
        .unwrap(),
        readback_id,
    )
}

fn storage_texture_graph() -> (GpuPreparedWorkGraph, [GpuReadbackId; 2]) {
    let mut resources = GpuResourceScope::new();
    let (first_texture, first_view) =
        fixed::zeroed_storage_texture(&mut resources, "r3 storage texture first");
    let (second_texture, second_view) =
        fixed::zeroed_storage_texture(&mut resources, "r3 storage texture second");

    let pipeline = fixed::compute_pipeline(
        "proof.r3.binding-array-non-uniform.storage-texture",
        STORAGE_TEXTURE_WGSL,
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    );
    assert!(matches!(
        pipeline
            .requirements()
            .get(GpuCapabilityFeature::StorageTextureBindingArrayNonUniformIndexing),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::StorageTextureBindingArrayNonUniformIndexing
        ))
    ));
    let bindings = GpuRuntimeBindingSet::new(
        pipeline.layout().clone(),
        [fixed::texture_array_binding(0, [&first_view, &second_view])],
    )
    .unwrap();
    let compute = GpuComputeOperation::new(
        pipeline,
        bindings,
        GpuDispatchIntent::direct(GpuDispatchSize::new(1, 1, 1)),
    )
    .unwrap();

    let readback = |texture: &GpuTextureHandle| {
        let region = GpuTextureCopyRegion::new(
            texture,
            0,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::Color,
            GpuCopyExtent::new(1, 1, 1).unwrap(),
        )
        .unwrap();
        let id = GpuReadbackId::allocate().unwrap();
        (id, GpuReadbackOperation::new(region.into(), id).unwrap())
    };
    let (first_id, first_readback) = readback(&first_texture);
    let (second_id, second_readback) = readback(&second_texture);
    let fragment = GpuWorkFragment::build("r3 divergent storage-texture array proof", |work| {
        work.operation("write divergent storage-texture elements", compute)?;
        work.operation("read first divergent storage texture", first_readback)?;
        work.operation("read second divergent storage texture", second_readback)?;
        Ok(())
    })
    .unwrap();
    (
        GpuPreparedWorkGraph::prepare(
            fixed::label("r3 divergent storage-texture array graph"),
            [fragment],
        )
        .unwrap(),
        [first_id, second_id],
    )
}

async fn execute_u32_pair(
    context: &GpuContext,
    graph: GpuPreparedWorkGraph,
    readback_id: GpuReadbackId,
    label: &str,
) -> [u32; 2] {
    let prepared = context.prepare_submission(graph).await.unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let bytes = readback_wait::wait_for_readback(context, &submission, readback_id, label).await;
    assert_eq!(bytes.as_bytes().len(), 8);
    [
        u32::from_le_bytes(bytes.as_bytes()[0..4].try_into().unwrap()),
        u32::from_le_bytes(bytes.as_bytes()[4..8].try_into().unwrap()),
    ]
}

fn supports_all(
    facts: &GpuAdapterFacts,
    features: impl IntoIterator<Item = GpuCapabilityFeature>,
) -> bool {
    features
        .into_iter()
        .all(|feature| facts.supported().supports(feature))
}

pub(crate) async fn run_storage_buffer_proof(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
    expected_adapter: &GpuAdapterFacts,
) -> bool {
    if !supports_all(
        expected_adapter,
        [
            GpuCapabilityFeature::BufferBindingArray,
            GpuCapabilityFeature::StorageResourceBindingArray,
            GpuCapabilityFeature::StorageBufferBindingArrayNonUniformIndexing,
        ],
    ) {
        println!("R3 non-uniform storage-buffer arrays: UNSUPPORTED");
        return false;
    }

    let (graph, readback_id) = storage_buffer_graph();
    let baseline = fixed::request_proof_context(
        fixed::context_descriptor(
            backend,
            fallback,
            GpuCapabilityProfile::ComputeBaseline.requirements(),
        ),
        Some(expected_adapter),
        "R3 non-uniform storage-buffer baseline context must remain available",
    )
    .await;
    assert!(
        baseline.prepare_submission(graph.clone()).await.is_err(),
        "non-uniform storage-buffer work must reject before realization on a baseline context"
    );

    let context = fixed::request_proof_context(
        fixed::context_descriptor(backend, fallback, graph.requirements().clone()),
        Some(expected_adapter),
        "R3 non-uniform storage-buffer requirements must admit the qualified adapter",
    )
    .await;
    assert!(
        context
            .device_facts()
            .is_enabled(GpuCapabilityFeature::StorageBufferBindingArrayNonUniformIndexing)
    );
    assert_eq!(
        execute_u32_pair(
            &context,
            graph,
            readback_id,
            "R3 non-uniform storage-buffer execution",
        )
        .await,
        [17, 101]
    );
    println!("R3 non-uniform storage-buffer arrays: EXERCISED");
    true
}

pub(crate) async fn run_texture_sampler_proof(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
    expected_adapter: &GpuAdapterFacts,
) -> bool {
    if !supports_all(
        expected_adapter,
        [
            GpuCapabilityFeature::TextureBindingArray,
            GpuCapabilityFeature::TextureBindingArrayNonUniformIndexing,
        ],
    ) {
        println!("R3 non-uniform texture/sampler arrays: UNSUPPORTED");
        return false;
    }

    let (graph, readback_id) = texture_sampler_graph();
    let context = fixed::request_proof_context(
        fixed::context_descriptor_with_roles(
            backend,
            fallback,
            graph.requirements().clone(),
            [
                (GpuTextureFormat::Rgba8Unorm, GpuFormatRole::Sampled),
                (GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopyDestination),
            ],
        ),
        Some(expected_adapter),
        "R3 non-uniform texture/sampler requirements must admit the qualified adapter",
    )
    .await;
    assert!(
        context
            .device_facts()
            .is_enabled(GpuCapabilityFeature::TextureBindingArrayNonUniformIndexing)
    );
    assert_eq!(
        execute_u32_pair(
            &context,
            graph,
            readback_id,
            "R3 non-uniform texture/sampler execution",
        )
        .await,
        [101, 41],
        "divergent texture/sampler routing must fail if elements are duplicated or swapped"
    );
    println!("R3 non-uniform texture/sampler arrays: EXERCISED");
    true
}

pub(crate) async fn run_storage_texture_proof(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
    expected_adapter: &GpuAdapterFacts,
) -> bool {
    if !supports_all(
        expected_adapter,
        [
            GpuCapabilityFeature::TextureBindingArray,
            GpuCapabilityFeature::StorageResourceBindingArray,
            GpuCapabilityFeature::StorageTextureBindingArrayNonUniformIndexing,
        ],
    ) {
        println!("R3 non-uniform storage-texture arrays: UNSUPPORTED");
        return false;
    }

    let (graph, readback_ids) = storage_texture_graph();
    let context = fixed::request_proof_context(
        fixed::context_descriptor_with_roles(
            backend,
            fallback,
            graph.requirements().clone(),
            [
                (GpuTextureFormat::R32Uint, GpuFormatRole::StorageWrite),
                (GpuTextureFormat::R32Uint, GpuFormatRole::CopySource),
            ],
        ),
        Some(expected_adapter),
        "R3 non-uniform storage-texture requirements must admit the qualified adapter",
    )
    .await;
    assert!(
        context
            .device_facts()
            .is_enabled(GpuCapabilityFeature::StorageTextureBindingArrayNonUniformIndexing)
    );

    let prepared = context.prepare_submission(graph).await.unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let first = readback_wait::wait_for_readback(
        &context,
        &submission,
        readback_ids[0],
        "R3 first non-uniform storage texture",
    )
    .await;
    let second = readback_wait::wait_for_readback(
        &context,
        &submission,
        readback_ids[1],
        "R3 second non-uniform storage texture",
    )
    .await;
    assert_eq!(first.as_bytes().len(), 4);
    assert_eq!(second.as_bytes().len(), 4);
    assert_eq!(u32::from_le_bytes(first.as_bytes().try_into().unwrap()), 17);
    assert_eq!(
        u32::from_le_bytes(second.as_bytes().try_into().unwrap()),
        101
    );
    println!("R3 non-uniform storage-texture arrays: EXERCISED");
    true
}

pub(crate) async fn run_suite_on_adapter(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
    expected_adapter: &GpuAdapterFacts,
) -> NonUniformBindingArrayProof {
    NonUniformBindingArrayProof {
        texture_sampler: run_texture_sampler_proof(backend, fallback, expected_adapter).await,
        storage_buffer: run_storage_buffer_proof(backend, fallback, expected_adapter).await,
        storage_texture: run_storage_texture_proof(backend, fallback, expected_adapter).await,
    }
}

#[allow(dead_code)]
pub(crate) async fn run_suite(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
) -> NonUniformBindingArrayProof {
    let anchor = GpuContext::request(fixed::context_descriptor(
        backend,
        fallback,
        GpuCapabilityProfile::ComputeBaseline.requirements(),
    ))
    .await
    .expect("R3 non-uniform suite requires one anchor adapter");

    let retained_lavapipe = backend == GpuBackendFamily::Vulkan
        && anchor
            .adapter_facts()
            .diagnostic_name()
            .is_some_and(|name| {
                let name = name.to_ascii_lowercase();
                name.contains("llvmpipe") || name.contains("lavapipe")
            });
    if retained_lavapipe {
        println!(
            "R3 non-uniform storage-buffer arrays: UNQUALIFIED (accepted retained Lavapipe fixed-buffer-array driver exception)"
        );
        return NonUniformBindingArrayProof {
            texture_sampler: run_texture_sampler_proof(backend, fallback, anchor.adapter_facts())
                .await,
            storage_buffer: false,
            storage_texture: run_storage_texture_proof(backend, fallback, anchor.adapter_facts())
                .await,
        };
    }

    run_suite_on_adapter(backend, fallback, anchor.adapter_facts()).await
}

#[allow(dead_code)]
pub(crate) async fn prove_browser_webgpu_unsupported_contract() {
    let baseline = GpuContext::request(fixed::context_descriptor(
        GpuBackendFamily::BrowserWebGpu,
        None,
        GpuCapabilityProfile::ComputeBaseline.requirements(),
    ))
    .await
    .expect("actual-browser R3 non-uniform proof requires the baseline WebGPU context");

    for feature in [
        GpuCapabilityFeature::TextureBindingArrayNonUniformIndexing,
        GpuCapabilityFeature::StorageBufferBindingArrayNonUniformIndexing,
        GpuCapabilityFeature::StorageTextureBindingArrayNonUniformIndexing,
    ] {
        assert!(
            !baseline.adapter_facts().supported().supports(feature),
            "browser WebGPU must not advertise native non-uniform array feature {feature:?}"
        );
        assert!(
            !baseline.device_facts().is_enabled(feature),
            "browser WebGPU must not enable native non-uniform array feature {feature:?}"
        );

        let mut requirements = GpuCapabilityProfile::ComputeBaseline.requirements();
        requirements
            .insert(GpuCapabilityRequirement::Required(feature))
            .unwrap();
        let rejected = GpuContext::request(fixed::context_descriptor(
            GpuBackendFamily::BrowserWebGpu,
            None,
            requirements,
        ))
        .await
        .expect_err("browser native non-uniform indexing must reject during normalized admission");
        assert_eq!(
            rejected.category(),
            GpuContextRequestErrorCategory::NoAdmissibleCandidate
        );
        assert!(
            rejected.candidate_dispositions().iter().any(|disposition| {
                matches!(
                    disposition,
                    GpuCandidateDisposition::Rejected(report)
                        if report
                            .capability_admission_error()
                            .is_some_and(|error| {
                                error.cause() == GpuCapabilityAdmissionCause::RequiredUnavailable
                                    && error.feature() == Some(feature)
                            })
                )
            }),
            "browser non-uniform rejection must retain typed feature evidence for {feature:?}"
        );
    }

    println!("R3 non-uniform binding arrays: UNSUPPORTED (browser WebGPU native features absent)");
}
