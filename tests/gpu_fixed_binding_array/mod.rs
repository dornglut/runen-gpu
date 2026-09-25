use runen_gpu::*;

#[path = "../support/readback_wait.rs"]
mod readback_wait;

const UNUSED_STORAGE_ARRAY_WGSL: &str = r#"
enable wgpu_binding_array;

struct Value {
    value: u32,
}

@group(0) @binding(0)
var<storage, read> unused_inputs: binding_array<Value, 2>;

@compute @workgroup_size(1)
fn cs_main() {}
"#;

const STORAGE_ARRAY_WGSL: &str = r#"
enable wgpu_binding_array;

struct Value {
    value: u32,
}

@group(0) @binding(0)
var<storage, read> inputs: binding_array<Value, 2>;

@group(0) @binding(1)
var<storage, read_write> output: Value;

@compute @workgroup_size(1)
fn cs_main() {
    output.value = inputs[0].value + inputs[1].value;
}
"#;

fn label(value: impl AsRef<str>) -> GpuResourceLabel {
    GpuResourceLabel::new(value.as_ref()).unwrap()
}

fn prepared_u32_buffer(
    resources: &mut GpuResourceScope,
    name: &str,
    value: u32,
    readback: bool,
) -> GpuBufferHandle {
    let data = PreparedGpuData::<TransferData>::ordinary_pod_transfer(name, &[value]).unwrap();
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

fn unused_storage_array_program() -> GpuProgramDescriptor {
    let [source] = admit_static_wgsl_sources([(
        "proof.fixed-binding-array.unused-storage",
        1,
        UNUSED_STORAGE_ARRAY_WGSL,
    )])
    .unwrap();
    let entry = GpuEntryPointName::new("cs_main").unwrap();
    let program = GpuProgramDescriptor::new(
        source,
        [entry],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();
    assert_eq!(program.interface().bindings().count(), 0);
    program
}

fn pipeline() -> GpuComputePipelineDescriptor {
    let [source] =
        admit_static_wgsl_sources([("proof.fixed-binding-array.storage", 1, STORAGE_ARRAY_WGSL)])
            .unwrap();
    let entry = GpuEntryPointName::new("cs_main").unwrap();
    let program = GpuProgramDescriptor::new(
        source,
        [entry.clone()],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();
    assert!(matches!(
        program
            .requirements()
            .get(GpuCapabilityFeature::BufferBindingArray),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::BufferBindingArray
        ))
    ));
    assert!(matches!(
        program
            .requirements()
            .get(GpuCapabilityFeature::StorageResourceBindingArray),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::StorageResourceBindingArray
        ))
    ));
    GpuComputePipelineDescriptor::new(program, entry, GpuPipelineConfiguration::default()).unwrap()
}

fn array_binding(inputs: [&GpuBufferHandle; 2]) -> GpuRuntimeBindingValue {
    GpuRuntimeBindingValue::new(
        GpuBindingKey::try_new(0, 0).unwrap(),
        inputs.into_iter().map(|buffer| {
            GpuRuntimeBindingResource::Buffer(GpuRuntimeBufferBinding::whole(buffer))
        }),
    )
    .unwrap()
}

fn proof_graph() -> (
    GpuPreparedWorkGraph,
    GpuReadbackId,
    GpuPipelineLayoutDescriptor,
) {
    let mut resources = GpuResourceScope::new();
    let left = prepared_u32_buffer(&mut resources, "fixed-array left", 17, false);
    let right = prepared_u32_buffer(&mut resources, "fixed-array right", 25, false);
    let output = prepared_u32_buffer(&mut resources, "fixed-array output", 0, true);

    let pipeline = pipeline();
    let layout = pipeline.layout().clone();
    let bindings = GpuRuntimeBindingSet::new(
        layout.clone(),
        [
            array_binding([&left, &right]),
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
    let output_region =
        GpuBufferRegion::new(&output, GpuBufferRange::whole(&output).unwrap()).unwrap();
    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(output_region.into(), readback_id).unwrap();
    let fragment = GpuWorkFragment::build("fixed binding-array storage proof", |work| {
        work.operation("sum fixed storage-buffer array", compute)?;
        work.operation("read fixed-array output", readback)?;
        Ok(())
    })
    .unwrap();
    (
        GpuPreparedWorkGraph::prepare(label("fixed binding-array storage graph"), [fragment])
            .unwrap(),
        readback_id,
        layout,
    )
}

fn context_descriptor(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
    requirements: GpuCapabilityRequirements,
) -> GpuContextDescriptor {
    let mut descriptor = GpuContextDescriptor::new(requirements)
        .with_allowed_backends([backend])
        .with_label("fixed binding-array execution proof");
    if let Some(fallback) = fallback {
        descriptor = descriptor.with_fallback_policy(fallback);
    }
    descriptor
}

pub(crate) async fn run_storage_buffer_array_proof(
    backend: GpuBackendFamily,
    fallback: Option<GpuSoftwareFallbackPolicy>,
) -> bool {
    let baseline = GpuContext::request(context_descriptor(
        backend,
        fallback,
        GpuCapabilityProfile::ComputeBaseline.requirements(),
    ))
    .await
    .expect("retained fixed-array proof requires the requested baseline context");

    if !baseline
        .adapter_facts()
        .supported()
        .supports(GpuCapabilityFeature::BufferBindingArray)
        || !baseline
            .adapter_facts()
            .supported()
            .supports(GpuCapabilityFeature::StorageResourceBindingArray)
    {
        println!("Fixed binding arrays: UNSUPPORTED (storage-buffer array capability absent)");
        return false;
    }

    let unused_program = unused_storage_array_program();
    let error = baseline
        .realize_program(&unused_program)
        .await
        .expect_err("baseline context must reject unused module-global array requirements");
    assert_eq!(
        error.category(),
        GpuProgramBindingRealizationErrorCategory::RequirementNotAdmitted
    );
    let unused_context = GpuContext::request(context_descriptor(
        backend,
        fallback,
        unused_program.requirements().clone(),
    ))
    .await
    .expect("unused module-global array requirements must admit the capability-bearing context");
    unused_context
        .realize_program(&unused_program)
        .await
        .expect("whole-module shader realization must succeed after admitting its exact requirements");

    let (graph, readback_id, layout) = proof_graph();
    assert!(
        baseline.prepare_submission(graph.clone()).await.is_err(),
        "a baseline context must reject graph-derived fixed-array capabilities"
    );

    let capped = GpuContext::request(
        context_descriptor(backend, fallback, graph.requirements().clone())
            .permit_limit(GpuLimitKind::MaxBindingArrayElementsPerShaderStage, 1),
    )
    .await
    .expect("capability-bearing adapter must admit an explicit one-element workload cap");
    let group = layout.group(0).unwrap();
    let error = capped
        .realize_bind_group_layout(group)
        .await
        .expect_err("a two-element public layout must reject against a one-element workload cap");
    assert_eq!(
        error.category(),
        GpuProgramBindingRealizationErrorCategory::LayoutDescriptorInvalid
    );

    let context = GpuContext::request(context_descriptor(
        backend,
        fallback,
        graph.requirements().clone(),
    ))
    .await
    .expect("graph-derived fixed-array requirements must admit the capability-bearing context");
    assert!(
        context
            .device_facts()
            .workload_budget()
            .limits()
            .max_binding_array_elements_per_shader_stage()
            >= 2
    );

    let prepared = context.prepare_submission(graph).await.unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let bytes = readback_wait::wait_for_readback(
        &context,
        &submission,
        readback_id,
        "fixed binding-array storage execution",
    )
    .await;
    assert_eq!(bytes.as_bytes().len(), 4);
    assert_eq!(u32::from_le_bytes(bytes.as_bytes().try_into().unwrap()), 42);
    println!("Fixed binding arrays: EXERCISED (storage-buffer array + exact readback)");
    true
}
