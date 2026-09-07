use runen_gpu::{
    GpuAdapterClass, GpuAlignmentKind, GpuBackendFamily, GpuBufferDescriptor,
    GpuBufferInitialization, GpuBufferUsage, GpuBufferUsages, GpuCapabilityFeature,
    GpuCapabilityProfile, GpuCapabilityRequirement, GpuComputeOperation,
    GpuComputePipelineDescriptor, GpuContext, GpuContextDescriptor, GpuDispatchIntent,
    GpuDispatchSize, GpuFormatRole, GpuLimitConstraint, GpuLimitKind, GpuMemoryIntent,
    GpuPipelineConfiguration, GpuPortabilityPolicy, GpuPowerPreference, GpuPreparedWorkGraph,
    GpuProgramDescriptor, GpuProgramSourceIdentity, GpuProgramSourceKey, GpuProgramSourceOwnerId,
    GpuProgramSourceProvenance, GpuProgramSourceRegistry, GpuProgramSourceRevision,
    GpuReconstruction, GpuResourceCommon, GpuResourceLabel, GpuResourceLifetime,
    GpuResourceProvenance, GpuRuntimeBindingSet, GpuShaderStage, GpuSoftwareFallbackPolicy,
    GpuTextureFormat, GpuWorkFragmentBuilder, GpuWorkOperation, GpuWorkResourceIdAllocator,
};

fn label(value: &str) -> GpuResourceLabel {
    GpuResourceLabel::new(value).expect("provider-proof labels should be valid")
}

fn provenance(value: &str) -> GpuResourceProvenance {
    GpuResourceProvenance::new(label(value), Some(7), Some(label("provider-source-r7")))
}

fn compute_operation() -> GpuWorkOperation {
    let mut registry = GpuProgramSourceRegistry::new(1, 1024).unwrap();
    let source = registry
        .admit_wgsl(
            GpuProgramSourceIdentity::new(
                GpuProgramSourceOwnerId::allocate().unwrap(),
                GpuProgramSourceKey::new("reproducibility.provider.compute").unwrap(),
                GpuProgramSourceRevision::try_from_raw(3).unwrap(),
            ),
            "@compute @workgroup_size(1) fn main() {}",
            GpuProgramSourceProvenance::new("reproducibility-provider-proof", None).unwrap(),
        )
        .unwrap();
    let entry_point = runen_gpu::GpuEntryPointName::new("main").unwrap();
    let program = GpuProgramDescriptor::new(source, [entry_point.clone()], []).unwrap();
    let pipeline = GpuComputePipelineDescriptor::new(
        program,
        entry_point,
        GpuPipelineConfiguration::default(),
    )
    .unwrap();
    let bindings = GpuRuntimeBindingSet::new(pipeline.layout().clone(), []).unwrap();
    GpuWorkOperation::Compute(
        GpuComputeOperation::new(
            pipeline,
            bindings,
            GpuDispatchIntent::direct(GpuDispatchSize::new(1, 1, 1)),
        )
        .unwrap(),
    )
}

#[test]
fn descriptor_exposes_normalized_reproducibility_request_facts() {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::ComputeBaseline.requirements())
            .with_label("provider request")
            .with_provenance("provider-proof")
            .with_power_preference(GpuPowerPreference::HighPerformance)
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Forbid)
            .with_allowed_backends([GpuBackendFamily::Vulkan, GpuBackendFamily::Metal])
            .with_backend_preference([GpuBackendFamily::Metal, GpuBackendFamily::Vulkan])
            .with_allowed_adapter_classes([GpuAdapterClass::Discrete, GpuAdapterClass::Integrated])
            .with_portability_policy(GpuPortabilityPolicy::RequirePortableBaseline)
            .require_limit(GpuLimitKind::MaxVertexBuffers, 4)
            .permit_limit(GpuLimitKind::MaxVertexBuffers, 8)
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
            .require_alignment(GpuAlignmentKind::BytesPerRow, 256);

    assert_eq!(descriptor.label(), Some("provider request"));
    assert_eq!(descriptor.provenance(), Some("provider-proof"));
    assert_eq!(
        descriptor.power_preference(),
        GpuPowerPreference::HighPerformance
    );
    assert_eq!(
        descriptor.fallback_policy(),
        GpuSoftwareFallbackPolicy::Forbid
    );
    assert_eq!(
        descriptor.backend_allowlist().collect::<Vec<_>>(),
        [GpuBackendFamily::Vulkan, GpuBackendFamily::Metal]
    );
    assert_eq!(
        descriptor.backend_preference_order().collect::<Vec<_>>(),
        [GpuBackendFamily::Metal, GpuBackendFamily::Vulkan]
    );
    assert_eq!(
        descriptor.adapter_class_allowlist().collect::<Vec<_>>(),
        [GpuAdapterClass::Discrete, GpuAdapterClass::Integrated]
    );
    assert_eq!(
        descriptor.limit_constraints().collect::<Vec<_>>(),
        [(
            GpuLimitKind::MaxVertexBuffers,
            GpuLimitConstraint {
                minimum: Some(4),
                maximum: Some(8),
            },
        )]
    );
    assert_eq!(
        descriptor.format_role_requirements().collect::<Vec<_>>(),
        [(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)]
    );
    assert_eq!(
        descriptor.alignment_requirements().collect::<Vec<_>>(),
        [(GpuAlignmentKind::BytesPerRow, 256)]
    );
    assert_eq!(
        descriptor.portability_policy(),
        GpuPortabilityPolicy::RequirePortableBaseline
    );
    assert!(descriptor.exact_candidate().is_none());
    assert!(descriptor.requirements().iter().any(|requirement| {
        matches!(
            requirement,
            GpuCapabilityRequirement::Required(GpuCapabilityFeature::Compute)
        )
    }));
}

#[test]
#[ignore = "requires the controlled RunenGPU native conformance GPU environment"]
fn reproducibility_provider_collects_public_context_work_and_identity_facts() {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::ComputeBaseline.requirements())
            .with_label("provider admitted context")
            .with_provenance("provider-proof")
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .with_backend_preference([GpuBackendFamily::Vulkan]);

    let context = pollster::block_on(GpuContext::request(descriptor.clone()))
        .expect("controlled native conformance must admit the Vulkan fallback context");

    assert!(context.id().is_nonzero());
    assert_eq!(
        context.generation(),
        runen_gpu::GpuDeviceGeneration::first()
    );
    assert_eq!(context.adapter_facts().backend(), GpuBackendFamily::Vulkan);
    assert!(
        descriptor
            .backend_allowlist()
            .any(|backend| backend == context.adapter_facts().backend())
    );
    assert_eq!(
        context.adapter_facts(),
        context.admission_report().candidate().adapter()
    );
    assert!(
        context
            .adapter_facts()
            .supported()
            .supports(GpuCapabilityFeature::Compute)
    );
    assert!(
        context
            .device_facts()
            .enabled_features()
            .any(|feature| feature == GpuCapabilityFeature::Compute)
    );
    assert_eq!(
        context.admission_report().candidate().workload_budget(),
        context.device_facts().workload_budget()
    );
    let adapter_limits = context.adapter_facts().adapter_limits().values();
    let device_limits = context.device_facts().device_limits().values();
    let workload_budget = context.device_facts().workload_budget().limits();
    assert!(
        adapter_limits.max_compute_workgroups_per_dimension()
            >= device_limits.max_compute_workgroups_per_dimension()
    );
    assert!(
        device_limits.max_compute_workgroups_per_dimension()
            >= workload_budget.max_compute_workgroups_per_dimension()
    );

    let mut fragment =
        GpuWorkFragmentBuilder::new(label("provider fragment"), provenance("provider"));
    fragment
        .operation(label("provider compute"), compute_operation())
        .unwrap();
    let graph = GpuPreparedWorkGraph::prepare(
        label("provider prepared graph"),
        [fragment.finish().unwrap()],
    )
    .unwrap();

    assert_eq!(graph.label().as_str(), "provider prepared graph");
    assert_eq!(graph.nodes().len(), 1);
    assert!(graph.dependencies().is_empty());
    assert!(graph.initialization().is_empty());
    assert!(graph.outputs().is_empty());
    assert!(graph.requirements().iter().any(|requirement| {
        matches!(
            requirement,
            GpuCapabilityRequirement::Required(GpuCapabilityFeature::Compute)
        )
    }));

    let prepared = &graph.nodes()[0];
    let workload_identity = prepared.id();
    assert_eq!(workload_identity.fragment_ordinal(), 0);
    assert_eq!(workload_identity.local_node(), 1);
    assert_eq!(graph.topological_order(), &[workload_identity]);
    assert_eq!(prepared.fragment_label().as_str(), "provider fragment");

    let GpuWorkOperation::Compute(operation) = prepared.node().operation() else {
        panic!("provider proof should retain the authored compute operation");
    };
    let program = operation.pipeline().program();
    assert!(program.is_same_record(&program.clone()));
    assert_eq!(program.entry_points().len(), 1);
    let entry_point = program
        .entry_points()
        .next()
        .expect("provider proof program should retain one compute entry point");
    assert_eq!(entry_point.name().as_str(), "main");
    assert_eq!(entry_point.stage(), GpuShaderStage::Compute);
    assert_eq!(operation.pipeline().entry_point().as_str(), "main");

    let source = program.source();
    assert_eq!(
        source.identity().key().as_str(),
        "reproducibility.provider.compute"
    );
    assert_eq!(source.identity().revision().get(), 3);
    assert_eq!(
        source.provenance().producer(),
        "reproducibility-provider-proof"
    );
    assert_ne!(source.identity().owner().diagnostic_raw(), 0);
    assert_ne!(source.digest().diagnostic_raw(), 0);

    let resource_label = label("provider resource");
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let resource = allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                GpuResourceCommon::owned(
                    resource_label.clone(),
                    GpuResourceLifetime::Transient,
                    GpuMemoryIntent::Device,
                    GpuReconstruction::SourceBacked,
                    provenance("provider resource"),
                )
                .unwrap(),
                16,
                GpuBufferUsages::new(&resource_label, [GpuBufferUsage::Storage]).unwrap(),
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let correlation = resource.diagnostic_identity();
    assert_eq!(resource.clone().diagnostic_identity(), correlation);
    assert_eq!(
        resource
            .descriptor()
            .common()
            .provenance()
            .source_generation(),
        Some(7)
    );
    assert_eq!(
        resource
            .descriptor()
            .common()
            .provenance()
            .source_revision()
            .map(GpuResourceLabel::as_str),
        Some("provider-source-r7")
    );
}
