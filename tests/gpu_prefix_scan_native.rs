//! Retained G5-C01 exact-conformance proof for canonical RunenGPU execution.
//!
//! The fixed 4,097-element fixture deliberately crosses three 64-element scan
//! hierarchy levels: 4,097 -> 65 -> 2 -> 1. The proof uses only public RunenGPU
//! resources, admitted WGSL programs, work authoring, prepare/submit, completion,
//! and readback semantics.

use runen_gpu::*;
use std::time::{Duration, Instant};

const ELEMENT_COUNT: u32 = 4_097;
const WORKGROUP_SIZE: u32 = 64;
const SOURCE_REVISION: u64 = 1;
const SCAN_WGSL: &str = include_str!("gpu_prefix_scan_native/scan.wgsl");
const APPLY_OFFSETS_WGSL: &str = include_str!("gpu_prefix_scan_native/apply_offsets.wgsl");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScanMode {
    Exclusive,
    Inclusive,
}

impl ScanMode {
    const fn key(self) -> &'static str {
        match self {
            Self::Exclusive => "exclusive",
            Self::Inclusive => "inclusive",
        }
    }

    const fn specialization(self) -> u32 {
        match self {
            Self::Exclusive => 0,
            Self::Inclusive => 1,
        }
    }
}

#[derive(Clone)]
pub(crate) struct ProgramSources {
    scan: GpuAdmittedProgramSource,
    apply_offsets: GpuAdmittedProgramSource,
}

fn label(value: impl AsRef<str>) -> GpuResourceLabel {
    GpuResourceLabel::new(value.as_ref()).unwrap()
}

fn blocks_for(element_count: u32) -> u32 {
    element_count.div_ceil(WORKGROUP_SIZE)
}

fn hierarchy_counts() -> [u32; 3] {
    let level_0 = blocks_for(ELEMENT_COUNT);
    let level_1 = blocks_for(level_0);
    let level_2 = blocks_for(level_1);
    [level_0, level_1, level_2]
}

pub(crate) fn admitted_sources() -> ProgramSources {
    let [scan, apply_offsets] = admit_static_wgsl_sources([
        ("proof.prefix-scan.scan", SOURCE_REVISION, SCAN_WGSL),
        (
            "proof.prefix-scan.apply-offsets",
            SOURCE_REVISION,
            APPLY_OFFSETS_WGSL,
        ),
    ])
    .unwrap();
    ProgramSources {
        scan,
        apply_offsets,
    }
}

fn specialization_key(value: &str) -> GpuSpecializationKey {
    GpuSpecializationKey::new(value).unwrap()
}

fn u32_specialization(values: &[(&str, u32)]) -> GpuSpecializationValueSet {
    let declarations = values.iter().map(|(name, _)| {
        GpuSpecializationDeclaration::new(
            specialization_key(name),
            GpuSpecializationValueType::U32,
            Some(GpuSpecializationValue::U32(0)),
            GpuCapabilityRequirements::new(),
        )
        .unwrap()
    });
    let schema = GpuSpecializationSchema::new(declarations).unwrap();
    let entries = values.iter().map(|(name, value)| {
        GpuSpecializationEntry::new(
            specialization_key(name),
            GpuSpecializationValue::U32(*value),
        )
    });
    GpuSpecializationValueSet::new(schema, entries).unwrap()
}

fn compute_pipeline(
    source: &GpuAdmittedProgramSource,
    specialization: GpuSpecializationValueSet,
) -> GpuComputePipelineDescriptor {
    let entry_point = GpuEntryPointName::new("cs_main").unwrap();
    let program = GpuProgramDescriptor::new(
        source.clone(),
        [entry_point.clone()],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();
    GpuComputePipelineDescriptor::new(
        program,
        entry_point,
        GpuPipelineConfiguration::new(Some(specialization), None),
    )
    .unwrap()
}

fn scan_pipeline(
    source: &GpuAdmittedProgramSource,
    element_count: u32,
    mode: ScanMode,
) -> GpuComputePipelineDescriptor {
    compute_pipeline(
        source,
        u32_specialization(&[
            ("ELEMENT_COUNT", element_count),
            ("INCLUSIVE", mode.specialization()),
        ]),
    )
}

fn apply_offsets_pipeline(
    source: &GpuAdmittedProgramSource,
    element_count: u32,
) -> GpuComputePipelineDescriptor {
    compute_pipeline(
        source,
        u32_specialization(&[("ELEMENT_COUNT", element_count)]),
    )
}

fn prepared_u32_buffer(
    resources: &mut GpuResourceScope,
    name: &str,
    values: &[u32],
) -> GpuBufferHandle {
    let data = PreparedGpuData::<TransferData>::ordinary_pod_transfer(name, values).unwrap();
    let byte_len = data.layout().byte_len();
    resources
        .buffer(
            GpuBufferDescriptor::ordinary_owned(
                name,
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                byte_len,
                [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
                GpuBufferInitialization::Prepared(data),
            )
            .unwrap(),
        )
        .unwrap()
}

fn zeroed_u32_buffer(
    resources: &mut GpuResourceScope,
    name: &str,
    element_count: u32,
    readback: bool,
) -> GpuBufferHandle {
    let mut usages = vec![GpuBufferUsage::Storage];
    if readback {
        usages.push(GpuBufferUsage::CopySource);
    }
    resources
        .buffer(
            GpuBufferDescriptor::ordinary_owned(
                name,
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                u64::from(element_count) * u64::try_from(std::mem::size_of::<u32>()).unwrap(),
                usages,
                GpuBufferInitialization::Zeroed,
            )
            .unwrap(),
        )
        .unwrap()
}

fn scan_operation(
    pipeline: &GpuComputePipelineDescriptor,
    input: &GpuBufferHandle,
    output: &GpuBufferHandle,
    block_sums: &GpuBufferHandle,
    workgroups: u32,
) -> GpuComputeOperation {
    let bindings = pipeline
        .runtime_bindings([
            GpuRuntimeBindingValue::whole_buffer(0, 0, input),
            GpuRuntimeBindingValue::whole_buffer(0, 1, output),
            GpuRuntimeBindingValue::whole_buffer(0, 2, block_sums),
        ])
        .unwrap();
    GpuComputeOperation::new(
        pipeline.clone(),
        bindings,
        GpuDispatchIntent::direct(GpuDispatchSize::new(workgroups, 1, 1)),
    )
    .unwrap()
}

fn apply_offsets_operation(
    pipeline: &GpuComputePipelineDescriptor,
    output: &GpuBufferHandle,
    offsets: &GpuBufferHandle,
    workgroups: u32,
) -> GpuComputeOperation {
    let bindings = pipeline
        .runtime_bindings([
            GpuRuntimeBindingValue::whole_buffer(0, 0, output),
            GpuRuntimeBindingValue::whole_buffer(0, 1, offsets),
        ])
        .unwrap();
    GpuComputeOperation::new(
        pipeline.clone(),
        bindings,
        GpuDispatchIntent::direct(GpuDispatchSize::new(workgroups, 1, 1)),
    )
    .unwrap()
}

fn readback_operation(buffer: &GpuBufferHandle) -> GpuReadbackOperation {
    GpuReadbackOperation::ordinary(GpuBufferRegion::whole(buffer).unwrap().into()).unwrap()
}

pub(crate) fn author_scan(
    sources: &ProgramSources,
    mode: ScanMode,
) -> (GpuWorkFragment, GpuReadbackId, GpuReadbackId) {
    let [level_0_blocks, level_1_blocks, level_2_blocks] = hierarchy_counts();
    assert_eq!([level_0_blocks, level_1_blocks, level_2_blocks], [65, 2, 1]);

    let mut resources = GpuResourceScope::new();
    let input_values = vec![1_u32; usize::try_from(ELEMENT_COUNT).unwrap()];
    let input = prepared_u32_buffer(
        &mut resources,
        &format!("prefix scan {} input", mode.key()),
        &input_values,
    );
    let output = zeroed_u32_buffer(
        &mut resources,
        &format!("prefix scan {} output", mode.key()),
        ELEMENT_COUNT,
        true,
    );

    // 4,097 -> 65 -> 2 -> 1 block-sum hierarchy. Upper-level scan outputs are
    // exclusive block offsets consumed during the reverse propagation pass.
    let block_sums_0 = zeroed_u32_buffer(
        &mut resources,
        &format!("prefix scan {} level 0 block sums", mode.key()),
        level_0_blocks,
        false,
    );
    let block_offsets_0 = zeroed_u32_buffer(
        &mut resources,
        &format!("prefix scan {} level 0 block offsets", mode.key()),
        level_0_blocks,
        false,
    );
    let block_sums_1 = zeroed_u32_buffer(
        &mut resources,
        &format!("prefix scan {} level 1 block sums", mode.key()),
        level_1_blocks,
        false,
    );
    let block_offsets_1 = zeroed_u32_buffer(
        &mut resources,
        &format!("prefix scan {} level 1 block offsets", mode.key()),
        level_1_blocks,
        false,
    );
    let final_total = zeroed_u32_buffer(
        &mut resources,
        &format!("prefix scan {} final total", mode.key()),
        level_2_blocks,
        true,
    );

    let scan_0 = scan_pipeline(&sources.scan, ELEMENT_COUNT, mode);
    let scan_1 = scan_pipeline(&sources.scan, level_0_blocks, ScanMode::Exclusive);
    let scan_2 = scan_pipeline(&sources.scan, level_1_blocks, ScanMode::Exclusive);
    let apply_1 = apply_offsets_pipeline(&sources.apply_offsets, level_0_blocks);
    let apply_0 = apply_offsets_pipeline(&sources.apply_offsets, ELEMENT_COUNT);

    let mut output_readback_id = None;
    let mut total_readback_id = None;
    let fragment = GpuWorkFragment::build(
        format!("prefix scan {} exact conformance", mode.key()),
        |work| {
            work.operation(
                "prefix scan level 0",
                scan_operation(&scan_0, &input, &output, &block_sums_0, level_0_blocks),
            )?;
            work.operation(
                "prefix scan level 1",
                scan_operation(
                    &scan_1,
                    &block_sums_0,
                    &block_offsets_0,
                    &block_sums_1,
                    level_1_blocks,
                ),
            )?;
            work.operation(
                "prefix scan level 2",
                scan_operation(
                    &scan_2,
                    &block_sums_1,
                    &block_offsets_1,
                    &final_total,
                    level_2_blocks,
                ),
            )?;
            work.operation(
                "prefix scan apply level 1 offsets",
                apply_offsets_operation(
                    &apply_1,
                    &block_offsets_0,
                    &block_offsets_1,
                    blocks_for(level_0_blocks),
                ),
            )?;
            work.operation(
                "prefix scan apply level 0 offsets",
                apply_offsets_operation(&apply_0, &output, &block_offsets_0, level_0_blocks),
            )?;

            let output_readback = readback_operation(&output);
            output_readback_id = Some(output_readback.id());
            work.operation("prefix scan read full output", output_readback)?;

            let total_readback = readback_operation(&final_total);
            total_readback_id = Some(total_readback.id());
            work.operation("prefix scan read exact total", total_readback)?;
            Ok(())
        },
    )
    .unwrap();

    assert_eq!(fragment.resources().len(), 7);
    assert_eq!(fragment.nodes().len(), 7);
    (
        fragment,
        output_readback_id.unwrap(),
        total_readback_id.unwrap(),
    )
}

fn native_compute_context() -> GpuContext {
    let mut requirements = GpuCapabilityRequirements::new();
    for feature in [GpuCapabilityFeature::Compute, GpuCapabilityFeature::Copy] {
        requirements
            .insert(GpuCapabilityRequirement::Required(feature))
            .unwrap();
    }
    let descriptor = GpuContextDescriptor::new(requirements)
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label("G5-C01 prefix scan exact-conformance proof");
    let context = pollster::block_on(GpuContext::request(descriptor))
        .expect("native conformance environment must provide a Vulkan fallback adapter");
    assert_eq!(context.adapter_facts().backend(), GpuBackendFamily::Vulkan);
    assert_eq!(
        context.adapter_facts().fallback(),
        GpuFallbackStatus::ConfirmedFallback
    );
    context
}

fn progress_to_readbacks(
    context: &GpuContext,
    submission: &GpuSubmission,
    ids: [GpuReadbackId; 2],
) -> Vec<GpuReadbackBytes> {
    let handles = ids
        .into_iter()
        .map(|id| {
            submission
                .readback(id)
                .expect("prefix-scan readback must remain observable")
                .clone()
        })
        .collect::<Vec<_>>();
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        context.progress();
        let mut all_ready = true;
        for handle in &handles {
            match handle.status() {
                GpuReadbackStatus::Ready(_) => {}
                GpuReadbackStatus::Failed(failure) => {
                    panic!("prefix-scan readback failed: {failure:?}")
                }
                GpuReadbackStatus::Pending => all_ready = false,
            }
        }
        match submission.status() {
            GpuSubmissionStatus::Failed(failure) => {
                panic!("prefix-scan submission failed: {failure:?}")
            }
            GpuSubmissionStatus::Completed if all_ready => break,
            GpuSubmissionStatus::Accepted | GpuSubmissionStatus::Completed => {}
        }
        assert!(Instant::now() < deadline, "prefix-scan proof timed out");
        std::thread::yield_now();
    }
    handles
        .into_iter()
        .map(|handle| match handle.status() {
            GpuReadbackStatus::Ready(bytes) => bytes,
            other => panic!("terminal prefix-scan readback must be ready, got {other:?}"),
        })
        .collect()
}

pub(crate) fn decode_u32(bytes: &GpuReadbackBytes) -> Vec<u32> {
    let mut chunks = bytes.as_bytes().chunks_exact(4);
    assert!(chunks.remainder().is_empty());
    chunks
        .by_ref()
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

pub(crate) fn assert_exact_output(mode: ScanMode, output: &[u32], total: &[u32]) {
    assert_eq!(output.len(), usize::try_from(ELEMENT_COUNT).unwrap());
    for (index, value) in output.iter().copied().enumerate() {
        let index = u32::try_from(index).unwrap();
        let expected = match mode {
            ScanMode::Exclusive => index,
            ScanMode::Inclusive => index + 1,
        };
        assert_eq!(
            value,
            expected,
            "{} prefix-scan mismatch at element {index}",
            mode.key()
        );
    }
    assert_eq!(total, [ELEMENT_COUNT]);
}

fn run_mode(context: &GpuContext, sources: &ProgramSources, mode: ScanMode) {
    let (fragment, output_id, total_id) = author_scan(sources, mode);
    let graph = GpuPreparedWorkGraph::prepare(
        label(format!("prefix scan {} prepared graph", mode.key())),
        [fragment],
    )
    .unwrap();
    assert_eq!(graph.nodes().len(), 7);
    assert_eq!(graph.topological_order().len(), 7);
    assert!(graph.initialization().len() >= 7);
    assert!(
        graph
            .dependencies()
            .iter()
            .flat_map(|dependency| dependency.reasons())
            .all(|reason| !matches!(reason, GpuDependencyReason::ExplicitNonData { .. })),
        "prefix-scan hierarchy must be ordered by typed resource hazards, not manual ordering"
    );
    for feature in [GpuCapabilityFeature::Compute, GpuCapabilityFeature::Copy] {
        assert!(graph.requirements().get(feature).is_some());
    }

    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let readbacks = progress_to_readbacks(context, &submission, [output_id, total_id]);
    assert_eq!(readbacks.len(), 2);
    let output = decode_u32(&readbacks[0]);
    let total = decode_u32(&readbacks[1]);
    assert_exact_output(mode, &output, &total);
}

#[test]
#[ignore = "requires a real Vulkan fallback adapter; executed by RunenGPU Conformance CI"]
fn native_prefix_scan_proves_exact_4097_element_inclusive_and_exclusive_results() {
    assert_eq!(hierarchy_counts(), [65, 2, 1]);
    let context = native_compute_context();
    let sources = admitted_sources();

    run_mode(&context, &sources, ScanMode::Exclusive);
    run_mode(&context, &sources, ScanMode::Inclusive);

    let stats = context.execution_stats();
    assert_eq!(stats.prepared_submissions(), 0);
    assert_eq!(stats.in_flight_submissions(), 0);
    assert_eq!(stats.upload_bytes_in_flight(), 0);
    assert_eq!(stats.readback_bytes_in_flight(), 0);
    assert_eq!(stats.pending_readbacks(), 0);
}
