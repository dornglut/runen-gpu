use super::common::{
    DirectWgpuContext, MEASURED_SAMPLES, Measurements, WARMUP_SAMPLES, micros, ratio_summary,
    submit_and_map,
};
use runen_gpu::*;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

const ELEMENT_COUNT: u32 = 4_097;
const WORKGROUP_SIZE: u32 = 64;
const SOURCE_REVISION: u64 = 1;
const SCAN_WGSL: &str = include_str!("../gpu_prefix_scan_native/scan.wgsl");
const APPLY_OFFSETS_WGSL: &str = include_str!("../gpu_prefix_scan_native/apply_offsets.wgsl");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScanMode {
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
struct ProgramSources {
    scan: GpuAdmittedProgramSource,
    apply_offsets: GpuAdmittedProgramSource,
}

struct RunenGpuPipelines {
    scan_0_exclusive: GpuComputePipelineDescriptor,
    scan_0_inclusive: GpuComputePipelineDescriptor,
    scan_1: GpuComputePipelineDescriptor,
    scan_2: GpuComputePipelineDescriptor,
    apply_1: GpuComputePipelineDescriptor,
    apply_0: GpuComputePipelineDescriptor,
}

impl RunenGpuPipelines {
    fn scan_0(&self, mode: ScanMode) -> &GpuComputePipelineDescriptor {
        match mode {
            ScanMode::Exclusive => &self.scan_0_exclusive,
            ScanMode::Inclusive => &self.scan_0_inclusive,
        }
    }
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

fn admitted_sources() -> ProgramSources {
    let [scan, apply_offsets] = admit_static_wgsl_sources([
        (
            "proof.direct-cost.prefix-scan.scan",
            SOURCE_REVISION,
            SCAN_WGSL,
        ),
        (
            "proof.direct-cost.prefix-scan.apply-offsets",
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

fn runengpu_pipelines(sources: &ProgramSources) -> RunenGpuPipelines {
    let [level_0_blocks, level_1_blocks, _] = hierarchy_counts();
    RunenGpuPipelines {
        scan_0_exclusive: scan_pipeline(&sources.scan, ELEMENT_COUNT, ScanMode::Exclusive),
        scan_0_inclusive: scan_pipeline(&sources.scan, ELEMENT_COUNT, ScanMode::Inclusive),
        scan_1: scan_pipeline(&sources.scan, level_0_blocks, ScanMode::Exclusive),
        scan_2: scan_pipeline(&sources.scan, level_1_blocks, ScanMode::Exclusive),
        apply_1: apply_offsets_pipeline(&sources.apply_offsets, level_0_blocks),
        apply_0: apply_offsets_pipeline(&sources.apply_offsets, ELEMENT_COUNT),
    }
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

fn author_scan(
    pipelines: &RunenGpuPipelines,
    mode: ScanMode,
) -> (GpuWorkFragment, GpuReadbackId, GpuReadbackId) {
    let [level_0_blocks, level_1_blocks, level_2_blocks] = hierarchy_counts();
    assert_eq!([level_0_blocks, level_1_blocks, level_2_blocks], [65, 2, 1]);

    let mut resources = GpuResourceScope::new();
    let input_values = vec![1_u32; usize::try_from(ELEMENT_COUNT).unwrap()];
    let input = prepared_u32_buffer(
        &mut resources,
        &format!("direct-cost prefix scan {} input", mode.key()),
        &input_values,
    );
    let output = zeroed_u32_buffer(
        &mut resources,
        &format!("direct-cost prefix scan {} output", mode.key()),
        ELEMENT_COUNT,
        true,
    );
    let block_sums_0 = zeroed_u32_buffer(
        &mut resources,
        &format!("direct-cost prefix scan {} level 0 block sums", mode.key()),
        level_0_blocks,
        false,
    );
    let block_offsets_0 = zeroed_u32_buffer(
        &mut resources,
        &format!(
            "direct-cost prefix scan {} level 0 block offsets",
            mode.key()
        ),
        level_0_blocks,
        false,
    );
    let block_sums_1 = zeroed_u32_buffer(
        &mut resources,
        &format!("direct-cost prefix scan {} level 1 block sums", mode.key()),
        level_1_blocks,
        false,
    );
    let block_offsets_1 = zeroed_u32_buffer(
        &mut resources,
        &format!(
            "direct-cost prefix scan {} level 1 block offsets",
            mode.key()
        ),
        level_1_blocks,
        false,
    );
    let final_total = zeroed_u32_buffer(
        &mut resources,
        &format!("direct-cost prefix scan {} final total", mode.key()),
        level_2_blocks,
        true,
    );

    let mut output_readback_id = None;
    let mut total_readback_id = None;
    let fragment = GpuWorkFragment::build(
        format!("direct-cost prefix scan {} comparison", mode.key()),
        |work| {
            work.operation(
                "prefix scan level 0",
                scan_operation(
                    pipelines.scan_0(mode),
                    &input,
                    &output,
                    &block_sums_0,
                    level_0_blocks,
                ),
            )?;
            work.operation(
                "prefix scan level 1",
                scan_operation(
                    &pipelines.scan_1,
                    &block_sums_0,
                    &block_offsets_0,
                    &block_sums_1,
                    level_1_blocks,
                ),
            )?;
            work.operation(
                "prefix scan level 2",
                scan_operation(
                    &pipelines.scan_2,
                    &block_sums_1,
                    &block_offsets_1,
                    &final_total,
                    level_2_blocks,
                ),
            )?;
            work.operation(
                "prefix scan apply level 1 offsets",
                apply_offsets_operation(
                    &pipelines.apply_1,
                    &block_offsets_0,
                    &block_offsets_1,
                    blocks_for(level_0_blocks),
                ),
            )?;
            work.operation(
                "prefix scan apply level 0 offsets",
                apply_offsets_operation(
                    &pipelines.apply_0,
                    &output,
                    &block_offsets_0,
                    level_0_blocks,
                ),
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

fn runengpu_context() -> GpuContext {
    let mut requirements = GpuCapabilityRequirements::new();
    for feature in [GpuCapabilityFeature::Compute, GpuCapabilityFeature::Copy] {
        requirements
            .insert(GpuCapabilityRequirement::Required(feature))
            .unwrap();
    }
    let descriptor = GpuContextDescriptor::new(requirements)
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label("G6-P01 prefix scan RunenGPU comparison");
    let context = pollster::block_on(GpuContext::request(descriptor)).unwrap();
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
        .map(|id| submission.readback(id).unwrap().clone())
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
        assert!(
            Instant::now() < deadline,
            "prefix-scan comparison timed out"
        );
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

fn decode_u32_bytes(bytes: &[u8]) -> Vec<u32> {
    let words = bytes.chunks_exact(4);
    assert!(words.remainder().is_empty());
    words
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

fn assert_exact_output(mode: ScanMode, output: &[u32], total: &[u32]) {
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

fn runengpu_mode_sample(
    context: &GpuContext,
    pipelines: &RunenGpuPipelines,
    mode: ScanMode,
) -> BTreeMap<String, f64> {
    let total_start = Instant::now();

    let author_start = Instant::now();
    let (fragment, output_id, total_id) = author_scan(pipelines, mode);
    let authoring_us = micros(author_start.elapsed());

    let prepare_start = Instant::now();
    let graph = GpuPreparedWorkGraph::prepare(
        label(format!("G6-P01 prefix scan {} graph", mode.key())),
        [fragment],
    )
    .unwrap();
    assert_eq!(graph.nodes().len(), 7);
    let graph_prepare_us = micros(prepare_start.elapsed());

    let backend_start = Instant::now();
    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let backend_prepare_us = micros(backend_start.elapsed());

    let submit_start = Instant::now();
    let submission = context.submit_prepared(prepared).unwrap();
    let submit_call_us = micros(submit_start.elapsed());

    let completion_start = Instant::now();
    let readbacks = progress_to_readbacks(context, &submission, [output_id, total_id]);
    let completion_readback_us = micros(completion_start.elapsed());

    let output = decode_u32_bytes(readbacks[0].as_bytes());
    let total = decode_u32_bytes(readbacks[1].as_bytes());
    assert_exact_output(mode, &output, &total);

    BTreeMap::from([
        ("authoring".to_owned(), authoring_us),
        ("graph_prepare".to_owned(), graph_prepare_us),
        ("backend_prepare".to_owned(), backend_prepare_us),
        (
            "boundary_prepare_or_record".to_owned(),
            graph_prepare_us + backend_prepare_us,
        ),
        ("submit_encode_and_queue".to_owned(), submit_call_us),
        (
            "boundary_prepare_record_submit".to_owned(),
            graph_prepare_us + backend_prepare_us + submit_call_us,
        ),
        ("completion_readback".to_owned(), completion_readback_us),
        ("total".to_owned(), micros(total_start.elapsed())),
    ])
}

fn sum_phases(
    first: BTreeMap<String, f64>,
    second: BTreeMap<String, f64>,
) -> BTreeMap<String, f64> {
    let mut result = first;
    for (phase, value) in second {
        *result.entry(phase).or_insert(0.0) += value;
    }
    result
}

fn runengpu_sample(context: &GpuContext, pipelines: &RunenGpuPipelines) -> BTreeMap<String, f64> {
    let total_start = Instant::now();
    let exclusive = runengpu_mode_sample(context, pipelines, ScanMode::Exclusive);
    let inclusive = runengpu_mode_sample(context, pipelines, ScanMode::Inclusive);
    let mut phases = sum_phases(exclusive, inclusive);
    phases.insert("total".to_owned(), micros(total_start.elapsed()));
    phases
}

struct DirectPipelines {
    scan_0_exclusive: wgpu::ComputePipeline,
    scan_0_inclusive: wgpu::ComputePipeline,
    scan_1: wgpu::ComputePipeline,
    scan_2: wgpu::ComputePipeline,
    apply_1: wgpu::ComputePipeline,
    apply_0: wgpu::ComputePipeline,
    cold_pipeline_us: f64,
}

impl DirectPipelines {
    fn scan_0(&self, mode: ScanMode) -> &wgpu::ComputePipeline {
        match mode {
            ScanMode::Exclusive => &self.scan_0_exclusive,
            ScanMode::Inclusive => &self.scan_0_inclusive,
        }
    }
}

fn direct_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    shader: &wgpu::ShaderModule,
    constants: &[(&str, f64)],
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: shader,
        entry_point: Some("cs_main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants,
            ..Default::default()
        },
        cache: None,
    })
}

fn direct_pipelines(context: &DirectWgpuContext) -> DirectPipelines {
    let start = Instant::now();
    let scan_shader = context
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("G6-P01 retained prefix scan shader"),
            source: wgpu::ShaderSource::Wgsl(SCAN_WGSL.into()),
        });
    let apply_shader = context
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("G6-P01 retained prefix apply-offsets shader"),
            source: wgpu::ShaderSource::Wgsl(APPLY_OFFSETS_WGSL.into()),
        });
    let [level_0_blocks, level_1_blocks, _] = hierarchy_counts();

    let scan_0_exclusive_constants = [
        ("ELEMENT_COUNT", f64::from(ELEMENT_COUNT)),
        ("INCLUSIVE", 0.0),
    ];
    let scan_0_inclusive_constants = [
        ("ELEMENT_COUNT", f64::from(ELEMENT_COUNT)),
        ("INCLUSIVE", 1.0),
    ];
    let scan_1_constants = [
        ("ELEMENT_COUNT", f64::from(level_0_blocks)),
        ("INCLUSIVE", 0.0),
    ];
    let scan_2_constants = [
        ("ELEMENT_COUNT", f64::from(level_1_blocks)),
        ("INCLUSIVE", 0.0),
    ];
    let apply_1_constants = [("ELEMENT_COUNT", f64::from(level_0_blocks))];
    let apply_0_constants = [("ELEMENT_COUNT", f64::from(ELEMENT_COUNT))];

    let scan_0_exclusive = direct_compute_pipeline(
        &context.device,
        "G6-P01 prefix scan level 0 exclusive",
        &scan_shader,
        &scan_0_exclusive_constants,
    );
    let scan_0_inclusive = direct_compute_pipeline(
        &context.device,
        "G6-P01 prefix scan level 0 inclusive",
        &scan_shader,
        &scan_0_inclusive_constants,
    );
    let scan_1 = direct_compute_pipeline(
        &context.device,
        "G6-P01 prefix scan level 1 exclusive",
        &scan_shader,
        &scan_1_constants,
    );
    let scan_2 = direct_compute_pipeline(
        &context.device,
        "G6-P01 prefix scan level 2 exclusive",
        &scan_shader,
        &scan_2_constants,
    );
    let apply_1 = direct_compute_pipeline(
        &context.device,
        "G6-P01 prefix scan apply level 1",
        &apply_shader,
        &apply_1_constants,
    );
    let apply_0 = direct_compute_pipeline(
        &context.device,
        "G6-P01 prefix scan apply level 0",
        &apply_shader,
        &apply_0_constants,
    );

    DirectPipelines {
        scan_0_exclusive,
        scan_0_inclusive,
        scan_1,
        scan_2,
        apply_1,
        apply_0,
        cold_pipeline_us: micros(start.elapsed()),
    }
}

struct DirectResources {
    input: wgpu::Buffer,
    input_upload: wgpu::Buffer,
    output: wgpu::Buffer,
    block_sums_0: wgpu::Buffer,
    block_offsets_0: wgpu::Buffer,
    block_sums_1: wgpu::Buffer,
    block_offsets_1: wgpu::Buffer,
    final_total: wgpu::Buffer,
    output_readback: wgpu::Buffer,
    total_readback: wgpu::Buffer,
}

fn buffer_size_u32(element_count: u32) -> u64 {
    u64::from(element_count) * u64::try_from(std::mem::size_of::<u32>()).unwrap()
}

fn direct_storage_buffer(
    context: &DirectWgpuContext,
    label: &str,
    element_count: u32,
    copy_source: bool,
) -> wgpu::Buffer {
    let mut usage = wgpu::BufferUsages::STORAGE;
    if copy_source {
        usage |= wgpu::BufferUsages::COPY_SRC;
    }
    context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: buffer_size_u32(element_count),
        usage,
        mapped_at_creation: false,
    })
}

fn direct_resources(context: &DirectWgpuContext, mode: ScanMode) -> DirectResources {
    let [level_0_blocks, level_1_blocks, level_2_blocks] = hierarchy_counts();
    let input_values = vec![1_u32; usize::try_from(ELEMENT_COUNT).unwrap()];
    let input_upload = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("G6-P01 prefix scan input upload"),
            contents: bytemuck::cast_slice(&input_values),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
    let input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("G6-P01 prefix scan input"),
        size: buffer_size_u32(ELEMENT_COUNT),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output = direct_storage_buffer(
        context,
        &format!("G6-P01 prefix scan {} output", mode.key()),
        ELEMENT_COUNT,
        true,
    );
    let block_sums_0 = direct_storage_buffer(
        context,
        "G6-P01 prefix scan level 0 block sums",
        level_0_blocks,
        false,
    );
    let block_offsets_0 = direct_storage_buffer(
        context,
        "G6-P01 prefix scan level 0 block offsets",
        level_0_blocks,
        false,
    );
    let block_sums_1 = direct_storage_buffer(
        context,
        "G6-P01 prefix scan level 1 block sums",
        level_1_blocks,
        false,
    );
    let block_offsets_1 = direct_storage_buffer(
        context,
        "G6-P01 prefix scan level 1 block offsets",
        level_1_blocks,
        false,
    );
    let final_total = direct_storage_buffer(
        context,
        "G6-P01 prefix scan final total",
        level_2_blocks,
        true,
    );
    let output_readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("G6-P01 prefix scan output readback"),
        size: buffer_size_u32(ELEMENT_COUNT),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let total_readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("G6-P01 prefix scan total readback"),
        size: buffer_size_u32(level_2_blocks),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    DirectResources {
        input,
        input_upload,
        output,
        block_sums_0,
        block_offsets_0,
        block_sums_1,
        block_offsets_1,
        final_total,
        output_readback,
        total_readback,
    }
}

fn direct_scan_bind_group(
    context: &DirectWgpuContext,
    pipeline: &wgpu::ComputePipeline,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    block_sums: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let layout = pipeline.get_bind_group_layout(0);
    context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("G6-P01 prefix scan bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: block_sums.as_entire_binding(),
                },
            ],
        })
}

fn direct_apply_bind_group(
    context: &DirectWgpuContext,
    pipeline: &wgpu::ComputePipeline,
    output: &wgpu::Buffer,
    offsets: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let layout = pipeline.get_bind_group_layout(0);
    context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("G6-P01 prefix apply-offsets bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: offsets.as_entire_binding(),
                },
            ],
        })
}

fn direct_dispatch(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(workgroups, 1, 1);
}

fn direct_mode_sample(
    context: &DirectWgpuContext,
    pipelines: &DirectPipelines,
    mode: ScanMode,
) -> BTreeMap<String, f64> {
    let total_start = Instant::now();
    let [level_0_blocks, level_1_blocks, level_2_blocks] = hierarchy_counts();

    let resource_start = Instant::now();
    let resources = direct_resources(context, mode);
    let scan_0_bindings = direct_scan_bind_group(
        context,
        pipelines.scan_0(mode),
        &resources.input,
        &resources.output,
        &resources.block_sums_0,
    );
    let scan_1_bindings = direct_scan_bind_group(
        context,
        &pipelines.scan_1,
        &resources.block_sums_0,
        &resources.block_offsets_0,
        &resources.block_sums_1,
    );
    let scan_2_bindings = direct_scan_bind_group(
        context,
        &pipelines.scan_2,
        &resources.block_sums_1,
        &resources.block_offsets_1,
        &resources.final_total,
    );
    let apply_1_bindings = direct_apply_bind_group(
        context,
        &pipelines.apply_1,
        &resources.block_offsets_0,
        &resources.block_offsets_1,
    );
    let apply_0_bindings = direct_apply_bind_group(
        context,
        &pipelines.apply_0,
        &resources.output,
        &resources.block_offsets_0,
    );
    let resource_setup_us = micros(resource_start.elapsed());

    let record_start = Instant::now();
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("G6-P01 prefix scan direct encoder"),
        });
    encoder.copy_buffer_to_buffer(
        &resources.input_upload,
        0,
        &resources.input,
        0,
        buffer_size_u32(ELEMENT_COUNT),
    );
    direct_dispatch(
        &mut encoder,
        "G6-P01 prefix scan level 0",
        pipelines.scan_0(mode),
        &scan_0_bindings,
        level_0_blocks,
    );
    direct_dispatch(
        &mut encoder,
        "G6-P01 prefix scan level 1",
        &pipelines.scan_1,
        &scan_1_bindings,
        level_1_blocks,
    );
    direct_dispatch(
        &mut encoder,
        "G6-P01 prefix scan level 2",
        &pipelines.scan_2,
        &scan_2_bindings,
        level_2_blocks,
    );
    direct_dispatch(
        &mut encoder,
        "G6-P01 prefix scan apply level 1",
        &pipelines.apply_1,
        &apply_1_bindings,
        blocks_for(level_0_blocks),
    );
    direct_dispatch(
        &mut encoder,
        "G6-P01 prefix scan apply level 0",
        &pipelines.apply_0,
        &apply_0_bindings,
        level_0_blocks,
    );
    encoder.copy_buffer_to_buffer(
        &resources.output,
        0,
        &resources.output_readback,
        0,
        buffer_size_u32(ELEMENT_COUNT),
    );
    encoder.copy_buffer_to_buffer(
        &resources.final_total,
        0,
        &resources.total_readback,
        0,
        buffer_size_u32(level_2_blocks),
    );
    let command_buffer = encoder.finish();
    let command_record_us = micros(record_start.elapsed());

    let submitted = submit_and_map(
        context,
        command_buffer,
        &[&resources.output_readback, &resources.total_readback],
    );
    let output = decode_u32_bytes(&submitted.mapped[0]);
    let total = decode_u32_bytes(&submitted.mapped[1]);
    assert_exact_output(mode, &output, &total);

    BTreeMap::from([
        ("resource_setup".to_owned(), resource_setup_us),
        (
            "boundary_prepare_or_record".to_owned(),
            resource_setup_us + command_record_us,
        ),
        ("command_record".to_owned(), command_record_us),
        (
            "readback_registration".to_owned(),
            submitted.readback_registration_us,
        ),
        ("queue_submit".to_owned(), submitted.submit_call_us),
        (
            "boundary_prepare_record_submit".to_owned(),
            resource_setup_us + command_record_us + submitted.boundary_submit_us(),
        ),
        (
            "completion_readback".to_owned(),
            submitted.completion_readback_us,
        ),
        ("total".to_owned(), micros(total_start.elapsed())),
    ])
}

fn direct_sample(
    context: &DirectWgpuContext,
    pipelines: &DirectPipelines,
) -> BTreeMap<String, f64> {
    let total_start = Instant::now();
    let exclusive = direct_mode_sample(context, pipelines, ScanMode::Exclusive);
    let inclusive = direct_mode_sample(context, pipelines, ScanMode::Inclusive);
    let mut phases = sum_phases(exclusive, inclusive);
    phases.insert("total".to_owned(), micros(total_start.elapsed()));
    phases
}

fn assert_equivalent_adapter_selection(
    runengpu_context: &GpuContext,
    direct_context: &DirectWgpuContext,
) {
    let runengpu = runengpu_context.adapter_facts();
    assert_eq!(runengpu.backend(), GpuBackendFamily::Vulkan);
    assert_eq!(runengpu.fallback(), GpuFallbackStatus::ConfirmedFallback);
    assert_eq!(direct_context.adapter_info.backend, wgpu::Backend::Vulkan);
    assert_eq!(runengpu.vendor(), Some(direct_context.adapter_info.vendor));
    assert_eq!(runengpu.device(), Some(direct_context.adapter_info.device));
}

fn runengpu_adapter_facts_json(context: &GpuContext) -> Value {
    let facts = context.adapter_facts();
    json!({
        "backend": format!("{:?}", facts.backend()),
        "class": format!("{:?}", facts.class()),
        "software": format!("{:?}", facts.software()),
        "fallback": format!("{:?}", facts.fallback()),
        "name": facts.diagnostic_name(),
        "driver": facts.driver(),
        "driver_info": facts.driver_info(),
        "vendor": facts.vendor(),
        "device": facts.device(),
    })
}

pub(crate) fn compare() -> Value {
    assert_eq!(hierarchy_counts(), [65, 2, 1]);

    let direct_context = DirectWgpuContext::request("G6-P01 prefix scan direct WGPU");
    let direct_pipelines = direct_pipelines(&direct_context);

    let runengpu_context_start = Instant::now();
    let runengpu_context = runengpu_context();
    let runengpu_context_us = micros(runengpu_context_start.elapsed());
    assert_equivalent_adapter_selection(&runengpu_context, &direct_context);

    let runengpu_pipeline_start = Instant::now();
    let runengpu_sources = admitted_sources();
    let runengpu_pipelines = runengpu_pipelines(&runengpu_sources);
    let runengpu_pipeline_descriptor_us = micros(runengpu_pipeline_start.elapsed());

    let mut runengpu_cold = None;
    let mut direct_cold = None;
    for _ in 0..WARMUP_SAMPLES {
        runengpu_cold = Some(runengpu_sample(&runengpu_context, &runengpu_pipelines));
        direct_cold = Some(direct_sample(&direct_context, &direct_pipelines));
    }
    let runengpu_cold = runengpu_cold.expect("one RunenGPU warm-up sample is required");
    let direct_cold = direct_cold.expect("one direct-WGPU warm-up sample is required");
    let runengpu_cold_end_to_end_us = runengpu_context_us
        + runengpu_pipeline_descriptor_us
        + runengpu_cold.get("total").copied().unwrap();
    let direct_cold_end_to_end_us = direct_context.setup_us
        + direct_pipelines.cold_pipeline_us
        + direct_cold.get("total").copied().unwrap();
    assert!(direct_cold_end_to_end_us > 0.0);

    let mut runengpu = Measurements::default();
    let mut direct = Measurements::default();
    for _ in 0..MEASURED_SAMPLES {
        runengpu.push(runengpu_sample(&runengpu_context, &runengpu_pipelines));
        direct.push(direct_sample(&direct_context, &direct_pipelines));
    }

    let per_mode_logical_resource_bytes = buffer_size_u32(ELEMENT_COUNT) * 2
        + buffer_size_u32(65) * 2
        + buffer_size_u32(2) * 2
        + buffer_size_u32(1);
    let per_mode_readback_bytes = buffer_size_u32(ELEMENT_COUNT) + buffer_size_u32(1);

    json!({
        "workload": "G5-C01-4097-u32-prefix-scan",
        "comparison_envelope": {
            "element_count": ELEMENT_COUNT,
            "input_value": 1,
            "modes": ["exclusive", "inclusive"],
            "workgroup_size": WORKGROUP_SIZE,
            "hierarchy": [ELEMENT_COUNT, 65, 2, 1],
            "dispatch_workgroups_per_mode": [65, 2, 1, 2, 65],
            "shader_sources": [
                "gpu_prefix_scan_native/scan.wgsl",
                "gpu_prefix_scan_native/apply_offsets.wgsl"
            ],
            "shader_entry_point": "cs_main",
            "specialization": {
                "level_0_exclusive": {"ELEMENT_COUNT": ELEMENT_COUNT, "INCLUSIVE": 0},
                "level_0_inclusive": {"ELEMENT_COUNT": ELEMENT_COUNT, "INCLUSIVE": 1},
                "level_1": {"ELEMENT_COUNT": 65, "INCLUSIVE": 0},
                "level_2": {"ELEMENT_COUNT": 2, "INCLUSIVE": 0},
                "apply_level_1": {"ELEMENT_COUNT": 65},
                "apply_level_0": {"ELEMENT_COUNT": ELEMENT_COUNT}
            },
            "submissions_per_sample_per_path": 2,
            "logical_resource_bytes_per_mode": per_mode_logical_resource_bytes,
            "logical_resource_bytes_per_sample": per_mode_logical_resource_bytes * 2,
            "upload_staging_bytes_per_mode": buffer_size_u32(ELEMENT_COUNT),
            "upload_staging_bytes_per_sample": buffer_size_u32(ELEMENT_COUNT) * 2,
            "readback_bytes_per_mode": per_mode_readback_bytes,
            "readback_bytes_per_sample": per_mode_readback_bytes * 2,
        },
        "adapter_equivalence": {
            "criteria": "Vulkan forced-fallback selection with equal vendor/device identity",
            "runengpu": runengpu_adapter_facts_json(&runengpu_context),
            "direct_wgpu": {
                "facts": direct_context.facts_json(),
                "vendor": direct_context.adapter_info.vendor,
                "device": direct_context.adapter_info.device,
            },
        },
        "cold": {
            "scope": "fresh path context/device plus path-specific program/pipeline setup plus first complete exclusive+inclusive retained workload sample",
            "construction_order": ["direct_wgpu", "runengpu"],
            "normalized_end_to_end_us": {
                "runengpu": runengpu_cold_end_to_end_us,
                "direct_wgpu": direct_cold_end_to_end_us,
                "runengpu_over_direct_ratio": runengpu_cold_end_to_end_us / direct_cold_end_to_end_us,
            },
            "component_observations": {
                "runengpu_context_us": runengpu_context_us,
                "runengpu_program_pipeline_descriptor_us": runengpu_pipeline_descriptor_us,
                "runengpu_first_workload_phases_us": runengpu_cold,
                "direct_context_us": direct_context.setup_us,
                "direct_physical_pipeline_us": direct_pipelines.cold_pipeline_us,
                "direct_first_workload_phases_us": direct_cold,
                "note": "RunenGPU physical pipeline realization occurs during first backend_prepare and submit_prepared owns readback callback registration, physical encoding, and queue submission. Direct WGPU exposes resource creation, command recording, callback registration, and queue submission separately. Compare normalized boundary/total fields, not unlike component fields pairwise."
            },
        },
        "warm_lifecycle": {
            "warmup_samples": WARMUP_SAMPLES,
            "measured_samples": MEASURED_SAMPLES,
            "runengpu_program_pipeline_identity_reused": true,
            "direct_wgpu_pipeline_reused": true,
            "per_sample_resources_recreated": true,
            "per_sample_exclusive_and_inclusive_submissions": true,
        },
        "phase_comparability": {
            "ratio_phases": ["boundary_prepare_record_submit", "completion_readback", "total"],
            "runengpu_submit_component": "submit_prepared combines acceptance, readback callback registration, physical encoding and queue submission and is not separable through the public boundary",
            "direct_submit_boundary_component": "readback callback registration plus queue.submit call",
            "direct_queue_submit_component": "queue.submit call only",
            "queue_submit_ratio_status": "unavailable because the RunenGPU public submit boundary intentionally combines additional execution work",
        },
        "runengpu": runengpu.to_json(),
        "direct_wgpu": direct.to_json(),
        "runengpu_over_direct_ratio": ratio_summary(
            &runengpu,
            &direct,
            &["boundary_prepare_record_submit", "completion_readback", "total"],
        ),
        "timestamp_evidence": {
            "supported_by_direct_adapter": direct_context.timestamp_supported,
            "status": "not-yet-instrumented",
        },
        "correctness": {
            "runengpu": "full exclusive+inclusive outputs and exact total passed on every sample",
            "direct_wgpu": "full exclusive+inclusive outputs and exact total passed on every sample",
        },
        "allocation_bytes": null,
        "allocation_bytes_status": "unavailable without new backend allocator instrumentation",
    })
}
