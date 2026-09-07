use super::*;
use crate::common::summarize_u64;

const PASS_NAMES: [&str; 5] = [
    "scan_level_0",
    "scan_level_1",
    "scan_level_2",
    "apply_level_1_offsets",
    "apply_level_0_offsets",
];
const QUERY_COUNT: u32 = 10;
const TIMESTAMP_BYTES: u64 = QUERY_COUNT as u64 * 8;

fn timestamp_context() -> GpuContext {
    let mut requirements = GpuCapabilityRequirements::new();
    for feature in [
        GpuCapabilityFeature::Compute,
        GpuCapabilityFeature::Copy,
        GpuCapabilityFeature::TimestampQuery,
    ] {
        requirements
            .insert(GpuCapabilityRequirement::Required(feature))
            .unwrap();
    }
    let descriptor = GpuContextDescriptor::new(requirements)
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label("G6-P01 prefix scan RunenGPU timestamp evidence");
    let context = pollster::block_on(GpuContext::request(descriptor)).unwrap();
    assert_eq!(context.adapter_facts().backend(), GpuBackendFamily::Vulkan);
    assert_eq!(
        context.adapter_facts().fallback(),
        GpuFallbackStatus::ConfirmedFallback
    );
    context
}

fn query_common(name: &str) -> GpuResourceCommon {
    let resource_label = label(name);
    GpuResourceCommon::owned(
        resource_label.clone(),
        GpuResourceLifetime::Transient,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        GpuResourceProvenance::new(resource_label, None, None),
    )
    .unwrap()
}

struct TimestampFragment {
    fragment: GpuWorkFragment,
    output_readback: GpuReadbackId,
    total_readback: GpuReadbackId,
    timestamp_readback: GpuReadbackId,
}

fn timestamp_fragment(pipelines: &RunenGpuPipelines, mode: ScanMode) -> TimestampFragment {
    let (base, output_readback, total_readback) = author_scan(pipelines, mode);
    assert_eq!(base.nodes().len(), 7);

    let mut resources = GpuResourceScope::new();
    let query_set = resources
        .query_set(
            GpuQuerySetDescriptor::new(
                query_common(&format!("prefix scan {} timestamp query set", mode.key())),
                GpuQueryKind::Timestamp,
                QUERY_COUNT,
            )
            .unwrap(),
        )
        .unwrap();
    let resolve = resources
        .buffer(
            GpuBufferDescriptor::ordinary_owned(
                format!("prefix scan {} timestamp resolve", mode.key()),
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                TIMESTAMP_BYTES,
                [GpuBufferUsage::QueryResolve, GpuBufferUsage::CopySource],
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let resolve_operation = GpuQueryResolveOperation::new(
        &query_set,
        GpuQueryRange::new(&query_set, 0, QUERY_COUNT).unwrap(),
        &resolve,
        0,
    )
    .unwrap();
    let timestamp_readback_operation = readback_operation(&resolve);
    let timestamp_readback = timestamp_readback_operation.id();

    let fragment = GpuWorkFragment::build(
        format!(
            "direct-cost prefix scan {} timestamp comparison",
            mode.key()
        ),
        |work| {
            for (index, node) in base.nodes().iter().enumerate() {
                if index < PASS_NAMES.len() {
                    let GpuWorkOperation::Compute(operation) = node.operation() else {
                        panic!("first five retained prefix-scan nodes must remain compute passes")
                    };
                    let beginning = u32::try_from(index * 2).unwrap();
                    let end = beginning + 1;
                    let writes =
                        GpuTimestampWrites::new(&query_set, Some(beginning), Some(end)).unwrap();
                    work.operation(
                        node.label().as_str(),
                        operation.clone().with_timestamp_writes(writes),
                    )?;
                    if index + 1 == PASS_NAMES.len() {
                        work.operation("prefix scan timestamp resolve", resolve_operation.clone())?;
                    }
                } else {
                    assert!(
                        matches!(node.operation(), GpuWorkOperation::Readback(_)),
                        "retained prefix-scan tail must remain the two correctness readbacks"
                    );
                    work.operation(node.label().as_str(), node.operation().clone())?;
                }
            }
            work.operation(
                "prefix scan timestamp readback",
                timestamp_readback_operation,
            )?;
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(fragment.nodes().len(), 9);

    TimestampFragment {
        fragment,
        output_readback,
        total_readback,
        timestamp_readback,
    }
}

fn progress_timestamp_readbacks(
    context: &GpuContext,
    submission: &GpuSubmission,
    ids: &[GpuReadbackId],
) -> Vec<GpuReadbackBytes> {
    let handles = ids
        .iter()
        .map(|id| submission.readback(*id).unwrap().clone())
        .collect::<Vec<_>>();
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        context.progress();
        let mut all_ready = true;
        for handle in &handles {
            match handle.status() {
                GpuReadbackStatus::Ready(_) => {}
                GpuReadbackStatus::Failed(failure) => {
                    panic!("prefix-scan timestamp readback failed: {failure:?}")
                }
                GpuReadbackStatus::Pending => all_ready = false,
            }
        }
        match submission.status() {
            GpuSubmissionStatus::Failed(failure) => {
                panic!("prefix-scan timestamp submission failed: {failure:?}")
            }
            GpuSubmissionStatus::Completed if all_ready => break,
            GpuSubmissionStatus::Accepted | GpuSubmissionStatus::Completed => {}
        }
        assert!(
            Instant::now() < deadline,
            "prefix-scan timestamp comparison timed out"
        );
        std::thread::yield_now();
    }
    handles
        .into_iter()
        .map(|handle| match handle.status() {
            GpuReadbackStatus::Ready(bytes) => bytes,
            other => panic!("terminal prefix-scan timestamp readback must be ready: {other:?}"),
        })
        .collect()
}

fn timestamp_deltas(bytes: &[u8]) -> Vec<u64> {
    let timestamps = bytes.chunks_exact(8);
    assert!(timestamps.remainder().is_empty());
    let timestamps = timestamps
        .map(|timestamp| u64::from_ne_bytes(timestamp.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(timestamps.len(), usize::try_from(QUERY_COUNT).unwrap());
    let pairs = timestamps.chunks_exact(2);
    assert!(pairs.remainder().is_empty());
    pairs
        .map(|pair| {
            let beginning = pair[0];
            let end = pair[1];
            assert!(end >= beginning, "timestamp end must not precede beginning");
            end - beginning
        })
        .collect()
}

fn runengpu_mode_timestamp_sample(
    context: &GpuContext,
    pipelines: &RunenGpuPipelines,
    mode: ScanMode,
) -> Vec<u64> {
    let authored = timestamp_fragment(pipelines, mode);
    let graph = GpuPreparedWorkGraph::prepare(
        label(format!("G6-P01 prefix scan {} timestamp graph", mode.key())),
        [authored.fragment],
    )
    .unwrap();
    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let readbacks = progress_timestamp_readbacks(
        context,
        &submission,
        &[
            authored.output_readback,
            authored.total_readback,
            authored.timestamp_readback,
        ],
    );
    let output = decode_u32_bytes(readbacks[0].as_bytes());
    let total = decode_u32_bytes(readbacks[1].as_bytes());
    assert_exact_output(mode, &output, &total);
    timestamp_deltas(readbacks[2].as_bytes())
}

fn direct_timestamp_dispatch(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups: u32,
    query_set: &wgpu::QuerySet,
    pass_index: usize,
) {
    let beginning = u32::try_from(pass_index * 2).unwrap();
    let end = beginning + 1;
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(beginning),
            end_of_pass_write_index: Some(end),
        }),
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(workgroups, 1, 1);
}

fn direct_mode_timestamp_sample(
    context: &DirectWgpuContext,
    pipelines: &DirectPipelines,
    mode: ScanMode,
) -> Vec<u64> {
    let [level_0_blocks, level_1_blocks, level_2_blocks] = hierarchy_counts();
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

    let query_set = context.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("G6-P01 prefix scan direct timestamp query set"),
        ty: wgpu::QueryType::Timestamp,
        count: QUERY_COUNT,
    });
    let resolve = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("G6-P01 prefix scan direct timestamp resolve"),
        size: TIMESTAMP_BYTES,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let timestamp_readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("G6-P01 prefix scan direct timestamp readback"),
        size: TIMESTAMP_BYTES,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("G6-P01 prefix scan direct timestamp encoder"),
        });
    encoder.copy_buffer_to_buffer(
        &resources.input_upload,
        0,
        &resources.input,
        0,
        buffer_size_u32(ELEMENT_COUNT),
    );
    direct_timestamp_dispatch(
        &mut encoder,
        "G6-P01 prefix scan level 0 timestamp",
        pipelines.scan_0(mode),
        &scan_0_bindings,
        level_0_blocks,
        &query_set,
        0,
    );
    direct_timestamp_dispatch(
        &mut encoder,
        "G6-P01 prefix scan level 1 timestamp",
        &pipelines.scan_1,
        &scan_1_bindings,
        level_1_blocks,
        &query_set,
        1,
    );
    direct_timestamp_dispatch(
        &mut encoder,
        "G6-P01 prefix scan level 2 timestamp",
        &pipelines.scan_2,
        &scan_2_bindings,
        level_2_blocks,
        &query_set,
        2,
    );
    direct_timestamp_dispatch(
        &mut encoder,
        "G6-P01 prefix scan apply level 1 timestamp",
        &pipelines.apply_1,
        &apply_1_bindings,
        blocks_for(level_0_blocks),
        &query_set,
        3,
    );
    direct_timestamp_dispatch(
        &mut encoder,
        "G6-P01 prefix scan apply level 0 timestamp",
        &pipelines.apply_0,
        &apply_0_bindings,
        level_0_blocks,
        &query_set,
        4,
    );
    encoder.resolve_query_set(&query_set, 0..QUERY_COUNT, &resolve, 0);
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
    encoder.copy_buffer_to_buffer(&resolve, 0, &timestamp_readback, 0, TIMESTAMP_BYTES);
    let command_buffer = encoder.finish();
    let submitted = submit_and_map(
        context,
        command_buffer,
        &[
            &resources.output_readback,
            &resources.total_readback,
            &timestamp_readback,
        ],
    );
    let output = decode_u32_bytes(&submitted.mapped[0]);
    let total = decode_u32_bytes(&submitted.mapped[1]);
    assert_exact_output(mode, &output, &total);
    timestamp_deltas(&submitted.mapped[2])
}

fn pass_evidence(samples: &[Vec<u64>], period_ns: Option<f64>) -> Value {
    assert_eq!(samples.len(), MEASURED_SAMPLES);
    assert!(
        samples
            .iter()
            .all(|sample| sample.len() == PASS_NAMES.len())
    );

    let mut raw = serde_json::Map::new();
    let mut summaries = serde_json::Map::new();
    let mut ns = serde_json::Map::new();
    for (index, pass_name) in PASS_NAMES.iter().enumerate() {
        let values = samples
            .iter()
            .map(|sample| sample[index])
            .collect::<Vec<_>>();
        raw.insert((*pass_name).to_owned(), json!(&values));
        summaries.insert((*pass_name).to_owned(), summarize_u64(&values));
        if let Some(period_ns) = period_ns {
            ns.insert(
                (*pass_name).to_owned(),
                json!(
                    values
                        .iter()
                        .map(|ticks| *ticks as f64 * period_ns)
                        .collect::<Vec<_>>()
                ),
            );
        }
    }
    let totals = samples
        .iter()
        .map(|sample| sample.iter().sum::<u64>())
        .collect::<Vec<_>>();

    json!({
        "raw_delta_ticks_by_pass": raw,
        "summary_ticks_by_pass": summaries,
        "total_pass_ticks_per_sample": totals,
        "total_pass_ticks_summary": summarize_u64(&totals),
        "delta_ns_by_pass": period_ns.map(|_| Value::Object(ns)),
    })
}

fn expected_adapter_identity(wall_evidence: &Value) -> (u64, u64) {
    let direct = &wall_evidence["adapter_equivalence"]["direct_wgpu"];
    (
        direct["vendor"].as_u64().unwrap(),
        direct["device"].as_u64().unwrap(),
    )
}

pub(crate) fn evidence(wall_evidence: &Value) -> Value {
    assert_eq!(
        wall_evidence["timestamp_evidence"]["supported_by_direct_adapter"], true,
        "accepted G6-P01 Lavapipe adapter reports timestamp-query support"
    );
    let (expected_vendor, expected_device) = expected_adapter_identity(wall_evidence);

    let runengpu_context = timestamp_context();
    let direct_context =
        DirectWgpuContext::request_timestamp("G6-P01 prefix scan direct WGPU timestamp evidence");
    assert_equivalent_adapter_selection(&runengpu_context, &direct_context);
    assert_eq!(
        runengpu_context.adapter_facts().vendor().map(u64::from),
        Some(expected_vendor)
    );
    assert_eq!(
        runengpu_context.adapter_facts().device().map(u64::from),
        Some(expected_device)
    );
    assert_eq!(
        u64::from(direct_context.adapter_info.vendor),
        expected_vendor
    );
    assert_eq!(
        u64::from(direct_context.adapter_info.device),
        expected_device
    );

    let sources = admitted_sources();
    let runengpu_pipelines = runengpu_pipelines(&sources);
    let direct_pipelines = direct_pipelines(&direct_context);

    for _ in 0..WARMUP_SAMPLES {
        let _ = runengpu_mode_timestamp_sample(
            &runengpu_context,
            &runengpu_pipelines,
            ScanMode::Exclusive,
        );
        let _ = runengpu_mode_timestamp_sample(
            &runengpu_context,
            &runengpu_pipelines,
            ScanMode::Inclusive,
        );
        let _ =
            direct_mode_timestamp_sample(&direct_context, &direct_pipelines, ScanMode::Exclusive);
        let _ =
            direct_mode_timestamp_sample(&direct_context, &direct_pipelines, ScanMode::Inclusive);
    }

    let mut runengpu_exclusive = Vec::with_capacity(MEASURED_SAMPLES);
    let mut runengpu_inclusive = Vec::with_capacity(MEASURED_SAMPLES);
    let mut direct_exclusive = Vec::with_capacity(MEASURED_SAMPLES);
    let mut direct_inclusive = Vec::with_capacity(MEASURED_SAMPLES);
    for _ in 0..MEASURED_SAMPLES {
        runengpu_exclusive.push(runengpu_mode_timestamp_sample(
            &runengpu_context,
            &runengpu_pipelines,
            ScanMode::Exclusive,
        ));
        runengpu_inclusive.push(runengpu_mode_timestamp_sample(
            &runengpu_context,
            &runengpu_pipelines,
            ScanMode::Inclusive,
        ));
        direct_exclusive.push(direct_mode_timestamp_sample(
            &direct_context,
            &direct_pipelines,
            ScanMode::Exclusive,
        ));
        direct_inclusive.push(direct_mode_timestamp_sample(
            &direct_context,
            &direct_pipelines,
            ScanMode::Inclusive,
        ));
    }

    let runengpu_period_ns = f64::from(
        runengpu_context
            .timestamp_period_ns()
            .expect("timestamp-query-admitted RunenGPU context must expose timestamp period"),
    );
    let direct_period_ns = f64::from(direct_context.queue.get_timestamp_period());
    assert!(runengpu_period_ns.is_finite() && runengpu_period_ns > 0.0);
    assert!(direct_period_ns.is_finite() && direct_period_ns > 0.0);
    json!({
        "status": "measured",
        "separate_from_wall_clock_samples": true,
        "pass_boundaries": PASS_NAMES,
        "queries_per_mode": QUERY_COUNT,
        "warmup_samples": WARMUP_SAMPLES,
        "measured_samples": MEASURED_SAMPLES,
        "runengpu": {
            "timestamp_period_ns": runengpu_period_ns,
            "timestamp_period_status": "reported by GpuContext::timestamp_period_ns()",
            "exclusive": pass_evidence(&runengpu_exclusive, Some(runengpu_period_ns)),
            "inclusive": pass_evidence(&runengpu_inclusive, Some(runengpu_period_ns)),
        },
        "direct_wgpu": {
            "timestamp_period_ns": direct_period_ns,
            "exclusive": pass_evidence(&direct_exclusive, Some(direct_period_ns)),
            "inclusive": pass_evidence(&direct_inclusive, Some(direct_period_ns)),
        },
        "correctness": "full exclusive+inclusive outputs and exact total passed on every timestamp sample for both paths",
        "interpretation": "Both paths write timestamps at the beginning/end of each of the five retained prefix-scan compute passes on the same Vulkan fallback adapter. Raw ticks remain retained for both paths; each path converts its own ticks with its own reported timestamp period, keeping GPU timestamp evidence separate from wall-clock samples.",
    })
}
