#![allow(clippy::duplicate_mod)] // Reuses retained standalone proof modules that each own the shared readback helper.

use runen_gpu::*;
use serde_json::{Map, Value, json};
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "support/readback_wait.rs"]
mod readback_wait;
#[path = "gpu_r2_blend_state.rs"]
mod retained_blend;
#[path = "gpu_r2_depth_bias.rs"]
mod retained_depth_bias;
#[path = "gpu_compute_generated_indirect_native.rs"]
mod retained_indirect;
#[path = "gpu_offscreen_indexed_native.rs"]
mod retained_offscreen;
#[path = "gpu_prefix_scan_native.rs"]
mod retained_prefix_scan;
#[path = "gpu_r2_sampler_anisotropy.rs"]
mod retained_sampler_anisotropy;
#[path = "gpu_r1_vertex16_formats.rs"]
mod retained_vertex16;
#[path = "gpu_r1_vertex8_formats.rs"]
mod retained_vertex8;
#[path = "gpu_r1_vertex_packed_formats.rs"]
mod retained_vertex_packed;

const FEATURES: [GpuCapabilityFeature; 14] = [
    GpuCapabilityFeature::Compute,
    GpuCapabilityFeature::RenderPipeline,
    GpuCapabilityFeature::Copy,
    GpuCapabilityFeature::IndirectExecution,
    GpuCapabilityFeature::StorageTexture,
    GpuCapabilityFeature::TextureBindingArray,
    GpuCapabilityFeature::BufferBindingArray,
    GpuCapabilityFeature::StorageResourceBindingArray,
    GpuCapabilityFeature::UniformBufferBindingArray,
    GpuCapabilityFeature::DepthAttachment,
    GpuCapabilityFeature::DepthBiasClamp,
    GpuCapabilityFeature::ShaderF16,
    GpuCapabilityFeature::TimestampQuery,
    GpuCapabilityFeature::Presentation,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QualificationMode {
    GenericMetal,
    ActualM3,
}

impl QualificationMode {
    fn from_environment() -> Self {
        match std::env::var("RUNEN_GPU_QUALIFICATION_MODE").as_deref() {
            Ok("generic") => Self::GenericMetal,
            Ok("m3") => Self::ActualM3,
            Ok(other) => panic!("unsupported RUNEN_GPU_QUALIFICATION_MODE={other:?}"),
            Err(error) => panic!("RUNEN_GPU_QUALIFICATION_MODE is required: {error}"),
        }
    }

    const fn report_name(self) -> &'static str {
        match self {
            Self::GenericMetal => "generic_metal",
            Self::ActualM3 => "actual_m3",
        }
    }
}

fn label(value: impl AsRef<str>) -> GpuResourceLabel {
    GpuResourceLabel::new(value.as_ref()).unwrap()
}

fn command_stdout(program: &str, arguments: &[&str]) -> String {
    let output = Command::new(program)
        .args(arguments)
        .output()
        .unwrap_or_else(|error| panic!("failed to execute {program}: {error}"));
    assert!(
        output.status.success(),
        "{program} {:?} failed: {}",
        arguments,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("qualification command output must be UTF-8")
        .trim()
        .to_owned()
}

fn repository_status() -> String {
    command_stdout(
        "git",
        &["status", "--porcelain", "--untracked-files=normal"],
    )
}

fn expected_revision() -> String {
    std::env::var("RUNEN_GPU_PROOF_REVISION")
        .expect("RUNEN_GPU_PROOF_REVISION must bind qualification to an exact revision")
}

fn report_path() -> PathBuf {
    std::env::var_os("RUNEN_GPU_QUALIFICATION_REPORT")
        .map(PathBuf::from)
        .expect("RUNEN_GPU_QUALIFICATION_REPORT must name the machine-readable report")
}

fn find_string_field(value: &Value, key: &str) -> Option<String> {
    match value {
        Value::Object(entries) => entries
            .get(key)
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| {
                entries
                    .values()
                    .find_map(|entry| find_string_field(entry, key))
            }),
        Value::Array(entries) => entries
            .iter()
            .find_map(|entry| find_string_field(entry, key)),
        _ => None,
    }
}

fn hardware_identity() -> (String, String) {
    let json = command_stdout("system_profiler", &["SPHardwareDataType", "-json"]);
    let value: Value =
        serde_json::from_str(&json).expect("system_profiler hardware JSON must parse");
    let machine_model = find_string_field(&value, "machine_model")
        .expect("system_profiler must publish a machine model");
    let chip_type =
        find_string_field(&value, "chip_type").expect("system_profiler must publish a chip type");
    (machine_model, chip_type)
}

fn prefix_graph(
    sources: &retained_prefix_scan::ProgramSources,
    mode: retained_prefix_scan::ScanMode,
) -> (GpuPreparedWorkGraph, GpuReadbackId, GpuReadbackId) {
    let (fragment, output_id, total_id) = retained_prefix_scan::author_scan(sources, mode);
    let mode_name = match mode {
        retained_prefix_scan::ScanMode::Exclusive => "exclusive",
        retained_prefix_scan::ScanMode::Inclusive => "inclusive",
    };
    (
        GpuPreparedWorkGraph::prepare(
            label(format!("Metal qualification {mode_name} prefix scan")),
            [fragment],
        )
        .unwrap(),
        output_id,
        total_id,
    )
}

fn qualification_context(
    exclusive: &GpuPreparedWorkGraph,
    inclusive: &GpuPreparedWorkGraph,
    render: &GpuPreparedWorkGraph,
    indirect: &GpuPreparedWorkGraph,
) -> GpuContext {
    let mut requirements = exclusive
        .requirements()
        .merge(inclusive.requirements())
        .unwrap()
        .merge(render.requirements())
        .unwrap()
        .merge(indirect.requirements())
        .unwrap();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::DepthAttachment,
        ))
        .unwrap();
    let descriptor = GpuContextDescriptor::new(requirements)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
        .require_format_role(GpuTextureFormat::Depth16Unorm, GpuFormatRole::DepthStencil)
        .with_allowed_backends([GpuBackendFamily::Metal])
        .with_label("RunenGPU generic Metal qualification");
    let context = pollster::block_on(GpuContext::request(descriptor))
        .expect("qualification requires a usable Metal adapter");
    assert_eq!(
        context.adapter_facts().backend(),
        GpuBackendFamily::Metal,
        "qualification must execute on Metal rather than a fallback backend"
    );
    context
}

async fn execute_prefix(
    context: &GpuContext,
    graph: GpuPreparedWorkGraph,
    output_id: GpuReadbackId,
    total_id: GpuReadbackId,
    mode: retained_prefix_scan::ScanMode,
) {
    let prepared = context.prepare_submission(graph).await.unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let output = readback_wait::wait_for_readback(
        context,
        &submission,
        output_id,
        "Metal qualification prefix output",
    )
    .await;
    let total = readback_wait::wait_for_readback(
        context,
        &submission,
        total_id,
        "Metal qualification prefix total",
    )
    .await;
    retained_prefix_scan::assert_exact_output(
        mode,
        &retained_prefix_scan::decode_u32(&output),
        &retained_prefix_scan::decode_u32(&total),
    );
}

async fn execute_render(
    context: &GpuContext,
    graph: GpuPreparedWorkGraph,
    readback_id: GpuReadbackId,
) {
    let prepared = context.prepare_submission(graph).await.unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let bytes = readback_wait::wait_for_readback(
        context,
        &submission,
        readback_id,
        "Metal qualification indexed offscreen render",
    )
    .await;
    retained_offscreen::assert_known_pattern(&bytes);
}

async fn execute_indirect(
    context: &GpuContext,
    graph: GpuPreparedWorkGraph,
    readback_id: GpuReadbackId,
) {
    let prepared = context.prepare_submission(graph).await.unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let bytes = readback_wait::wait_for_readback(
        context,
        &submission,
        readback_id,
        "Metal qualification compute-generated indirect draw",
    )
    .await;
    retained_indirect::assert_rendered_pixels(&bytes);
}

fn limits_report(limits: GpuLimits) -> Value {
    json!({
        "max_uniform_buffer_binding_size": limits.max_uniform_buffer_binding_size(),
        "max_storage_buffer_binding_size": limits.max_storage_buffer_binding_size(),
        "max_color_attachments": limits.max_color_attachments(),
        "max_vertex_buffers": limits.max_vertex_buffers(),
        "max_bindings_per_group": limits.max_bindings_per_group(),
        "max_texture_dimension_2d": limits.max_texture_dimension_2d(),
        "max_bind_groups": limits.max_bind_groups(),
        "max_bind_groups_plus_vertex_buffers": limits.max_bind_groups_plus_vertex_buffers(),
        "max_dynamic_uniform_buffers_per_pipeline_layout":
            limits.max_dynamic_uniform_buffers_per_pipeline_layout(),
        "max_dynamic_storage_buffers_per_pipeline_layout":
            limits.max_dynamic_storage_buffers_per_pipeline_layout(),
        "max_compute_workgroups_per_dimension": limits.max_compute_workgroups_per_dimension(),
        "max_buffer_size": limits.max_buffer_size(),
        "max_texture_dimension_1d": limits.max_texture_dimension_1d(),
        "max_texture_dimension_3d": limits.max_texture_dimension_3d(),
        "max_texture_array_layers": limits.max_texture_array_layers(),
        "max_vertex_attributes": limits.max_vertex_attributes(),
        "max_vertex_buffer_array_stride": limits.max_vertex_buffer_array_stride(),
    })
}

fn capability_report(context: &GpuContext) -> Value {
    let adapter = context.adapter_facts();
    let device = context.device_facts();
    let mut capabilities = Map::new();
    for feature in FEATURES {
        capabilities.insert(
            format!("{feature:?}"),
            json!({
                "supported": adapter.supported().supports(feature),
                "enabled": device.is_enabled(feature),
            }),
        );
    }
    Value::Object(capabilities)
}

fn write_report(path: &Path, report: &Value) {
    let parent = path
        .parent()
        .expect("qualification report path must have a parent directory");
    std::fs::create_dir_all(parent).expect("qualification report directory must be creatable");
    let bytes = serde_json::to_vec_pretty(report).expect("qualification report must serialize");
    std::fs::write(path, bytes).expect("qualification report must be writable");
    assert!(
        std::fs::metadata(path)
            .expect("qualification report metadata must be readable")
            .len()
            > 0
    );
}

#[test]
#[ignore = "requires an Apple Metal adapter; executed by Metal qualification CI or the trusted M3 harness"]
fn metal_qualification_records_exact_public_api_evidence() {
    let mode = QualificationMode::from_environment();
    let status_before = repository_status();
    assert!(
        status_before.is_empty(),
        "qualification requires a clean checkout; found:\n{status_before}"
    );

    let revision = command_stdout("git", &["rev-parse", "HEAD"]);
    assert_eq!(
        revision,
        expected_revision(),
        "qualification revision must equal the explicitly expected revision"
    );

    let os_version = command_stdout("sw_vers", &["-productVersion"]);
    let architecture = command_stdout("uname", &["-m"]);
    assert_eq!(
        architecture, "arm64",
        "Metal qualification requires an Apple-Silicon arm64 environment"
    );
    let (machine_model, chip_type) = hardware_identity();
    if mode == QualificationMode::ActualM3 {
        assert!(
            chip_type.to_ascii_lowercase().contains("apple m3"),
            "actual-M3 qualification requires Apple M3/M3 Pro/M3 Max hardware, observed {chip_type:?}"
        );
    }

    let sources = retained_prefix_scan::admitted_sources();
    let (exclusive, exclusive_output, exclusive_total) =
        prefix_graph(&sources, retained_prefix_scan::ScanMode::Exclusive);
    let (inclusive, inclusive_output, inclusive_total) =
        prefix_graph(&sources, retained_prefix_scan::ScanMode::Inclusive);
    let (render, render_readback) = retained_offscreen::render_graph();
    let (indirect, indirect_readback, indirect_args, indirect_vertices) =
        retained_indirect::graph();
    retained_indirect::assert_graph_contract(&indirect, &indirect_args, &indirect_vertices);

    let context = qualification_context(&exclusive, &inclusive, &render, &indirect);
    let adapter = context.adapter_facts();
    let adapter_name = adapter
        .diagnostic_name()
        .expect("Metal qualification requires a recorded adapter name")
        .to_owned();

    assert!(
        !adapter
            .supported()
            .supports(GpuCapabilityFeature::TimestampQuery),
        "TimestampQuery must remain conservatively suppressed on Metal under #89"
    );
    assert!(
        !context
            .device_facts()
            .is_enabled(GpuCapabilityFeature::TimestampQuery),
        "Metal qualification must not enable the unproven timestamp contract"
    );

    pollster::block_on(execute_prefix(
        &context,
        exclusive,
        exclusive_output,
        exclusive_total,
        retained_prefix_scan::ScanMode::Exclusive,
    ));
    pollster::block_on(execute_prefix(
        &context,
        inclusive,
        inclusive_output,
        inclusive_total,
        retained_prefix_scan::ScanMode::Inclusive,
    ));
    pollster::block_on(execute_render(&context, render, render_readback));
    pollster::block_on(execute_indirect(&context, indirect, indirect_readback));
    let vertex8_mask = pollster::block_on(retained_vertex8::run_suite(&context));
    let vertex16_mask = pollster::block_on(retained_vertex16::run_suite(&context));
    let vertex_packed_mask = pollster::block_on(retained_vertex_packed::run_suite(&context));
    let blend_mask = pollster::block_on(retained_blend::run_suite(&context));
    let depth_bias_mask = pollster::block_on(retained_depth_bias::run_baseline(&context));
    retained_sampler_anisotropy::realize_anisotropic_sampler(&context);

    let stats = context.execution_stats();
    assert_eq!(stats.prepared_submissions(), 0);
    assert_eq!(stats.in_flight_submissions(), 0);
    assert_eq!(stats.upload_bytes_in_flight(), 0);
    assert_eq!(stats.readback_bytes_in_flight(), 0);
    assert_eq!(stats.pending_readbacks(), 0);

    let report = json!({
        "schema_version": 1,
        "qualification_level": mode.report_name(),
        "revision": revision,
        "environment": {
            "macos_version": os_version,
            "architecture": architecture,
            "machine_model": machine_model,
            "chip_type": chip_type,
        },
        "adapter": {
            "backend": format!("{:?}", adapter.backend()),
            "class": format!("{:?}", adapter.class()),
            "software": format!("{:?}", adapter.software()),
            "fallback": format!("{:?}", adapter.fallback()),
            "name": adapter_name,
            "driver": adapter.driver(),
            "driver_info": adapter.driver_info(),
            "vendor": adapter.vendor(),
            "device": adapter.device(),
        },
        "capabilities": capability_report(&context),
        "limits": {
            "adapter": limits_report(adapter.adapter_limits().values()),
            "device": limits_report(context.device_facts().device_limits().values()),
        },
        "proofs": {
            "prefix_scan_exclusive": "EXERCISED",
            "prefix_scan_inclusive": "EXERCISED",
            "indexed_offscreen_render": "EXERCISED",
            "compute_generated_indirect_draw": "EXERCISED",
            "vertex8_mask": vertex8_mask,
            "vertex16_mask": vertex16_mask,
            "vertex_packed_mask": vertex_packed_mask,
            "blend_state_mask": blend_mask,
            "depth_bias_baseline_mask": depth_bias_mask,
            "sampler_anisotropy": "EXERCISED",
            "timestamp_query": "UNSUPPORTED_SUPPRESSED",
        },
    });

    let report_path = report_path();
    write_report(&report_path, &report);

    let status_after = repository_status();
    assert!(
        status_after.is_empty(),
        "qualification must leave tracked/untracked repository state clean; found:\n{status_after}"
    );

    println!(
        "RunenGPU {} qualification: PASS (revision={}, chip={}, adapter={}, report={})",
        mode.report_name(),
        expected_revision(),
        report["environment"]["chip_type"].as_str().unwrap(),
        report["adapter"]["name"].as_str().unwrap(),
        report_path.display()
    );
}
