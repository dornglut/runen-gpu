from pathlib import Path


def replace(path: str, old: str, new: str, count: int = 1) -> None:
    source = Path(path).read_text()
    observed = source.count(old)
    if observed != count:
        raise SystemExit(
            f"{path}: expected {count} exact matches, found {observed}: {old[:100]!r}"
        )
    Path(path).write_text(source.replace(old, new))


api = "src/api/execution.rs"
replace(
    api,
    "    GpuReadbackId, GpuResourceRef, GpuSurfaceLeaseError, GpuWorkOperationError,\n",
    "    GpuReadbackId, GpuResourceRef, GpuSurfaceLeaseError, GpuTransferRegion,\n    GpuWorkOperationError,\n",
)
replace(
    api,
    "pub struct GpuReadback {\n    id: GpuReadbackId,\n    status: Arc<Mutex<GpuReadbackStatus>>,\n}\n",
    "pub struct GpuReadback {\n    id: GpuReadbackId,\n    source: GpuTransferRegion,\n    status: Arc<Mutex<GpuReadbackStatus>>,\n}\n",
)
replace(
    api,
    "    pub(crate) fn new(id: GpuReadbackId, status: Arc<Mutex<GpuReadbackStatus>>) -> Self {\n        Self { id, status }\n    }\n",
    "    pub(crate) fn new(\n        id: GpuReadbackId,\n        source: GpuTransferRegion,\n        status: Arc<Mutex<GpuReadbackStatus>>,\n    ) -> Self {\n        Self { id, source, status }\n    }\n",
)
replace(
    api,
    "    pub const fn id(&self) -> GpuReadbackId {\n        self.id\n    }\n\n    pub fn status(&self) -> GpuReadbackStatus {\n",
    "    pub const fn id(&self) -> GpuReadbackId {\n        self.id\n    }\n\n    pub fn source(&self) -> &GpuTransferRegion {\n        &self.source\n    }\n\n    pub fn status(&self) -> GpuReadbackStatus {\n",
)
replace(
    api,
    "            .debug_struct(\"GpuReadback\")\n            .field(\"id\", &self.id)\n            .field(\"status\", &self.status())\n",
    "            .debug_struct(\"GpuReadback\")\n            .field(\"id\", &self.id)\n            .field(\"source\", &self.source)\n            .field(\"status\", &self.status())\n",
)

backend = "src/backend/wgpu/execution.rs"
replace(
    backend,
    "    GpuBufferInitialization, GpuBufferTextureLayout, GpuCapabilityAdmission, GpuClearOperation,\n",
    "    GpuBufferInitialization, GpuBufferRegion, GpuBufferTextureLayout, GpuCapabilityAdmission,\n    GpuClearOperation,\n",
)
replace(
    backend,
    "    Readback {\n        id: GpuReadbackId,\n        source: GpuRealizedBuffer,\n        source_offset: u64,\n        size: u64,\n        metadata: BufferReadbackMetadata,\n    },\n",
    "    Readback {\n        id: GpuReadbackId,\n        source: GpuRealizedBuffer,\n        region: GpuBufferRegion,\n        metadata: BufferReadbackMetadata,\n    },\n",
)
replace(
    backend,
    """        for operation in &plan.operations {
            let (readback_id, size, metadata) = match operation {
                PreparedExecutionOperation::Readback {
                    id, size, metadata, ..
                } => (*id, *size, ReadbackMetadata::Buffer(metadata.clone())),
                PreparedExecutionOperation::TextureReadback {
                    id,
                    staging,
                    metadata,
                    ..
                } => (
                    *id,
                    staging.staging_byte_len,
                    ReadbackMetadata::Texture(metadata.clone()),
                ),
                _ => continue,
            };
""",
    """        for operation in &plan.operations {
            let (readback_id, source, size, metadata) = match operation {
                PreparedExecutionOperation::Readback {
                    id, region, metadata, ..
                } => (
                    *id,
                    GpuTransferRegion::Buffer(region.clone()),
                    region.range().size(),
                    ReadbackMetadata::Buffer(metadata.clone()),
                ),
                PreparedExecutionOperation::TextureReadback {
                    id,
                    region,
                    staging,
                    metadata,
                    ..
                } => (
                    *id,
                    GpuTransferRegion::Texture(region.clone()),
                    staging.staging_byte_len,
                    ReadbackMetadata::Texture(metadata.clone()),
                ),
                _ => continue,
            };
""",
)
replace(
    backend,
    "            public_readbacks.push(GpuReadback::new(readback_id, readback_status));\n",
    "            public_readbacks.push(GpuReadback::new(readback_id, source, readback_status));\n",
)
replace(
    backend,
    """                        operations.push(PreparedExecutionOperation::Readback {
                            id: readback.id(),
                            source: realized_buffer(context, &mut buffer_cache, source.buffer())?,
                            source_offset: source.range().offset(),
                            size,
                            metadata,
                        });
""",
    """                        operations.push(PreparedExecutionOperation::Readback {
                            id: readback.id(),
                            source: realized_buffer(context, &mut buffer_cache, source.buffer())?,
                            region: source.clone(),
                            metadata,
                        });
""",
)
replace(
    backend,
    """            PreparedExecutionOperation::Readback {
                id: readback_id,
                size,
                ..
            } => {
                let staging = Arc::new(backend.device.create_buffer(&BufferDescriptor {
                    label: Some("RunenGPU readback staging"),
                    size: *size,
""",
    """            PreparedExecutionOperation::Readback {
                id: readback_id,
                region,
                ..
            } => {
                let staging = Arc::new(backend.device.create_buffer(&BufferDescriptor {
                    label: Some("RunenGPU readback staging"),
                    size: region.range().size(),
""",
)
replace(
    backend,
    """            PreparedExecutionOperation::Readback {
                id: readback_id,
                source,
                source_offset,
                size,
                ..
            } => {
""",
    """            PreparedExecutionOperation::Readback {
                id: readback_id,
                source,
                region,
                ..
            } => {
""",
)
replace(
    backend,
    """                encoder.copy_buffer_to_buffer(
                    &source.record.object,
                    *source_offset,
                    staging_buffer,
                    0,
                    *size,
                );
""",
    """                encoder.copy_buffer_to_buffer(
                    &source.record.object,
                    region.range().offset(),
                    staging_buffer,
                    0,
                    region.range().size(),
                );
""",
)

test = "tests/gpu_g5b_native_runtime.rs"
replace(
    test,
    "fn native_texture_round_trip_graph() -> (GpuPreparedWorkGraph, GpuReadbackId, Vec<u8>) {\n",
    "fn native_texture_round_trip_graph() -> (\n    GpuPreparedWorkGraph,\n    GpuReadbackId,\n    GpuTransferRegion,\n    Vec<u8>,\n) {\n",
)
replace(
    test,
    "    let readback_id = GpuReadbackId::allocate().unwrap();\n    let readback = GpuReadbackOperation::new(region.into(), readback_id).unwrap();\n",
    "    let readback_id = GpuReadbackId::allocate().unwrap();\n    let readback_source = GpuTransferRegion::Texture(region.clone());\n    let readback = GpuReadbackOperation::new(readback_source.clone(), readback_id).unwrap();\n",
    count=1,
)
replace(
    test,
    "        readback_id,\n        expected,\n    )\n}\n\n#[test]\n#[ignore = \"requires a real Vulkan fallback adapter; executed by RunenGPU Native Conformance CI\"]\nfn native_texture_preparation_accounts_for_private_staging_and_drop_releases_capacity() {\n",
    "        readback_id,\n        readback_source,\n        expected,\n    )\n}\n\n#[test]\n#[ignore = \"requires a real Vulkan fallback adapter; executed by RunenGPU Native Conformance CI\"]\nfn native_texture_preparation_accounts_for_private_staging_and_drop_releases_capacity() {\n",
)
replace(
    test,
    "    let (upload_graph, _, expected) = native_texture_round_trip_graph();\n",
    "    let (upload_graph, _, _, expected) = native_texture_round_trip_graph();\n",
)
replace(
    test,
    "    let (readback_graph, _, _) = native_texture_round_trip_graph();\n",
    "    let (readback_graph, _, _, _) = native_texture_round_trip_graph();\n",
)
replace(
    test,
    "    let (graph, _, _) = native_texture_round_trip_graph();\n",
    "    let (graph, _, _, _) = native_texture_round_trip_graph();\n",
)
replace(
    test,
    "    let (graph, readback_id, expected) = native_texture_round_trip_graph();\n",
    "    let (graph, readback_id, expected_source, expected) = native_texture_round_trip_graph();\n",
)
replace(
    test,
    "        .expect(\"accepted native texture readback must remain observable\")\n        .clone();\n    let bytes = progress_to_readback(&context, &submission, &readback);\n",
    "        .expect(\"accepted native texture readback must remain observable\")\n        .clone();\n    assert_eq!(readback.source(), &expected_source);\n    let bytes = progress_to_readback(&context, &submission, &readback);\n",
)
replace(
    test,
    "fn direct_texture_copy_graph(\n    bytes_per_row: u32,\n) -> (GpuPreparedWorkGraph, Vec<GpuReadbackId>, Vec<Vec<u8>>, u64) {\n",
    "fn direct_texture_copy_graph(\n    bytes_per_row: u32,\n) -> (\n    GpuPreparedWorkGraph,\n    Vec<GpuReadbackId>,\n    Vec<GpuTransferRegion>,\n    Vec<Vec<u8>>,\n    u64,\n) {\n",
)
replace(
    test,
    "    let mut readback_ids = Vec::new();\n    let mut readbacks = Vec::new();\n",
    "    let mut readback_ids = Vec::new();\n    let mut readback_sources = Vec::new();\n    let mut readbacks = Vec::new();\n",
    count=1,
)
replace(
    test,
    "        let id = GpuReadbackId::allocate().unwrap();\n        readback_ids.push(id);\n        readbacks.push(GpuReadbackOperation::new(region.into(), id).unwrap());\n",
    "        let id = GpuReadbackId::allocate().unwrap();\n        let source = GpuTransferRegion::Buffer(region);\n        readback_ids.push(id);\n        readback_sources.push(source.clone());\n        readbacks.push(GpuReadbackOperation::new(source, id).unwrap());\n",
    count=1,
)
replace(
    test,
    "        readback_ids,\n        expected_rows,\n        footprint,\n    )\n}\n\n#[test]\n#[ignore = \"requires a real Vulkan fallback adapter; executed by RunenGPU Native Conformance CI\"]\nfn native_texture_copy_executes_all_directions_without_copy_scratch() {\n",
    "        readback_ids,\n        readback_sources,\n        expected_rows,\n        footprint,\n    )\n}\n\n#[test]\n#[ignore = \"requires a real Vulkan fallback adapter; executed by RunenGPU Native Conformance CI\"]\nfn native_texture_copy_executes_all_directions_without_copy_scratch() {\n",
)
replace(
    test,
    "    let (graph, readback_ids, expected_rows, footprint) = direct_texture_copy_graph(BYTES_PER_ROW);\n",
    "    let (graph, readback_ids, expected_sources, expected_rows, footprint) =\n        direct_texture_copy_graph(BYTES_PER_ROW);\n",
)
replace(
    test,
    "    for (readback_id, expected) in readback_ids.into_iter().zip(expected_rows) {\n",
    "    for ((readback_id, expected_source), expected) in readback_ids\n        .into_iter()\n        .zip(expected_sources)\n        .zip(expected_rows)\n    {\n",
)
replace(
    test,
    "            .expect(\"accepted direct-copy row readback must remain observable\")\n            .clone();\n        let bytes = progress_to_readback(&context, &submission, &readback);\n",
    "            .expect(\"accepted direct-copy row readback must remain observable\")\n            .clone();\n        assert_eq!(readback.source(), &expected_source);\n        let bytes = progress_to_readback(&context, &submission, &readback);\n",
)
replace(
    test,
    "    let (graph, _, _, _) = direct_texture_copy_graph(DIRECT_COPY_ROW_BYTES);\n",
    "    let (graph, _, _, _, _) = direct_texture_copy_graph(DIRECT_COPY_ROW_BYTES);\n",
)
