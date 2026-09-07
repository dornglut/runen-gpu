mod observability;
mod render;
mod retained_continuity;
mod surface_resources;

use self::observability::PreparedExecutionObservability;
use self::render::{PreparedRenderOperation, encode_render_operation, prepare_render_operation};
use self::retained_continuity::{PreparedRetainedContinuity, RetainedContinuityState};
use self::surface_resources::{
    PreparedSurfaceUse, PreparedTexture, PreparedTextureView, prepare_present_source,
    prepare_texture,
};
use super::WgpuContextState;
use super::health::{WgpuDeviceFaultClass, WgpuDeviceFaultEvidence};
use super::resource_realization::map_texture_aspect;
use super::surface::execution::WgpuSurfaceLeaseGuard;
use crate::{
    GpuBufferInitialization, GpuBufferTextureLayout, GpuCapabilityAdmission, GpuClearOperation,
    GpuContext, GpuContextAffinity, GpuCopyExtent, GpuCopyOperation, GpuDataLayout,
    GpuDispatchSize, GpuExecutionLifecycleState, GpuExecutionPolicy, GpuExecutionStats,
    GpuPipelineRealizationError, GpuPipelineRealizationErrorCategory, GpuPreparedInitialContent,
    GpuPreparedSubmission, GpuPreparedSubmissionRejected, GpuPreparedTextureData,
    GpuPreparedWorkGraph, GpuProgramBindingRealizationError,
    GpuProgramBindingRealizationErrorCategory, GpuReadback, GpuReadbackBytes, GpuReadbackId,
    GpuReadbackStatus, GpuRealizedBindGroup, GpuRealizedBuffer, GpuRealizedComputePipeline,
    GpuRealizedQuerySet, GpuRealizedTexture, GpuResourceLabel, GpuResourceProvenance,
    GpuRetainedResourceContinuity, GpuRuntimeBindingResource, GpuSubmission, GpuSubmissionFailure,
    GpuSubmissionFailureKind, GpuSubmissionId, GpuSubmissionPreparationError,
    GpuSubmissionPreparationErrorKind, GpuSubmissionRejectionKind, GpuSubmissionRejectionReason,
    GpuSubmissionStatus, GpuSurfaceLeaseError, GpuSurfaceLeaseErrorCategory, GpuSurfaceLeaseId,
    GpuTextureAspect, GpuTextureCopyRegion, GpuTextureFormat, GpuTextureInitialization,
    GpuTextureOrigin, GpuTransferRegion, GpuValidatedBindGroupBindings, GpuWorkFragment,
    GpuWorkGraphError, GpuWorkOperation, GpuWorkResourceId, PreparedGpuData, TransferData,
};
use core::num::NonZeroU64;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, COPY_BUFFER_ALIGNMENT, COPY_BYTES_PER_ROW_ALIGNMENT,
    CommandEncoder, CommandEncoderDescriptor, ComputePassDescriptor, ComputePassTimestampWrites,
    Extent3d, MapMode, Origin3d, PollType, QUERY_RESOLVE_BUFFER_ALIGNMENT, TexelCopyBufferInfo,
    TexelCopyBufferLayout, TexelCopyTextureInfo,
};

#[derive(Debug)]
pub(crate) struct WgpuExecutionState {
    affinity: GpuContextAffinity,
    policy: GpuExecutionPolicy,
    next_prepared: AtomicU64,
    next_submission: AtomicU64,
    submission_order: Mutex<()>,
    retained: RetainedContinuityState,
    inner: Mutex<ExecutionInner>,
    events: Arc<Mutex<VecDeque<ExecutionEvent>>>,
}

#[derive(Debug)]
struct ExecutionInner {
    lifecycle: GpuExecutionLifecycleState,
    prepared: BTreeMap<NonZeroU64, Option<PreparedExecutionPlan>>,
    in_flight: BTreeMap<GpuSubmissionId, InFlightSubmission>,
    upload_bytes_in_flight: u64,
    readback_bytes_in_flight: u64,
    pending_readbacks: usize,
}

impl Default for ExecutionInner {
    fn default() -> Self {
        Self {
            lifecycle: GpuExecutionLifecycleState::Running,
            prepared: BTreeMap::new(),
            in_flight: BTreeMap::new(),
            upload_bytes_in_flight: 0,
            readback_bytes_in_flight: 0,
            pending_readbacks: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct PreparedExecutionPlan {
    graph_label: GpuResourceLabel,
    operations: Vec<PreparedExecutionOperation>,
    retained_writes: Vec<BTreeSet<GpuWorkResourceId>>,
    retained_continuity: PreparedRetainedContinuity,
    surface_uses: Vec<PreparedSurfaceUse>,
    upload_bytes: u64,
    readback_bytes: u64,
    readback_ids: Vec<GpuReadbackId>,
    initial_content: Vec<PreparedInitialContentTransfer>,
}

#[derive(Debug, Clone)]
struct PreparedInitialContentTransfer {
    operation: PreparedExecutionOperation,
    staging_bytes: u64,
    record: PreparedInitialContentRecord,
    retained_write: Option<GpuWorkResourceId>,
}

#[derive(Debug, Clone)]
enum PreparedInitialContentRecord {
    Buffer(GpuRealizedBuffer),
    Texture(GpuRealizedTexture),
}

impl PreparedInitialContentRecord {
    fn needs_transfer(&self) -> bool {
        match self {
            Self::Buffer(buffer) => buffer.record.needs_initial_content(),
            Self::Texture(texture) => texture.record.needs_initial_content(),
        }
    }

    fn mark_queued(&self) -> bool {
        match self {
            Self::Buffer(buffer) => buffer.record.mark_initial_content_queued(),
            Self::Texture(texture) => texture.record.mark_initial_content_queued(),
        }
    }

    fn mark_completed(&self) {
        match self {
            Self::Buffer(buffer) => buffer.record.mark_initial_content_completed(),
            Self::Texture(texture) => texture.record.mark_initial_content_completed(),
        }
    }
}

impl PreparedExecutionPlan {
    fn effective_for_acceptance(&self) -> Result<Self, GpuSubmissionRejectionReason> {
        let initial_content = self
            .initial_content
            .iter()
            .filter(|candidate| candidate.record.needs_transfer())
            .cloned()
            .collect::<Vec<_>>();
        let mut upload_bytes = self.upload_bytes;
        for candidate in &initial_content {
            upload_bytes = upload_bytes
                .checked_add(candidate.staging_bytes)
                .ok_or_else(|| {
                    GpuSubmissionRejectionReason::new(
                        GpuSubmissionRejectionKind::UploadBytesInFlightExceeded,
                        "conditional prepared initial-content demand overflowed the normalized upload-byte domain",
                    )
                })?;
        }
        let mut operations = Vec::with_capacity(initial_content.len() + self.operations.len());
        let mut retained_writes =
            Vec::with_capacity(initial_content.len() + self.retained_writes.len());
        for candidate in &initial_content {
            operations.push(candidate.operation.clone());
            retained_writes.push(candidate.retained_write.into_iter().collect());
        }
        operations.extend(self.operations.iter().cloned());
        retained_writes.extend(self.retained_writes.iter().cloned());
        Ok(Self {
            graph_label: self.graph_label.clone(),
            operations,
            retained_writes,
            retained_continuity: self.retained_continuity.clone(),
            surface_uses: self.surface_uses.clone(),
            upload_bytes,
            readback_bytes: self.readback_bytes,
            readback_ids: self.readback_ids.clone(),
            initial_content,
        })
    }

    fn mark_initial_content_queued(&self) -> bool {
        let mut all_queued = true;
        for candidate in &self.initial_content {
            all_queued &= candidate.record.mark_queued();
        }
        all_queued
    }

    fn mark_initial_content_completed(&self) {
        for candidate in &self.initial_content {
            candidate.record.mark_completed();
        }
    }
}

#[derive(Debug, Clone)]
struct PreparedBindGroup {
    index: u32,
    realization: GpuRealizedBindGroup,
    dynamic_offsets: Vec<u32>,
}

#[derive(Debug, Clone)]
enum PreparedComputeDispatch {
    Direct(GpuDispatchSize),
    Indirect {
        arguments: GpuRealizedBuffer,
        offset: u64,
    },
}

#[derive(Debug, Clone)]
struct PreparedTimestampWrites {
    query_set: GpuRealizedQuerySet,
    beginning_of_pass: Option<u32>,
    end_of_pass: Option<u32>,
}

#[derive(Debug, Clone)]
struct BufferReadbackMetadata {
    label: String,
    layout: GpuDataLayout,
    provenance: GpuResourceProvenance,
}

#[derive(Debug, Clone)]
struct TextureReadbackMetadata {
    label: String,
    layout: GpuDataLayout,
    format: GpuTextureFormat,
    provenance: GpuResourceProvenance,
    staging: TextureStagingLayout,
}

#[derive(Debug, Clone)]
enum ReadbackMetadata {
    Buffer(BufferReadbackMetadata),
    Texture(TextureReadbackMetadata),
}

#[derive(Debug, Clone, Copy)]
struct TextureStagingLayout {
    logical_bytes_per_row: u32,
    physical_bytes_per_row: u32,
    rows_per_image: u32,
    image_count: u32,
    logical_byte_len: u64,
    staging_byte_len: u64,
    requires_bytes_per_row: bool,
}

impl TextureStagingLayout {
    fn new(region: &GpuTextureCopyRegion) -> Result<Self, GpuSubmissionPreparationError> {
        let extent = region.extent();
        let bytes_per_texel = region.texture().descriptor().format().bytes_per_texel();
        let logical_bytes_per_row =
            extent.width().checked_mul(bytes_per_texel).ok_or_else(|| {
                texture_staging_preparation_error("texture logical row byte count overflowed")
            })?;
        let rows_per_image = extent.height();
        let image_count = extent.depth_or_layers();
        let row_count = u64::from(rows_per_image)
            .checked_mul(u64::from(image_count))
            .ok_or_else(|| texture_staging_preparation_error("texture row count overflowed"))?;
        let logical_byte_len = u64::from(logical_bytes_per_row)
            .checked_mul(row_count)
            .ok_or_else(|| {
                texture_staging_preparation_error("texture logical staging byte count overflowed")
            })?;
        let requires_bytes_per_row = rows_per_image > 1 || image_count > 1;
        let physical_bytes_per_row = if requires_bytes_per_row {
            u32::try_from(
                align_up(
                    u64::from(logical_bytes_per_row),
                    u64::from(COPY_BYTES_PER_ROW_ALIGNMENT),
                )
                .ok_or_else(|| {
                    texture_staging_preparation_error("texture physical row stride overflowed")
                })?,
            )
            .map_err(|_| {
                texture_staging_preparation_error(
                    "texture physical row stride exceeds the private WGPU u32 domain",
                )
            })?
        } else {
            logical_bytes_per_row
        };
        let copy_footprint = u64::from(physical_bytes_per_row)
            .checked_mul(row_count.saturating_sub(1))
            .and_then(|value| value.checked_add(u64::from(logical_bytes_per_row)))
            .ok_or_else(|| {
                texture_staging_preparation_error("texture staging copy footprint overflowed")
            })?;
        let staging_byte_len =
            align_up(copy_footprint, COPY_BUFFER_ALIGNMENT).ok_or_else(|| {
                texture_staging_preparation_error("texture staging buffer size overflowed")
            })?;
        Ok(Self {
            logical_bytes_per_row,
            physical_bytes_per_row,
            rows_per_image,
            image_count,
            logical_byte_len,
            staging_byte_len,
            requires_bytes_per_row,
        })
    }

    fn row_count(self) -> u64 {
        self.rows_per_image as u64 * self.image_count as u64
    }

    const fn buffer_layout(self) -> TexelCopyBufferLayout {
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: if self.requires_bytes_per_row {
                Some(self.physical_bytes_per_row)
            } else {
                None
            },
            rows_per_image: if self.image_count > 1 {
                Some(self.rows_per_image)
            } else {
                None
            },
        }
    }

    fn write_tightly_packed(
        self,
        destination: &mut wgpu::BufferViewMut,
        source: &[u8],
    ) -> Result<(), GpuSubmissionFailure> {
        let expected_source = usize::try_from(self.logical_byte_len).map_err(|_| {
            texture_staging_submission_error("logical texture payload length exceeds usize")
        })?;
        let expected_destination = usize::try_from(self.staging_byte_len).map_err(|_| {
            texture_staging_submission_error("physical texture staging length exceeds usize")
        })?;
        if source.len() != expected_source || destination.len() != expected_destination {
            return Err(texture_staging_submission_error(
                "texture upload staging lengths no longer match the prepared layout",
            ));
        }
        destination.slice(..).fill(0);
        let logical_row = usize::try_from(self.logical_bytes_per_row).map_err(|_| {
            texture_staging_submission_error("logical texture row length exceeds usize")
        })?;
        let physical_row = usize::try_from(self.physical_bytes_per_row).map_err(|_| {
            texture_staging_submission_error("physical texture row length exceeds usize")
        })?;
        let row_count = usize::try_from(self.row_count()).map_err(|_| {
            texture_staging_submission_error("texture staging row count exceeds usize")
        })?;
        for row in 0..row_count {
            let source_start = row.checked_mul(logical_row).ok_or_else(|| {
                texture_staging_submission_error("texture upload source offset overflowed")
            })?;
            let destination_start = row.checked_mul(physical_row).ok_or_else(|| {
                texture_staging_submission_error("texture upload staging offset overflowed")
            })?;
            let source_end = source_start.checked_add(logical_row).ok_or_else(|| {
                texture_staging_submission_error("texture upload source range overflowed")
            })?;
            let destination_end = destination_start.checked_add(logical_row).ok_or_else(|| {
                texture_staging_submission_error("texture upload staging range overflowed")
            })?;
            let source_row = source.get(source_start..source_end).ok_or_else(|| {
                texture_staging_submission_error("texture upload source row is out of bounds")
            })?;
            destination
                .slice(destination_start..destination_end)
                .copy_from_slice(source_row);
        }
        Ok(())
    }

    fn normalize_mapped(self, source: &[u8]) -> Result<Vec<u8>, GpuSubmissionFailure> {
        let expected_source = usize::try_from(self.staging_byte_len).map_err(|_| {
            texture_staging_submission_error("physical texture readback length exceeds usize")
        })?;
        if source.len() != expected_source {
            return Err(texture_staging_submission_error(
                "mapped texture readback length no longer matches the prepared staging layout",
            ));
        }
        let logical_len = usize::try_from(self.logical_byte_len).map_err(|_| {
            texture_staging_submission_error("logical texture readback length exceeds usize")
        })?;
        let logical_row = usize::try_from(self.logical_bytes_per_row).map_err(|_| {
            texture_staging_submission_error("logical texture row length exceeds usize")
        })?;
        let physical_row = usize::try_from(self.physical_bytes_per_row).map_err(|_| {
            texture_staging_submission_error("physical texture row length exceeds usize")
        })?;
        let row_count = usize::try_from(self.row_count()).map_err(|_| {
            texture_staging_submission_error("texture readback row count exceeds usize")
        })?;
        let mut normalized = Vec::with_capacity(logical_len);
        for row in 0..row_count {
            let source_start = row.checked_mul(physical_row).ok_or_else(|| {
                texture_staging_submission_error("texture readback staging offset overflowed")
            })?;
            let source_end = source_start.checked_add(logical_row).ok_or_else(|| {
                texture_staging_submission_error("texture readback staging range overflowed")
            })?;
            normalized.extend_from_slice(source.get(source_start..source_end).ok_or_else(
                || {
                    texture_staging_submission_error(
                        "texture readback staging row is out of bounds",
                    )
                },
            )?);
        }
        if normalized.len() != logical_len {
            return Err(texture_staging_submission_error(
                "normalized texture readback length disagrees with the prepared logical layout",
            ));
        }
        Ok(normalized)
    }
}

#[derive(Debug, Clone)]
enum PreparedExecutionOperation {
    Upload {
        destination: GpuRealizedBuffer,
        offset: u64,
        payload: PreparedGpuData<TransferData>,
    },
    TextureUpload {
        destination: PreparedTexture,
        region: GpuTextureCopyRegion,
        staging: TextureStagingLayout,
        payload: PreparedGpuData<TransferData>,
    },
    Compute {
        observability: PreparedExecutionObservability,
        pipeline: GpuRealizedComputePipeline,
        bind_groups: Vec<PreparedBindGroup>,
        dispatch: PreparedComputeDispatch,
        timestamp_writes: Option<PreparedTimestampWrites>,
    },
    Render {
        observability: PreparedExecutionObservability,
        operation: PreparedRenderOperation,
    },
    Copy {
        source: GpuRealizedBuffer,
        source_offset: u64,
        destination: GpuRealizedBuffer,
        destination_offset: u64,
        size: u64,
    },
    BufferToTextureCopy {
        source: GpuRealizedBuffer,
        layout: GpuBufferTextureLayout,
        destination: PreparedTexture,
        region: GpuTextureCopyRegion,
    },
    TextureToBufferCopy {
        source: PreparedTexture,
        region: GpuTextureCopyRegion,
        destination: GpuRealizedBuffer,
        layout: GpuBufferTextureLayout,
    },
    TextureToTextureCopy {
        source: PreparedTexture,
        source_region: GpuTextureCopyRegion,
        destination: PreparedTexture,
        destination_region: GpuTextureCopyRegion,
    },
    BufferZero {
        destination: GpuRealizedBuffer,
        offset: u64,
        size: u64,
    },
    Resolve {
        source: GpuRealizedQuerySet,
        query_range: std::ops::Range<u32>,
        destination: GpuRealizedBuffer,
        destination_offset: u64,
    },
    Readback {
        id: GpuReadbackId,
        source: GpuRealizedBuffer,
        source_offset: u64,
        size: u64,
        metadata: BufferReadbackMetadata,
    },
    TextureReadback {
        id: GpuReadbackId,
        source: PreparedTexture,
        region: GpuTextureCopyRegion,
        staging: TextureStagingLayout,
        metadata: TextureReadbackMetadata,
    },
    Present {
        source: PreparedSurfaceUse,
    },
}

#[derive(Debug)]
struct InFlightSubmission {
    status: Arc<Mutex<GpuSubmissionStatus>>,
    readbacks: BTreeMap<GpuReadbackId, InFlightReadback>,
    plan: Option<PreparedExecutionPlan>,
    submitted_retained_writes: BTreeSet<GpuWorkResourceId>,
    upload_staging: Vec<Arc<Buffer>>,
    upload_bytes: u64,
    submission_terminal: bool,
}

#[derive(Debug)]
struct InFlightReadback {
    status: Arc<Mutex<GpuReadbackStatus>>,
    staging: Option<Arc<Buffer>>,
    size: u64,
    metadata: ReadbackMetadata,
    terminal: bool,
}

#[derive(Debug)]
enum ExecutionEvent {
    SubmissionCompleted(GpuSubmissionId),
    ReadbackMapped {
        submission: GpuSubmissionId,
        readback: GpuReadbackId,
        result: Result<(), String>,
    },
}

struct PreparationReservation {
    execution: Arc<WgpuExecutionState>,
    ticket: NonZeroU64,
    committed: bool,
}

impl PreparationReservation {
    fn commit(
        mut self,
        plan: PreparedExecutionPlan,
    ) -> Result<NonZeroU64, GpuSubmissionPreparationError> {
        self.execution.commit_prepared(self.ticket, plan)?;
        self.committed = true;
        Ok(self.ticket)
    }
}

impl Drop for PreparationReservation {
    fn drop(&mut self) {
        if !self.committed {
            self.execution.release_prepared(self.ticket);
        }
    }
}

struct AcceptedPlan {
    id: GpuSubmissionId,
    plan: PreparedExecutionPlan,
    status: Arc<Mutex<GpuSubmissionStatus>>,
    readbacks: Vec<GpuReadback>,
}

struct EncodedSubmission {
    upload_staging: Vec<Arc<Buffer>>,
    readback_staging: Vec<(GpuReadbackId, Arc<Buffer>)>,
}

struct MaterializedStaging {
    encoded: EncodedSubmission,
    uploads: BTreeMap<usize, Arc<Buffer>>,
    readbacks: BTreeMap<usize, Arc<Buffer>>,
}

struct EncodedSegment {
    command_buffer: wgpu::CommandBuffer,
    readback_staging: Vec<(GpuReadbackId, Arc<Buffer>)>,
    retained_writes: BTreeSet<GpuWorkResourceId>,
    present_after: Option<PreparedSurfaceUse>,
}

impl WgpuExecutionState {
    pub(crate) fn new(affinity: GpuContextAffinity, policy: GpuExecutionPolicy) -> Self {
        Self {
            affinity,
            policy,
            next_prepared: AtomicU64::new(1),
            next_submission: AtomicU64::new(1),
            submission_order: Mutex::new(()),
            retained: RetainedContinuityState::new(affinity),
            inner: Mutex::new(ExecutionInner::default()),
            events: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    pub(crate) const fn policy(&self) -> GpuExecutionPolicy {
        self.policy
    }

    pub(crate) fn lifecycle_state(&self) -> GpuExecutionLifecycleState {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .lifecycle
    }

    pub(crate) fn begin_shutdown(&self) -> GpuExecutionLifecycleState {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if inner.lifecycle == GpuExecutionLifecycleState::Running {
            inner.lifecycle = GpuExecutionLifecycleState::ShuttingDown;
            inner.prepared.clear();
        }
        advance_shutdown_if_drained(&mut inner);
        inner.lifecycle
    }

    pub(crate) fn stats(&self) -> GpuExecutionStats {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        GpuExecutionStats::new(
            inner.prepared.len(),
            inner
                .in_flight
                .values()
                .filter(|record| !record.submission_terminal)
                .count(),
            inner.upload_bytes_in_flight,
            inner.readback_bytes_in_flight,
            inner.pending_readbacks,
        )
    }

    fn reserve_prepared(
        self: &Arc<Self>,
    ) -> Result<PreparationReservation, GpuSubmissionPreparationError> {
        let mut inner = self.inner.lock().map_err(|_| {
            GpuSubmissionPreparationError::new(
                GpuSubmissionPreparationErrorKind::ContextOrDeviceUnavailableOrLost,
                "execution preparation authority is unavailable",
            )
        })?;
        if inner.lifecycle != GpuExecutionLifecycleState::Running {
            return Err(preparation_not_running(inner.lifecycle));
        }
        if inner.prepared.len() >= self.policy.max_prepared_submissions().get() {
            return Err(GpuSubmissionPreparationError::new(
                GpuSubmissionPreparationErrorKind::PreparedCapacityExceeded,
                format!(
                    "prepared submissions: {}/{}",
                    inner.prepared.len(),
                    self.policy.max_prepared_submissions().get()
                ),
            ));
        }
        let ticket = allocate_nonzero(&self.next_prepared).ok_or_else(|| {
            GpuSubmissionPreparationError::new(
                GpuSubmissionPreparationErrorKind::IdentityExhausted,
                "prepared-submission identity space is exhausted",
            )
        })?;
        inner.prepared.insert(ticket, None);
        drop(inner);
        Ok(PreparationReservation {
            execution: Arc::clone(self),
            ticket,
            committed: false,
        })
    }

    fn commit_prepared(
        &self,
        ticket: NonZeroU64,
        plan: PreparedExecutionPlan,
    ) -> Result<(), GpuSubmissionPreparationError> {
        let mut inner = self.inner.lock().map_err(|_| {
            GpuSubmissionPreparationError::new(
                GpuSubmissionPreparationErrorKind::ContextOrDeviceUnavailableOrLost,
                "execution preparation authority is unavailable",
            )
        })?;
        if inner.lifecycle != GpuExecutionLifecycleState::Running {
            return Err(preparation_not_running(inner.lifecycle));
        }
        let Some(slot) = inner.prepared.get_mut(&ticket) else {
            return Err(GpuSubmissionPreparationError::new(
                GpuSubmissionPreparationErrorKind::InternalInvariant,
                "prepared reservation disappeared before publication",
            ));
        };
        if slot.is_some() {
            return Err(GpuSubmissionPreparationError::new(
                GpuSubmissionPreparationErrorKind::InternalInvariant,
                "prepared reservation was published more than once",
            ));
        }
        *slot = Some(plan);
        Ok(())
    }

    pub(crate) fn release_prepared(&self, ticket: NonZeroU64) {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .prepared
            .remove(&ticket);
    }

    fn prepared_surface_uses(
        &self,
        prepared: &GpuPreparedSubmission,
    ) -> Result<Vec<PreparedSurfaceUse>, GpuSubmissionRejectionReason> {
        let inner = self.inner.lock().map_err(|_| {
            GpuSubmissionRejectionReason::new(
                GpuSubmissionRejectionKind::ContextOrDeviceUnavailableOrLost,
                "execution preparation authority is unavailable during surface revalidation",
            )
        })?;
        if inner.lifecycle != GpuExecutionLifecycleState::Running {
            return Err(rejection_not_running(inner.lifecycle));
        }
        let Some(Some(plan)) = inner.prepared.get(&prepared.ticket) else {
            return Err(GpuSubmissionRejectionReason::new(
                GpuSubmissionRejectionKind::PreparedRecordUnavailable,
                "prepared submission is absent or was already consumed",
            ));
        };
        Ok(plan.surface_uses.clone())
    }

    fn accept_prepared(
        &self,
        prepared: &GpuPreparedSubmission,
    ) -> Result<AcceptedPlan, GpuSubmissionRejectionReason> {
        if prepared.affinity.context() != self.affinity.context() {
            return Err(GpuSubmissionRejectionReason::new(
                GpuSubmissionRejectionKind::ForeignContext,
                "prepared submission belongs to another GPU context",
            ));
        }
        if prepared.affinity.generation() != self.affinity.generation() {
            return Err(GpuSubmissionRejectionReason::new(
                GpuSubmissionRejectionKind::StaleDeviceGeneration,
                "prepared submission belongs to a stale device generation",
            ));
        }

        let mut inner = self.inner.lock().map_err(|_| {
            GpuSubmissionRejectionReason::new(
                GpuSubmissionRejectionKind::ContextOrDeviceUnavailableOrLost,
                "execution acceptance authority is unavailable",
            )
        })?;
        if inner.lifecycle != GpuExecutionLifecycleState::Running {
            return Err(rejection_not_running(inner.lifecycle));
        }
        let Some(Some(stored_plan)) = inner.prepared.get(&prepared.ticket) else {
            return Err(GpuSubmissionRejectionReason::new(
                GpuSubmissionRejectionKind::PreparedRecordUnavailable,
                "prepared submission is absent or was already consumed",
            ));
        };
        if inner
            .in_flight
            .values()
            .filter(|record| !record.submission_terminal)
            .count()
            >= self.policy.max_in_flight_submissions().get()
        {
            return Err(GpuSubmissionRejectionReason::new(
                GpuSubmissionRejectionKind::InFlightCapacityExceeded,
                "in-flight submission capacity is occupied",
            ));
        }
        let plan = stored_plan.effective_for_acceptance()?;
        let next_upload = inner
            .upload_bytes_in_flight
            .checked_add(plan.upload_bytes)
            .filter(|value| *value <= self.policy.max_upload_bytes_in_flight())
            .ok_or_else(|| {
                GpuSubmissionRejectionReason::new(
                    GpuSubmissionRejectionKind::UploadBytesInFlightExceeded,
                    "upload staging demand exceeds remaining in-flight capacity",
                )
            })?;
        let next_readback = inner
            .readback_bytes_in_flight
            .checked_add(plan.readback_bytes)
            .filter(|value| *value <= self.policy.max_readback_bytes_in_flight())
            .ok_or_else(|| {
                GpuSubmissionRejectionReason::new(
                    GpuSubmissionRejectionKind::ReadbackBytesInFlightExceeded,
                    "readback staging demand exceeds remaining in-flight capacity",
                )
            })?;
        let next_pending = inner
            .pending_readbacks
            .checked_add(plan.readback_ids.len())
            .filter(|value| *value <= self.policy.max_pending_readbacks())
            .ok_or_else(|| {
                GpuSubmissionRejectionReason::new(
                    GpuSubmissionRejectionKind::PendingReadbacksExceeded,
                    "pending readback count exceeds remaining capacity",
                )
            })?;

        let mut readbacks = BTreeMap::new();
        let mut public_readbacks = Vec::with_capacity(plan.readback_ids.len());
        for operation in &plan.operations {
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
            let readback_status = Arc::new(Mutex::new(GpuReadbackStatus::Pending));
            readbacks.insert(
                readback_id,
                InFlightReadback {
                    status: Arc::clone(&readback_status),
                    staging: None,
                    size,
                    metadata,
                    terminal: false,
                },
            );
            public_readbacks.push(GpuReadback::new(readback_id, readback_status));
        }
        if readbacks.len() != plan.readback_ids.len() {
            return Err(GpuSubmissionRejectionReason::new(
                GpuSubmissionRejectionKind::PreparedRecordUnavailable,
                "prepared readback metadata is incomplete",
            ));
        }

        let Some(stored_plan) = inner.prepared.remove(&prepared.ticket).flatten() else {
            return Err(GpuSubmissionRejectionReason::new(
                GpuSubmissionRejectionKind::PreparedRecordUnavailable,
                "prepared submission disappeared before acceptance",
            ));
        };
        let Some(raw_id) = allocate_nonzero(&self.next_submission) else {
            inner.prepared.insert(prepared.ticket, Some(stored_plan));
            return Err(GpuSubmissionRejectionReason::new(
                GpuSubmissionRejectionKind::IdentityExhausted,
                "submission identity space is exhausted",
            ));
        };
        if let Err(reason) = self
            .retained
            .validate_and_reserve(&plan.retained_continuity)
        {
            inner.prepared.insert(prepared.ticket, Some(stored_plan));
            return Err(reason);
        }
        let id = GpuSubmissionId::from_nonzero(raw_id);
        let status = Arc::new(Mutex::new(GpuSubmissionStatus::Accepted));

        inner.upload_bytes_in_flight = next_upload;
        inner.readback_bytes_in_flight = next_readback;
        inner.pending_readbacks = next_pending;
        inner.in_flight.insert(
            id,
            InFlightSubmission {
                status: Arc::clone(&status),
                readbacks,
                plan: Some(plan.clone()),
                submitted_retained_writes: BTreeSet::new(),
                upload_staging: Vec::new(),
                upload_bytes: plan.upload_bytes,
                submission_terminal: false,
            },
        );

        Ok(AcceptedPlan {
            id,
            plan,
            status,
            readbacks: public_readbacks,
        })
    }

    fn attach_staging(
        &self,
        id: GpuSubmissionId,
        encoded: &EncodedSubmission,
    ) -> Result<(), GpuSubmissionFailure> {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(record) = inner.in_flight.get_mut(&id) else {
            return Err(GpuSubmissionFailure::new(
                GpuSubmissionFailureKind::InternalInvariant,
                "accepted submission disappeared before staging attachment",
            ));
        };
        record.upload_staging = encoded.upload_staging.clone();
        for (readback_id, staging) in &encoded.readback_staging {
            let Some(readback) = record.readbacks.get_mut(readback_id) else {
                return Err(GpuSubmissionFailure::new(
                    GpuSubmissionFailureKind::InternalInvariant,
                    "accepted readback disappeared before staging attachment",
                ));
            };
            readback.staging = Some(Arc::clone(staging));
        }
        Ok(())
    }

    fn mark_segment_may_execute(
        &self,
        id: GpuSubmissionId,
        retained_writes: &BTreeSet<GpuWorkResourceId>,
    ) {
        if retained_writes.is_empty() {
            return;
        }
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(record) = inner.in_flight.get_mut(&id) {
            record
                .submitted_retained_writes
                .extend(retained_writes.iter().copied());
            if let Some(plan) = record.plan.as_ref() {
                self.retained
                    .mark_may_execute(&plan.retained_continuity, retained_writes);
            }
        }
    }

    fn drain_events(&self) {
        loop {
            let event = self
                .events
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .pop_front();
            let Some(event) = event else {
                break;
            };
            match event {
                ExecutionEvent::SubmissionCompleted(id) => self.complete_submission(id),
                ExecutionEvent::ReadbackMapped {
                    submission,
                    readback,
                    result,
                } => self.complete_readback_mapping(submission, readback, result),
            }
        }
    }

    fn complete_submission(&self, id: GpuSubmissionId) {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let upload_release = {
            let Some(record) = inner.in_flight.get_mut(&id) else {
                return;
            };
            if record.submission_terminal {
                return;
            }
            let mut status = record
                .status
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if matches!(*status, GpuSubmissionStatus::Accepted) {
                *status = GpuSubmissionStatus::Completed;
            }
            drop(status);
            record.submission_terminal = true;
            if let Some(plan) = record.plan.as_ref() {
                plan.mark_initial_content_completed();
                self.retained.complete(
                    id,
                    &plan.retained_continuity,
                    &record.submitted_retained_writes,
                );
            }
            record.plan = None;
            record.upload_staging.clear();
            let release = record.upload_bytes;
            record.upload_bytes = 0;
            release
        };
        inner.upload_bytes_in_flight = inner.upload_bytes_in_flight.saturating_sub(upload_release);
        cleanup_submission_if_terminal(&mut inner, id);
        advance_shutdown_if_drained(&mut inner);
    }

    fn complete_readback_mapping(
        &self,
        submission: GpuSubmissionId,
        readback_id: GpuReadbackId,
        result: Result<(), String>,
    ) {
        let (staging, metadata) = {
            let inner = self
                .inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let Some(readback) = inner
                .in_flight
                .get(&submission)
                .and_then(|record| record.readbacks.get(&readback_id))
            else {
                return;
            };
            (
                readback.staging.as_ref().cloned(),
                readback.metadata.clone(),
            )
        };

        let materialized = match (result, staging) {
            (Ok(()), Some(staging)) => {
                let bytes = match staging.slice(..).get_mapped_range() {
                    Ok(view) => {
                        let bytes = view.to_vec();
                        drop(view);
                        Ok(bytes)
                    }
                    Err(error) => Err(GpuSubmissionFailure::new(
                        GpuSubmissionFailureKind::ReadbackMapping,
                        format!("obtain mapped readback range: {error}"),
                    )),
                };
                staging.unmap();
                bytes.and_then(|bytes| materialize_readback(bytes, metadata))
            }
            (Err(detail), _) => Err(GpuSubmissionFailure::new(
                GpuSubmissionFailureKind::ReadbackMapping,
                detail,
            )),
            (Ok(()), None) => Err(GpuSubmissionFailure::new(
                GpuSubmissionFailureKind::InternalInvariant,
                "mapped readback staging record is absent",
            )),
        };

        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let release = {
            let Some(record) = inner.in_flight.get_mut(&submission) else {
                return;
            };
            let Some(readback) = record.readbacks.get_mut(&readback_id) else {
                return;
            };
            if readback.terminal {
                return;
            }
            let mut status = readback
                .status
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *status = match materialized {
                Ok(bytes) => GpuReadbackStatus::Ready(bytes),
                Err(failure) => GpuReadbackStatus::Failed(failure),
            };
            drop(status);
            readback.terminal = true;
            readback.staging = None;
            let release = readback.size;
            readback.size = 0;
            release
        };
        inner.readback_bytes_in_flight = inner.readback_bytes_in_flight.saturating_sub(release);
        inner.pending_readbacks = inner.pending_readbacks.saturating_sub(1);
        cleanup_submission_if_terminal(&mut inner, submission);
        advance_shutdown_if_drained(&mut inner);
    }

    fn fail_submission(&self, id: GpuSubmissionId, failure: GpuSubmissionFailure) {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (upload_release, readback_release, pending_release) = {
            let Some(record) = inner.in_flight.get_mut(&id) else {
                return;
            };
            let upload_release = if record.submission_terminal {
                0
            } else {
                let mut status = record
                    .status
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if matches!(*status, GpuSubmissionStatus::Accepted) {
                    *status = GpuSubmissionStatus::Failed(failure.clone());
                }
                drop(status);
                record.submission_terminal = true;
                if let Some(plan) = record.plan.as_ref() {
                    self.retained.fail_after_acceptance(
                        &plan.retained_continuity,
                        &record.submitted_retained_writes,
                    );
                }
                record.plan = None;
                record.upload_staging.clear();
                let release = record.upload_bytes;
                record.upload_bytes = 0;
                release
            };

            let mut readback_release = 0_u64;
            let mut pending_release = 0_usize;
            for readback in record.readbacks.values_mut() {
                if readback.terminal {
                    continue;
                }
                *readback
                    .status
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) =
                    GpuReadbackStatus::Failed(failure.clone());
                readback.terminal = true;
                readback.staging = None;
                readback_release = readback_release.saturating_add(readback.size);
                pending_release = pending_release.saturating_add(1);
                readback.size = 0;
            }
            (upload_release, readback_release, pending_release)
        };

        inner.upload_bytes_in_flight = inner.upload_bytes_in_flight.saturating_sub(upload_release);
        inner.readback_bytes_in_flight = inner
            .readback_bytes_in_flight
            .saturating_sub(readback_release);
        inner.pending_readbacks = inner.pending_readbacks.saturating_sub(pending_release);
        cleanup_submission_if_terminal(&mut inner, id);
        advance_shutdown_if_drained(&mut inner);
    }

    fn fail_active_for_fault(&self, fault: WgpuDeviceFaultEvidence) {
        let failure = failure_from_device_fault(&fault);
        let ids = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .in_flight
            .keys()
            .copied()
            .collect::<Vec<_>>();
        for id in ids {
            self.fail_submission(id, failure.clone());
        }
    }
}

impl Drop for WgpuExecutionState {
    fn drop(&mut self) {
        let failure = GpuSubmissionFailure::new(
            GpuSubmissionFailureKind::ContextDropped,
            "GPU context owner was dropped before execution reached a terminal observation",
        );
        let inner = self
            .inner
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for record in inner.in_flight.values_mut() {
            let mut submission = record
                .status
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if matches!(*submission, GpuSubmissionStatus::Accepted) {
                *submission = GpuSubmissionStatus::Failed(failure.clone());
            }
            drop(submission);
            for readback in record.readbacks.values_mut() {
                let mut status = readback
                    .status
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if matches!(*status, GpuReadbackStatus::Pending) {
                    *status = GpuReadbackStatus::Failed(failure.clone());
                }
            }
        }
    }
}

impl GpuContext {
    pub fn execution_policy(&self) -> GpuExecutionPolicy {
        self.backend.execution.policy()
    }

    pub fn execution_stats(&self) -> GpuExecutionStats {
        self.backend.execution.stats()
    }

    pub fn execution_lifecycle_state(&self) -> GpuExecutionLifecycleState {
        self.backend.execution.lifecycle_state()
    }

    pub fn begin_shutdown(&self) -> GpuExecutionLifecycleState {
        self.backend.execution.begin_shutdown()
    }

    /// Prepares immutable work against this context generation's completed retained coverage.
    ///
    /// This is the context-aware entry point for work whose graph-entry read safety depends on
    /// retained prior submissions. It lowers through the same canonical graph preparation and
    /// initialization implementation as [`GpuPreparedWorkGraph::prepare`].
    pub fn prepare_work_graph(
        &self,
        label: GpuResourceLabel,
        fragments: impl IntoIterator<Item = GpuWorkFragment>,
    ) -> Result<GpuPreparedWorkGraph, GpuWorkGraphError> {
        let retained = self.backend.execution.retained.coverage_seed();
        GpuPreparedWorkGraph::prepare_with_retained_coverage(label, fragments, &retained)
    }

    /// Returns current retained lifecycle facts for one logical storage identity.
    pub fn retained_resource_continuity(
        &self,
        resource: GpuWorkResourceId,
    ) -> Option<GpuRetainedResourceContinuity> {
        self.backend.execution.retained.snapshot(resource)
    }

    pub async fn prepare_submission(
        &self,
        graph: GpuPreparedWorkGraph,
    ) -> Result<GpuPreparedSubmission, GpuSubmissionPreparationError> {
        if let Some(fault) = self.backend.health.terminal_fault() {
            return Err(GpuSubmissionPreparationError::new(
                GpuSubmissionPreparationErrorKind::ContextOrDeviceUnavailableOrLost,
                fault.detail,
            ));
        }
        GpuCapabilityAdmission::evaluate(
            graph.label().as_str(),
            graph.requirements(),
            self.adapter_facts().supported(),
            self.device_facts().enabled_features(),
        )
        .map_err(|error| {
            GpuSubmissionPreparationError::new(
                GpuSubmissionPreparationErrorKind::CapabilityNotAdmitted,
                error.to_string(),
            )
        })?;

        self.validate_prepared_work_device_facts(&graph)?;
        let reservation = self.backend.execution.reserve_prepared()?;
        let plan = prepare_execution_plan(self, &graph).await?;
        validate_plan_policy(
            plan.upload_bytes,
            plan.readback_bytes,
            plan.readback_ids.len(),
            self.execution_policy(),
        )?;
        let planned_readbacks = plan.readback_ids.clone();
        let ticket = reservation.commit(plan)?;
        Ok(GpuPreparedSubmission::new(
            ticket,
            self.affinity(),
            Arc::downgrade(&self.backend.execution),
            planned_readbacks,
        ))
    }

    pub fn submit_prepared(
        &self,
        mut prepared: GpuPreparedSubmission,
    ) -> Result<GpuSubmission, GpuPreparedSubmissionRejected> {
        let expected_affinity = self.affinity();
        if prepared.affinity.context() != expected_affinity.context() {
            return Err(GpuPreparedSubmissionRejected::new(
                prepared,
                GpuSubmissionRejectionReason::new(
                    GpuSubmissionRejectionKind::ForeignContext,
                    "prepared submission belongs to another GPU context",
                ),
            ));
        }
        if prepared.affinity.generation() != expected_affinity.generation() {
            return Err(GpuPreparedSubmissionRejected::new(
                prepared,
                GpuSubmissionRejectionReason::new(
                    GpuSubmissionRejectionKind::StaleDeviceGeneration,
                    "prepared submission belongs to a stale device generation",
                ),
            ));
        }
        if !prepared
            .execution
            .ptr_eq(&Arc::downgrade(&self.backend.execution))
        {
            return Err(GpuPreparedSubmissionRejected::new(
                prepared,
                GpuSubmissionRejectionReason::new(
                    GpuSubmissionRejectionKind::PreparedRecordUnavailable,
                    "prepared submission belongs to a different execution owner for this context generation",
                ),
            ));
        }

        // Submission IDs define this context owner's execution order. Keep revalidation,
        // irreversible acceptance, physical encoding, segmented queue submission, and Present in
        // one owner-local interval so concurrent callers cannot reorder logical and physical work.
        let _submission_order = self
            .backend
            .execution
            .submission_order
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(fault) = self.backend.health.terminal_fault() {
            return Err(GpuPreparedSubmissionRejected::new(
                prepared,
                GpuSubmissionRejectionReason::new(
                    GpuSubmissionRejectionKind::ContextOrDeviceUnavailableOrLost,
                    fault.detail,
                ),
            ));
        }

        let surface_uses = match self.backend.execution.prepared_surface_uses(&prepared) {
            Ok(uses) => uses,
            Err(reason) => return Err(GpuPreparedSubmissionRejected::new(prepared, reason)),
        };
        let _attribution_gate = self.backend.error_attribution_gate.acquire();
        let mut surface_guard = if let Some(representative) = surface_uses.first() {
            match self
                .backend
                .surfaces
                .execution_lease_guard(representative.lease(), &self.backend.health)
            {
                Ok(guard) => Some(guard),
                Err(error) => {
                    return Err(GpuPreparedSubmissionRejected::new(
                        prepared,
                        GpuSubmissionRejectionReason::from_surface_lease(error),
                    ));
                }
            }
        } else {
            None
        };
        if let Some(guard) = surface_guard.as_mut() {
            for surface in &surface_uses {
                if let Err(error) = guard.validate_and_pin(surface.lease(), surface.resource()) {
                    return Err(GpuPreparedSubmissionRejected::new(
                        prepared,
                        GpuSubmissionRejectionReason::from_surface_lease(error),
                    ));
                }
            }
        }
        if let Some(fault) = self.backend.health.terminal_fault() {
            return Err(GpuPreparedSubmissionRejected::new(
                prepared,
                GpuSubmissionRejectionReason::new(
                    GpuSubmissionRejectionKind::ContextOrDeviceUnavailableOrLost,
                    fault.detail,
                ),
            ));
        }

        let accepted = match self.backend.execution.accept_prepared(&prepared) {
            Ok(accepted) => accepted,
            Err(reason) => return Err(GpuPreparedSubmissionRejected::new(prepared, reason)),
        };
        prepared.disarm();
        let submission = GpuSubmission::new(
            accepted.id,
            self.affinity(),
            Arc::clone(&accepted.status),
            accepted.readbacks,
        );

        if let Err(failure) = encode_submit_and_register(
            &self.backend,
            &self.backend.execution,
            accepted.id,
            &accepted.plan,
            surface_guard.as_mut(),
        ) {
            self.backend.execution.fail_submission(accepted.id, failure);
        }
        Ok(submission)
    }

    pub fn progress(&self) -> GpuExecutionStats {
        let _submission_order = self
            .backend
            .execution
            .submission_order
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.backend.execution.drain_events();
        if let Err(error) = self.backend.device.poll(PollType::Poll) {
            self.backend
                .health
                .mark_scoped_internal(format!("nonblocking WGPU progress poll failed: {error}"));
        }
        self.backend.execution.drain_events();
        if let Some(fault) = self.backend.health.terminal_fault() {
            self.backend.execution.fail_active_for_fault(fault);
        }
        self.backend.execution.stats()
    }
}

async fn prepare_execution_plan(
    context: &GpuContext,
    graph: &GpuPreparedWorkGraph,
) -> Result<PreparedExecutionPlan, GpuSubmissionPreparationError> {
    let mut buffer_cache = BTreeMap::<GpuWorkResourceId, GpuRealizedBuffer>::new();
    let mut texture_cache = BTreeMap::<GpuWorkResourceId, PreparedTexture>::new();
    let mut texture_view_cache = BTreeMap::<GpuWorkResourceId, PreparedTextureView>::new();
    let mut query_set_cache = BTreeMap::<GpuWorkResourceId, GpuRealizedQuerySet>::new();
    let retained_continuity = PreparedRetainedContinuity::from_graph(graph);
    let initial_content =
        prepare_initial_content(context, graph, &mut buffer_cache, &mut texture_cache)?;
    let mut operations = Vec::with_capacity(graph.topological_order().len());
    let mut retained_writes = Vec::with_capacity(graph.topological_order().len());
    let mut surface_uses = Vec::new();
    let mut presented_surface_leases = BTreeSet::<GpuSurfaceLeaseId>::new();
    let mut upload_bytes = 0_u64;
    let mut readback_bytes = 0_u64;
    let mut readback_ids = Vec::new();
    let mut seen_readbacks = BTreeSet::new();

    for id in graph.topological_order() {
        let prepared = graph
            .nodes()
            .iter()
            .find(|prepared| prepared.id() == *id)
            .ok_or_else(|| {
                GpuSubmissionPreparationError::new(
                    GpuSubmissionPreparationErrorKind::InternalInvariant,
                    "prepared topological order references an absent work node",
                )
            })?;
        let retained_node_writes = prepared
            .node()
            .accesses()
            .iter()
            .filter(|access| access.writes())
            .map(|access| access.resource_identity())
            .filter(|identity| retained_continuity.contains(*identity))
            .collect::<BTreeSet<_>>();
        let operation_count = operations.len();
        match prepared.node().operation() {
            GpuWorkOperation::Upload(upload) => match upload.destination() {
                GpuTransferRegion::Buffer(destination) => {
                    let alignment = copy_alignment(context)?;
                    validate_copy_range(
                        destination.range().offset(),
                        destination.range().size(),
                        alignment,
                    )?;
                    let realized =
                        realized_buffer(context, &mut buffer_cache, destination.buffer())?;
                    upload_bytes = checked_staging_demand(
                        upload_bytes,
                        destination.range().size(),
                        GpuSubmissionPreparationErrorKind::UploadDemandExceedsPolicy,
                        "upload",
                    )?;
                    operations.push(PreparedExecutionOperation::Upload {
                        destination: realized,
                        offset: destination.range().offset(),
                        payload: upload.payload().clone(),
                    });
                }
                GpuTransferRegion::Texture(destination) => {
                    let prepared_texture =
                        prepare_texture(context, &mut texture_cache, destination.texture())?;
                    append_texture_surface_use(
                        &mut surface_uses,
                        &presented_surface_leases,
                        &prepared_texture,
                    )?;
                    let staging = TextureStagingLayout::new(destination)?;
                    if upload.payload().layout().byte_len() != staging.logical_byte_len {
                        return Err(GpuSubmissionPreparationError::new(
                            GpuSubmissionPreparationErrorKind::InternalInvariant,
                            "validated texture Upload payload no longer matches its logical region",
                        ));
                    }
                    upload_bytes = checked_staging_demand(
                        upload_bytes,
                        staging.staging_byte_len,
                        GpuSubmissionPreparationErrorKind::UploadDemandExceedsPolicy,
                        "upload",
                    )?;
                    operations.push(PreparedExecutionOperation::TextureUpload {
                        destination: prepared_texture,
                        region: destination.clone(),
                        staging,
                        payload: upload.payload().clone(),
                    });
                }
            },
            GpuWorkOperation::Compute(compute) => {
                let observability = PreparedExecutionObservability::new(
                    prepared.fragment_label().clone(),
                    prepared.node().label().clone(),
                    prepared.node().provenance().clone(),
                );
                operations.push(
                    prepare_compute_operation(
                        context,
                        &mut buffer_cache,
                        &mut query_set_cache,
                        observability,
                        compute,
                    )
                    .await?,
                );
            }
            GpuWorkOperation::Render(render) => {
                let observability = PreparedExecutionObservability::new(
                    prepared.fragment_label().clone(),
                    prepared.node().label().clone(),
                    prepared.node().provenance().clone(),
                );
                let render = prepare_render_operation(
                    context,
                    &mut texture_cache,
                    &mut texture_view_cache,
                    render,
                )
                .await?;
                let mut render_surface_uses = Vec::new();
                render.append_surface_uses(&mut render_surface_uses);
                for surface in render_surface_uses {
                    append_surface_use(&mut surface_uses, &presented_surface_leases, &surface)?;
                }
                operations.push(PreparedExecutionOperation::Render {
                    observability,
                    operation: render,
                });
            }
            GpuWorkOperation::Copy(copy) => match copy {
                GpuCopyOperation::BufferToBuffer {
                    source,
                    destination,
                } => {
                    let alignment = copy_alignment(context)?;
                    validate_copy_range(source.range().offset(), source.range().size(), alignment)?;
                    validate_copy_range(
                        destination.range().offset(),
                        destination.range().size(),
                        alignment,
                    )?;
                    operations.push(PreparedExecutionOperation::Copy {
                        source: realized_buffer(context, &mut buffer_cache, source.buffer())?,
                        source_offset: source.range().offset(),
                        destination: realized_buffer(
                            context,
                            &mut buffer_cache,
                            destination.buffer(),
                        )?,
                        destination_offset: destination.range().offset(),
                        size: source.range().size(),
                    });
                }
                GpuCopyOperation::BufferToTexture {
                    source,
                    destination,
                } => {
                    let realized_source =
                        realized_buffer(context, &mut buffer_cache, source.buffer())?;
                    let prepared_destination =
                        prepare_texture(context, &mut texture_cache, destination.texture())?;
                    append_texture_surface_use(
                        &mut surface_uses,
                        &presented_surface_leases,
                        &prepared_destination,
                    )?;
                    validate_buffer_texture_copy_layout(source, destination)?;
                    operations.push(PreparedExecutionOperation::BufferToTextureCopy {
                        source: realized_source,
                        layout: source.clone(),
                        destination: prepared_destination,
                        region: destination.clone(),
                    });
                }
                GpuCopyOperation::TextureToBuffer {
                    source,
                    destination,
                } => {
                    let prepared_source =
                        prepare_texture(context, &mut texture_cache, source.texture())?;
                    append_texture_surface_use(
                        &mut surface_uses,
                        &presented_surface_leases,
                        &prepared_source,
                    )?;
                    let realized_destination =
                        realized_buffer(context, &mut buffer_cache, destination.buffer())?;
                    validate_buffer_texture_copy_layout(destination, source)?;
                    operations.push(PreparedExecutionOperation::TextureToBufferCopy {
                        source: prepared_source,
                        region: source.clone(),
                        destination: realized_destination,
                        layout: destination.clone(),
                    });
                }
                GpuCopyOperation::TextureToTexture {
                    source,
                    destination,
                } => {
                    let prepared_source =
                        prepare_texture(context, &mut texture_cache, source.texture())?;
                    let prepared_destination =
                        prepare_texture(context, &mut texture_cache, destination.texture())?;
                    append_texture_surface_use(
                        &mut surface_uses,
                        &presented_surface_leases,
                        &prepared_source,
                    )?;
                    append_texture_surface_use(
                        &mut surface_uses,
                        &presented_surface_leases,
                        &prepared_destination,
                    )?;
                    operations.push(PreparedExecutionOperation::TextureToTextureCopy {
                        source: prepared_source,
                        source_region: source.clone(),
                        destination: prepared_destination,
                        destination_region: destination.clone(),
                    });
                }
            },
            GpuWorkOperation::Clear(GpuClearOperation::BufferZero(region)) => {
                let destination = realized_buffer(context, &mut buffer_cache, region.buffer())?;
                validate_copy_range(
                    region.range().offset(),
                    region.range().size(),
                    COPY_BUFFER_ALIGNMENT,
                )?;
                operations.push(PreparedExecutionOperation::BufferZero {
                    destination,
                    offset: region.range().offset(),
                    size: region.range().size(),
                });
            }
            GpuWorkOperation::Resolve(resolve) => {
                validate_query_resolve_offset(resolve.destination_offset())?;
                operations.push(PreparedExecutionOperation::Resolve {
                    source: realized_query_set(context, &mut query_set_cache, resolve.source())?,
                    query_range: resolve.source_range().first()..resolve.source_range().end(),
                    destination: realized_buffer(
                        context,
                        &mut buffer_cache,
                        resolve.destination(),
                    )?,
                    destination_offset: resolve.destination_offset(),
                });
            }
            GpuWorkOperation::Readback(readback) => {
                if !seen_readbacks.insert(readback.id()) {
                    return Err(GpuSubmissionPreparationError::new(
                        GpuSubmissionPreparationErrorKind::InternalInvariant,
                        format!(
                            "duplicate readback identity in one prepared graph: {:?}",
                            readback.id()
                        ),
                    ));
                }
                match readback.source() {
                    GpuTransferRegion::Buffer(source) => {
                        let alignment = copy_alignment(context)?;
                        validate_copy_range(
                            source.range().offset(),
                            source.range().size(),
                            alignment,
                        )?;
                        let size = source.range().size();
                        let common = source.buffer().descriptor().common();
                        let metadata = BufferReadbackMetadata {
                            label: common.label().as_str().to_string(),
                            layout: GpuDataLayout::new(common.label().as_str(), size, 1, size, 1)
                                .map_err(|error| {
                                GpuSubmissionPreparationError::new(
                                    GpuSubmissionPreparationErrorKind::InternalInvariant,
                                    error.to_string(),
                                )
                            })?,
                            provenance: common.provenance().clone(),
                        };
                        readback_bytes = checked_staging_demand(
                            readback_bytes,
                            size,
                            GpuSubmissionPreparationErrorKind::ReadbackDemandExceedsPolicy,
                            "readback",
                        )?;
                        readback_ids.push(readback.id());
                        operations.push(PreparedExecutionOperation::Readback {
                            id: readback.id(),
                            source: realized_buffer(context, &mut buffer_cache, source.buffer())?,
                            source_offset: source.range().offset(),
                            size,
                            metadata,
                        });
                    }
                    GpuTransferRegion::Texture(source) => {
                        let prepared_source =
                            prepare_texture(context, &mut texture_cache, source.texture())?;
                        append_texture_surface_use(
                            &mut surface_uses,
                            &presented_surface_leases,
                            &prepared_source,
                        )?;
                        let staging = TextureStagingLayout::new(source)?;
                        let common = source.texture().descriptor().common();
                        let format = source.texture().descriptor().format();
                        let metadata = TextureReadbackMetadata {
                            label: common.label().as_str().to_string(),
                            layout: GpuDataLayout::new(
                                common.label().as_str(),
                                staging.logical_byte_len,
                                u64::from(format.bytes_per_texel()),
                                u64::from(staging.logical_bytes_per_row),
                                staging.row_count(),
                            )
                            .map_err(|error| {
                                GpuSubmissionPreparationError::new(
                                    GpuSubmissionPreparationErrorKind::InternalInvariant,
                                    error.to_string(),
                                )
                            })?,
                            format,
                            provenance: common.provenance().clone(),
                            staging,
                        };
                        readback_bytes = checked_staging_demand(
                            readback_bytes,
                            staging.staging_byte_len,
                            GpuSubmissionPreparationErrorKind::ReadbackDemandExceedsPolicy,
                            "readback",
                        )?;
                        readback_ids.push(readback.id());
                        operations.push(PreparedExecutionOperation::TextureReadback {
                            id: readback.id(),
                            source: prepared_source,
                            region: source.clone(),
                            staging,
                            metadata,
                        });
                    }
                }
            }
            GpuWorkOperation::Present(present) => {
                let surface = prepare_present_source(context, present.source())?;
                append_surface_use(&mut surface_uses, &presented_surface_leases, &surface)?;
                presented_surface_leases.insert(surface.lease().lease_id());
                operations.push(PreparedExecutionOperation::Present { source: surface });
            }
        }
        if operations.len() != operation_count + 1 {
            return Err(GpuSubmissionPreparationError::new(
                GpuSubmissionPreparationErrorKind::InternalInvariant,
                "one prepared work node no longer lowers to exactly one execution operation",
            ));
        }
        retained_writes.push(retained_node_writes);
    }

    Ok(PreparedExecutionPlan {
        graph_label: graph.label().clone(),
        operations,
        retained_writes,
        retained_continuity,
        surface_uses,
        upload_bytes,
        readback_bytes,
        readback_ids,
        initial_content,
    })
}

fn prepare_initial_content(
    context: &GpuContext,
    graph: &GpuPreparedWorkGraph,
    buffer_cache: &mut BTreeMap<GpuWorkResourceId, GpuRealizedBuffer>,
    texture_cache: &mut BTreeMap<GpuWorkResourceId, PreparedTexture>,
) -> Result<Vec<PreparedInitialContentTransfer>, GpuSubmissionPreparationError> {
    let mut transfers = Vec::with_capacity(graph.initial_content().len());
    for candidate in graph.initial_content() {
        match candidate {
            GpuPreparedInitialContent::Buffer(buffer) => {
                let GpuBufferInitialization::Prepared(payload) =
                    buffer.descriptor().initialization()
                else {
                    return Err(GpuSubmissionPreparationError::new(
                        GpuSubmissionPreparationErrorKind::InternalInvariant,
                        "prepared initial-content candidate no longer carries prepared buffer data",
                    ));
                };
                let size = buffer.descriptor().size_bytes();
                validate_copy_range(0, size, copy_alignment(context)?)?;
                let realized = realized_buffer(context, buffer_cache, buffer)?;
                transfers.push(PreparedInitialContentTransfer {
                    operation: PreparedExecutionOperation::Upload {
                        destination: realized.clone(),
                        offset: 0,
                        payload: payload.clone(),
                    },
                    staging_bytes: size,
                    record: PreparedInitialContentRecord::Buffer(realized),
                    retained_write: buffer
                        .descriptor()
                        .common()
                        .lifetime()
                        .is_retained()
                        .then_some(buffer.diagnostic_identity()),
                });
            }
            GpuPreparedInitialContent::Texture(texture) => {
                let GpuTextureInitialization::Prepared(source) =
                    texture.descriptor().initialization()
                else {
                    return Err(GpuSubmissionPreparationError::new(
                        GpuSubmissionPreparationErrorKind::InternalInvariant,
                        "prepared initial-content candidate no longer carries prepared texture data",
                    ));
                };
                let extent = texture.descriptor().extent();
                let region = GpuTextureCopyRegion::new(
                    texture,
                    0,
                    GpuTextureOrigin::new(0, 0, 0),
                    GpuTextureAspect::All,
                    GpuCopyExtent::new(extent.width(), extent.height(), extent.depth_or_layers())
                        .map_err(|error| {
                        GpuSubmissionPreparationError::new(
                            GpuSubmissionPreparationErrorKind::InternalInvariant,
                            error.to_string(),
                        )
                    })?,
                )
                .map_err(|error| {
                    GpuSubmissionPreparationError::new(
                        GpuSubmissionPreparationErrorKind::InternalInvariant,
                        error.to_string(),
                    )
                })?;
                let staging = TextureStagingLayout::new(&region)?;
                let payload = normalize_prepared_texture_payload(source)?;
                if payload.layout().byte_len() != staging.logical_byte_len {
                    return Err(GpuSubmissionPreparationError::new(
                        GpuSubmissionPreparationErrorKind::InternalInvariant,
                        "normalized prepared texture payload no longer matches the canonical upload region",
                    ));
                }
                let prepared_texture = prepare_texture(context, texture_cache, texture)?;
                let PreparedTexture::Realized(realized) = prepared_texture else {
                    return Err(GpuSubmissionPreparationError::new(
                        GpuSubmissionPreparationErrorKind::InternalInvariant,
                        "prepared initial content cannot target a surface-acquired texture",
                    ));
                };
                transfers.push(PreparedInitialContentTransfer {
                    operation: PreparedExecutionOperation::TextureUpload {
                        destination: PreparedTexture::Realized(realized.clone()),
                        region,
                        staging,
                        payload,
                    },
                    staging_bytes: staging.staging_byte_len,
                    record: PreparedInitialContentRecord::Texture(realized),
                    retained_write: texture
                        .descriptor()
                        .common()
                        .lifetime()
                        .is_retained()
                        .then_some(texture.diagnostic_identity()),
                });
            }
        }
    }
    Ok(transfers)
}

fn normalize_prepared_texture_payload(
    source: &GpuPreparedTextureData,
) -> Result<PreparedGpuData<TransferData>, GpuSubmissionPreparationError> {
    let extent = source.extent();
    let logical_row = extent
        .width()
        .checked_mul(source.format().bytes_per_texel())
        .ok_or_else(|| {
            GpuSubmissionPreparationError::new(
                GpuSubmissionPreparationErrorKind::InternalInvariant,
                "prepared texture logical row byte count overflowed during normalization",
            )
        })?;
    let logical_len = u64::from(logical_row)
        .checked_mul(u64::from(extent.height()))
        .and_then(|value| value.checked_mul(u64::from(extent.depth_or_layers())))
        .ok_or_else(|| {
            GpuSubmissionPreparationError::new(
                GpuSubmissionPreparationErrorKind::InternalInvariant,
                "prepared texture logical byte count overflowed during normalization",
            )
        })?;
    let capacity = usize::try_from(logical_len).map_err(|_| {
        GpuSubmissionPreparationError::new(
            GpuSubmissionPreparationErrorKind::InternalInvariant,
            "prepared texture logical byte count exceeds usize during normalization",
        )
    })?;
    let logical_row = usize::try_from(logical_row).map_err(|_| {
        GpuSubmissionPreparationError::new(
            GpuSubmissionPreparationErrorKind::InternalInvariant,
            "prepared texture logical row byte count exceeds usize during normalization",
        )
    })?;
    let bytes_per_row = usize::try_from(source.bytes_per_row()).map_err(|_| {
        GpuSubmissionPreparationError::new(
            GpuSubmissionPreparationErrorKind::InternalInvariant,
            "prepared texture source row stride exceeds usize during normalization",
        )
    })?;
    let rows_per_image = usize::try_from(source.rows_per_image()).map_err(|_| {
        GpuSubmissionPreparationError::new(
            GpuSubmissionPreparationErrorKind::InternalInvariant,
            "prepared texture source rows-per-image exceeds usize during normalization",
        )
    })?;
    let image_count = usize::try_from(extent.depth_or_layers()).map_err(|_| {
        GpuSubmissionPreparationError::new(
            GpuSubmissionPreparationErrorKind::InternalInvariant,
            "prepared texture image count exceeds usize during normalization",
        )
    })?;
    let row_count = usize::try_from(extent.height()).map_err(|_| {
        GpuSubmissionPreparationError::new(
            GpuSubmissionPreparationErrorKind::InternalInvariant,
            "prepared texture row count exceeds usize during normalization",
        )
    })?;
    let image_stride = if image_count > 1 {
        bytes_per_row.checked_mul(rows_per_image).ok_or_else(|| {
            GpuSubmissionPreparationError::new(
                GpuSubmissionPreparationErrorKind::InternalInvariant,
                "prepared texture source image stride overflowed during normalization",
            )
        })?
    } else {
        0
    };
    let source_bytes = source.data().as_bytes();
    let mut normalized = Vec::with_capacity(capacity);
    for image in 0..image_count {
        let image_base = if image_count > 1 {
            image.checked_mul(image_stride).ok_or_else(|| {
                GpuSubmissionPreparationError::new(
                    GpuSubmissionPreparationErrorKind::InternalInvariant,
                    "prepared texture source image offset overflowed during normalization",
                )
            })?
        } else {
            0
        };
        for row in 0..row_count {
            let row_offset = row.checked_mul(bytes_per_row).ok_or_else(|| {
                GpuSubmissionPreparationError::new(
                    GpuSubmissionPreparationErrorKind::InternalInvariant,
                    "prepared texture source row offset overflowed during normalization",
                )
            })?;
            let start = image_base.checked_add(row_offset).ok_or_else(|| {
                GpuSubmissionPreparationError::new(
                    GpuSubmissionPreparationErrorKind::InternalInvariant,
                    "prepared texture source offset overflowed during normalization",
                )
            })?;
            let end = start.checked_add(logical_row).ok_or_else(|| {
                GpuSubmissionPreparationError::new(
                    GpuSubmissionPreparationErrorKind::InternalInvariant,
                    "prepared texture source row range overflowed during normalization",
                )
            })?;
            normalized.extend_from_slice(source_bytes.get(start..end).ok_or_else(|| {
                GpuSubmissionPreparationError::new(
                    GpuSubmissionPreparationErrorKind::InternalInvariant,
                    "checked prepared texture source row became out of bounds during normalization",
                )
            })?);
        }
    }
    if normalized.len() != capacity {
        return Err(GpuSubmissionPreparationError::new(
            GpuSubmissionPreparationErrorKind::InternalInvariant,
            "normalized prepared texture byte length disagrees with the logical texture extent",
        ));
    }
    PreparedGpuData::<TransferData>::from_pod_transfer(
        "prepared texture initial content",
        &normalized,
        source.data().provenance().clone(),
    )
    .map_err(|error| {
        GpuSubmissionPreparationError::new(
            GpuSubmissionPreparationErrorKind::InternalInvariant,
            error.to_string(),
        )
    })
}

fn append_texture_surface_use(
    uses: &mut Vec<PreparedSurfaceUse>,
    presented: &BTreeSet<GpuSurfaceLeaseId>,
    texture: &PreparedTexture,
) -> Result<(), GpuSubmissionPreparationError> {
    if let Some(surface) = texture.surface_use() {
        append_surface_use(uses, presented, surface)?;
    }
    Ok(())
}

fn append_surface_use(
    uses: &mut Vec<PreparedSurfaceUse>,
    presented: &BTreeSet<GpuSurfaceLeaseId>,
    surface: &PreparedSurfaceUse,
) -> Result<(), GpuSubmissionPreparationError> {
    if presented.contains(&surface.lease().lease_id()) {
        return Err(GpuSubmissionPreparationError::from_surface_lease(
            GpuSurfaceLeaseError::new(
                GpuSurfaceLeaseErrorCategory::AlreadyConsumed,
                surface.lease().surface().id(),
                surface.lease().lease_id(),
                "prepared graph uses a surface acquisition lease after an earlier Present consumed it",
            ),
        ));
    }
    uses.push(surface.clone());
    Ok(())
}

async fn prepare_compute_operation(
    context: &GpuContext,
    buffer_cache: &mut BTreeMap<GpuWorkResourceId, GpuRealizedBuffer>,
    query_set_cache: &mut BTreeMap<GpuWorkResourceId, GpuRealizedQuerySet>,
    observability: PreparedExecutionObservability,
    compute: &crate::GpuComputeOperation,
) -> Result<PreparedExecutionOperation, GpuSubmissionPreparationError> {
    let dispatch = if let Some(dispatch) = compute.dispatch().direct_size() {
        PreparedComputeDispatch::Direct(dispatch)
    } else if let Some(arguments) = compute.dispatch().indirect_access() {
        PreparedComputeDispatch::Indirect {
            arguments: realized_buffer(context, buffer_cache, arguments.buffer())?,
            offset: arguments.range().offset(),
        }
    } else {
        return Err(GpuSubmissionPreparationError::new(
            GpuSubmissionPreparationErrorKind::InternalInvariant,
            "validated compute dispatch lost both direct and indirect execution intent",
        ));
    };
    let timestamp_writes = compute
        .timestamp_writes()
        .map(|writes| {
            Ok(PreparedTimestampWrites {
                query_set: realized_query_set(context, query_set_cache, writes.query_set())?,
                beginning_of_pass: writes.beginning_of_pass(),
                end_of_pass: writes.end_of_pass(),
            })
        })
        .transpose()?;

    let descriptor = compute.pipeline();
    let program = context
        .realize_program(descriptor.program())
        .await
        .map_err(preparation_program_binding_failure)?;
    let pipeline_layout = context
        .realize_pipeline_layout(descriptor.layout())
        .await
        .map_err(preparation_program_binding_failure)?;
    let pipeline = context
        .realize_compute_pipeline(descriptor, &program, &pipeline_layout)
        .await
        .map_err(preparation_pipeline_failure)?;

    let mut bind_groups = Vec::with_capacity(compute.bindings().groups().len());
    for group in compute.bindings().groups() {
        let layout = context
            .realize_bind_group_layout(group.layout())
            .await
            .map_err(preparation_program_binding_failure)?;
        let realization = context
            .realize_validated_bind_group(&layout, group.clone())
            .await
            .map_err(preparation_program_binding_failure)?;
        bind_groups.push(PreparedBindGroup {
            index: group.layout().group(),
            realization,
            dynamic_offsets: checked_dynamic_offsets(group)?,
        });
    }

    Ok(PreparedExecutionOperation::Compute {
        observability,
        pipeline,
        bind_groups,
        dispatch,
        timestamp_writes,
    })
}

fn checked_dynamic_offsets(
    group: &GpuValidatedBindGroupBindings,
) -> Result<Vec<u32>, GpuSubmissionPreparationError> {
    let mut offsets = Vec::new();
    for declaration in group.layout().bindings() {
        if !declaration.kind().uses_dynamic_offset() {
            continue;
        }
        let value = group.value(declaration.key().binding()).ok_or_else(|| {
            GpuSubmissionPreparationError::new(
                GpuSubmissionPreparationErrorKind::InternalInvariant,
                format!(
                    "validated dynamic binding {} disappeared before execution preparation",
                    declaration.key()
                ),
            )
        })?;
        for resource in value.resources() {
            let GpuRuntimeBindingResource::Buffer(binding) = resource else {
                return Err(GpuSubmissionPreparationError::new(
                    GpuSubmissionPreparationErrorKind::InternalInvariant,
                    format!(
                        "validated dynamic binding {} no longer contains a buffer",
                        declaration.key()
                    ),
                ));
            };
            let offset = binding.dynamic_offset().ok_or_else(|| {
                GpuSubmissionPreparationError::new(
                    GpuSubmissionPreparationErrorKind::InternalInvariant,
                    format!(
                        "validated dynamic binding {} lost its per-use offset",
                        declaration.key()
                    ),
                )
            })?;
            offsets.push(u32::try_from(offset).map_err(|_| {
                GpuSubmissionPreparationError::new(
                    GpuSubmissionPreparationErrorKind::DynamicOffsetNotEncodable,
                    format!(
                        "logical dynamic offset {offset} for {} exceeds the private WGPU u32 domain",
                        declaration.key()
                    ),
                )
            })?);
        }
    }
    Ok(offsets)
}

fn copy_alignment(context: &GpuContext) -> Result<u64, GpuSubmissionPreparationError> {
    context
        .device_facts()
        .device_limits()
        .alignments()
        .copy_buffer_offset
        .ok_or_else(|| {
            GpuSubmissionPreparationError::new(
                GpuSubmissionPreparationErrorKind::InternalInvariant,
                "created device did not publish its required buffer-copy alignment fact",
            )
        })
}

fn realized_buffer(
    context: &GpuContext,
    cache: &mut BTreeMap<GpuWorkResourceId, GpuRealizedBuffer>,
    handle: &crate::GpuBufferHandle,
) -> Result<GpuRealizedBuffer, GpuSubmissionPreparationError> {
    let identity = handle.diagnostic_identity();
    if let Some(realized) = cache.get(&identity) {
        return Ok(realized.clone());
    }
    let realized = context.realize_buffer(handle).map_err(|error| {
        GpuSubmissionPreparationError::new(
            GpuSubmissionPreparationErrorKind::ResourceRealizationFailed,
            error.to_string(),
        )
    })?;
    cache.insert(identity, realized.clone());
    Ok(realized)
}

fn realized_query_set(
    context: &GpuContext,
    cache: &mut BTreeMap<GpuWorkResourceId, GpuRealizedQuerySet>,
    handle: &crate::GpuQuerySetHandle,
) -> Result<GpuRealizedQuerySet, GpuSubmissionPreparationError> {
    let identity = handle.diagnostic_identity();
    if let Some(realized) = cache.get(&identity) {
        return Ok(realized.clone());
    }
    let realized = context.realize_query_set(handle).map_err(|error| {
        GpuSubmissionPreparationError::new(
            GpuSubmissionPreparationErrorKind::ResourceRealizationFailed,
            error.to_string(),
        )
    })?;
    cache.insert(identity, realized.clone());
    Ok(realized)
}

fn preparation_program_binding_failure(
    error: GpuProgramBindingRealizationError,
) -> GpuSubmissionPreparationError {
    let kind = if error.category()
        == GpuProgramBindingRealizationErrorCategory::ContextOrDeviceUnavailableOrLost
    {
        GpuSubmissionPreparationErrorKind::ContextOrDeviceUnavailableOrLost
    } else {
        GpuSubmissionPreparationErrorKind::ProgramBindingRealizationFailed
    };
    GpuSubmissionPreparationError::new(kind, error.to_string())
}

fn preparation_pipeline_failure(
    error: GpuPipelineRealizationError,
) -> GpuSubmissionPreparationError {
    let kind = if error.category()
        == GpuPipelineRealizationErrorCategory::ContextOrDeviceUnavailableOrLost
    {
        GpuSubmissionPreparationErrorKind::ContextOrDeviceUnavailableOrLost
    } else {
        GpuSubmissionPreparationErrorKind::PipelineRealizationFailed
    };
    GpuSubmissionPreparationError::new(kind, error.to_string())
}

fn validate_copy_range(
    offset: u64,
    size: u64,
    alignment: u64,
) -> Result<(), GpuSubmissionPreparationError> {
    if alignment == 0 || !offset.is_multiple_of(alignment) || !size.is_multiple_of(alignment) {
        return Err(GpuSubmissionPreparationError::new(
            GpuSubmissionPreparationErrorKind::TransferAlignmentNotAdmitted,
            format!(
                "buffer transfer range offset={offset} size={size} is not encodable at admitted copy alignment {alignment}"
            ),
        ));
    }
    Ok(())
}

fn validate_query_resolve_offset(offset: u64) -> Result<(), GpuSubmissionPreparationError> {
    if !offset.is_multiple_of(QUERY_RESOLVE_BUFFER_ALIGNMENT) {
        return Err(GpuSubmissionPreparationError::new(
            GpuSubmissionPreparationErrorKind::TransferAlignmentNotAdmitted,
            format!(
                "query-resolve destination offset {offset} is not encodable at private WGPU alignment {QUERY_RESOLVE_BUFFER_ALIGNMENT}"
            ),
        ));
    }
    Ok(())
}

fn validate_buffer_texture_copy_layout(
    layout: &GpuBufferTextureLayout,
    region: &GpuTextureCopyRegion,
) -> Result<(), GpuSubmissionPreparationError> {
    let extent = region.extent();
    if (extent.height() > 1 || extent.depth_or_layers() > 1)
        && !layout
            .bytes_per_row()
            .is_multiple_of(COPY_BYTES_PER_ROW_ALIGNMENT)
    {
        return Err(GpuSubmissionPreparationError::new(
            GpuSubmissionPreparationErrorKind::TransferAlignmentNotAdmitted,
            format!(
                "buffer-texture copy bytes_per_row {} is not encodable at WGPU command-copy row alignment {}",
                layout.bytes_per_row(),
                COPY_BYTES_PER_ROW_ALIGNMENT
            ),
        ));
    }
    Ok(())
}

fn wgpu_buffer_texture_copy_layout(
    layout: &GpuBufferTextureLayout,
    region: &GpuTextureCopyRegion,
) -> TexelCopyBufferLayout {
    let extent = region.extent();
    let requires_bytes_per_row = extent.height() > 1 || extent.depth_or_layers() > 1;
    TexelCopyBufferLayout {
        offset: layout.byte_offset(),
        bytes_per_row: requires_bytes_per_row.then_some(layout.bytes_per_row()),
        rows_per_image: (extent.depth_or_layers() > 1).then_some(layout.rows_per_image()),
    }
}

fn checked_staging_demand(
    current: u64,
    additional: u64,
    kind: GpuSubmissionPreparationErrorKind,
    label: &'static str,
) -> Result<u64, GpuSubmissionPreparationError> {
    current.checked_add(additional).ok_or_else(|| {
        GpuSubmissionPreparationError::new(
            kind,
            format!("{label} staging byte demand overflowed the normalized u64 domain"),
        )
    })
}

fn validate_plan_policy(
    upload_bytes: u64,
    readback_bytes: u64,
    pending_readbacks: usize,
    policy: GpuExecutionPolicy,
) -> Result<(), GpuSubmissionPreparationError> {
    if upload_bytes > policy.max_upload_bytes_in_flight() {
        return Err(GpuSubmissionPreparationError::new(
            GpuSubmissionPreparationErrorKind::UploadDemandExceedsPolicy,
            format!(
                "submission upload demand {upload_bytes} exceeds policy {}",
                policy.max_upload_bytes_in_flight()
            ),
        ));
    }
    if readback_bytes > policy.max_readback_bytes_in_flight() {
        return Err(GpuSubmissionPreparationError::new(
            GpuSubmissionPreparationErrorKind::ReadbackDemandExceedsPolicy,
            format!(
                "submission readback demand {readback_bytes} exceeds policy {}",
                policy.max_readback_bytes_in_flight()
            ),
        ));
    }
    if pending_readbacks > policy.max_pending_readbacks() {
        return Err(GpuSubmissionPreparationError::new(
            GpuSubmissionPreparationErrorKind::PendingReadbacksExceedPolicy,
            format!(
                "submission readback count {pending_readbacks} exceeds policy {}",
                policy.max_pending_readbacks()
            ),
        ));
    }
    Ok(())
}

fn materialize_staging(
    backend: &WgpuContextState,
    plan: &PreparedExecutionPlan,
) -> Result<MaterializedStaging, GpuSubmissionFailure> {
    let mut upload_staging = Vec::new();
    let mut readback_staging = Vec::new();
    let mut uploads = BTreeMap::new();
    let mut readbacks = BTreeMap::new();

    for (index, operation) in plan.operations.iter().enumerate() {
        match operation {
            PreparedExecutionOperation::Upload { payload, .. } => {
                let staging = Arc::new(backend.device.create_buffer(&BufferDescriptor {
                    label: Some("RunenGPU upload staging"),
                    size: payload.layout().byte_len(),
                    usage: BufferUsages::COPY_SRC,
                    mapped_at_creation: true,
                }));
                {
                    let mut mapped = staging.slice(..).get_mapped_range_mut().map_err(|error| {
                        GpuSubmissionFailure::new(
                            GpuSubmissionFailureKind::BackendValidation,
                            format!("obtain mapped upload staging range: {error}"),
                        )
                    })?;
                    mapped.copy_from_slice(payload.as_bytes());
                }
                staging.unmap();
                uploads.insert(index, Arc::clone(&staging));
                upload_staging.push(staging);
            }
            PreparedExecutionOperation::TextureUpload {
                staging: staging_layout,
                payload,
                ..
            } => {
                let staging = Arc::new(backend.device.create_buffer(&BufferDescriptor {
                    label: Some("RunenGPU texture upload staging"),
                    size: staging_layout.staging_byte_len,
                    usage: BufferUsages::COPY_SRC,
                    mapped_at_creation: true,
                }));
                {
                    let mut mapped = staging.slice(..).get_mapped_range_mut().map_err(|error| {
                        GpuSubmissionFailure::new(
                            GpuSubmissionFailureKind::BackendValidation,
                            format!("obtain mapped texture upload staging range: {error}"),
                        )
                    })?;
                    staging_layout.write_tightly_packed(&mut mapped, payload.as_bytes())?;
                }
                staging.unmap();
                uploads.insert(index, Arc::clone(&staging));
                upload_staging.push(staging);
            }
            PreparedExecutionOperation::Readback {
                id: readback_id,
                size,
                ..
            } => {
                let staging = Arc::new(backend.device.create_buffer(&BufferDescriptor {
                    label: Some("RunenGPU readback staging"),
                    size: *size,
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }));
                readbacks.insert(index, Arc::clone(&staging));
                readback_staging.push((*readback_id, staging));
            }
            PreparedExecutionOperation::TextureReadback {
                id: readback_id,
                staging: staging_layout,
                ..
            } => {
                let staging = Arc::new(backend.device.create_buffer(&BufferDescriptor {
                    label: Some("RunenGPU texture readback staging"),
                    size: staging_layout.staging_byte_len,
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }));
                readbacks.insert(index, Arc::clone(&staging));
                readback_staging.push((*readback_id, staging));
            }
            _ => {}
        }
    }

    Ok(MaterializedStaging {
        encoded: EncodedSubmission {
            upload_staging,
            readback_staging,
        },
        uploads,
        readbacks,
    })
}

fn new_submission_encoder(
    backend: &WgpuContextState,
    graph_label: &GpuResourceLabel,
) -> CommandEncoder {
    backend
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some(graph_label.as_str()),
        })
}

fn encode_submit_and_register(
    backend: &WgpuContextState,
    execution: &Arc<WgpuExecutionState>,
    submission: GpuSubmissionId,
    plan: &PreparedExecutionPlan,
    mut surface_guard: Option<&mut WgpuSurfaceLeaseGuard<'_>>,
) -> Result<(), GpuSubmissionFailure> {
    if let Some(fault) = backend.health.terminal_fault() {
        return Err(failure_from_device_fault(&fault));
    }

    let staging = materialize_staging(backend, plan)?;
    execution.attach_staging(submission, &staging.encoded)?;

    let mut encoder = new_submission_encoder(backend, &plan.graph_label);
    let mut segment_readbacks = Vec::new();
    let mut segment_retained_writes = BTreeSet::new();
    let mut segments = Vec::new();

    for (index, operation) in plan.operations.iter().enumerate() {
        match operation {
            PreparedExecutionOperation::Upload {
                destination,
                offset,
                payload,
            } => {
                let staging_buffer = staging.uploads.get(&index).ok_or_else(|| {
                    GpuSubmissionFailure::new(
                        GpuSubmissionFailureKind::InternalInvariant,
                        "materialized upload staging is absent during encoding",
                    )
                })?;
                encoder.copy_buffer_to_buffer(
                    staging_buffer,
                    0,
                    &destination.record.object,
                    *offset,
                    payload.layout().byte_len(),
                );
            }
            PreparedExecutionOperation::TextureUpload {
                destination,
                region,
                staging: staging_layout,
                ..
            } => {
                let staging_buffer = staging.uploads.get(&index).ok_or_else(|| {
                    GpuSubmissionFailure::new(
                        GpuSubmissionFailureKind::InternalInvariant,
                        "materialized texture upload staging is absent during encoding",
                    )
                })?;
                encoder.copy_buffer_to_texture(
                    TexelCopyBufferInfo {
                        buffer: staging_buffer,
                        layout: staging_layout.buffer_layout(),
                    },
                    texture_copy_info(destination, region, surface_guard.as_deref())?,
                    texture_copy_extent(region),
                );
            }
            PreparedExecutionOperation::Compute {
                observability,
                pipeline,
                bind_groups,
                dispatch,
                timestamp_writes,
            } => {
                let zero_direct = matches!(dispatch, PreparedComputeDispatch::Direct(size) if size.as_array().contains(&0));
                if zero_direct && timestamp_writes.is_none() {
                    continue;
                }
                let realized_groups = bind_groups
                    .iter()
                    .map(|group| &group.realization)
                    .collect::<Vec<_>>();
                let debug_label = observability.debug_label();
                backend
                    .pipeline_realization
                    .with_execution_compute_pipeline(
                        pipeline,
                        &backend.program_binding_realization,
                        |pipeline_object| {
                            backend
                                .program_binding_realization
                                .with_execution_bind_groups(&realized_groups, |group_objects| {
                                    let timestamp_writes =
                                        timestamp_writes.as_ref().map(|writes| {
                                            ComputePassTimestampWrites {
                                                query_set: &writes.query_set.record.object,
                                                beginning_of_pass_write_index: writes
                                                    .beginning_of_pass,
                                                end_of_pass_write_index: writes.end_of_pass,
                                            }
                                        });
                                    let mut pass =
                                        encoder.begin_compute_pass(&ComputePassDescriptor {
                                            label: Some(debug_label.as_str()),
                                            timestamp_writes,
                                        });
                                    pass.set_pipeline(pipeline_object);
                                    for (prepared, object) in bind_groups.iter().zip(group_objects)
                                    {
                                        pass.set_bind_group(
                                            prepared.index,
                                            *object,
                                            &prepared.dynamic_offsets,
                                        );
                                    }
                                    match dispatch {
                                        PreparedComputeDispatch::Direct(size)
                                            if !size.as_array().contains(&0) =>
                                        {
                                            let [x, y, z] = size.as_array();
                                            pass.dispatch_workgroups(x, y, z);
                                        }
                                        PreparedComputeDispatch::Direct(_) => {}
                                        PreparedComputeDispatch::Indirect { arguments, offset } => {
                                            pass.dispatch_workgroups_indirect(
                                                &arguments.record.object,
                                                *offset,
                                            );
                                        }
                                    }
                                })
                        },
                    )
                    .map_err(submission_pipeline_failure)?
                    .map_err(submission_program_binding_failure)?;
            }
            PreparedExecutionOperation::Render {
                observability,
                operation,
            } => {
                encode_render_operation(
                    backend,
                    &mut encoder,
                    observability,
                    operation,
                    surface_guard.as_deref(),
                )?;
            }
            PreparedExecutionOperation::Copy {
                source,
                source_offset,
                destination,
                destination_offset,
                size,
            } => encoder.copy_buffer_to_buffer(
                &source.record.object,
                *source_offset,
                &destination.record.object,
                *destination_offset,
                *size,
            ),
            PreparedExecutionOperation::BufferToTextureCopy {
                source,
                layout,
                destination,
                region,
            } => encoder.copy_buffer_to_texture(
                TexelCopyBufferInfo {
                    buffer: &source.record.object,
                    layout: wgpu_buffer_texture_copy_layout(layout, region),
                },
                texture_copy_info(destination, region, surface_guard.as_deref())?,
                texture_copy_extent(region),
            ),
            PreparedExecutionOperation::TextureToBufferCopy {
                source,
                region,
                destination,
                layout,
            } => encoder.copy_texture_to_buffer(
                texture_copy_info(source, region, surface_guard.as_deref())?,
                TexelCopyBufferInfo {
                    buffer: &destination.record.object,
                    layout: wgpu_buffer_texture_copy_layout(layout, region),
                },
                texture_copy_extent(region),
            ),
            PreparedExecutionOperation::TextureToTextureCopy {
                source,
                source_region,
                destination,
                destination_region,
            } => encoder.copy_texture_to_texture(
                texture_copy_info(source, source_region, surface_guard.as_deref())?,
                texture_copy_info(destination, destination_region, surface_guard.as_deref())?,
                texture_copy_extent(source_region),
            ),
            PreparedExecutionOperation::BufferZero {
                destination,
                offset,
                size,
            } => encoder.clear_buffer(&destination.record.object, *offset, Some(*size)),
            PreparedExecutionOperation::Resolve {
                source,
                query_range,
                destination,
                destination_offset,
            } => encoder.resolve_query_set(
                &source.record.object,
                query_range.clone(),
                &destination.record.object,
                *destination_offset,
            ),
            PreparedExecutionOperation::Readback {
                id: readback_id,
                source,
                source_offset,
                size,
                ..
            } => {
                let staging_buffer = staging.readbacks.get(&index).ok_or_else(|| {
                    GpuSubmissionFailure::new(
                        GpuSubmissionFailureKind::InternalInvariant,
                        "materialized readback staging is absent during encoding",
                    )
                })?;
                encoder.copy_buffer_to_buffer(
                    &source.record.object,
                    *source_offset,
                    staging_buffer,
                    0,
                    *size,
                );
                segment_readbacks.push((*readback_id, Arc::clone(staging_buffer)));
            }
            PreparedExecutionOperation::TextureReadback {
                id: readback_id,
                source,
                region,
                staging: staging_layout,
                ..
            } => {
                let staging_buffer = staging.readbacks.get(&index).ok_or_else(|| {
                    GpuSubmissionFailure::new(
                        GpuSubmissionFailureKind::InternalInvariant,
                        "materialized texture readback staging is absent during encoding",
                    )
                })?;
                encoder.copy_texture_to_buffer(
                    texture_copy_info(source, region, surface_guard.as_deref())?,
                    TexelCopyBufferInfo {
                        buffer: staging_buffer,
                        layout: staging_layout.buffer_layout(),
                    },
                    texture_copy_extent(region),
                );
                segment_readbacks.push((*readback_id, Arc::clone(staging_buffer)));
            }
            PreparedExecutionOperation::Present { source } => {
                let next = new_submission_encoder(backend, &plan.graph_label);
                let current = std::mem::replace(&mut encoder, next);
                segments.push(EncodedSegment {
                    command_buffer: current.finish(),
                    readback_staging: std::mem::take(&mut segment_readbacks),
                    retained_writes: std::mem::take(&mut segment_retained_writes),
                    present_after: Some(source.clone()),
                });
            }
        }
        let operation_retained_writes = plan.retained_writes.get(index).ok_or_else(|| {
            GpuSubmissionFailure::new(
                GpuSubmissionFailureKind::InternalInvariant,
                "retained write metadata no longer aligns with prepared execution operations",
            )
        })?;
        segment_retained_writes.extend(operation_retained_writes.iter().copied());
    }

    segments.push(EncodedSegment {
        command_buffer: encoder.finish(),
        readback_staging: segment_readbacks,
        retained_writes: segment_retained_writes,
        present_after: None,
    });

    if let Some(fault) = backend.health.terminal_fault() {
        return Err(failure_from_device_fault(&fault));
    }

    let segment_count = segments.len();
    for (index, segment) in segments.into_iter().enumerate() {
        register_readback_callbacks(
            execution,
            submission,
            &segment.readback_staging,
            &segment.command_buffer,
        );
        if index + 1 == segment_count {
            register_submission_completion(execution, submission, &segment.command_buffer);
        }
        backend.queue.submit([segment.command_buffer]);
        execution.mark_segment_may_execute(submission, &segment.retained_writes);
        if index == 0 && !plan.mark_initial_content_queued() {
            backend.health.mark_scoped_internal(
                "prepared initial-content state changed outside serialized submission acceptance",
            );
            if let Some(fault) = backend.health.terminal_fault() {
                return Err(failure_from_device_fault(&fault));
            }
            return Err(GpuSubmissionFailure::new(
                GpuSubmissionFailureKind::InternalInvariant,
                "prepared initial-content state could not be marked queued after physical submission",
            ));
        }
        if let Some(surface) = segment.present_after {
            let guard = surface_guard.as_deref_mut().ok_or_else(|| {
                GpuSubmissionFailure::new(
                    GpuSubmissionFailureKind::InternalInvariant,
                    "prepared Present reached physical submission without the validated G7 lease guard",
                )
            })?;
            guard
                .present(&backend.queue, surface.lease(), surface.resource())
                .map_err(GpuSubmissionFailure::from_surface_lease)?;
        }
        if let Some(fault) = backend.health.terminal_fault() {
            return Err(failure_from_device_fault(&fault));
        }
    }
    Ok(())
}

fn texture_copy_info<'a>(
    texture: &'a PreparedTexture,
    region: &GpuTextureCopyRegion,
    surface_guard: Option<&'a WgpuSurfaceLeaseGuard<'_>>,
) -> Result<TexelCopyTextureInfo<'a>, GpuSubmissionFailure> {
    let origin = region.origin();
    Ok(TexelCopyTextureInfo {
        texture: texture.resolve(surface_guard)?,
        mip_level: region.mip_level(),
        origin: Origin3d {
            x: origin.x(),
            y: origin.y(),
            z: origin.z(),
        },
        aspect: map_texture_aspect(region.aspect()),
    })
}

fn texture_copy_extent(region: &GpuTextureCopyRegion) -> Extent3d {
    let extent = region.extent();
    Extent3d {
        width: extent.width(),
        height: extent.height(),
        depth_or_array_layers: extent.depth_or_layers(),
    }
}

fn materialize_readback(
    bytes: Vec<u8>,
    metadata: ReadbackMetadata,
) -> Result<GpuReadbackBytes, GpuSubmissionFailure> {
    match metadata {
        ReadbackMetadata::Buffer(metadata) => GpuReadbackBytes::from_normalized_bytes(
            &metadata.label,
            bytes,
            metadata.layout,
            None,
            metadata.provenance,
        )
        .map_err(|error| {
            GpuSubmissionFailure::new(
                GpuSubmissionFailureKind::InternalInvariant,
                error.to_string(),
            )
        }),
        ReadbackMetadata::Texture(metadata) => {
            let normalized = metadata.staging.normalize_mapped(&bytes)?;
            GpuReadbackBytes::from_normalized_bytes(
                &metadata.label,
                normalized,
                metadata.layout,
                Some(metadata.format),
                metadata.provenance,
            )
            .map_err(|error| {
                GpuSubmissionFailure::new(
                    GpuSubmissionFailureKind::InternalInvariant,
                    error.to_string(),
                )
            })
        }
    }
}

fn align_up(value: u64, alignment: u64) -> Option<u64> {
    if alignment == 0 {
        return None;
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - remainder)
    }
}

fn texture_staging_preparation_error(detail: &'static str) -> GpuSubmissionPreparationError {
    GpuSubmissionPreparationError::new(GpuSubmissionPreparationErrorKind::InternalInvariant, detail)
}

fn texture_staging_submission_error(detail: &'static str) -> GpuSubmissionFailure {
    GpuSubmissionFailure::new(GpuSubmissionFailureKind::InternalInvariant, detail)
}

fn submission_program_binding_failure(
    error: GpuProgramBindingRealizationError,
) -> GpuSubmissionFailure {
    let kind = match error.category() {
        GpuProgramBindingRealizationErrorCategory::BackendResourceExhaustion => {
            GpuSubmissionFailureKind::BackendResourceExhaustion
        }
        GpuProgramBindingRealizationErrorCategory::ContextOrDeviceUnavailableOrLost => {
            GpuSubmissionFailureKind::ContextOrDeviceUnavailableOrLost
        }
        GpuProgramBindingRealizationErrorCategory::ForeignContext
        | GpuProgramBindingRealizationErrorCategory::StaleDeviceGeneration
        | GpuProgramBindingRealizationErrorCategory::ExecutionAuthorityViolation => {
            GpuSubmissionFailureKind::InternalInvariant
        }
        _ => GpuSubmissionFailureKind::BackendValidation,
    };
    GpuSubmissionFailure::new(kind, error.to_string())
}

fn submission_pipeline_failure(error: GpuPipelineRealizationError) -> GpuSubmissionFailure {
    let kind = match error.category() {
        GpuPipelineRealizationErrorCategory::BackendResourceExhaustion => {
            GpuSubmissionFailureKind::BackendResourceExhaustion
        }
        GpuPipelineRealizationErrorCategory::ContextOrDeviceUnavailableOrLost => {
            GpuSubmissionFailureKind::ContextOrDeviceUnavailableOrLost
        }
        GpuPipelineRealizationErrorCategory::ForeignContext
        | GpuPipelineRealizationErrorCategory::StaleDeviceGeneration
        | GpuPipelineRealizationErrorCategory::ExecutionAuthorityViolation => {
            GpuSubmissionFailureKind::InternalInvariant
        }
        _ => GpuSubmissionFailureKind::BackendValidation,
    };
    GpuSubmissionFailure::new(kind, error.to_string())
}

fn register_readback_callbacks(
    execution: &Arc<WgpuExecutionState>,
    submission: GpuSubmissionId,
    readback_staging: &[(GpuReadbackId, Arc<Buffer>)],
    command_buffer: &wgpu::CommandBuffer,
) {
    for (readback, staging) in readback_staging {
        let events = Arc::clone(&execution.events);
        let readback = *readback;
        command_buffer.map_buffer_on_submit(staging, MapMode::Read, .., move |result| {
            events
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push_back(ExecutionEvent::ReadbackMapped {
                    submission,
                    readback,
                    result: result.map_err(|error| error.to_string()),
                });
        });
    }
}

fn register_submission_completion(
    execution: &Arc<WgpuExecutionState>,
    submission: GpuSubmissionId,
    command_buffer: &wgpu::CommandBuffer,
) {
    let events = Arc::clone(&execution.events);
    command_buffer.on_submitted_work_done(move || {
        events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push_back(ExecutionEvent::SubmissionCompleted(submission));
    });
}

fn cleanup_submission_if_terminal(inner: &mut ExecutionInner, id: GpuSubmissionId) {
    let should_remove = inner.in_flight.get(&id).is_some_and(|record| {
        record.submission_terminal && record.readbacks.values().all(|readback| readback.terminal)
    });
    if should_remove {
        inner.in_flight.remove(&id);
    }
}

fn advance_shutdown_if_drained(inner: &mut ExecutionInner) {
    if inner.lifecycle == GpuExecutionLifecycleState::ShuttingDown
        && inner.prepared.is_empty()
        && inner.in_flight.is_empty()
    {
        inner.lifecycle = GpuExecutionLifecycleState::Closed;
    }
}

fn preparation_not_running(state: GpuExecutionLifecycleState) -> GpuSubmissionPreparationError {
    GpuSubmissionPreparationError::new(
        GpuSubmissionPreparationErrorKind::ExecutionNotRunning,
        format!("GPU execution lifecycle is {state:?}; preparation requires Running"),
    )
}

fn rejection_not_running(state: GpuExecutionLifecycleState) -> GpuSubmissionRejectionReason {
    GpuSubmissionRejectionReason::new(
        GpuSubmissionRejectionKind::ExecutionNotRunning,
        format!("GPU execution lifecycle is {state:?}; acceptance requires Running"),
    )
}

fn allocate_nonzero(counter: &AtomicU64) -> Option<NonZeroU64> {
    let value = counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            (current != 0).then_some(if current == u64::MAX { 0 } else { current + 1 })
        })
        .ok()?;
    NonZeroU64::new(value)
}

fn failure_from_device_fault(fault: &WgpuDeviceFaultEvidence) -> GpuSubmissionFailure {
    let kind = match fault.class {
        WgpuDeviceFaultClass::UnexpectedValidation => GpuSubmissionFailureKind::BackendValidation,
        WgpuDeviceFaultClass::OutOfMemory => GpuSubmissionFailureKind::BackendResourceExhaustion,
        WgpuDeviceFaultClass::InternalOrDeviceLost => {
            GpuSubmissionFailureKind::ContextOrDeviceUnavailableOrLost
        }
    };
    GpuSubmissionFailure::new(kind, fault.detail.clone())
}
