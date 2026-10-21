//! Only existing public RunenGPU execution contracts are used here.
use super::frame::Frame;
use runen_gpu::*;
use std::time::{Duration, Instant};

const SOURCE: &str = include_str!("step.wgsl");

pub(crate) enum BatchError {
    Build(String),
    Submission(GpuWorkSubmissionError),
}
impl std::fmt::Display for BatchError {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Build(message) => write!(out, "{message}"),
            Self::Submission(error) => write!(out, "{error}"),
        }
    }
}
impl std::fmt::Debug for BatchError {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, out)
    }
}

pub(crate) struct Batch {
    pub(crate) submission: GpuSubmission,
    pub(crate) ids: Vec<GpuReadbackId>,
}

pub(crate) struct GpuWorkload {
    pub(crate) context: GpuContext,
    resources: GpuResourceScope,
    background: GpuBufferHandle,
    pipeline: GpuComputePipelineDescriptor,
    pixel_count: u32,
}

impl GpuWorkload {
    pub(crate) fn new(width: u32, height: u32) -> Result<Self, String> {
        let pixel_count = width.checked_mul(height).ok_or("pixel count overflow")?;
        let mut requirements = GpuCapabilityRequirements::new();
        for feature in [GpuCapabilityFeature::Compute, GpuCapabilityFeature::Copy] {
            requirements
                .insert(GpuCapabilityRequirement::Required(feature))
                .map_err(|e| e.to_string())?;
        }
        let mut descriptor = GpuContextDescriptor::new(requirements)
            .with_label("RunenGPU #172 streaming adaptive background proof");
        if std::env::var("RUNEN_GPU_STREAMING_REQUIRE_GPU")
            .ok()
            .as_deref()
            == Some("1")
        {
            descriptor = descriptor
                .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
                .with_allowed_backends([GpuBackendFamily::Vulkan]);
        }
        let context =
            pollster::block_on(GpuContext::request(descriptor)).map_err(|e| e.to_string())?;
        let [admitted] = admit_static_wgsl_sources([("example.streaming-background", 1, SOURCE)])
            .map_err(|e| e.to_string())?;
        let pipeline = GpuComputePipelineDescriptor::ordinary(admitted, "cs_main")
            .map_err(|e| e.to_string())?;
        let mut resources = GpuResourceScope::new();
        let background = resources
            .buffer(
                GpuBufferDescriptor::ordinary_owned(
                    "streaming background persistent state",
                    GpuResourceLifetime::Retained,
                    GpuReconstruction::SourceBacked,
                    u64::from(pixel_count) * 4,
                    [GpuBufferUsage::Storage, GpuBufferUsage::CopySource],
                    GpuBufferInitialization::Zeroed,
                )
                .map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
        Ok(Self {
            context,
            resources,
            background,
            pipeline,
            pixel_count,
        })
    }

    fn transient_buffer(
        &mut self,
        label: String,
        len: u64,
        usages: &[GpuBufferUsage],
        initialization: GpuBufferInitialization,
    ) -> Result<GpuBufferHandle, BatchError> {
        let descriptor = GpuBufferDescriptor::ordinary_owned(
            &label,
            GpuResourceLifetime::Transient,
            GpuReconstruction::SourceBacked,
            len,
            usages.iter().copied(),
            initialization,
        )
        .map_err(|e| BatchError::Build(e.to_string()))?;
        self.resources
            .buffer(descriptor)
            .map_err(|e| BatchError::Build(e.to_string()))
    }

    pub(crate) fn submit(&mut self, frames: &[Frame]) -> Result<Batch, BatchError> {
        if frames.is_empty() {
            return Err(BatchError::Build("empty batch".into()));
        }
        let mut items = Vec::with_capacity(frames.len());
        for frame in frames {
            if frame.pixels.len() != self.pixel_count as usize || !matches!(frame.q, 2 | 4) {
                return Err(BatchError::Build(
                    "bad frame dimensions or adaptation".into(),
                ));
            }
            let base = format!("frame-{:06}", frame.id);
            let payload = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
                format!("{base}.pixels"),
                &frame.pixels,
            )
            .map_err(|e| BatchError::Build(e.to_string()))?;
            let parameters = [self.pixel_count, frame.q, 32u32, 0u32];
            let parameter_payload = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
                format!("{base}.params"),
                &parameters,
            )
            .map_err(|e| BatchError::Build(e.to_string()))?;
            let input = self.transient_buffer(
                format!("{base}.input"),
                payload.layout().byte_len(),
                &[GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
                GpuBufferInitialization::Uninitialized,
            )?;
            let params = self.transient_buffer(
                format!("{base}.parameters"),
                parameter_payload.layout().byte_len(),
                &[GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
                GpuBufferInitialization::Uninitialized,
            )?;
            let mask = self.transient_buffer(
                format!("{base}.mask"),
                u64::from(self.pixel_count) * 4,
                &[GpuBufferUsage::Storage, GpuBufferUsage::CopySource],
                // Generic shader writes do not prove whole-buffer initialization coverage.
                GpuBufferInitialization::Zeroed,
            )?;
            items.push((base, input, params, mask, payload, parameter_payload));
        }

        let mut ids = Vec::with_capacity(items.len());
        let dispatch =
            GpuDispatchIntent::direct(GpuDispatchSize::new(self.pixel_count.div_ceil(64), 1, 1));
        let fragment =
            GpuWorkFragment::build("RunenGPU #172 video background update batch", |work| {
                for (name, input, params, mask, pixels, parameters) in &items {
                    work.operation(
                        format!("{name}.upload-input"),
                        GpuUploadOperation::whole_buffer(input, pixels.clone()).unwrap(),
                    )?;
                    work.operation(
                        format!("{name}.upload-parameters"),
                        GpuUploadOperation::whole_buffer(params, parameters.clone()).unwrap(),
                    )?;
                    let bindings = self
                        .pipeline
                        .runtime_bindings([
                            GpuRuntimeBindingValue::whole_buffer(0, 0, &self.background),
                            GpuRuntimeBindingValue::whole_buffer(0, 1, input),
                            GpuRuntimeBindingValue::whole_buffer(0, 2, mask),
                            GpuRuntimeBindingValue::whole_buffer(0, 3, params),
                        ])
                        .unwrap();
                    work.operation(
                        format!("{name}.background-step"),
                        GpuComputeOperation::new(self.pipeline.clone(), bindings, dispatch.clone())
                            .unwrap(),
                    )?;
                    let readback = GpuReadbackOperation::ordinary(
                        GpuBufferRegion::whole(mask).unwrap().into(),
                    )
                    .unwrap();
                    ids.push(readback.id());
                    work.operation(format!("{name}.mask-readback"), readback)?;
                }
                Ok(())
            })
            .map_err(|e| BatchError::Build(e.to_string()))?;
        let submission = pollster::block_on(
            self.context
                .submit_work("RunenGPU #172 GPU background batch", [fragment]),
        )
        .map_err(BatchError::Submission)?;
        Ok(Batch { submission, ids })
    }

    pub(crate) fn ready_masks(batch: &Batch) -> Result<Option<Vec<Vec<u32>>>, String> {
        match batch.submission.status() {
            GpuSubmissionStatus::Accepted => return Ok(None),
            GpuSubmissionStatus::Failed(e) => return Err(format!("GPU submission failed: {e:?}")),
            GpuSubmissionStatus::Completed => {}
        }
        let mut result = Vec::with_capacity(batch.ids.len());
        for &id in &batch.ids {
            let readback = batch
                .submission
                .readback(id)
                .ok_or("missing mask readback")?;
            match readback.status() {
                GpuReadbackStatus::Pending => return Ok(None),
                GpuReadbackStatus::Failed(e) => return Err(format!("mask readback failed: {e:?}")),
                GpuReadbackStatus::Ready(bytes) => result.push(decode(&bytes)?),
            }
        }
        Ok(Some(result))
    }

    pub(crate) fn completed_background(&self) -> Result<Vec<u32>, String> {
        let mut id = None;
        let fragment = GpuWorkFragment::build("streaming background final state", |work| {
            let read = GpuReadbackOperation::ordinary(
                GpuBufferRegion::whole(&self.background).unwrap().into(),
            )
            .unwrap();
            id = Some(read.id());
            work.operation("read final background", read)?;
            Ok(())
        })
        .map_err(|e| e.to_string())?;
        let submitted = pollster::block_on(
            self.context
                .submit_work("streaming background final readback", [fragment]),
        )
        .map_err(|e| e.to_string())?;
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            self.context.progress();
            match submitted.status() {
                GpuSubmissionStatus::Failed(e) => return Err(format!("final H failed: {e:?}")),
                GpuSubmissionStatus::Completed => {
                    match submitted
                        .readback(id.expect("readback ID"))
                        .expect("readback")
                        .status()
                    {
                        GpuReadbackStatus::Ready(bytes) => return decode(&bytes),
                        GpuReadbackStatus::Failed(e) => {
                            return Err(format!("final H readback failed: {e:?}"));
                        }
                        GpuReadbackStatus::Pending => {}
                    }
                }
                GpuSubmissionStatus::Accepted => {}
            }
            if Instant::now() >= deadline {
                return Err("final H readback timed out".into());
            }
            std::thread::yield_now();
        }
    }

    pub(crate) fn background_identity(&self) -> GpuWorkResourceId {
        self.background.diagnostic_identity()
    }
}

fn decode(bytes: &GpuReadbackBytes) -> Result<Vec<u32>, String> {
    let bytes = bytes.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return Err("unaligned u32 readback".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect())
}
