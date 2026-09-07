#[cfg(target_arch = "wasm32")]
#[path = "gpu_offscreen_indexed_native.rs"]
mod retained_offscreen_indexed;
#[cfg(target_arch = "wasm32")]
#[path = "gpu_prefix_scan_native.rs"]
mod retained_prefix_scan;

#[cfg(target_arch = "wasm32")]
mod browser {
    use super::{retained_offscreen_indexed, retained_prefix_scan};
    use runen_gpu::*;
    use std::cell::RefCell;
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll, Waker};

    thread_local! {
        static BROWSER_PROOF: RefCell<Option<Pin<Box<dyn Future<Output = ()>>>>> = RefCell::new(None);
    }

    struct YieldOnce(bool);

    impl Future for YieldOnce {
        type Output = ();

        fn poll(mut self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Self::Output> {
            if self.0 {
                Poll::Ready(())
            } else {
                self.0 = true;
                Poll::Pending
            }
        }
    }

    async fn browser_yield() {
        YieldOnce(false).await;
    }

    async fn wait_for_terminal_readbacks(
        context: &GpuContext,
        submission: &GpuSubmission,
        ids: &[GpuReadbackId],
    ) -> Vec<GpuReadbackBytes> {
        const MAX_PROGRESS_TICKS: usize = 2_000;

        let readbacks = ids
            .iter()
            .map(|id| {
                submission
                    .readback(*id)
                    .expect("retained browser readback must remain observable")
                    .clone()
            })
            .collect::<Vec<_>>();

        for _ in 0..MAX_PROGRESS_TICKS {
            context.progress();

            let mut ready = Vec::with_capacity(readbacks.len());
            let mut all_ready = true;
            for readback in &readbacks {
                match readback.status() {
                    GpuReadbackStatus::Ready(bytes) => ready.push(bytes),
                    GpuReadbackStatus::Failed(failure) => {
                        panic!("RunenGPU browser readback failed: {failure:?}")
                    }
                    GpuReadbackStatus::Pending => all_ready = false,
                }
            }

            match submission.status() {
                GpuSubmissionStatus::Failed(failure) => {
                    panic!("RunenGPU browser submission failed: {failure:?}")
                }
                GpuSubmissionStatus::Completed if all_ready => return ready,
                GpuSubmissionStatus::Accepted | GpuSubmissionStatus::Completed => {}
            }

            browser_yield().await;
        }

        panic!("RunenGPU browser proof exceeded its bounded progress budget")
    }

    fn assert_execution_drained(context: &GpuContext) {
        let stats = context.execution_stats();
        assert_eq!(stats.prepared_submissions(), 0);
        assert_eq!(stats.in_flight_submissions(), 0);
        assert_eq!(stats.upload_bytes_in_flight(), 0);
        assert_eq!(stats.readback_bytes_in_flight(), 0);
        assert_eq!(stats.pending_readbacks(), 0);
    }

    async fn browser_compute_context() -> GpuContext {
        let mut requirements = GpuCapabilityRequirements::new();
        for feature in [GpuCapabilityFeature::Compute, GpuCapabilityFeature::Copy] {
            requirements
                .insert(GpuCapabilityRequirement::Required(feature))
                .unwrap();
        }
        let descriptor = GpuContextDescriptor::new(requirements)
            .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
            .with_label("retained prefix-scan browser conformance");
        let context = GpuContext::request(descriptor)
            .await
            .expect("declared browser-conformance environment must provide WebGPU");
        assert_eq!(
            context.adapter_facts().backend(),
            GpuBackendFamily::BrowserWebGpu,
            "browser compute evidence must execute through BrowserWebGpu"
        );
        context
    }

    fn retained_prefix_scan_graph_label(mode: retained_prefix_scan::ScanMode) -> GpuResourceLabel {
        let mode = match mode {
            retained_prefix_scan::ScanMode::Exclusive => "exclusive",
            retained_prefix_scan::ScanMode::Inclusive => "inclusive",
        };
        GpuResourceLabel::new(format!("prefix scan {mode} prepared graph")).unwrap()
    }

    async fn run_browser_prefix_scan_mode(
        context: &GpuContext,
        sources: &retained_prefix_scan::ProgramSources,
        mode: retained_prefix_scan::ScanMode,
    ) {
        let (fragment, output_id, total_id) = retained_prefix_scan::author_scan(sources, mode);
        let graph =
            GpuPreparedWorkGraph::prepare(retained_prefix_scan_graph_label(mode), [fragment])
                .unwrap();

        let prepared = context.prepare_submission(graph).await.unwrap();
        let submission = context.submit_prepared(prepared).unwrap();
        let readbacks =
            wait_for_terminal_readbacks(context, &submission, &[output_id, total_id]).await;
        assert_eq!(readbacks.len(), 2);
        let output = retained_prefix_scan::decode_u32(&readbacks[0]);
        let total = retained_prefix_scan::decode_u32(&readbacks[1]);
        retained_prefix_scan::assert_exact_output(mode, &output, &total);
    }

    async fn run_browser_prefix_scan() {
        let context = browser_compute_context().await;
        let sources = retained_prefix_scan::admitted_sources();
        run_browser_prefix_scan_mode(
            &context,
            &sources,
            retained_prefix_scan::ScanMode::Exclusive,
        )
        .await;
        run_browser_prefix_scan_mode(
            &context,
            &sources,
            retained_prefix_scan::ScanMode::Inclusive,
        )
        .await;
        assert_execution_drained(&context);
    }

    async fn browser_offscreen_context() -> GpuContext {
        let descriptor = GpuContextDescriptor::new(
            GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements(),
        )
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
        .with_allowed_backends([GpuBackendFamily::BrowserWebGpu])
        .with_label("retained indexed-offscreen browser conformance");
        let context = GpuContext::request(descriptor)
            .await
            .expect("declared browser-conformance environment must provide offscreen WebGPU");
        assert_eq!(
            context.adapter_facts().backend(),
            GpuBackendFamily::BrowserWebGpu,
            "browser offscreen evidence must execute through BrowserWebGpu"
        );
        context
    }

    async fn run_browser_offscreen_indexed() {
        let context = browser_offscreen_context().await;
        let (graph, readback_id) = retained_offscreen_indexed::render_graph();
        let prepared = context.prepare_submission(graph).await.unwrap();
        let submission = context.submit_prepared(prepared).unwrap();
        let readbacks = wait_for_terminal_readbacks(&context, &submission, &[readback_id]).await;
        assert_eq!(readbacks.len(), 1);
        retained_offscreen_indexed::assert_known_pattern(&readbacks[0]);
        assert_execution_drained(&context);
    }

    async fn run_browser_webgpu_conformance() {
        run_browser_prefix_scan().await;
        run_browser_offscreen_indexed().await;
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn runengpu_browser_start() {
        BROWSER_PROOF.with(|slot| {
            let mut slot = slot.borrow_mut();
            assert!(slot.is_none(), "RunenGPU browser proof already started");
            *slot = Some(Box::pin(run_browser_webgpu_conformance()));
        });
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn runengpu_browser_poll() -> u32 {
        BROWSER_PROOF.with(|slot| {
            let mut slot = slot.borrow_mut();
            let Some(future) = slot.as_mut() else {
                return 2;
            };
            let waker = Waker::noop();
            let mut context = Context::from_waker(waker);
            let status = match future.as_mut().poll(&mut context) {
                Poll::Pending => 0,
                Poll::Ready(()) => 1,
            };
            if status == 1 {
                slot.take();
            }
            status
        })
    }
}

fn main() {}
