use runen_gpu::{
    GpuContext, GpuReadbackBytes, GpuReadbackId, GpuReadbackStatus, GpuSubmission,
    GpuSubmissionStatus,
};

#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

#[cfg(target_arch = "wasm32")]
const MAX_PROGRESS_TICKS: usize = 4_000;
#[cfg(not(target_arch = "wasm32"))]
const MAX_PROGRESS_DURATION: Duration = Duration::from_secs(15);

#[cfg(target_arch = "wasm32")]
struct YieldOnce(bool);

#[cfg(target_arch = "wasm32")]
impl std::future::Future for YieldOnce {
    type Output = ();

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        _context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.0 {
            std::task::Poll::Ready(())
        } else {
            self.0 = true;
            std::task::Poll::Pending
        }
    }
}

async fn progress_yield() {
    #[cfg(target_arch = "wasm32")]
    YieldOnce(false).await;
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::yield_now();
}

pub(crate) async fn wait_for_readback(
    context: &GpuContext,
    submission: &GpuSubmission,
    id: GpuReadbackId,
    proof: impl Into<String>,
) -> GpuReadbackBytes {
    let proof = proof.into();
    let readback = submission.readback(id).unwrap().clone();

    #[cfg(target_arch = "wasm32")]
    let mut progress_ticks = 0_usize;
    #[cfg(not(target_arch = "wasm32"))]
    let deadline = Instant::now() + MAX_PROGRESS_DURATION;

    loop {
        context.progress();
        match readback.status() {
            GpuReadbackStatus::Ready(bytes)
                if matches!(submission.status(), GpuSubmissionStatus::Completed) =>
            {
                return bytes;
            }
            GpuReadbackStatus::Ready(_) | GpuReadbackStatus::Pending => {}
            GpuReadbackStatus::Failed(error) => {
                panic!("{proof} readback failed: {error:?}")
            }
        }
        if let GpuSubmissionStatus::Failed(error) = submission.status() {
            panic!("{proof} submission failed: {error:?}");
        }

        #[cfg(not(target_arch = "wasm32"))]
        assert!(
            Instant::now() < deadline,
            "{proof} proof timed out after {MAX_PROGRESS_DURATION:?}"
        );

        progress_yield().await;

        #[cfg(target_arch = "wasm32")]
        {
            progress_ticks += 1;
            assert!(
                progress_ticks < MAX_PROGRESS_TICKS,
                "{proof} proof exceeded its bounded browser progress budget"
            );
        }
    }
}
