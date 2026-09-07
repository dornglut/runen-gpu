use crate::{GpuCapabilityFeature, GpuContext};

impl GpuContext {
    /// Returns the backend-neutral nanoseconds represented by one timestamp-query tick.
    ///
    /// The scale is observable only when timestamp queries were admitted for this context. This
    /// keeps renderer timing interpretation independent from private WGPU `Queue` authority while
    /// leaving timestamp selection, labeling, and evidence policy outside RunenGPU.
    pub fn timestamp_period_ns(&self) -> Option<f32> {
        if !self
            .device_facts()
            .is_enabled(GpuCapabilityFeature::TimestampQuery)
        {
            return None;
        }
        let period = self.backend.queue.get_timestamp_period();
        (period.is_finite() && period > 0.0).then_some(period)
    }
}
