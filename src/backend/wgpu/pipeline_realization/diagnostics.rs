use super::PipelineRealizationState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PipelineCacheFamily {
    Compute,
    Render,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PipelineCacheObservation {
    Hit,
    Miss,
    Rejected,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct PipelineCacheDiagnosticRecord {
    hits: u64,
    misses: u64,
    rejected: u64,
}

impl PipelineCacheDiagnosticRecord {
    fn observe(&mut self, observation: PipelineCacheObservation) {
        match observation {
            PipelineCacheObservation::Hit => self.hits = self.hits.saturating_add(1),
            PipelineCacheObservation::Miss => self.misses = self.misses.saturating_add(1),
            PipelineCacheObservation::Rejected => self.rejected = self.rejected.saturating_add(1),
        }
    }
}

impl core::fmt::Debug for PipelineCacheDiagnosticRecord {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("PipelineCacheDiagnosticRecord")
            .field("hits", &self.hits)
            .field("misses", &self.misses)
            .field("rejected", &self.rejected)
            .finish()
    }
}

#[derive(Debug, Default)]
pub(super) struct PipelineCacheDiagnosticRegistry {
    compute: PipelineCacheDiagnosticRecord,
    render: PipelineCacheDiagnosticRecord,
}

impl PipelineCacheDiagnosticRegistry {
    fn record_mut(&mut self, family: PipelineCacheFamily) -> &mut PipelineCacheDiagnosticRecord {
        match family {
            PipelineCacheFamily::Compute => &mut self.compute,
            PipelineCacheFamily::Render => &mut self.render,
        }
    }

    #[cfg(test)]
    fn record(&self, family: PipelineCacheFamily) -> PipelineCacheDiagnosticRecord {
        match family {
            PipelineCacheFamily::Compute => self.compute,
            PipelineCacheFamily::Render => self.render,
        }
    }
}

impl PipelineRealizationState {
    pub(super) fn observe_cache(
        &self,
        family: PipelineCacheFamily,
        observation: PipelineCacheObservation,
        request: &str,
    ) {
        self.cache_diagnostics
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .record_mut(family)
            .observe(observation);
        tracing::trace!(
            target: "runengpu.pipeline_cache",
            ?family,
            ?observation,
            request,
            "pipeline cache observation"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_diagnostic_records_keep_hit_miss_and_rejected_distinct() {
        let mut registry = PipelineCacheDiagnosticRegistry::default();
        for observation in [
            PipelineCacheObservation::Miss,
            PipelineCacheObservation::Hit,
            PipelineCacheObservation::Hit,
            PipelineCacheObservation::Rejected,
        ] {
            registry
                .record_mut(PipelineCacheFamily::Compute)
                .observe(observation);
        }

        let record = registry.record(PipelineCacheFamily::Compute);
        assert_eq!(record.hits, 2);
        assert_eq!(record.misses, 1);
        assert_eq!(record.rejected, 1);
        assert_eq!(
            registry.record(PipelineCacheFamily::Render),
            PipelineCacheDiagnosticRecord::default()
        );
    }
}
