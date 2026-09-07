use crate::{GpuResourceLabel, GpuResourceProvenance};
use core::fmt::Write;

/// Private carrier for already-authoritative semantic work identity at the WGPU encoding boundary.
///
/// This does not define a second naming or provenance model. It retains the accepted RunenGPU
/// fragment/node labels and provenance so backend diagnostics can correlate physical passes with
/// the semantic work that produced them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PreparedExecutionObservability {
    fragment_label: GpuResourceLabel,
    node_label: GpuResourceLabel,
    provenance: GpuResourceProvenance,
}

impl PreparedExecutionObservability {
    pub(super) fn new(
        fragment_label: GpuResourceLabel,
        node_label: GpuResourceLabel,
        provenance: GpuResourceProvenance,
    ) -> Self {
        Self {
            fragment_label,
            node_label,
            provenance,
        }
    }

    /// Renders the existing semantic identity for a private backend debug label.
    ///
    /// The structured `GpuResourceProvenance` remains the authority; this string is only a WGPU
    /// diagnostic rendering and is not persisted or exposed as public RunenGPU semantics.
    pub(super) fn debug_label(&self) -> String {
        let mut label = format!(
            "fragment '{}' node '{}' from '{}'",
            self.fragment_label.as_str(),
            self.node_label.as_str(),
            self.provenance.producer().as_str(),
        );
        if let Some(generation) = self.provenance.source_generation() {
            write!(&mut label, " generation {generation}")
                .expect("writing to an owned String cannot fail");
        }
        if let Some(revision) = self.provenance.source_revision() {
            write!(&mut label, " revision '{}'", revision.as_str())
                .expect("writing to an owned String cannot fail");
        }
        label
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn label(value: &str) -> GpuResourceLabel {
        GpuResourceLabel::new(value).unwrap()
    }

    #[test]
    fn backend_debug_label_is_only_a_rendering_of_existing_semantic_facts() {
        let provenance = GpuResourceProvenance::new(
            label("semantic producer"),
            Some(17),
            Some(label("source revision 4")),
        );
        let observability = PreparedExecutionObservability::new(
            label("semantic fragment"),
            label("semantic node"),
            provenance.clone(),
        );

        assert_eq!(observability.fragment_label, label("semantic fragment"));
        assert_eq!(observability.node_label, label("semantic node"));
        assert_eq!(observability.provenance, provenance);
        assert_eq!(
            observability.debug_label(),
            "fragment 'semantic fragment' node 'semantic node' from 'semantic producer' generation 17 revision 'source revision 4'"
        );
    }

    #[test]
    fn backend_debug_label_handles_provenance_without_optional_source_facts() {
        let observability = PreparedExecutionObservability::new(
            label("fragment"),
            label("node"),
            GpuResourceProvenance::new(label("producer"), None, None),
        );

        assert_eq!(
            observability.debug_label(),
            "fragment 'fragment' node 'node' from 'producer'"
        );
    }
}
