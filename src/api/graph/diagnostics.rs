use super::super::{
    GpuExportKey, GpuWorkGraphCause, GpuWorkGraphError, GpuWorkGraphErrorContext, GpuWorkResourceId,
};
use super::{
    authoring::{GpuWorkFragment, GpuWorkNode},
    dependency::GpuWorkDependency,
    identity::GpuPreparedWorkNodeId,
    initialization::GpuPreparedResourceInitialization,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuPreparedWorkDiagnostic {
    Dependency(GpuWorkDependency),
    ResourceInitialization(GpuPreparedResourceInitialization),
    Output {
        export_key: GpuExportKey,
        resource: GpuWorkResourceId,
    },
}

#[derive(Clone, Copy)]
pub(super) struct GraphErrorOrigin<'a> {
    fragment: Option<&'a GpuWorkFragment>,
    node: Option<&'a GpuWorkNode>,
}

impl<'a> GraphErrorOrigin<'a> {
    pub(super) const fn new(
        fragment: Option<&'a GpuWorkFragment>,
        node: Option<&'a GpuWorkNode>,
    ) -> Self {
        Self { fragment, node }
    }
}

pub(super) fn graph_error(
    operation: &'static str,
    graph_label: &str,
    origin: GraphErrorOrigin<'_>,
    prepared_node: Option<GpuPreparedWorkNodeId>,
    resource: Option<GpuWorkResourceId>,
    cause: GpuWorkGraphCause,
    correction: &'static str,
) -> GpuWorkGraphError {
    GpuWorkGraphError::invalid(
        operation,
        GpuWorkGraphErrorContext::new(
            graph_label,
            origin
                .fragment
                .map(|fragment| fragment.label().as_str().to_string()),
            origin.node.map(|node| node.label().as_str().to_string()),
            prepared_node,
            resource,
            None,
            origin
                .node
                .map(|node| node.provenance().clone())
                .or_else(|| {
                    origin
                        .fragment
                        .map(|fragment| fragment.provenance().clone())
                }),
        ),
        cause,
        correction,
    )
}

pub(super) fn graph_error_with_region(
    operation: &'static str,
    graph_label: &str,
    origin: GraphErrorOrigin<'_>,
    prepared_node: Option<GpuPreparedWorkNodeId>,
    region: (GpuWorkResourceId, String),
    cause: GpuWorkGraphCause,
    correction: &'static str,
) -> GpuWorkGraphError {
    let (resource, region) = region;
    GpuWorkGraphError::invalid(
        operation,
        GpuWorkGraphErrorContext::new(
            graph_label,
            origin
                .fragment
                .map(|fragment| fragment.label().as_str().to_string()),
            origin.node.map(|node| node.label().as_str().to_string()),
            prepared_node,
            Some(resource),
            Some(region),
            origin
                .node
                .map(|node| node.provenance().clone())
                .or_else(|| {
                    origin
                        .fragment
                        .map(|fragment| fragment.provenance().clone())
                }),
        ),
        cause,
        correction,
    )
}
