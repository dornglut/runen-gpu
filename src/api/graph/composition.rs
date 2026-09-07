use super::super::{
    GpuExportKey, GpuResourceAccess, GpuResourceAccessIntent, GpuWorkGraphCause, GpuWorkGraphError,
    GpuWorkGraphErrorContext, GpuWorkResourceId,
};
use super::{
    authoring::GpuWorkFragment,
    coverage::storage_identity,
    diagnostics::{GraphErrorOrigin, graph_error},
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) type OutputBindings = BTreeMap<GpuExportKey, (usize, usize)>;
pub(super) type ImportBindings = BTreeMap<(usize, usize), (usize, usize)>;
pub(super) type FragmentRelations = BTreeSet<(usize, usize, GpuWorkResourceId)>;

pub(super) fn collect_output_bindings(
    graph_label: &str,
    fragments: &[GpuWorkFragment],
) -> Result<OutputBindings, GpuWorkGraphError> {
    let mut bindings = BTreeMap::new();
    for (fragment_index, fragment) in fragments.iter().enumerate() {
        for (output_index, output) in fragment.outputs().iter().enumerate() {
            let key = output.relationship().export_key().clone();
            if bindings
                .insert(key.clone(), (fragment_index, output_index))
                .is_some()
            {
                return Err(graph_error(
                    "bind GPU work output",
                    graph_label,
                    GraphErrorOrigin::new(Some(fragment), None),
                    None,
                    Some(output.relationship().resource().diagnostic_identity()),
                    GpuWorkGraphCause::DuplicateExportKey,
                    "use each typed export key for exactly one producer output",
                ));
            }
        }
    }
    Ok(bindings)
}

pub(super) fn bind_imports(
    graph_label: &str,
    fragments: &[GpuWorkFragment],
    outputs: &OutputBindings,
) -> Result<(ImportBindings, FragmentRelations), GpuWorkGraphError> {
    let mut bindings = BTreeMap::new();
    let mut relations = BTreeSet::new();
    let mut consumer_sources = BTreeMap::<(usize, GpuWorkResourceId), usize>::new();
    for (consumer_index, fragment) in fragments.iter().enumerate() {
        for (import_index, import) in fragment.imports().iter().enumerate() {
            let Some(&(producer_index, output_index)) = outputs.get(import.producer()) else {
                return Err(graph_error(
                    "bind GPU work import",
                    graph_label,
                    GraphErrorOrigin::new(Some(fragment), None),
                    None,
                    Some(import.resource().diagnostic_identity()),
                    GpuWorkGraphCause::ImportExportMismatch,
                    "provide exactly one output with the imported typed export key",
                ));
            };
            let output = &fragments[producer_index].outputs()[output_index];
            if producer_index == consumer_index
                || output.relationship().resource() != import.resource()
            {
                return Err(graph_error(
                    "bind GPU work import",
                    graph_label,
                    GraphErrorOrigin::new(Some(fragment), None),
                    None,
                    Some(import.resource().diagnostic_identity()),
                    GpuWorkGraphCause::ImportExportMismatch,
                    "bind the exact kind-preserving producer resource from another fragment",
                ));
            }
            let resource = storage_identity(import.resource());
            if consumer_sources
                .insert((consumer_index, resource), producer_index)
                .is_some_and(|existing| existing != producer_index)
            {
                return Err(graph_error(
                    "bind GPU work import",
                    graph_label,
                    GraphErrorOrigin::new(Some(fragment), None),
                    None,
                    Some(resource),
                    GpuWorkGraphCause::AmbiguousWriter,
                    "select one typed producer for each imported storage resource",
                ));
            }
            bindings.insert(
                (consumer_index, import_index),
                (producer_index, output_index),
            );
            relations.insert((producer_index, consumer_index, resource));
        }
    }
    Ok((bindings, relations))
}

pub(super) fn validate_boundary_access_intents(
    graph_label: &str,
    fragments: &[GpuWorkFragment],
) -> Result<(), GpuWorkGraphError> {
    for fragment in fragments {
        for import in fragment.imports() {
            let resource = storage_identity(import.resource());
            if first_fragment_access_intent(fragment, resource)
                != Some(import.required_initial_access())
            {
                return Err(graph_error(
                    "validate GPU work import access",
                    graph_label,
                    GraphErrorOrigin::new(Some(fragment), None),
                    None,
                    Some(resource),
                    GpuWorkGraphCause::ImportExportMismatch,
                    "make the import's required initial access match the consumer fragment's first actual access",
                ));
            }
        }
        for output in fragment.outputs() {
            let resource = storage_identity(output.relationship().resource());
            if last_fragment_access_intent(fragment, resource)
                != Some(output.relationship().required_final_access())
            {
                return Err(graph_error(
                    "validate GPU work output access",
                    graph_label,
                    GraphErrorOrigin::new(Some(fragment), None),
                    None,
                    Some(resource),
                    GpuWorkGraphCause::ImportExportMismatch,
                    "make the output's required final access match the producer fragment's last actual access",
                ));
            }
        }
    }
    Ok(())
}

fn first_fragment_access_intent(
    fragment: &GpuWorkFragment,
    resource: GpuWorkResourceId,
) -> Option<GpuResourceAccessIntent> {
    fragment.nodes().iter().find_map(|node| {
        combined_access_intent(
            node.accesses()
                .iter()
                .filter(|access| access.resource_identity() == resource),
        )
    })
}

fn last_fragment_access_intent(
    fragment: &GpuWorkFragment,
    resource: GpuWorkResourceId,
) -> Option<GpuResourceAccessIntent> {
    fragment.nodes().iter().rev().find_map(|node| {
        combined_access_intent(
            node.accesses()
                .iter()
                .filter(|access| access.resource_identity() == resource),
        )
    })
}

fn combined_access_intent<'a>(
    accesses: impl Iterator<Item = &'a GpuResourceAccess>,
) -> Option<GpuResourceAccessIntent> {
    let mut reads = false;
    let mut writes = false;
    let mut any = false;
    for access in accesses {
        any = true;
        reads |= access.reads();
        writes |= access.writes();
    }
    if !any {
        None
    } else if reads && writes {
        Some(GpuResourceAccessIntent::ReadWrite)
    } else if writes {
        Some(GpuResourceAccessIntent::Write)
    } else {
        Some(GpuResourceAccessIntent::Read)
    }
}

pub(super) fn topological_fragment_order(
    graph_label: &str,
    fragments: &[GpuWorkFragment],
    imports: &ImportBindings,
) -> Result<Vec<usize>, GpuWorkGraphError> {
    let mut outgoing = vec![BTreeSet::<usize>::new(); fragments.len()];
    let mut indegree = vec![0_usize; fragments.len()];
    for &(producer, _) in imports.values() {
        // The consumer index is retained in the import-binding key.
        let consumers = imports
            .iter()
            .filter_map(|(&(consumer, _), &(bound_producer, _))| {
                (bound_producer == producer).then_some(consumer)
            })
            .collect::<BTreeSet<_>>();
        for consumer in consumers {
            if outgoing[producer].insert(consumer) {
                indegree[consumer] += 1;
            }
        }
    }
    let mut ready = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, degree)| (*degree == 0).then_some(index))
        .collect::<BTreeSet<_>>();
    let mut order = Vec::with_capacity(fragments.len());
    while let Some(fragment) = ready.pop_first() {
        order.push(fragment);
        for &consumer in &outgoing[fragment] {
            indegree[consumer] -= 1;
            if indegree[consumer] == 0 {
                ready.insert(consumer);
            }
        }
    }
    if order.len() != fragments.len() {
        return Err(GpuWorkGraphError::invalid(
            "order GPU work-fragment imports",
            GpuWorkGraphErrorContext::new(graph_label, None, None, None, None, None, None),
            GpuWorkGraphCause::Cycle,
            "remove cyclic typed import/export relationships",
        ));
    }
    Ok(order)
}
