use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

fn collect_rust_sources(root: &Path, paths: &mut Vec<PathBuf>) {
    for entry in
        fs::read_dir(root).unwrap_or_else(|error| panic!("cannot read {}: {error}", root.display()))
    {
        let path = entry.expect("source entry should be readable").path();
        if path.is_dir() {
            collect_rust_sources(&path, paths);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            paths.push(path);
        }
    }
}

fn compact(source: &str) -> String {
    source
        .lines()
        .filter(|line| {
            let line = line.trim_start();
            !line.starts_with("//") && !line.starts_with("* ") && !line.starts_with("*/")
        })
        .flat_map(str::chars)
        .filter(|character| !character.is_whitespace())
        .collect()
}

fn read(manifest: &Path, relative: &str) -> String {
    fs::read_to_string(manifest.join(relative))
        .unwrap_or_else(|error| panic!("cannot read {relative}: {error}"))
}

#[test]
fn g5b_mapping_and_completion_remain_command_buffer_local_across_g7a_segments() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let backend_root = manifest.join("src/backend/wgpu");
    let execution_path = "src/backend/wgpu/execution.rs";

    let mut paths = Vec::new();
    collect_rust_sources(&backend_root, &mut paths);
    let direct_submit_paths = paths
        .into_iter()
        .filter_map(|path| {
            let source =
                compact(&fs::read_to_string(&path).expect("WGPU source should be readable"));
            source.contains(".queue.submit(").then(|| {
                path.strip_prefix(&manifest)
                    .expect("WGPU source stays in engine")
                    .to_string_lossy()
                    .into_owned()
            })
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        direct_submit_paths,
        BTreeSet::from([execution_path.to_owned()]),
        "direct private WGPU queue submission must remain owned by the G5 executor; G7 surface presentation may segment that one logical submission but does not create another queue owner"
    );

    let execution = compact(&read(&manifest, execution_path));
    assert!(
        !execution.contains("current_render_execution_bridge("),
        "reusable G5 execution must consume private G4/G7 authority directly, not the residual renderer bridge"
    );
    assert!(
        !execution.contains("backend.queue.on_submitted_work_done("),
        "G5 completion must not regress to queue-relative previous-submit ownership"
    );
    assert!(
        !execution.contains(".slice(..).map_async("),
        "G5 readback mapping must be deferred on the owning command buffer before submission"
    );
    assert!(execution.contains("command_buffer.map_buffer_on_submit("));
    assert!(execution.contains("command_buffer.on_submitted_work_done("));

    let submit_start = execution
        .find("pubfnsubmit_prepared(")
        .expect("public prepared submission entrypoint must remain explicit");
    let submit_end = execution[submit_start..]
        .find("pubfnprogress(")
        .map(|offset| submit_start + offset)
        .expect("submission entrypoint must end before progress");
    let submit = &execution[submit_start..submit_end];
    let attribution_gate = submit
        .find("let_attribution_gate=self.backend.error_attribution_gate.acquire();")
        .expect("accepted physical submission must retain the shared backend-operation attribution gate");
    let surface_guard = submit.find(".execution_lease_guard(").expect(
        "G7 surface execution must acquire its lease guard inside the attributed submit interval",
    );
    let accept = submit
        .find("self.backend.execution.accept_prepared(&prepared)")
        .expect("irreversible acceptance must remain explicit");
    let encode_submit = submit
        .find("encode_submit_and_register(")
        .expect("accepted physical execution must remain in the same submit entrypoint");
    assert!(
        attribution_gate < surface_guard && surface_guard < accept && accept < encode_submit,
        "lock order must remain attribution gate -> G7 surface authority -> irreversible acceptance -> physical execution"
    );
    assert!(
        !submit.contains("drop(_attribution_gate)"),
        "the attribution gate must remain live through segmented physical submission and Present"
    );

    let encode_start = execution
        .find("fnencode_submit_and_register(")
        .expect("private physical execution owner must remain explicit");
    let encode_end = execution[encode_start..]
        .find("fntexture_copy_info")
        .map(|offset| encode_start + offset)
        .expect("physical execution helper must end before texture copy lowering");
    let encode = &execution[encode_start..encode_end];
    let attach = encode
        .find("execution.attach_staging(submission,&staging.encoded)?;")
        .expect("accepted staging must be published once before any physical segment is submitted");
    let map_callbacks = encode
        .find("register_readback_callbacks(execution,submission,&segment.readback_staging,&segment.command_buffer,);")
        .expect("each physical segment must attach its readback callbacks to its own command buffer");
    let completion = encode
        .find("register_submission_completion(execution,submission,&segment.command_buffer);")
        .expect("logical completion must remain command-buffer-local on the final segment");
    let queue_submit = encode
        .find("backend.queue.submit([segment.command_buffer]);")
        .expect("each owned segment must be submitted by the one G5 executor");
    let initial_content_queued = encode
        .find("plan.mark_initial_content_queued()")
        .expect("G5R initial-content state must advance only at the physical submission boundary");
    let present = encode
        .find(".present(&backend.queue,surface.lease(),surface.resource())")
        .expect(
            "Present must consume the lease only after its preceding command segment is submitted",
        );
    assert!(
        attach < map_callbacks
            && map_callbacks < completion
            && completion < queue_submit
            && queue_submit < initial_content_queued
            && initial_content_queued < present,
        "staging publication and command-buffer-local callbacks must precede physical submit; G5R seed state must advance only after submit and before any later Present"
    );
    assert!(
        encode.contains("ifindex+1==segment_count{register_submission_completion("),
        "only the final physical segment may own logical submission completion"
    );
    assert!(
        encode.contains("present_after:Some(source.clone())")
            && encode.contains("present_after:None"),
        "Present must terminate a physical segment and leave one final completion segment, including the terminal-Present case"
    );

    let fail_start = execution
        .find("fnfail_submission(")
        .expect("accepted submission failure cleanup must remain explicit");
    let fail_end = execution[fail_start..]
        .find("fnfail_active_for_fault(")
        .map(|offset| fail_start + offset)
        .expect("submission failure cleanup must end before fault-wide cleanup");
    let fail = &execution[fail_start..fail_end];
    assert!(
        !fail.contains("mark_initial_content_queued(")
            && !fail.contains("mark_initial_content_completed("),
        "failure cleanup must not fabricate or reset physical initial-content state: pre-submit failure leaves it required, while failure after first submit preserves the queued seed"
    );
}

#[test]
fn g5b_rejection_identity_and_owner_local_execution_order_preserve_acceptance() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let execution = compact(&read(&manifest, "src/backend/wgpu/execution.rs"));

    assert!(
        execution.contains("submission_order:Mutex<()>"),
        "G5B execution state must retain a context-local acceptance/execution ordering gate"
    );

    let submit_start = execution
        .find("pubfnsubmit_prepared(")
        .expect("public prepared submission entrypoint must remain explicit");
    let submit_end = execution[submit_start..]
        .find("pubfnprogress(")
        .map(|offset| submit_start + offset)
        .expect("submission entrypoint must end before progress");
    let submit = &execution[submit_start..submit_end];
    let foreign = submit
        .find("prepared.affinity.context()!=expected_affinity.context()")
        .expect("foreign-context rejection must be explicit");
    let stale = submit
        .find("prepared.affinity.generation()!=expected_affinity.generation()")
        .expect("stale-generation rejection must be explicit");
    let owner = submit
        .find(".ptr_eq(&Arc::downgrade(&self.backend.execution))")
        .expect("same-affinity prepared ownership must still be checked");
    let order_gate = submit
        .find("let_submission_order=self.backend.execution.submission_order.lock().unwrap_or_else(std::sync::PoisonError::into_inner);")
        .expect("owner-local submission ordering must begin before irreversible acceptance");
    let accept = submit
        .find("self.backend.execution.accept_prepared(&prepared)")
        .expect("irreversible acceptance must remain explicit");
    let encode_submit = submit
        .find("encode_submit_and_register(")
        .expect("accepted physical execution must remain in the same submit entrypoint");
    assert!(
        foreign < stale
            && stale < owner
            && owner < order_gate
            && order_gate < accept
            && accept < encode_submit,
        "classification must precede one owner-local interval spanning irreversible acceptance through physical execution"
    );
    assert!(
        !submit.contains("drop(_submission_order)"),
        "the owner-local execution-order gate must remain held through physical encode/submit"
    );

    let accept_start = execution
        .find("fnaccept_prepared(")
        .expect("private acceptance owner must remain explicit");
    let accept_end = execution[accept_start..]
        .find("fnattach_staging(")
        .map(|offset| accept_start + offset)
        .expect("acceptance must end before staging attachment");
    let accept = &execution[accept_start..accept_end];
    let metadata = accept
        .find("ifreadbacks.len()!=plan.readback_ids.len()")
        .expect("readback metadata completeness must be validated pre-acceptance");
    let remove_prepared = accept
        .find("inner.prepared.remove(&prepared.ticket)")
        .expect("irreversible acceptance must remove retryable prepared authority");
    let allocate_id = accept
        .find("allocate_nonzero(&self.next_submission)")
        .expect("submission identity must be allocated only at irreversible acceptance");
    let restore_prepared = accept
        .find("inner.prepared.insert(prepared.ticket,Some(stored_plan))")
        .expect("submission-ID exhaustion must restore retryable prepared authority");
    assert!(
        metadata < remove_prepared
            && remove_prepared < allocate_id
            && allocate_id < restore_prepared,
        "all rollback-capable metadata validation must precede prepared removal and submission-ID publication, and identity exhaustion must restore retryable prepared authority"
    );
}

#[test]
fn g5b_compute_preparation_owns_checked_offsets_and_retained_g4_realizations() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let execution = compact(&read(&manifest, "src/backend/wgpu/execution.rs"));

    assert!(execution.contains("dynamic_offsets:Vec<u32>"));
    assert!(
        execution.contains("u32::try_from(offset)"),
        "logical u64 dynamic offsets must narrow exactly once during preparation"
    );
    assert!(
        execution.contains("pipeline:GpuRealizedComputePipeline")
            && execution.contains("realization:GpuRealizedBindGroup")
            && execution.contains("query_set:GpuRealizedQuerySet"),
        "prepared compute work must retain exact G4 pipeline, bind-group, and optional timestamp query-set records"
    );
    assert!(
        execution.contains("Indirect{arguments:GpuRealizedBuffer,offset:u64"),
        "prepared indirect compute must retain the exact realized G4 argument buffer and accepted byte offset"
    );
    assert!(
        execution.contains("letzero_direct=matches!(dispatch,PreparedComputeDispatch::Direct(size)ifsize.as_array().contains(&0));")
            && execution.contains("ifzero_direct&&timestamp_writes.is_none(){continue;}")
            && execution.contains("PreparedComputeDispatch::Direct(size)if!size.as_array().contains(&0)=>")
            && execution.contains("PreparedComputeDispatch::Direct(_)=>{}"),
        "zero direct dispatch must emit no shader work while preserving an otherwise observable timestamped pass"
    );
    assert!(
        execution.contains("arguments:realized_buffer(context,buffer_cache,arguments.buffer())?")
            && execution.contains("offset:arguments.range().offset()"),
        "indirect compute preparation must reuse the execution plan's G4 buffer cache and preserve the accepted offset"
    );
    assert!(
        execution
            .contains("query_set:realized_query_set(context,query_set_cache,writes.query_set())?"),
        "timestamped compute preparation must retain the exact G4 query-set realization"
    );
    assert!(
        execution.contains("pass.dispatch_workgroups_indirect(")
            && execution.contains("&arguments.record.object")
            && execution.contains("*offset"),
        "private execution must encode indirect dispatch from the retained G4 buffer and accepted offset without host-reading runtime arguments"
    );
    assert!(
        execution.contains("backend.program_binding_realization.with_execution_bind_groups("),
        "G5B compute must consume the generic private G4 lexical lending owner"
    );
}
