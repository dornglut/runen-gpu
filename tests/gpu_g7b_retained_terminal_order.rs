use std::fs;
use std::path::{Path, PathBuf};

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

fn execution_source(manifest: &Path) -> String {
    compact(
        &fs::read_to_string(manifest.join("src/backend/wgpu/execution.rs"))
            .expect("WGPU execution source should be readable"),
    )
}

#[test]
fn retained_may_execute_transition_stays_after_physical_queue_submission() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let execution = execution_source(&manifest);
    let start = execution
        .find("fnencode_submit_and_register(")
        .expect("physical execution owner must remain explicit");
    let end = execution[start..]
        .find("fntexture_copy_info")
        .map(|offset| start + offset)
        .expect("physical execution owner must end before texture copy lowering");
    let body = &execution[start..end];
    let queue_submit = body
        .find("backend.queue.submit([segment.command_buffer]);")
        .expect("physical queue submission must remain explicit");
    let mark_may_execute = body
        .find("execution.mark_segment_may_execute(submission,&segment.retained_writes);")
        .expect("retained may-execute transition must remain explicit");

    assert!(
        queue_submit < mark_may_execute,
        "retained writes may become indeterminate only after physical queue admission"
    );
}

#[test]
fn progress_drains_poll_produced_completion_before_terminal_fault_failure() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let execution = execution_source(&manifest);
    let start = execution
        .find("pubfnprogress(&self)->GpuExecutionStats{")
        .expect("GpuContext::progress must remain explicit");
    let end = execution[start..]
        .find("asyncfnprepare_execution_plan")
        .map(|offset| start + offset)
        .expect("GpuContext::progress must end before execution-plan preparation");
    let body = &execution[start..end];
    let submission_order = body
        .find(".submission_order")
        .expect("progress must share submission-order serialization");
    let poll = body
        .find(".device.poll(PollType::Poll)")
        .expect("progress must poll the device");
    let drains = body
        .match_indices("self.backend.execution.drain_events();")
        .map(|(offset, _)| offset)
        .collect::<Vec<_>>();
    let terminal_fault = body
        .find("self.backend.health.terminal_fault()")
        .expect("progress must inspect terminal device fault state");

    assert_eq!(drains.len(), 2);
    assert!(submission_order < drains[0]);
    assert!(drains[0] < poll);
    assert!(poll < drains[1]);
    assert!(
        drains[1] < terminal_fault,
        "completion evidence produced by the poll must be consumed before unresolved terminal fault failure"
    );
}
