use std::fs;

fn read(path: &str) -> String {
    fs::read_to_string(path).unwrap_or_else(|err| panic!("failed to read {path}: {err}"))
}

#[test]
fn execution_limits_consume_positional_binding_and_vertex_slots() {
    let bindings = read("src/api/program/runtime_binding/set.rs");
    assert!(
        bindings.contains("required_bind_group_slots > u64::from(device_facts.max_bind_groups())"),
        "complete runtime bindings must admit the highest positional bind-group slot before backend realization"
    );
    assert!(
        bindings.contains("u64::from(group.group()) + 1"),
        "sparse logical group indices must count the positional slots that private realization requires"
    );

    let render = read("src/api/render_execution.rs");
    assert!(
        render.contains("u64::from(binding.slot()) + 1")
            && render.contains("limits.max_vertex_buffers()")
            && render.contains("bindings.required_bind_group_slots()")
            && render.contains("limits.max_bind_groups_plus_vertex_buffers()"),
        "render limit admission must consume positional vertex-buffer and bind-group slots"
    );
    assert!(
        !render.contains("vertex_buffers.len() + bindings.groups().len()"),
        "render combined-limit admission must not regress to declared cardinality for sparse slots"
    );
}
