override ELEMENT_COUNT: u32 = 0u;

struct U32Cell {
    value: u32,
};

@group(0) @binding(0)
var<storage, read_write> output_values: array<U32Cell>;

@group(0) @binding(1)
var<storage, read> block_offsets: array<U32Cell>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let total_count = min(ELEMENT_COUNT, arrayLength(&output_values));
    let block_index = index / 64u;
    if (index < total_count && block_index < arrayLength(&block_offsets)) {
        output_values[index].value = output_values[index].value + block_offsets[block_index].value;
    }
}
