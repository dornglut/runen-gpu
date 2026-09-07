override ELEMENT_COUNT: u32 = 0u;
override INCLUSIVE: u32 = 0u;

struct U32Cell {
    value: u32,
};

@group(0) @binding(0)
var<storage, read> input_values: array<U32Cell>;

@group(0) @binding(1)
var<storage, read_write> output_values: array<U32Cell>;

@group(0) @binding(2)
var<storage, read_write> block_sums: array<U32Cell>;

var<workgroup> scratch: array<u32, 64>;

@compute @workgroup_size(64)
fn cs_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let index = global_id.x;
    let local_index = local_id.x;
    let total_count = min(ELEMENT_COUNT, min(arrayLength(&input_values), arrayLength(&output_values)));
    var input_value = 0u;
    if (index < total_count) {
        input_value = input_values[index].value;
    }
    scratch[local_index] = input_value;
    workgroupBarrier();

    var offset = 1u;
    loop {
        if (offset >= 64u) {
            break;
        }
        var addend = 0u;
        if (local_index >= offset) {
            addend = scratch[local_index - offset];
        }
        workgroupBarrier();
        scratch[local_index] = scratch[local_index] + addend;
        workgroupBarrier();
        offset = offset * 2u;
    }

    if (index < total_count) {
        let exclusive_value = scratch[local_index] - input_value;
        output_values[index].value = select(exclusive_value, scratch[local_index], INCLUSIVE != 0u);
    }

    let block_start = workgroup_id.x * 64u;
    let block_end = min(block_start + 64u, total_count);
    if (block_end > block_start) {
        let last_local_index = block_end - block_start - 1u;
        if (local_index == last_local_index && workgroup_id.x < arrayLength(&block_sums)) {
            block_sums[workgroup_id.x].value = scratch[local_index];
        }
    }
}
