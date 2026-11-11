// Exact integer, per-pixel adaptive background and foreground mask.
// Input arrays contain one u32 per grayscale pixel in [0, 255].
@group(0) @binding(0) var<storage, read_write> background: array<u32>;
@group(0) @binding(1) var<storage, read> frame: array<u32>;
@group(0) @binding(2) var<storage, read_write> mask: array<u32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params[0]) { return; }

    let old = background[i];
    let sample = frame[i];
    let delta = select(sample - old, old - sample, old > sample);
    mask[i] = select(0u, 255u, delta >= params[2]);
    let q = params[1];
    background[i] = ((q - 1u) * old + sample) / q;
}
