
struct Params {
    width: u32,
    height: u32,
    dt: f32,
    feed: f32,
    kill: f32,
    diffusion_a: f32,
    diffusion_b: f32,
    _pad: f32,
}

@group(0) @binding(0)
var<storage, read> state_in: array<vec2<f32>>;

@group(0) @binding(1)
var<storage, read_write> state_out: array<vec2<f32>>;

@group(0) @binding(2)
var<storage, read> params: Params;

fn wrapped_index(x: i32, y: i32) -> u32 {
    let width = i32(params.width);
    let height = i32(params.height);
    let wx = (x + width) % width;
    let wy = (y + height) % height;
    return u32(wy) * params.width + u32(wx);
}

fn sample(x: i32, y: i32) -> vec2<f32> {
    return state_in[wrapped_index(x, y)];
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let x = i32(gid.x);
    let y = i32(gid.y);
    let center = sample(x, y);

    let cardinal =
        sample(x - 1, y) + sample(x + 1, y) +
        sample(x, y - 1) + sample(x, y + 1);
    let diagonal =
        sample(x - 1, y - 1) + sample(x + 1, y - 1) +
        sample(x - 1, y + 1) + sample(x + 1, y + 1);
    let laplacian = center * -1.0 + cardinal * 0.2 + diagonal * 0.05;

    let a = center.x;
    let b = center.y;
    let reaction = a * b * b;
    let next_a = clamp(
        a + (params.diffusion_a * laplacian.x - reaction + params.feed * (1.0 - a)) * params.dt,
        0.0,
        1.0,
    );
    let next_b = clamp(
        b + (params.diffusion_b * laplacian.y + reaction - (params.kill + params.feed) * b) * params.dt,
        0.0,
        1.0,
    );
    state_out[gid.y * params.width + gid.x] = vec2<f32>(next_a, next_b);
}
