
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
var<storage, read> state: array<vec2<f32>>;

@group(0) @binding(1)
var<storage, read> params: Params;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(positions[vertex_index], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let x = min(u32(position.x), params.width - 1u);
    let y = min(u32(position.y), params.height - 1u);
    let cell = state[y * params.width + x];
    let b = clamp(cell.y, 0.0, 1.0);
    let contrast = clamp((cell.x - cell.y) * 0.5 + 0.5, 0.0, 1.0);
    return vec4<f32>(b, b * b, contrast, 1.0);
}
