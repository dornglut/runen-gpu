struct Cell {
    alive: u32,
};

@group(0) @binding(0)
var<storage, read> state: array<Cell>;

const WIDTH: u32 = 160u;
const HEIGHT: u32 = 90u;

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
    let x = min(u32(position.x), WIDTH - 1u);
    let y = min(u32(position.y), HEIGHT - 1u);
    let alive = select(0.0, 1.0, state[y * WIDTH + x].alive == 1u);
    return vec4<f32>(alive, alive, alive, 1.0);
}
