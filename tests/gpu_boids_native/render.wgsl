struct DrawParams {
    world_to_clip: vec4<f32>,
    viewport: vec4<f32>,
    visible_world: vec4<f32>,
    sprite: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local_position: vec2<f32>,
    @location(1) local_velocity: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> draw_params: DrawParams;

fn normalize_or_up(value: vec2<f32>) -> vec2<f32> {
    let len = length(value);
    if (len <= 0.000001) {
        return vec2<f32>(0.0, -1.0);
    }
    return value / len;
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) instance_position: vec2<f32>,
    @location(1) instance_velocity: vec2<f32>,
    @location(2) instance_visual_heading: vec2<f32>,
) -> VsOut {
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );

    let local = corners[vertex_index];
    let direction = normalize_or_up(instance_visual_heading);
    let right = vec2<f32>(direction.y, -direction.x);
    let world_offset = right * local.x * draw_params.sprite.x
        + direction * local.y * draw_params.sprite.y;
    let center_clip = instance_position * draw_params.world_to_clip.xy + draw_params.world_to_clip.zw;
    let clip_offset = world_offset * draw_params.world_to_clip.xy;

    var out: VsOut;
    out.clip_position = vec4<f32>(center_clip + clip_offset, 0.0, 1.0);
    out.local_position = local;
    out.local_velocity = normalize_or_up(instance_velocity);
    return out;
}

@fragment
fn fs_main(input: VsOut) -> @location(0) vec4<f32> {
    let local = input.local_position;
    let body = 1.0 - smoothstep(0.46, 0.58, length(vec2<f32>(local.x * 1.16, local.y + 0.10)));
    let nose = 1.0 - smoothstep(0.22, 0.32, length(vec2<f32>(local.x * 1.75, local.y - 0.48)));
    let tail_axis = clamp((-local.y - 0.08) * 1.6, 0.0, 1.0);
    let tail_width = mix(0.34, 0.08, tail_axis);
    let tail = (1.0 - smoothstep(tail_width, tail_width + 0.08, abs(local.x)))
        * smoothstep(-0.86, -0.08, local.y);
    let halo = 1.0 - smoothstep(0.60, 1.05, length(local));

    let mask = clamp(max(max(body, nose), tail * 0.78), 0.0, 1.0);
    let glow = halo * 0.34;
    let base = vec3<f32>(0.30, 0.95, 0.82);
    let hot = vec3<f32>(0.88, 1.0, 0.92);
    let color = mix(base * 0.45, hot, clamp(mask + glow, 0.0, 1.0));
    let alpha = clamp(mask + glow * 0.62, 0.0, 1.0);
    return vec4<f32>(color, alpha);
}
