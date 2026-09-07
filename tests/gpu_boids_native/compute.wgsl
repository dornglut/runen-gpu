struct BoidAgent {
    position: vec2<f32>,
    velocity: vec2<f32>,
    visual_heading: vec2<f32>,
};

struct AtomicCounter {
    value: atomic<u32>,
};

struct U32Word {
    value: u32,
};

struct ComputeParams {
    frame_meta: vec4<u32>, // tick, mode, boid_count, grid_cell_count
    grid: vec4<u32>, // cells_x, cells_y, cell_count, padding
    sim_a: vec4<f32>, // delta_seconds, max_speed, max_force, neighbor_radius
    sim_b: vec4<f32>, // separation_radius, alignment_weight, cohesion_weight, separation_weight
    sim_c: vec4<f32>, // center_weight, jitter_strength, simulation_fps, padding
};

@group(0) @binding(0)
var<uniform> params: ComputeParams;

@group(0) @binding(1)
var<storage, read_write> boids_a: array<BoidAgent>;

@group(0) @binding(2)
var<storage, read_write> boids_b: array<BoidAgent>;

@group(0) @binding(3)
var<storage, read_write> cell_counts: array<AtomicCounter>;

@group(0) @binding(4)
var<storage, read_write> cell_offsets: array<U32Word>;

@group(0) @binding(5)
var<storage, read_write> cell_cursors: array<AtomicCounter>;

@group(0) @binding(6)
var<storage, read_write> sorted_indices: array<U32Word>;

const MODE_CLEAR_COUNTS: u32 = 1u;
const MODE_COUNT_CELLS: u32 = 2u;
const MODE_SCAN_COUNTS: u32 = 3u;
const MODE_RESET_CURSORS: u32 = 4u;
const MODE_SCATTER_INDICES: u32 = 5u;
const MODE_SIMULATE_GRID: u32 = 6u;
const MODE_PUBLISH: u32 = 7u;

fn hash_u32(value: u32) -> u32 {
    var x = value * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    x = (x >> 22u) ^ x;
    return x;
}

fn rand01(seed: u32) -> f32 {
    return f32(hash_u32(seed) & 0x00ffffffu) / 16777215.0;
}

fn wrap_delta(delta: vec2<f32>) -> vec2<f32> {
    return delta - round(delta);
}

fn limit_magnitude(value: vec2<f32>, max_value: f32) -> vec2<f32> {
    let len = length(value);
    if (len <= max_value || len == 0.0) {
        return value;
    }
    return value * (max_value / len);
}

fn normalize_or_zero(value: vec2<f32>) -> vec2<f32> {
    let len = length(value);
    if (len <= 0.000001) {
        return vec2<f32>(0.0, 0.0);
    }
    return value / len;
}

fn wrap_cell_index(value: i32, limit: u32) -> u32 {
    let signed_limit = i32(max(limit, 1u));
    var wrapped = value % signed_limit;
    if (wrapped < 0) {
        wrapped = wrapped + signed_limit;
    }
    return u32(wrapped);
}

fn cell_for_position(position: vec2<f32>) -> u32 {
    let cells_x = max(params.grid.x, 1u);
    let cells_y = max(params.grid.y, 1u);
    let wrapped = fract(position);
    let x = min(u32(floor(wrapped.x * f32(cells_x))), cells_x - 1u);
    let y = min(u32(floor(wrapped.y * f32(cells_y))), cells_y - 1u);
    return y * cells_x + x;
}

fn cell_xy(cell: u32) -> vec2<u32> {
    let cells_x = max(params.grid.x, 1u);
    return vec2<u32>(cell % cells_x, cell / cells_x);
}

fn scan_counts(cell_count: u32) {
    var prefix_sum = 0u;
    var cell = 0u;
    loop {
        if (cell >= cell_count) {
            break;
        }
        let count = atomicLoad(&cell_counts[cell].value);
        cell_offsets[cell].value = prefix_sum;
        prefix_sum = prefix_sum + count;
        cell = cell + 1u;
    }
}

fn simulate_boid(index: u32, boid_count: u32, tick: u32) -> BoidAgent {
    let delta_seconds = clamp(params.sim_a.x, 0.0001, 0.05);
    let max_speed = max(params.sim_a.y, 0.02);
    let max_force = max(params.sim_a.z, 0.05);
    let neighbor_radius = max(params.sim_a.w, 0.02);
    let separation_radius = max(params.sim_b.x, 0.005);
    let alignment_weight = params.sim_b.y;
    let cohesion_weight = params.sim_b.z;
    let separation_weight = params.sim_b.w;
    let center_weight = params.sim_c.x;
    let jitter_strength = params.sim_c.y;

    var boid = boids_a[index];
    var alignment = vec2<f32>(0.0, 0.0);
    var cohesion_delta_sum = vec2<f32>(0.0, 0.0);
    var separation = vec2<f32>(0.0, 0.0);
    var neighbor_count = 0u;
    var separation_count = 0u;

    let cells_x = max(params.grid.x, 1u);
    let cells_y = max(params.grid.y, 1u);
    let base_cell = cell_for_position(boid.position);
    let base_xy = cell_xy(base_cell);

    for (var oy: i32 = -1; oy <= 1; oy = oy + 1) {
        for (var ox: i32 = -1; ox <= 1; ox = ox + 1) {
            let nx = wrap_cell_index(i32(base_xy.x) + ox, cells_x);
            let ny = wrap_cell_index(i32(base_xy.y) + oy, cells_y);
            let neighbor_cell = ny * cells_x + nx;
            let start = cell_offsets[neighbor_cell].value;
            let count = atomicLoad(&cell_counts[neighbor_cell].value);

            var offset = 0u;
            loop {
                if (offset >= count) {
                    break;
                }
                let other_index = sorted_indices[start + offset].value;
                offset = offset + 1u;
                if (other_index == index || other_index >= boid_count) {
                    continue;
                }

                let other = boids_a[other_index];
                let delta = wrap_delta(other.position - boid.position);
                let dist = length(delta);
                if (dist < neighbor_radius) {
                    alignment = alignment + other.velocity;
                    cohesion_delta_sum = cohesion_delta_sum + delta;
                    neighbor_count = neighbor_count + 1u;
                }
                if (dist < separation_radius) {
                    let safe_dist = max(dist, 0.0005);
                    separation = separation - delta / safe_dist;
                    separation_count = separation_count + 1u;
                }
            }
        }
    }

    var acceleration = vec2<f32>(0.0, 0.0);
    if (neighbor_count > 0u) {
        let inv_neighbor_count = 1.0 / f32(neighbor_count);
        let mean_velocity = alignment * inv_neighbor_count;
        let desired_alignment = normalize_or_zero(mean_velocity) * max_speed;
        acceleration = acceleration + (desired_alignment - boid.velocity) * alignment_weight;

        let mean_cohesion_delta = cohesion_delta_sum * inv_neighbor_count;
        let desired_cohesion = normalize_or_zero(mean_cohesion_delta) * max_speed;
        acceleration = acceleration + (desired_cohesion - boid.velocity) * cohesion_weight;
    }

    if (separation_count > 0u) {
        let inv_separation_count = 1.0 / f32(separation_count);
        let desired_separation = normalize_or_zero(separation * inv_separation_count) * max_speed;
        acceleration = acceleration + (desired_separation - boid.velocity) * separation_weight;
    }

    let center_delta = wrap_delta(vec2<f32>(0.5, 0.5) - boid.position);
    acceleration = acceleration + center_delta * center_weight;

    let jitter_seed = tick * 4099u + index * 23u;
    let jitter = vec2<f32>(
        rand01(jitter_seed) - 0.5,
        rand01(jitter_seed + 1u) - 0.5
    ) * jitter_strength;
    acceleration = acceleration + jitter;

    acceleration = limit_magnitude(acceleration, max_force);
    boid.velocity = limit_magnitude(boid.velocity + acceleration * delta_seconds, max_speed);

    let min_speed = max_speed * 0.25;
    if (length(boid.velocity) < min_speed) {
        let stable_dir = normalize_or_zero(boid.velocity + vec2<f32>(0.0001, 0.0001));
        boid.velocity = stable_dir * min_speed;
    }

    boid.position = fract(boid.position + boid.velocity * delta_seconds);
    let desired_heading = normalize_or_zero(boid.velocity);
    let blended_heading = mix(boid.visual_heading, desired_heading, 0.16);
    let visual_heading = normalize_or_zero(blended_heading);
    if (length(visual_heading) > 0.000001) {
        boid.visual_heading = visual_heading;
    } else {
        boid.visual_heading = desired_heading;
    }
    return boid;
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let tick = params.frame_meta.x;
    let mode = params.frame_meta.y;
    let boid_count = params.frame_meta.z;
    let cell_count = params.frame_meta.w;

    if (mode == MODE_CLEAR_COUNTS) {
        if (index < cell_count) {
            atomicStore(&cell_counts[index].value, 0u);
            atomicStore(&cell_cursors[index].value, 0u);
            cell_offsets[index].value = 0u;
        }
        return;
    }

    if (mode == MODE_SCAN_COUNTS) {
        if (index == 0u) {
            scan_counts(cell_count);
        }
        return;
    }

    if (mode == MODE_RESET_CURSORS) {
        if (index < cell_count) {
            atomicStore(&cell_cursors[index].value, 0u);
        }
        return;
    }

    if (index >= boid_count) {
        return;
    }

    if (mode == MODE_COUNT_CELLS) {
        let cell = cell_for_position(boids_a[index].position);
        atomicAdd(&cell_counts[cell].value, 1u);
        return;
    }

    if (mode == MODE_SCATTER_INDICES) {
        let cell = cell_for_position(boids_a[index].position);
        let local_index = atomicAdd(&cell_cursors[cell].value, 1u);
        let sorted_index = cell_offsets[cell].value + local_index;
        if (sorted_index < boid_count) {
            sorted_indices[sorted_index].value = index;
        }
        return;
    }

    if (mode == MODE_SIMULATE_GRID) {
        boids_b[index] = simulate_boid(index, boid_count, tick);
        return;
    }

    if (mode == MODE_PUBLISH) {
        boids_a[index] = boids_b[index];
    }
}
