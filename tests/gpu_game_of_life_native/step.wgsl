struct Cell {
    alive: u32,
};

@group(0) @binding(0)
var<storage, read> input_cells: array<Cell>;

@group(0) @binding(1)
var<storage, read_write> output_cells: array<Cell>;

const WIDTH: u32 = 160u;
const HEIGHT: u32 = 90u;

fn linear_index(x: u32, y: u32) -> u32 {
    return y * WIDTH + x;
}

fn wrap_signed(value: i32, limit: u32) -> u32 {
    let limit_i = i32(limit);
    let m = value % limit_i;
    return u32(select(m + limit_i, m, m >= 0));
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= WIDTH || gid.y >= HEIGHT) {
        return;
    }

    let x = gid.x;
    let y = gid.y;
    let index = linear_index(x, y);
    let current = input_cells[index].alive;

    var neighbors: u32 = 0u;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            if (dx == 0 && dy == 0) {
                continue;
            }

            let nx = wrap_signed(i32(x) + dx, WIDTH);
            let ny = wrap_signed(i32(y) + dy, HEIGHT);
            neighbors = neighbors + input_cells[linear_index(nx, ny)].alive;
        }
    }

    var next_alive = 0u;
    if (current == 1u && (neighbors == 2u || neighbors == 3u)) {
        next_alive = 1u;
    } else if (current == 0u && neighbors == 3u) {
        next_alive = 1u;
    }

    output_cells[index].alive = next_alive;
}
