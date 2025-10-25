const TILE_DIM = 32u;

struct MatmulInfo {
    m: u32,
    k: u32,
    n: u32,
};

var<workgroup> a_tile: array<f32, TILE_DIM * TILE_DIM>;
var<workgroup> b_tile: array<f32, TILE_DIM * TILE_DIM>;


@group(0) @binding(0) var<uniform> info: MatmulInfo;
@group(0) @binding(1) var<storage, read> a_in: array<f32>;
@group(0) @binding(2) var<storage, read> b_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> c_out: array<f32>;

@compute @workgroup_size(TILE_DIM, TILE_DIM, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let a = &a_in;
    let b = &b_in;
    let c = &c_out;

    let global_row = group_id.y * TILE_DIM + local_id.y;
    let global_col = group_id.x * TILE_DIM + local_id.x;

    var acc = 0.0; // Accumulator for the final sum.
    let num_tiles = (info.k + TILE_DIM - 1u) / TILE_DIM;

    // Loop over the tiles of the input matrices.
    for (var t = 0u; t < num_tiles; t = t + 1u) {

        // Calculate the source coordinates for this thread's load.
        let a_col = t * TILE_DIM + local_id.x;
        let a_row = group_id.y * TILE_DIM + local_id.y;
        
        let b_col = group_id.x * TILE_DIM + local_id.x;
        let b_row = t * TILE_DIM + local_id.y;

        // Load the data into the shared tiles.
        if (a_row < info.m && a_col < info.k) {
            a_tile[local_id.y * TILE_DIM + local_id.x] = (*a)[a_row * info.k + a_col];
        } else {
            a_tile[local_id.y * TILE_DIM + local_id.x] = 0.0;
        }

        if (b_row < info.k && b_col < info.n) {
            b_tile[local_id.y * TILE_DIM + local_id.x] = (*b)[b_row * info.n + b_col];
        } else {
            b_tile[local_id.y * TILE_DIM + local_id.x] = 0.0;
        }

        workgroupBarrier();

        for (var i = 0u; i < TILE_DIM; i = i + 1u) {
            acc = acc + a_tile[local_id.y * TILE_DIM + i] * b_tile[i * TILE_DIM + local_id.x];
        }

        workgroupBarrier();
    }
    if (global_row < info.m && global_col < info.n) {
        (*c)[global_row * info.n + global_col] = acc;
    }
}