struct MatmulInfo {
    b: u32, // Batch size
    m: u32, // M
    k: u32, // K
    n: u32, // N
};

@group(0) @binding(0) var<uniform> info: MatmulInfo;
@group(0) @binding(1) var<storage, read> a_in: array<f32>;
@group(0) @binding(2) var<storage, read> b_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> c_out: array<f32>;

var<workgroup> a_tile: array<array<f32, 32>, 32>;
var<workgroup> b_tile: array<array<f32, 32>, 32>;

@compute @workgroup_size(32, 32, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let b_idx = group_id.z; // The batch we are working on

    // Offsets for this batch
    let a_batch_offset = b_idx * info.m * info.k;
    let b_batch_offset = b_idx * info.k * info.n;
    let c_batch_offset = b_idx * info.m * info.n;

    let global_row = group_id.y * 32u + local_id.y;
    let global_col = group_id.x * 32u + local_id.x;

    var acc = 0.0;
    let num_tiles = (info.k + 15u) / 32u;

    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * 32u + local_id.x;
        let b_row = t * 32u + local_id.y;

        if (global_row < info.m && a_col < info.k) {
            a_tile[local_id.y][local_id.x] = a_in[a_batch_offset + global_row * info.k + a_col];
        } else {
            a_tile[local_id.y][local_id.x] = 0.0;
        }

        if (b_row < info.k && global_col < info.n) {
            b_tile[local_id.y][local_id.x] = b_in[b_batch_offset + b_row * info.n + global_col];
        } else {
            b_tile[local_id.y][local_id.x] = 0.0;
        }

        workgroupBarrier();

        for (var i = 0u; i < 32u; i = i + 1u) {
            acc += a_tile[local_id.y][i] * b_tile[i][local_id.x];
        }
        workgroupBarrier();
    }
    
    if (global_row < info.m && global_col < info.n) {
        c_out[c_batch_offset + global_row * info.n + global_col] = acc;
    }
}