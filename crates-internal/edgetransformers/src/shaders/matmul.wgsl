struct MatmulInfo {
    m: u32,
    k: u32,
    n: u32,
};

@group(0) @binding(0) var<uniform> info: MatmulInfo;

@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(8, 8, 1) 
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y; 

    // Boundary check to avoid writing out of bounds
    if (row >= info.m || col >= info.n) {
        return;
    }

    var sum = 0.0;
    // Perform the dot product for the single output cell (c[row, col])
    for (var i = 0u; i < info.k; i = i + 1u) {
        sum = sum + a[row * info.k + i] * b[i * info.n + col];
    }

    // Write the result to the output buffer
    c[row * info.n + col] = sum;
}