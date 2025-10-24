struct SoftmaxUniforms {
    rows: u32,
    cols: u32,
    scale: f32,
};

@group(0) @binding(0) var<uniform> uniforms: SoftmaxUniforms;
@group(0) @binding(1) var<storage, read_write> data: array<f32>; // In-place softmax

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_idx = global_id.x;
    if (row_idx >= uniforms.rows) { return; }

    let row_offset = row_idx * uniforms.cols;

    // 1. Scale and find max for numerical stability
    var max_val = -3.4e+38; // f32.min
    for (var i = 0u; i < uniforms.cols; i = i + 1u) {
        data[row_offset + i] = data[row_offset + i] * uniforms.scale;
        max_val = max(max_val, data[row_offset + i]);
    }

    // 2. Exponentiate and find sum
    var exp_sum = 0.0;
    for (var i = 0u; i < uniforms.cols; i = i + 1u) {
        let val = exp(data[row_offset + i] - max_val);
        data[row_offset + i] = val;
        exp_sum += val;
    }

    // 3. Normalize
    for (var i = 0u; i < uniforms.cols; i = i + 1u) {
        data[row_offset + i] /= exp_sum;
    }
}