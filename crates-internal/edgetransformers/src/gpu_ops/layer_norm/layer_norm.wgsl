struct NormUniforms {
    m: u32, // batch_size * seq_len
    n: u32, // hidden_size
    eps: f32,
};

@group(0) @binding(0) var<uniform> uniforms: NormUniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>; // The 'weight'
@group(0) @binding(3) var<storage, read> beta: array<f32>;  // The 'bias'
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_idx = global_id.x;
    if (row_idx >= uniforms.m) {
        return;
    }

    let hidden_size = uniforms.n;
    let row_offset = row_idx * hidden_size;

    // 1. Calculate mean
    var mean = 0.0;
    for (var i = 0u; i < hidden_size; i = i + 1u) {
        mean += input[row_offset + i];
    }
    mean /= f32(hidden_size);

    // 2. Calculate variance
    var variance = 0.0;
    for (var i = 0u; i < hidden_size; i = i + 1u) {
        let val = input[row_offset + i] - mean;
        variance += val * val;
    }
    variance /= f32(hidden_size);

    let inv_std = 1.0 / sqrt(variance + uniforms.eps);

    // 3. Normalize and apply gamma/beta
    for (var i = 0u; i < hidden_size; i = i + 1u) {
        let normalized = (input[row_offset + i] - mean) * inv_std;
        output[row_offset + i] = normalized * gamma[i] + beta[i];
    }
}