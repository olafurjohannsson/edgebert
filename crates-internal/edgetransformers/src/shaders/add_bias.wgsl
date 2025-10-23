struct Uniforms {
    size: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.size) { return; }
    // This assumes the bias is a 1D vector being added to a 2D matrix,
    // so we use modulo to get the correct bias index.
    let bias_dim = arrayLength(&bias);
    output[idx] = input[idx] + bias[idx % bias_dim];
}