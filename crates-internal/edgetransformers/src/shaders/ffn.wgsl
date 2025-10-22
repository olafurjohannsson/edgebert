struct FfnWeights {
    fc1_weight: array<f32>, // Shape: [hidden_size, intermediate_size]
    fc1_bias: array<f32>,   // Shape: [intermediate_size]
    fc2_weight: array<f32>, // Shape: [intermediate_size, hidden_size]
    fc2_bias: array<f32>,   // Shape: [hidden_size]
};

@group(0) @binding(0) var<uniform> info: MatmulInfo; // We can reuse MatmulInfo
@group(0) @binding(1) var<storage, read> weights: FfnWeights;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// GELU activation
fn gelu(x: f32) -> f32 {
    let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(256, 1, 1) // Workgroup of 256 threads
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_idx = global_id.x;

    if (row_idx >= info.m) { // info.m is sequence_length
        return;
    }

    let hidden_size = info.k;
    let intermediate_size = info.n;

    // FC1 Layer
    var intermediate_vec: array<f32, 4096>; // Max intermediate size
    for (var j = 0u; j < intermediate_size; j = j + 1u) {
        var sum = 0.0;
        for (var i = 0u; i < hidden_size; i = i + 1u) {
            sum = sum + input[row_idx * hidden_size + i] * weights.fc1_weight[i * intermediate_size + j];
        }
        // Apply GELU immediately after adding bias
        intermediate_vec[j] = gelu(sum + weights.fc1_bias[j]);
    }

    // FC2 Layer
    for (var j = 0u; j < hidden_size; j = j + 1u) {
        var sum = 0.0;
        for (var i = 0u; i < intermediate_size; i = i + 1u) {
            sum = sum + intermediate_vec[i] * weights.fc2_weight[i * hidden_size + j];
        }
        output[row_idx * hidden_size + j] = sum + weights.fc2_bias[j];
    }
}