// Define the uniform struct at the top of the file.
// Renamed to FfnUniforms to be specific to this shader's purpose.
struct FfnUniforms {
    m: u32, // sequence_length
    k: u32, // hidden_size
    n: u32, // intermediate_size
};

@group(0) @binding(0) var<uniform> info: FfnUniforms;
@group(0) @binding(1) var<storage, read> weights: array<f32>; // A single flat buffer
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// Helper function to get a value from the fc1_weight part of the buffer
fn get_fc1_weight(row: u32, col: u32) -> f32 {
    // fc1_weight is at the beginning of the buffer
    let intermediate_size = info.n;
    return weights[row * intermediate_size + col];
}

// Helper function to get a value from the fc1_bias part of the buffer
fn get_fc1_bias(index: u32) -> f32 {
    let offset = info.k * info.n; // Offset starts after the fc1_weight matrix
    return weights[offset + index];
}

// Helper function to get a value from the fc2_weight part of the buffer
fn get_fc2_weight(row: u32, col: u32) -> f32 {
    let hidden_size = info.k;
    // Offset starts after fc1_weight AND fc1_bias
    let offset = (info.k * info.n) + info.n;
    return weights[offset + row * hidden_size + col];
}

// Helper function to get a value from the fc2_bias part of the buffer
fn get_fc2_bias(index: u32) -> f32 {
    // Offset starts after fc1_weight, fc1_bias, AND fc2_weight
    let offset = (info.k * info.n) + info.n + (info.n * info.k);
    return weights[offset + index];
}

// GELU activation
fn gelu(x: f32) -> f32 {
    // Using the common tanh approximation for GELU
    return 0.5 * x * (1.0 + tanh(0.79788456 * (x + 0.044715 * pow(x, 3.0))));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_idx = global_id.x; // Token index in the sequence

    if (row_idx >= info.m) {
        return;
    }

    let hidden_size = info.k;
    let intermediate_size = info.n;

    var intermediate_vec: array<f32, 4096>; // Using a fixed max size for simplicity

    // FC1 Layer
    for (var j = 0u; j < intermediate_size; j = j + 1u) {
        var sum = 0.0;
        for (var i = 0u; i < hidden_size; i = i + 1u) {
            sum += input[row_idx * hidden_size + i] * get_fc1_weight(i, j);
        }
        intermediate_vec[j] = gelu(sum + get_fc1_bias(j));
    }

    // FC2 Layer
    for (var j = 0u; j < hidden_size; j = j + 1u) {
        var sum = 0.0;
        for (var i = 0u; i < intermediate_size; i = i + 1u) {
            sum += intermediate_vec[i] * get_fc2_weight(i, j);
        }
        output[row_idx * hidden_size + j] = sum + get_fc2_bias(j);
    }
}