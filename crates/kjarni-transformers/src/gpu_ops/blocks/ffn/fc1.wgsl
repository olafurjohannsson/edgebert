//! First feedforward layer (FC1) with configurable activation function.


struct FfnUniforms {
    /// Number of input rows (batch * seq_len).
    m: u32,
    /// Input dimension (hidden_size).
    k: u32,
    /// Output dimension (intermediate_size, typically 4x hidden_size).
    n: u32,
    /// Padding for 16-byte alignment.
    _padding: u32,
};

/// Pipeline constant for activation function selection
///
/// - 0: GELU (default)
/// - 1: GELU_NEW (tanh approx)
/// - 2: ReLU
/// - 3: SiLU (Swish)
/// - 4: Tanh
@id(0) override COMPUTE_ACT_TYPE: u32 = 0;

@group(0) @binding(0) var<uniform> info: FfnUniforms;
@group(0) @binding(1) var<storage, read> fc1_weight: array<f32>;
@group(0) @binding(2) var<storage, read> fc1_bias: array<f32>;
@group(0) @binding(3) var<storage, read> input: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;


// GeluNew
const SQRT_2_OVER_PI: f32 = 0.7978845608;
const GELU_COEFF: f32 = 0.044715;

// Erf Approx
const SQRT_2_INV: f32 = 0.7071067811865476; // 1/sqrt(2)
const A1: f32 =  0.254829592;
const A2: f32 = -0.284496736;
const A3: f32 =  1.421413741;
const A4: f32 = -1.453152027;
const A5: f32 =  1.061405429;
const P: f32  =  0.3275911;



// GELU 
fn gelu_new(x: f32) -> f32 {
    let x_cubed = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);
    return 0.5 * x * (1.0 + tanh(inner));
}

// Abramowitz and Stegun approximation (accurate to ~1.5e-7)
fn erf_approx(x: f32) -> f32 {
    let sign = select(-1.0, 1.0, x >= 0.0);
    let abs_x = abs(x);
    let t = 1.0 / (1.0 + P * abs_x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let y = 1.0 - (A1 * t + A2 * t2 + A3 * t3 + A4 * t4 + A5 * t5) * exp(-abs_x * abs_x);
    return sign * y;
}

fn gelu(x: f32) -> f32 {
    return 0.5 * x * (1.0 + erf_approx(x * SQRT_2_INV));
}

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

@compute @workgroup_size(512, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_outputs = info.m * info.n;
    if (idx >= total_outputs) { return; }

    let row = idx / info.n;
    let col = idx % info.n;

    var sum: f32 = 0.0;

    let input_offset = row * info.k;
    let weight_offset = col * info.k;

    let k_vec = info.k / 4u;
    for (var t = 0u; t < k_vec; t = t + 1u) {
        let base = t * 4u;
        
        let input_vec = vec4<f32>(
            input[input_offset + base + 0u],
            input[input_offset + base + 1u],
            input[input_offset + base + 2u],
            input[input_offset + base + 3u]
        );
        
        let weight_vec = vec4<f32>(
            fc1_weight[weight_offset + base + 0u],
            fc1_weight[weight_offset + base + 1u],
            fc1_weight[weight_offset + base + 2u],
            fc1_weight[weight_offset + base + 3u]
        );

        sum = fma(input_vec.x, weight_vec.x, sum);
        sum = fma(input_vec.y, weight_vec.y, sum);
        sum = fma(input_vec.z, weight_vec.z, sum);
        sum = fma(input_vec.w, weight_vec.w, sum);
    }

    // Remainder loop
    let remainder_start = k_vec * 4u;
    for (var kk = remainder_start; kk < info.k; kk = kk + 1u) {
        sum = fma(input[input_offset + kk], fc1_weight[weight_offset + kk], sum);
    }

    sum = sum + fc1_bias[col];

    if (COMPUTE_ACT_TYPE == 0u) {
        output[idx] = gelu(sum);
    } else if (COMPUTE_ACT_TYPE == 1u) {
        output[idx] = gelu_new(sum);
    } else if (COMPUTE_ACT_TYPE == 2u) {
        output[idx] = max(0.0, sum); // ReLU
    } else if (COMPUTE_ACT_TYPE == 3u) {
        output[idx] = silu(sum);
    } else if (COMPUTE_ACT_TYPE == 4u) {
        output[idx] = tanh(sum);
    } else if (COMPUTE_ACT_TYPE == 5u) {
        // QuickGELU: x * sigmoid(1.702x), what CLIP was trained with.
        output[idx] = sum / (1.0 + exp(-1.702 * sum));
    }
}