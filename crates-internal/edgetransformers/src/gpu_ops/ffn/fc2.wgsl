struct FfnUniforms {
    m: u32,
    k: u32,
    n: u32,
}

@group(0) @binding(0) var<uniform> info: FfnUniforms;
@group(0) @binding(1) var<storage, read> fc2_weight: array<f32>;
@group(0) @binding(2) var<storage, read> fc2_bias: array<f32>;
@group(0) @binding(3) var<storage, read> input: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(512, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_outputs = info.m * info.k;
    
    if (idx >= total_outputs) {
        return;
    }
    
    let row = idx / info.k;
    let col = idx % info.k;
    
    var sum = 0.0;
    let n_vec = info.n / 4u;
    let input_offset = row * info.n;
    
    for (var n = 0u; n < n_vec; n = n + 1u) {
        let n4 = n * 4u;
        
        let input_vec = vec4<f32>(
            input[input_offset + n4],
            input[input_offset + n4 + 1u],
            input[input_offset + n4 + 2u],
            input[input_offset + n4 + 3u]
        );
        
        let weight_vec = vec4<f32>(
            fc2_weight[n4 * info.k + col],
            fc2_weight[(n4 + 1u) * info.k + col],
            fc2_weight[(n4 + 2u) * info.k + col],
            fc2_weight[(n4 + 3u) * info.k + col]
        );
        
        sum = sum + dot(input_vec, weight_vec);
    }
    
    let remainder_start = n_vec * 4u;
    for (var n = remainder_start; n < info.n; n = n + 1u) {
        sum = sum + input[input_offset + n] * fc2_weight[n * info.k + col];
    }
    
    sum = sum + fc2_bias[col];
    output[idx] = sum;
}