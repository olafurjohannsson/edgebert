struct ReshapeUniforms {
    b: u32, // batch_size
    s: u32, // seq_len
    h: u32, // num_heads
    d: u32, // head_dim
};

@group(0) @binding(0) var<uniform> uniforms: ReshapeUniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Reshapes [B, H, S, D] -> [B, S, H*D]
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let s_idx = global_id.x; // sequence index
    let h_idx = global_id.y; // head index
    let b_idx = global_id.z; // batch index

    if (s_idx >= uniforms.s || h_idx >= uniforms.h || b_idx >= uniforms.b) {
        return;
    }

    let head_dim = uniforms.d;
    let hidden_size = uniforms.h * head_dim;
    let seq_len = uniforms.s;

    let batch_offset = b_idx * seq_len * hidden_size;

    for (var d_idx = 0u; d_idx < head_dim; d_idx = d_idx + 1u) {
        // Index for the flat input buffer [B, H, S, D]
        let in_idx = batch_offset + h_idx * (seq_len * head_dim) + s_idx * head_dim + d_idx;
        
        // Index for the flat output buffer [B, S, H*D]
        let out_idx = batch_offset + s_idx * hidden_size + h_idx * head_dim + d_idx;
        
        output[out_idx] = input[in_idx];
    }
}