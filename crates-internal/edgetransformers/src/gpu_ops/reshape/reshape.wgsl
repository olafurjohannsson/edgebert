struct ReshapeUniforms {
b: u32, // batch_size
s: u32, // seq_len
h: u32, // num_heads
d: u32, // head_dim
transpose_k: u32, // 0 for false, 1 for true
};
@group(0) @binding(0) var<uniform> uniforms: ReshapeUniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
// Reshapes [B, S, HD] -> [B, H, S, D] (for Q, V)
// or [B, S, HD] -> [B, H, D, S] (for K^T)
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

    let batch_offset_in = b_idx * seq_len * hidden_size;
    let batch_offset_out = b_idx * seq_len * hidden_size;

    for (var d_idx = 0u; d_idx < head_dim; d_idx = d_idx + 1u) {
        // Index for the flat input buffer [B, S, H*D]
        let in_idx = batch_offset_in + s_idx * hidden_size + h_idx * head_dim + d_idx;

        var out_idx = 0u;
        if (uniforms.transpose_k == 1u) { // For K^T: [B, H, D, S]
            out_idx = batch_offset_out + h_idx * (seq_len * head_dim) + d_idx * seq_len + s_idx;
        } else { // For Q, V: [B, H, S, D]
            out_idx = batch_offset_out + h_idx * (seq_len * head_dim) + s_idx * head_dim + d_idx;
        }
        
        output[out_idx] = input[in_idx];
    }
}