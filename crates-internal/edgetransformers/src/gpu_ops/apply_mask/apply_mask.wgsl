struct MaskUniforms {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
};

@group(0) @binding(0) var<uniform> uniforms: MaskUniforms;
@group(0) @binding(1) var<storage, read_write> scores: array<f32>; // Shape: [B, H, S, S]
@group(0) @binding(2) var<storage, read> mask: array<f32>;         // Shape: [B, S]

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let query_pos = global_id.x;
    let key_pos = global_id.y;
    let batch_and_head_idx = global_id.z; // Combined batch and head index

    let total_heads = uniforms.batch_size * uniforms.num_heads;
    if (query_pos >= uniforms.seq_len || key_pos >= uniforms.seq_len || batch_and_head_idx >= total_heads) {
        return;
    }

    let batch_idx = batch_and_head_idx / uniforms.num_heads;
    
    // If the mask for this batch item at the KEY position is 0, apply the mask.
    if (mask[batch_idx * uniforms.seq_len + key_pos] == 0.0) {
        let score_idx = batch_and_head_idx * uniforms.seq_len * uniforms.seq_len 
                      + query_pos * uniforms.seq_len 
                      + key_pos;
        scores[score_idx] = -1.0e9; // Large negative float
    }
}