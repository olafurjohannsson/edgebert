use ndarray::Array2;

/// Create a causal attention mask where position i can only attend to positions 0..=i
/// Returns a [seq_len, seq_len] mask with 1.0 for allowed positions, 0.0 for masked
pub fn create_causal_mask(seq_len: usize) -> Array2<f32> {
    let mut mask = Array2::zeros((seq_len, seq_len));
    for i in 0..seq_len {
        for j in 0..=i {
            mask[[i, j]] = 1.0;
        }
    }
    mask
}

/// Create a padding mask from attention mask
pub fn create_padding_mask(attention_mask: &Array2<f32>) -> Array2<f32> {
    attention_mask.clone()
}
