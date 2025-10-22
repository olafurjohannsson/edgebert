//! Pooling strategies for transformer outputs

use anyhow::Result;
use ndarray::{Array2, Array3, Axis};

/// Pooling strategies for sequence outputs
pub enum PoolingStrategy {
    Mean,
    Max,
    Cls,
    LastToken,
}

/// Perform mean pooling over sequence dimension
pub fn mean_pool(hidden: &Array3<f32>, attention_mask: &Array2<f32>) -> Result<Array2<f32>> {
    let mask_expanded = attention_mask.clone().insert_axis(Axis(2));
    let masked_hidden = hidden * &mask_expanded;
    let sum = masked_hidden.sum_axis(Axis(1));
    let count = attention_mask
        .sum_axis(Axis(1))
        .mapv(|x| x.max(1.0))
        .insert_axis(Axis(1));
    
    Ok(sum / &count)
}

/// Extract CLS token embedding (first token)
pub fn cls_pool(hidden: &Array3<f32>) -> Result<Array2<f32>> {
    Ok(hidden.slice(ndarray::s![.., 0, ..]).to_owned())
}

/// Max pooling over sequence dimension
pub fn max_pool(hidden: &Array3<f32>, attention_mask: &Array2<f32>) -> Result<Array2<f32>> {
    let mask_expanded = attention_mask.clone().insert_axis(Axis(2));
    // Set masked positions to -inf before max pooling
    let mut masked_hidden = hidden.clone();
    masked_hidden.zip_mut_with(&mask_expanded, |h, &m| {
        if m == 0.0 {
            *h = f32::NEG_INFINITY;
        }
    });
    
    Ok(masked_hidden.fold_axis(Axis(1), f32::NEG_INFINITY, |&acc, &x| acc.max(x)))
}

/// Extract last token embedding
pub fn last_token_pool(hidden: &Array3<f32>, attention_mask: &Array2<f32>) -> Result<Array2<f32>> {
    let batch_size = hidden.shape()[0];
    let hidden_size = hidden.shape()[2];
    let mut output = Array2::<f32>::zeros((batch_size, hidden_size));
    
    for i in 0..batch_size {
        // Find last non-padding position
        let seq_mask = attention_mask.row(i);
        let last_pos = seq_mask
            .iter()
            .rposition(|&x| x > 0.0)
            .unwrap_or(0);
        
        output.row_mut(i).assign(&hidden.slice(ndarray::s![i, last_pos, ..]));
    }
    
    Ok(output)
}