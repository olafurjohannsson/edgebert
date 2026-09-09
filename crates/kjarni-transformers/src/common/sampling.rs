use crate::activations::softmax_1d_inplace;
pub use crate::common::{DecodingStrategy, GenerationConfig};
use anyhow::{Result, anyhow};
use ndarray::Array1;
use ndarray::{ArrayBase, DataMut, Ix1};
use rand::Rng;

pub fn apply_repetition_penalty_inplace<S>(
    logits: &mut ArrayBase<S, Ix1>,
    tokens: &[u32],
    penalty: f32,
) where
    S: DataMut<Elem = f32>,
{
    if penalty == 1.0 {
        return;
    }
    for &token in tokens {
        let idx = token as usize;
        if idx < logits.len() {
            let score = logits[idx];
            if score < 0.0 {
                logits[idx] = score * penalty;
            } else {
                logits[idx] = score / penalty;
            }
        }
    }
}

/// Apply no-repeat n-gram blocking in-place - works with both Array1 and ArrayViewMut1
pub fn apply_no_repeat_ngram_inplace<S>(
    logits: &mut ArrayBase<S, Ix1>,
    tokens: &[u32],
    ngram_size: usize,
) where
    S: DataMut<Elem = f32>,
{
    let n = ngram_size;
    // Need at least n-1 tokens to form a prefix
    if tokens.len() < n - 1 {
        return;
    }

    // The last n-1 tokens form the current prefix
    let current_prefix = &tokens[tokens.len() - (n - 1)..];

    // Look for any historical n-gram that starts with this prefix
    for window in tokens.windows(n) {
        if &window[..n - 1] == current_prefix {
            // This n-gram would be repeated - ban the completing token
            let banned_token = window[n - 1] as usize;
            if banned_token < logits.len() {
                logits[banned_token] = f32::NEG_INFINITY;
            }
        }
    }
}
pub fn apply_repetition_penalty(
    mut logits: Array1<f32>,
    past_tokens: &[u32],
    penalty: f32,
) -> Array1<f32> {
    if penalty == 1.0 {
        return logits;
    }

    for &token_id in past_tokens {
        // log::info!("Applying repetition penalty to token {}: logits: {} logits: {:?}", token_id, logits.len(), logits);
        let score = logits[token_id as usize];

        if score > 0.0 {
            logits[token_id as usize] = score / penalty;
        } else {
            logits[token_id as usize] = score * penalty;
        }
    }
    logits
}

pub fn sample_token(mut logits: Array1<f32>, strategy: &DecodingStrategy) -> Result<u32> {
    match strategy {
        // `partial_cmp` returns `None` for NaN, and unwrapping it turned any upstream
        // numerical fault into a bare panic with no clue where it came from. A GPU
        // kernel misreading its weights surfaces here first, so say so.
        DecodingStrategy::Greedy => {
            let mut best: Option<(usize, f32)> = None;
            for (i, &v) in logits.iter().enumerate() {
                if v.is_nan() {
                    return Err(anyhow!(
                        "logit {i} of {} is NaN; the forward pass produced no usable \
                         distribution",
                        logits.len()
                    ));
                }
                if best.is_none_or(|(_, b)| v > b) {
                    best = Some((i, v));
                }
            }
            best.map(|(i, _)| i as u32)
                .ok_or_else(|| anyhow!("cannot sample from an empty logit vector"))
        }
        DecodingStrategy::Sample(params) => {
            // Apply filters sequentially
            if let Some(k) = params.top_k {
                logits = top_k_filtering(logits, k);
            }
            if let Some(p) = params.top_p {
                logits = top_p_filtering(logits, p);
            }
            if let Some(mp) = params.min_p {
                logits = min_p_filtering(logits, mp);
            }
            let temp = if params.temperature < 1e-5 {
                1.0
            } else {
                params.temperature
            };
            logits /= temp;

            softmax_1d_inplace(&mut logits);
            sample_from_probs(&logits)
        }
        DecodingStrategy::BeamSearch(_) => {
            anyhow::bail!("Beam search is not supported in this generator.")
        }
    }
}

/// Get probability of a specific token after softmax.
#[inline]
pub fn softmax_prob(logits: &Array1<f32>, token: u32) -> f32 {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
    let token_exp = (logits[token as usize] - max_logit).exp();
    token_exp / exp_sum
}

fn softmax_1d(logits: &Array1<f32>) -> Array1<f32> {
    let mut probs = logits.clone();
    softmax_1d_inplace(&mut probs);
    probs
}

pub fn min_p_filtering(mut logits: Array1<f32>, min_p: f32) -> Array1<f32> {
    let probs = softmax_1d(&logits);
    let max_prob = probs.fold(0.0f32, |a, &b| a.max(b));
    let cutoff = max_prob * min_p;

    for (i, &prob) in probs.iter().enumerate() {
        if prob < cutoff {
            logits[i] = f32::NEG_INFINITY;
        }
    }
    logits
}

pub fn top_k_filtering(mut logits: Array1<f32>, k: usize) -> Array1<f32> {
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    for &idx in &indices[k..] {
        logits[idx] = f32::NEG_INFINITY;
    }
    logits
}

pub fn top_p_filtering(mut logits: Array1<f32>, p: f32) -> Array1<f32> {
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());

    let probs = softmax_1d(&logits);

    let mut cumulative = 0.0;
    for (i, &idx) in indices.iter().enumerate() {
        cumulative += probs[idx];
        if cumulative > p {
            for &invalid_idx in &indices[i + 1..] {
                logits[invalid_idx] = f32::NEG_INFINITY;
            }
            break;
        }
    }
    logits
}

pub fn sample_from_probs(probs: &Array1<f32>) -> Result<u32> {
    let mut rng = rand::thread_rng();
    let uniform: f32 = rng.r#gen();
    let mut cumulative = 0.0;
    for (idx, &prob) in probs.iter().enumerate() {
        cumulative += prob;
        if cumulative >= uniform {
            return Ok(idx as u32);
        }
    }
    Ok((probs.len() - 1) as u32)
}

pub fn get_top_k_from_log_probs(log_probs: &Array1<f32>, k: usize) -> Vec<(u32, f32)> {
    let mut indexed_log_probs: Vec<(usize, f32)> = log_probs
        .iter()
        .enumerate()
        .map(|(i, &lp)| (i, lp))
        .collect();
    indexed_log_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed_log_probs.truncate(k);
    indexed_log_probs
        .into_iter()
        .map(|(i, lp)| (i as u32, lp))
        .collect()
}

pub fn log_softmax_1d(logits: &Array1<f32>) -> Array1<f32> {
    let max_val = logits.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let scaled_logits = logits - max_val;
    let exp_sum = scaled_logits.mapv(f32::exp).sum();
    scaled_logits - exp_sum.ln()
}

pub fn apply_repetition_penalty_mut(logits: &mut Array1<f32>, tokens: &[u32], penalty: f32) {
    for &token in tokens {
        let idx = token as usize;
        if idx < logits.len() {
            let score = logits[idx];
            if score < 0.0 {
                logits[idx] = score * penalty;
            } else {
                logits[idx] = score / penalty;
            }
        }
    }
}

pub fn apply_no_repeat_ngram(logits: &mut Array1<f32>, tokens: &[u32], ngram_size: usize) {
    let n = ngram_size;
    if tokens.len() < n - 1 {
        return;
    }
    let current_prefix = &tokens[tokens.len() - (n - 1)..];
    for window in tokens.windows(n) {
        if &window[..n - 1] == current_prefix {
            let banned_token = window[n - 1] as usize;
            if banned_token < logits.len() {
                logits[banned_token] = f32::NEG_INFINITY;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_softmax_1d_basic() {
        let mut logits = array![1.0, 2.0, 3.0];
        softmax_1d_inplace(&mut logits);
        let probs = logits.clone();
        // Sum should be 1.0
        assert!((probs.sum() - 1.0).abs() < 1e-6);
        // All probabilities should be positive
        assert!(probs.iter().all(|&p| p > 0.0));
        // Higher logit = higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_1d_uniform() {
        let mut logits = array![1.0, 1.0, 1.0, 1.0];
        softmax_1d_inplace(&mut logits);

        // Equal logits = equal probabilities
        for &p in logits.iter() {
            assert!((p - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_1d_numerical_stability() {
        // Large values that could overflow without proper implementation
        let mut logits = array![1000.0, 1001.0, 1002.0];
        softmax_1d_inplace(&mut logits);
        let probs = logits.clone();

        assert!((probs.sum() - 1.0).abs() < 1e-6);
        assert!(probs.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn test_softmax_1d_negative_values() {
        let mut logits = array![-2.0, -1.0, 0.0, 1.0];
        softmax_1d_inplace(&mut logits);
        let probs = logits.clone();

        assert!((probs.sum() - 1.0).abs() < 1e-6);
        assert!(probs[3] > probs[2]);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_log_softmax_1d_basic() {
        let mut logits = array![1.0, 2.0, 3.0];
        let log_probs = log_softmax_1d(&logits);

        softmax_1d_inplace(&mut logits);
        let probs = logits.clone();
        for i in 0..3 {
            assert!((log_probs[i] - probs[i].ln()).abs() < 1e-5);
        }
    }

    #[test]
    fn test_log_softmax_1d_all_negative() {
        let logits = array![1.0, 2.0, 3.0];
        let log_probs = log_softmax_1d(&logits);

        assert!(log_probs.iter().all(|&lp| lp <= 0.0));
    }

    #[test]
    fn test_top_k_filtering_basic() {
        let logits = array![1.0, 5.0, 3.0, 4.0, 2.0];
        let filtered = top_k_filtering(logits, 3);

        // Top 3 are indices 1 (5.0), 3 (4.0), 2 (3.0)
        assert!(filtered[1].is_finite());
        assert!(filtered[3].is_finite());
        assert!(filtered[2].is_finite());
        assert!(filtered[0] == f32::NEG_INFINITY);
        assert!(filtered[4] == f32::NEG_INFINITY);
    }

    #[test]
    fn test_top_k_filtering_k_equals_len() {
        let logits = array![1.0, 2.0, 3.0];
        let filtered = top_k_filtering(logits.clone(), 3);

        // Nothing should be filtered
        assert!(filtered.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_top_k_filtering_k_is_one() {
        let logits = array![1.0, 5.0, 3.0];
        let filtered = top_k_filtering(logits, 1);

        assert!(filtered[1].is_finite());
        assert!(filtered[0] == f32::NEG_INFINITY);
        assert!(filtered[2] == f32::NEG_INFINITY);
    }
    #[test]
    fn test_top_p_filtering_basic() {
        let logits = array![0.0, 1.0, 2.0, 3.0];
        let filtered = top_p_filtering(logits, 0.9);
        assert!(filtered[3].is_finite());
    }

    #[test]
    fn test_top_p_filtering_p_is_one() {
        let logits = array![1.0, 2.0, 3.0, 4.0];
        let filtered = top_p_filtering(logits.clone(), 1.0);

        // p=1.0 should keep everything
        assert!(filtered.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_top_p_filtering_very_small_p() {
        let logits = array![1.0, 2.0, 10.0]; // 10.0 dominates after softmax
        let filtered = top_p_filtering(logits, 0.01);

        assert!(filtered[2].is_finite());
    }

    #[test]
    fn test_repetition_penalty_no_penalty() {
        let logits = array![1.0, 2.0, 3.0];
        let result = apply_repetition_penalty(logits.clone(), &[0, 1], 1.0);

        assert_eq!(logits, result);
    }

    #[test]
    fn test_repetition_penalty_positive_logits() {
        let logits = array![2.0, 4.0, 6.0];
        let result = apply_repetition_penalty(logits, &[1], 2.0);

        assert_eq!(result[0], 2.0); // unchanged
        assert_eq!(result[1], 2.0); // 4.0 / 2.0
        assert_eq!(result[2], 6.0); // unchanged
    }

    #[test]
    fn test_repetition_penalty_negative_logits() {
        let logits = array![-2.0, -4.0, 1.0];
        let result = apply_repetition_penalty(logits, &[0, 1], 2.0);
        assert_eq!(result[0], -4.0); // -2.0 * 2.0
        assert_eq!(result[1], -8.0); // -4.0 * 2.0
        assert_eq!(result[2], 1.0); // unchanged
    }

    #[test]
    fn test_repetition_penalty_mixed_logits() {
        let logits = array![-1.0, 0.0, 2.0];
        let result = apply_repetition_penalty(logits, &[0, 2], 2.0);

        assert_eq!(result[0], -2.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 1.0);
    }

    #[test]
    fn test_repetition_penalty_mut_basic() {
        let mut logits = array![2.0, 4.0, 6.0];
        apply_repetition_penalty_mut(&mut logits, &[1], 2.0);

        assert_eq!(logits[0], 2.0);
        assert_eq!(logits[1], 2.0); // 4.0 / 2.0
        assert_eq!(logits[2], 6.0);
    }

    #[test]
    fn test_repetition_penalty_mut_out_of_bounds() {
        let mut logits = array![1.0, 2.0, 3.0];
        apply_repetition_penalty_mut(&mut logits, &[100], 2.0);

        assert_eq!(logits, array![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_no_repeat_ngram_basic() {
        let mut logits = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let tokens = vec![0, 1, 2, 0, 1];
        apply_no_repeat_ngram(&mut logits, &tokens, 3);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], 1.0);
        assert_eq!(logits[3], 1.0);
        assert_eq!(logits[4], 1.0);
    }

    #[test]
    fn test_no_repeat_ngram_too_short() {
        let mut logits = array![1.0, 1.0, 1.0];
        let tokens = vec![0]; // Only 1 token, can't form bigram prefix
        apply_no_repeat_ngram(&mut logits, &tokens, 3);

        // Nothing should change - sequence too short
        assert!(logits.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_no_repeat_ngram_no_repeat() {
        let mut logits = array![1.0, 1.0, 1.0, 1.0];
        let tokens = vec![0, 1, 2, 3]; // All unique, no repeats possible
        apply_no_repeat_ngram(&mut logits, &tokens, 3);
        assert!(logits.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_no_repeat_ngram_bigram() {
        let mut logits = array![1.0, 1.0, 1.0, 1.0];
        let tokens = vec![0, 1, 0];
        apply_no_repeat_ngram(&mut logits, &tokens, 2);
        assert_eq!(logits[1], f32::NEG_INFINITY);
    }

    #[test]
    fn test_sample_from_probs_deterministic() {
        let probs = array![0.0, 0.0, 1.0, 0.0];

        for _ in 0..10 {
            let result = sample_from_probs(&probs).unwrap();
            assert_eq!(result, 2);
        }
    }

    #[test]
    fn test_sample_from_probs_valid_range() {
        let probs = array![0.25, 0.25, 0.25, 0.25];

        for _ in 0..100 {
            let result = sample_from_probs(&probs).unwrap();
            assert!(result < 4);
        }
    }

    #[test]
    fn test_sample_token_greedy() {
        let logits = array![1.0, 5.0, 3.0, 2.0];
        let token = sample_token(logits, &DecodingStrategy::Greedy).unwrap();

        // Should always pick index 1 (highest logit)
        assert_eq!(token, 1);
    }

    #[test]
    fn test_sample_token_greedy_tie() {
        let logits = array![5.0, 5.0, 1.0];
        let token = sample_token(logits, &DecodingStrategy::Greedy).unwrap();

        // Should pick first occurrence of max
        assert!(token == 0 || token == 1);
    }

    #[test]
    fn test_sample_token_with_temperature() {
        use crate::common::SamplingParams;

        let logits = array![1.0, 2.0, 3.0];
        let strategy = DecodingStrategy::Sample(SamplingParams {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            min_p: None,
        });

        let token = sample_token(logits, &strategy).unwrap();
        assert!(token < 3);
    }

    #[test]
    fn test_sample_token_with_top_k() {
        use crate::common::SamplingParams;

        let logits = array![1.0, 2.0, 10.0, 0.5, 0.1];
        let strategy = DecodingStrategy::Sample(SamplingParams {
            temperature: 0.1, // Low temp = nearly greedy
            top_k: Some(1),
            top_p: None,
            min_p: None,
        });

        // With top_k=1 and low temperature, should always pick max
        for _ in 0..10 {
            let token = sample_token(logits.clone(), &strategy).unwrap();
            assert_eq!(token, 2);
        }
    }

    #[test]
    fn test_sample_token_beam_search_unsupported() {
        use crate::common::BeamSearchParams;

        let logits = array![1.0, 2.0, 3.0];
        let strategy = DecodingStrategy::BeamSearch(BeamSearchParams {
            num_beams: 4,
            length_penalty: 1.0,
            early_stopping: true,
        });

        let result = sample_token(logits, &strategy);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_top_k_from_log_probs_basic() {
        let log_probs = array![-2.0, -1.0, -3.0, -0.5, -4.0];
        let top_k = get_top_k_from_log_probs(&log_probs, 3);

        assert_eq!(top_k.len(), 3);
        assert_eq!(top_k[0].0, 3);
        assert_eq!(top_k[1].0, 1);
        assert_eq!(top_k[2].0, 0);
    }

    #[test]
    fn test_get_top_k_from_log_probs_k_greater_than_len() {
        let log_probs = array![-1.0, -2.0];
        let top_k = get_top_k_from_log_probs(&log_probs, 10);
        assert_eq!(top_k.len(), 2);
    }

    #[test]
    fn test_get_top_k_from_log_probs_returns_correct_values() {
        let log_probs = array![-1.0, -2.0, -3.0];
        let top_k = get_top_k_from_log_probs(&log_probs, 2);

        assert_eq!(top_k[0], (0, -1.0));
        assert_eq!(top_k[1], (1, -2.0));
    }
    #[test]
    fn test_no_repeat_ngram_inplace_actually_works() {
        use ndarray::Array1;

        let mut logits = Array1::from_vec(vec![1.0; 100]);
        let tokens = vec![10, 20, 30, 10, 20]; // Prefix is [10, 20], which appeared before 30

        apply_no_repeat_ngram_inplace(&mut logits, &tokens, 3);

        // Token 30 should be banned because [10, 20, 30] would repeat
        assert_eq!(logits[30], f32::NEG_INFINITY, "Token 30 should be banned!");
        assert_eq!(logits[0], 1.0, "Other tokens should be unchanged");
    }

    #[test]
    fn test_no_repeat_ngram_inplace_with_view() {
        use ndarray::Array2;

        let mut logits_2d = Array2::from_elem((4, 100), 1.0f32);
        let tokens = vec![10, 20, 30, 10, 20];

        for mut row in logits_2d.outer_iter_mut() {
            apply_no_repeat_ngram_inplace(&mut row, &tokens, 3);
        }

        // Check that all rows have token 30 banned
        for row in logits_2d.rows() {
            assert_eq!(
                row[30],
                f32::NEG_INFINITY,
                "Token 30 should be banned in all rows!"
            );
        }
    }
}
