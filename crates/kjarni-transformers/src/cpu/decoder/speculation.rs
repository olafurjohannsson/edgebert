use crate::{
    Cache,
    common::{
        CancellationToken, GenerationConfig, SpeculationParams, StreamedToken, TokenType,
        apply_no_repeat_ngram, apply_repetition_penalty_mut, sample_token,
    },
    cpu::decoder::CpuDecoderBackend,
    decoder::{
        backend::AnyDecoderBackend,
        traits::{DecoderGenerationBackend, DecoderLanguageModel},
    },
    stats::GenerationStats,
};

use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2, s};
use std::sync::Arc;

/// Cached draft model context for speculative decoding.
pub struct DraftModelContext {
    pub model: Arc<dyn DecoderLanguageModel + Send + Sync>,
    pub backend: AnyDecoderBackend,
}

impl DraftModelContext {
    pub fn load(model: Arc<dyn DecoderLanguageModel + Send + Sync>) -> Result<Self> {
        log::info!("Loading draft model from");

        let backend = AnyDecoderBackend::Cpu(CpuDecoderBackend::new());

        log::info!(
            "Draft model loaded: vocab_size={}, context_size={}",
            model.vocab_size(),
            model.context_size()
        );

        Ok(Self { model, backend })
    }
}

/// Runs speculative decoding generation loop.
pub async fn run_speculative_generation_loop(
    target: Arc<dyn DecoderLanguageModel + Send + Sync>,
    target_backend: AnyDecoderBackend,
    draft: Arc<dyn DecoderLanguageModel + Send + Sync>,
    draft_backend: AnyDecoderBackend,
    input_tokens: Vec<u32>,
    config: GenerationConfig,
    spec_params: &SpeculationParams,
    tx: tokio::sync::mpsc::Sender<Result<StreamedToken>>,
    cancellation: Option<CancellationToken>,
) -> Result<()> {
    let prompt_len = input_tokens.len();
    if prompt_len == 0 {
        return Err(anyhow!("Empty prompt"));
    }

    let max_len = config
        .max_new_tokens
        .map(|n| prompt_len + n)
        .unwrap_or(config.max_length);
    let context_limit = target.context_size();
    let cache_capacity = max_len + spec_params.num_tokens + 1;
    let num_speculative = spec_params.num_tokens;
    let probabilistic = spec_params.probabilistic;

    log::info!(
        "Speculative decoding: prompt={}, max={}, depth={}, probabilistic={}",
        prompt_len,
        max_len,
        num_speculative,
        probabilistic
    );

    // Allocate caches
    let mut target_cache = target.new_cache(1, cache_capacity, 1)?;
    let mut draft_cache = draft.new_cache(1, cache_capacity, 1)?;

    let mut all_tokens = input_tokens.clone();
    let mut stats = GenerationStats::new();

    stats.start_prefill(prompt_len);

    let tokens_array = Array2::from_shape_vec((1, prompt_len), input_tokens.clone())?;

    let (target_result, draft_result) = tokio::join!(
        target_backend.prefill(target.as_ref(), &tokens_array, target_cache.as_mut()),
        draft_backend.prefill(draft.as_ref(), &tokens_array, draft_cache.as_mut()),
    );

    // The prefill result is awaited for its error; the target logits themselves are
    // recomputed per verification round below.
    target_result?;
    let mut draft_logits = draft_result?;

    stats.end_prefill();

    let tokenizer = target.tokenizer();
    for &token_id in &input_tokens {
        if Some(token_id) == target.bos_token_id() {
            continue;
        }

        // Check cancellation
        if let Some(ref c) = cancellation
            && c.is_cancelled()
        {
            log::debug!("Cancelled during prompt emission");
            return Ok(());
        }

        let text = tokenizer
            .decode(&[token_id], false)
            .map_err(|e| anyhow!("Decode error: {}", e))?;

        if tx
            .send(Ok(StreamedToken {
                text,
                id: token_id,
                token_type: TokenType::Prompt,
            }))
            .await
            .is_err()
        {
            return Ok(());
        }
    }

    let stop_tokens = target.stop_token_ids();
    let mut draft_token_tensor = draft_backend.new_decode_token()?;

    'outer: while all_tokens.len() < max_len && all_tokens.len() < context_limit {
        // Check cancellation
        if let Some(ref c) = cancellation
            && c.is_cancelled()
        {
            log::debug!("Cancelled in speculation loop");
            break;
        }

        let mut draft_tokens: Vec<u32> = Vec::with_capacity(num_speculative);
        let mut draft_probs: Vec<Array1<f32>> = Vec::with_capacity(num_speculative);

        let mut current_logits = draft_logits.clone();
        let mut seq_len = all_tokens.len() + 1;

        #[allow(
            clippy::explicit_counter_loop,
            reason = "seq_len is also advanced inside the body, so it is not a plain loop counter"
        )]
        for _ in 0..num_speculative {
            // The plain loop penalises repetition before sampling; without the
            // same treatment here the draft proposes tokens the target would
            // never pick, every one is rejected, and the output drifts toward
            // exactly the repetition the penalty exists to stop.
            let mut draft_logits_step = current_logits.clone();
            apply_penalties(&mut draft_logits_step, &all_tokens, &config);
            let probs = softmax(&draft_logits_step);
            let token = sample_token(draft_logits_step, &crate::common::DecodingStrategy::Greedy)?;

            draft_tokens.push(token);
            draft_probs.push(probs);

            if stop_tokens.contains(&token) {
                break;
            }

            draft_backend.update_decode_token(&mut draft_token_tensor, token)?;
            current_logits = draft_backend
                .decode_one(
                    draft.as_ref(),
                    &draft_token_tensor,
                    seq_len,
                    draft_cache.as_mut(),
                )
                .await?;
            seq_len += 1;
        }

        if draft_tokens.is_empty() {
            break;
        }

        log::trace!(
            "Draft proposed {} tokens: {:?}",
            draft_tokens.len(),
            draft_tokens
        );

        // What verifies a draft token is the target's prediction from the state
        // *before* that token. Feeding only the draft tokens gives logits shifted
        // by one — row i is the prediction after d_i, not the one that produced
        // it — and the accept loop then compares every token against the wrong
        // distribution. The output degenerates into repetition rather than
        // failing outright, which is why this survived without a test.
        //
        // Prepending the last accepted token restores the alignment: row i is the
        // prediction that should have produced d_i, and the final row is the bonus
        // token that comes free when every draft token is accepted.
        let last_accepted = *all_tokens
            .last()
            .ok_or_else(|| anyhow!("speculation with an empty context"))?;
        let mut verify_input = Vec::with_capacity(draft_tokens.len() + 1);
        verify_input.push(last_accepted);
        verify_input.extend_from_slice(&draft_tokens);

        // Re-scoring that last token means rewinding one position, or it would be
        // written into the cache twice.
        let verify_offset = all_tokens.len() - 1;
        target_cache.set_seq_length(verify_offset);

        let verify_logits = verify_batch(
            target.as_ref(),
            &verify_input,
            verify_offset,
            target_cache.as_mut(),
        )?;

        // Same penalties on the verifying side, or the target scores each draft
        // token against a different distribution than plain decoding would use
        // and the two paths disagree for the same prompt.
        let mut verify_logits = verify_logits;
        for i in 0..verify_logits.shape()[0] {
            let mut row = verify_logits.slice(s![i, ..]).to_owned();
            apply_penalties(&mut row, &all_tokens, &config);
            verify_logits.slice_mut(s![i, ..]).assign(&row);
        }

        let (accepted_tokens, _final_logits) = if probabilistic {
            accept_probabilistic(&draft_tokens, &draft_probs, &verify_logits)
        } else {
            accept_greedy(&draft_tokens, &verify_logits)
        };

        let num_accepted_from_draft = accepted_tokens
            .len()
            .saturating_sub(1)
            .min(draft_tokens.len());
        log::trace!(
            "Accepted {}/{} draft tokens",
            num_accepted_from_draft,
            draft_tokens.len()
        );

        for &token in &accepted_tokens {
            // Before emitting, not after. The plain loop stops on a stop token
            // without sending it; sending first put `<|im_end|>` on the end of
            // every speculative reply.
            if stop_tokens.contains(&token) {
                all_tokens.push(token);
                break 'outer;
            }

            all_tokens.push(token);
            stats.record_token();

            let text = tokenizer
                .decode(&[token], false)
                .map_err(|e| anyhow!("Decode error: {}", e))?;

            if tx
                .send(Ok(StreamedToken {
                    text,
                    id: token,
                    token_type: TokenType::Generated,
                }))
                .await
                .is_err()
            {
                break 'outer;
            }
        }

        target_cache.set_seq_length(all_tokens.len());

        draft_cache.clear();
        let resync_array = Array2::from_shape_vec((1, all_tokens.len()), all_tokens.clone())?;
        draft_logits = draft_backend
            .prefill(draft.as_ref(), &resync_array, draft_cache.as_mut())
            .await?;
    }

    stats.print_summary();
    Ok(())
}

fn verify_batch(
    model: &dyn DecoderLanguageModel,
    tokens: &[u32],
    offset: usize,
    cache: &mut dyn Cache,
) -> Result<Array2<f32>> {
    let ops = model
        .decoder_cpu_ops()
        .ok_or_else(|| anyhow!("CPU ops required for speculative decoding"))?;

    let num_tokens = tokens.len();
    let tokens_array = Array2::from_shape_vec((1, num_tokens), tokens.to_vec())?;

    // Forward pass
    let hidden = ops.embed(&tokens_array, offset)?;
    let mask = ops.get_attention_mask(num_tokens, offset)?;
    let output = ops.decoder().forward(&hidden, &mask, offset, Some(cache))?;
    let logits_3d = ops.project_to_logits(&output)?;

    // [1, num_tokens, vocab] -> [num_tokens, vocab]
    let vocab_size = model.vocab_size();
    Ok(logits_3d.into_shape_with_order((num_tokens, vocab_size))?)
}

/// Greedy acceptance: accept while target argmax matches draft.
fn accept_greedy(draft_tokens: &[u32], verify_logits: &Array2<f32>) -> (Vec<u32>, Array1<f32>) {
    let mut accepted = Vec::new();

    for (i, &draft_token) in draft_tokens.iter().enumerate() {
        let logits_i = verify_logits.slice(s![i, ..]).to_owned();
        let target_token =
            sample_token(logits_i.clone(), &crate::common::DecodingStrategy::Greedy).unwrap();

        if target_token == draft_token {
            accepted.push(draft_token);
        } else {
            accepted.push(target_token);
            return (accepted, logits_i);
        }
    }

    let last_logits = verify_logits.slice(s![-1, ..]).to_owned();
    let bonus = sample_token(
        last_logits.clone(),
        &crate::common::DecodingStrategy::Greedy,
    )
    .unwrap();
    accepted.push(bonus);

    (accepted, last_logits)
}

/// preserves target distribution exactly.
fn accept_probabilistic(
    draft_tokens: &[u32],
    draft_probs: &[Array1<f32>],
    verify_logits: &Array2<f32>,
) -> (Vec<u32>, Array1<f32>) {
    let mut accepted = Vec::new();
    let mut rng = rand::thread_rng();

    for (i, &draft_token) in draft_tokens.iter().enumerate() {
        let target_logits = verify_logits.slice(s![i, ..]).to_owned();
        let target_probs = softmax(&target_logits);

        let p_draft = draft_probs[i][draft_token as usize];
        let p_target = target_probs[draft_token as usize];

        // Accept with probability min(1, p_target / p_draft)
        let accept_prob = (p_target / p_draft.max(1e-10)).min(1.0);

        if rand::Rng::r#gen::<f32>(&mut rng) < accept_prob {
            accepted.push(draft_token);
        } else {
            // Reject - sample from residual distribution
            let residual = compute_residual(&target_probs, &draft_probs[i]);
            let resampled = sample_from_distribution(&residual);
            accepted.push(resampled);
            return (accepted, target_logits);
        }
    }
    let last_logits = verify_logits.slice(s![-1, ..]).to_owned();
    let bonus = sample_token(
        last_logits.clone(),
        &crate::common::DecodingStrategy::Greedy,
    )
    .unwrap();
    accepted.push(bonus);

    (accepted, last_logits)
}

/// Computes residual distribution: normalize(max(0, p_target - p_draft))
fn compute_residual(target: &Array1<f32>, draft: &Array1<f32>) -> Array1<f32> {
    let residual: Array1<f32> = target
        .iter()
        .zip(draft.iter())
        .map(|(&t, &d)| (t - d).max(0.0))
        .collect();

    let sum: f32 = residual.sum();
    if sum > 1e-10 {
        residual / sum
    } else {
        target.clone()
    }
}

#[inline]
fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Array1<f32> = logits.mapv(|x| (x - max).exp());
    let sum: f32 = exp.sum();
    exp / sum
}

#[inline]
fn sample_from_distribution(probs: &Array1<f32>) -> u32 {
    let mut rng = rand::thread_rng();
    let r: f32 = rand::Rng::r#gen(&mut rng);

    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i as u32;
        }
    }

    (probs.len() - 1) as u32
}

/// Applies the same logit penalties the plain decoding loop applies.
///
/// Speculative decoding must produce what plain decoding would; anything applied
/// on one side and not the other makes `--draft` quietly change the output.
fn apply_penalties(logits: &mut Array1<f32>, history: &[u32], config: &GenerationConfig) {
    if config.repetition_penalty != 1.0 {
        apply_repetition_penalty_mut(logits, history, config.repetition_penalty);
    }
    if config.no_repeat_ngram_size > 0 {
        apply_no_repeat_ngram(logits, history, config.no_repeat_ngram_size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_softmax() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let probs = softmax(&logits);

        assert!((probs.sum() - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_residual() {
        let target = Array1::from_vec(vec![0.5, 0.3, 0.2]);
        let draft = Array1::from_vec(vec![0.3, 0.5, 0.2]);

        let residual = compute_residual(&target, &draft);

        assert!((residual.sum() - 1.0).abs() < 1e-5);
        assert!(residual[0] > 0.9); // Should be ~1.0 after normalization
    }

    #[test]
    fn test_greedy_accept_all() {
        let draft = vec![1, 2, 3];
        let mut logits = Array2::zeros((3, 10));
        logits[[0, 1]] = 10.0;
        logits[[1, 2]] = 10.0;
        logits[[2, 3]] = 10.0;

        let (accepted, _) = accept_greedy(&draft, &logits);

        assert_eq!(accepted.len(), 4);
        assert_eq!(&accepted[..3], &[1, 2, 3]);
    }

    #[test]
    fn test_greedy_reject_middle() {
        let draft = vec![1, 2, 3];
        let mut logits = Array2::zeros((3, 10));
        logits[[0, 1]] = 10.0; // Matches draft[0]
        logits[[1, 5]] = 10.0; // Doesn't match draft[1]=2
        logits[[2, 3]] = 10.0;

        let (accepted, _) = accept_greedy(&draft, &logits);

        // Accept first, reject second with correction
        assert_eq!(accepted.len(), 2);
        assert_eq!(accepted[0], 1);
        assert_eq!(accepted[1], 5); // Target's choice
    }
}
