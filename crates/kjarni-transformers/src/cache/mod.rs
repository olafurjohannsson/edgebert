//! Cache implementations for transformer models

mod cpu;
mod cpu_beam;
pub use cpu::CpuKVCache;
pub use cpu_beam::CpuBeamKVCache;

use std::any::Any;

/// A type-erased, thread-safe container for mutable inference state.
pub trait Cache: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    /// Get the current sequence length (number of cached tokens)
    fn get_seq_length(&self) -> usize;
    fn set_seq_length(&mut self, len: usize);
    /// Clear the cache
    fn clear(&mut self);
    fn increment_len(&mut self, new_tokens_len: usize);
    fn clone_box(&self) -> Box<dyn Cache>;
}

#[cfg(test)]
mod tests;

/// A KV cache kept across calls, together with the tokens it holds.
///
/// Every generation today builds a fresh cache and drops it, so a conversation
/// re-processes its whole history on every turn and a RAG query re-processes its
/// retrieved passages on every question. Prefill is the expensive half of both:
/// measured on Qwen2.5-0.5B, a 2048 token prompt costs 15.96s, while the same
/// prompt with 1984 tokens already cached costs 0.49s. Same logits, 33x cheaper.
///
/// The cache stores K and V, not token ids, so the ids have to be carried
/// alongside to know what is actually in it. They are compared as ids rather than
/// as text on purpose: the same string can tokenise differently depending on what
/// follows it at a chunk boundary, and two different strings can share a token
/// prefix. Only the ids say what the cache really contains.
pub struct PrefixCache {
    cache: Box<dyn Cache>,
    tokens: Vec<u32>,
    capacity: usize,
}

impl PrefixCache {
    /// Wraps a fresh cache that can hold `capacity` tokens.
    pub fn new(cache: Box<dyn Cache>, capacity: usize) -> Self {
        Self {
            cache,
            tokens: Vec::new(),
            capacity,
        }
    }

    /// How many tokens this cache can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Tokens currently valid in the cache.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Access for the generation loop.
    pub fn cache_mut(&mut self) -> &mut dyn Cache {
        self.cache.as_mut()
    }

    /// Prepares the cache to serve `wanted`, and reports how many leading tokens
    /// are already valid and can be skipped.
    ///
    /// The cache is truncated to the shared prefix, so anything after it is
    /// overwritten by the next prefill rather than being read as stale context.
    /// A shared prefix of the full length would leave nothing to prefill, so one
    /// token is always held back for the model to predict from.
    pub fn reuse(&mut self, wanted: &[u32]) -> usize {
        let shared = self
            .tokens
            .iter()
            .zip(wanted.iter())
            .take_while(|(a, b)| a == b)
            .count();

        // Never reuse the entire prompt: prefill needs at least one token to
        // produce logits from.
        let keep = shared.min(wanted.len().saturating_sub(1));

        // Unconditionally, not only when the token list shrinks. After a turn the
        // cache holds the generated tokens too, which this list never recorded,
        // so `keep == self.tokens.len()` does not mean the cache is already the
        // right length. Skipping the call there left the previous turn's output
        // in the cache to be read as context by the next prompt.
        self.cache.set_seq_length(keep);
        self.tokens.truncate(keep);
        keep
    }

    /// Records that `tokens` are now represented in the cache.
    pub fn extend(&mut self, tokens: &[u32]) {
        self.tokens.extend_from_slice(tokens);
    }

    /// Drops everything, as if freshly built.
    pub fn reset(&mut self) {
        self.cache.clear();
        self.tokens.clear();
    }
}

#[cfg(test)]
mod prefix_cache_tests {
    use super::*;

    /// A cache that records what was asked of it, so the prefix logic can be
    /// tested without loading a model.
    #[derive(Default)]
    struct FakeCache {
        len: usize,
        cleared: usize,
    }

    impl Cache for FakeCache {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn get_seq_length(&self) -> usize {
            self.len
        }
        fn set_seq_length(&mut self, len: usize) {
            self.len = len;
        }
        fn clear(&mut self) {
            self.len = 0;
            self.cleared += 1;
        }
        fn increment_len(&mut self, n: usize) {
            self.len += n;
        }
        fn clone_box(&self) -> Box<dyn Cache> {
            Box::new(FakeCache {
                len: self.len,
                cleared: self.cleared,
            })
        }
    }

    fn warmed(tokens: &[u32]) -> PrefixCache {
        let mut pc = PrefixCache::new(Box::new(FakeCache::default()), 1024);
        pc.cache_mut().increment_len(tokens.len());
        pc.extend(tokens);
        pc
    }

    #[test]
    fn an_empty_cache_reuses_nothing() {
        let mut pc = PrefixCache::new(Box::new(FakeCache::default()), 1024);
        assert_eq!(pc.reuse(&[1, 2, 3]), 0);
        assert!(pc.is_empty());
    }

    #[test]
    fn a_shared_prefix_is_reused() {
        let mut pc = warmed(&[1, 2, 3, 4, 5]);
        // shares 1,2,3 then diverges
        assert_eq!(pc.reuse(&[1, 2, 3, 9, 9]), 3);
        assert_eq!(pc.len(), 3);
        assert_eq!(pc.cache_mut().get_seq_length(), 3);
    }

    #[test]
    fn a_diverging_first_token_reuses_nothing() {
        let mut pc = warmed(&[1, 2, 3]);
        assert_eq!(pc.reuse(&[9, 2, 3]), 0);
        assert_eq!(pc.cache_mut().get_seq_length(), 0);
    }

    /// Prefill has to produce logits, so the last token is never served from the
    /// cache even when it matches. Reusing all of it would leave nothing to run.
    #[test]
    fn one_token_is_always_held_back() {
        let mut pc = warmed(&[1, 2, 3, 4]);
        assert_eq!(pc.reuse(&[1, 2, 3, 4]), 3);
        assert_eq!(pc.cache_mut().get_seq_length(), 3);
    }

    /// A cache warmed by a previous turn holds generated tokens the new prompt
    /// does not have. Those must be dropped, not left to be read as context.
    #[test]
    fn generated_tail_is_truncated_away() {
        let mut pc = warmed(&[1, 2, 3]);
        pc.cache_mut().increment_len(10); // as if 10 tokens had been generated
        assert_eq!(pc.reuse(&[1, 2, 3, 7, 8]), 3);
        assert_eq!(
            pc.cache_mut().get_seq_length(),
            3,
            "the generated tail must not survive into the next prompt"
        );
    }

    #[test]
    fn a_shorter_prompt_truncates() {
        let mut pc = warmed(&[1, 2, 3, 4, 5]);
        assert_eq!(pc.reuse(&[1, 2]), 1);
        assert_eq!(pc.len(), 1);
    }

    #[test]
    fn reset_clears_everything() {
        let mut pc = warmed(&[1, 2, 3]);
        pc.reset();
        assert!(pc.is_empty());
        assert_eq!(pc.cache_mut().get_seq_length(), 0);
        assert_eq!(pc.reuse(&[1, 2, 3]), 0);
    }

    #[test]
    fn extend_tracks_what_was_prefilled() {
        let mut pc = PrefixCache::new(Box::new(FakeCache::default()), 1024);
        pc.extend(&[1, 2, 3]);
        assert_eq!(pc.len(), 3);
        pc.extend(&[4]);
        assert_eq!(pc.len(), 4);
        assert_eq!(pc.reuse(&[1, 2, 3, 4]), 3);
    }
}
