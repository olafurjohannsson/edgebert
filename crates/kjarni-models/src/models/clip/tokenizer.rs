//! CLIP's BPE tokenizer.
//!
//! Not the GPT-2 tokenizer already in the tree. Both are byte-pair encodings over
//! the same byte-to-unicode table, but they mark word boundaries at opposite ends:
//! GPT-2 prefixes a leading space as `Ġ`, CLIP appends `</w>` to the final
//! character of each word. Their vocabularies are not interchangeable, and feeding
//! CLIP text through the GPT-2 path yields ids that exist but mean nothing.
//!
//! The pre-tokenizer pattern is hand-rolled rather than pulled in as a regex
//! dependency. It is small enough to read, and the whole point of this engine is
//! that it ships without a tree of crates behind it:
//!
//! ```text
//! <|startoftext|>|<|endoftext|>|'s|'t|'re|'ve|'m|'ll|'d
//!   |[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+
//! ```

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde_json::Value;

use super::nfc_table::CANONICAL_COMPOSITIONS;

pub const BOS: &str = "<|startoftext|>";
pub const EOS: &str = "<|endoftext|>";

/// CLIP's context length. Captions longer than this are truncated, which is what
/// every reference implementation does rather than erroring.
pub const MAX_TOKENS: usize = 77;

pub struct ClipTokenizer {
    vocab: HashMap<String, u32>,
    /// Merge pair to rank; a lower rank is applied first.
    ranks: HashMap<(String, String), usize>,
    byte_encoder: HashMap<u8, char>,
    bos_id: u32,
    eos_id: u32,
}

impl ClipTokenizer {
    pub fn from_json_str(content: &str) -> Result<Self> {
        let json: Value = serde_json::from_str(content).context("parsing tokenizer.json")?;

        let vocab: HashMap<String, u32> = json["model"]["vocab"]
            .as_object()
            .context("tokenizer.json has no model.vocab")?
            .iter()
            .filter_map(|(t, id)| id.as_u64().map(|v| (t.clone(), v as u32)))
            .collect();

        // Merges are stored either as "a b" or as ["a", "b"] depending on the
        // version of `tokenizers` that wrote the file.
        let mut ranks = HashMap::new();
        for (rank, merge) in json["model"]["merges"]
            .as_array()
            .context("tokenizer.json has no model.merges")?
            .iter()
            .enumerate()
        {
            let pair = match merge {
                Value::String(s) => {
                    let mut it = s.split(' ');
                    match (it.next(), it.next()) {
                        (Some(a), Some(b)) => Some((a.to_string(), b.to_string())),
                        _ => None,
                    }
                }
                Value::Array(a) if a.len() == 2 => match (a[0].as_str(), a[1].as_str()) {
                    (Some(x), Some(y)) => Some((x.to_string(), y.to_string())),
                    _ => None,
                },
                _ => None,
            };
            if let Some(p) = pair {
                ranks.entry(p).or_insert(rank);
            }
        }

        let bos_id = *vocab
            .get(BOS)
            .context("tokenizer.json lacks <|startoftext|>")?;
        let eos_id = *vocab
            .get(EOS)
            .context("tokenizer.json lacks <|endoftext|>")?;

        Ok(Self {
            vocab,
            ranks,
            byte_encoder: byte_to_unicode(),
            bos_id,
            eos_id,
        })
    }

    pub fn bos_id(&self) -> u32 {
        self.bos_id
    }

    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }

    /// A caption to token ids, wrapped in start and end markers.
    ///
    /// Truncates to [`MAX_TOKENS`] including both markers, so the end marker is
    /// always present: the text tower pools that position, and a caption that lost
    /// it would be read at whatever token happened to land last.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let normalized = normalize(text);
        let mut ids = vec![self.bos_id];

        for piece in pre_tokenize(&normalized) {
            if piece == BOS || piece == EOS {
                continue;
            }
            // Bytes, not chars: the vocabulary is defined over the byte-level
            // alphabet, so anything outside ASCII is several symbols here.
            let mut symbols: Vec<String> = piece
                .bytes()
                .map(|b| self.byte_encoder[&b].to_string())
                .collect();
            if symbols.is_empty() {
                continue;
            }
            // The word-final marker is what distinguishes CLIP from GPT-2.
            let last = symbols.len() - 1;
            symbols[last].push_str("</w>");

            for token in self.merge(symbols) {
                if let Some(&id) = self.vocab.get(&token) {
                    ids.push(id);
                }
            }
        }

        if ids.len() >= MAX_TOKENS {
            ids.truncate(MAX_TOKENS - 1);
        }
        ids.push(self.eos_id);
        ids
    }

    /// Greedy pair merging: repeatedly join the adjacent pair with the lowest rank.
    fn merge(&self, mut word: Vec<String>) -> Vec<String> {
        while word.len() > 1 {
            let mut best: Option<(usize, usize)> = None;
            for i in 0..word.len() - 1 {
                if let Some(&rank) = self.ranks.get(&(word[i].clone(), word[i + 1].clone()))
                    && best.is_none_or(|(r, _)| rank < r)
                {
                    best = Some((rank, i));
                }
            }
            let Some((_, at)) = best else { break };

            let mut next = Vec::with_capacity(word.len() - 1);
            next.extend_from_slice(&word[..at]);
            next.push(format!("{}{}", word[at], word[at + 1]));
            next.extend_from_slice(&word[at + 2..]);
            word = next;
        }
        word
    }
}

/// Collapse runs of whitespace, trim, and lowercase, matching the normalizer
/// chain in `tokenizer.json`.
///
/// NFC runs first. It matters more than it looks: macOS stores filenames
/// decomposed, so "café" can arrive as `cafe` plus a combining acute, and the two
/// forms tokenize to completely different ids.
fn normalize(text: &str) -> String {
    let composed = compose(text);
    let mut out = String::with_capacity(composed.len());
    let mut in_space = false;
    for ch in composed.chars() {
        if ch.is_whitespace() {
            in_space = true;
            continue;
        }
        if in_space && !out.is_empty() {
            out.push(' ');
        }
        in_space = false;
        for lower in ch.to_lowercase() {
            out.push(lower);
        }
    }
    out
}

/// The pre-tokenizer pattern, by hand.
///
/// Note that numbers split one digit at a time: the pattern is `[\p{N}]`, not
/// `[\p{N}]+`, so "2024" becomes four pieces. Letters and punctuation both run.
fn pre_tokenize(text: &str) -> Vec<String> {
    const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];

        if ch.is_whitespace() {
            i += 1;
            continue;
        }

        if ch == '\'' {
            let rest: String = chars[i..].iter().take(3).collect();
            if let Some(c) = CONTRACTIONS
                .iter()
                .find(|c| rest.to_lowercase().starts_with(**c))
            {
                out.push((*c).to_string());
                i += c.chars().count();
                continue;
            }
        }

        if ch.is_alphabetic() {
            let start = i;
            while i < chars.len() && chars[i].is_alphabetic() {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
            continue;
        }

        if ch.is_numeric() {
            out.push(ch.to_string());
            i += 1;
            continue;
        }

        // Everything else runs together until whitespace or an alphanumeric.
        let start = i;
        while i < chars.len()
            && !chars[i].is_whitespace()
            && !chars[i].is_alphabetic()
            && !chars[i].is_numeric()
            && chars[i] != '\''
        {
            i += 1;
        }
        if i == start {
            // A quote that did not begin a contraction still has to advance.
            i += 1;
        }
        out.push(chars[start..i].iter().collect());
    }

    out
}

/// Canonical composition: fold each base plus combining mark into its single
/// precomposed character, repeatedly, so stacked marks compose too.
///
/// This is the composition half of NFC and not the whole algorithm. Full NFC also
/// canonically reorders combining marks by combining class, which only changes the
/// result when two marks are applied to one base out of order. That does not occur
/// in text people type, and carrying the reordering table would cost far more than
/// it buys here.
fn compose(text: &str) -> String {
    if text.is_ascii() {
        return text.to_string();
    }

    let mut chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i + 1 < chars.len() {
        let key = (chars[i] as u32, chars[i + 1] as u32);
        match CANONICAL_COMPOSITIONS.binary_search_by(|(a, b, _)| (*a, *b).cmp(&key)) {
            Ok(hit) => {
                let composed = CANONICAL_COMPOSITIONS[hit].2;
                chars[i] = char::from_u32(composed).expect("table holds valid scalars");
                chars.remove(i + 1);
                // Do not advance: the result may compose again with what follows.
            }
            Err(_) => i += 1,
        }
    }
    chars.into_iter().collect()
}

/// GPT-2's byte-to-unicode table, which CLIP reuses.
///
/// Printable ASCII maps to itself; everything else is lifted into a private range
/// so that any byte sequence becomes valid, lossless text the merge table can be
/// written over.
fn byte_to_unicode() -> HashMap<u8, char> {
    let mut map = HashMap::new();
    let mut extra = 0u32;
    for b in 0u16..=255 {
        let b = b as u8;
        let printable =
            (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
        let ch = if printable {
            char::from_u32(b as u32).expect("byte is a valid scalar")
        } else {
            let c = char::from_u32(256 + extra).expect("shifted byte is a valid scalar");
            extra += 1;
            c
        };
        map.insert(b, ch);
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitespace_collapses_and_text_lowercases() {
        assert_eq!(normalize("A Photo"), "a photo");
        assert_eq!(normalize("  a   photo  "), "a photo");
        assert_eq!(normalize("a\t\nphoto"), "a photo");
        assert_eq!(normalize(""), "");
        assert_eq!(normalize("   "), "");
        assert_eq!(normalize("ÉCOLE"), "école");
    }

    /// macOS stores filenames decomposed, so a caption really can arrive as `e`
    /// plus a combining acute. Without composing first it tokenizes to different
    /// ids than the same word typed on Linux.
    #[test]
    fn decomposed_accents_compose_before_anything_else() {
        let decomposed = "cafe\u{301}";
        let precomposed = "café";
        assert_ne!(decomposed, precomposed, "the inputs really do differ");
        assert_eq!(compose(decomposed), precomposed);
        assert_eq!(normalize(decomposed), normalize(precomposed));

        // Several marks in one string, and one already-composed for contrast.
        assert_eq!(compose("nai\u{308}ve re\u{301}sume\u{301}"), "naïve résumé");
    }

    #[test]
    fn composition_leaves_ascii_and_unrelated_text_alone() {
        assert_eq!(compose("a photo of a dog"), "a photo of a dog");
        assert_eq!(compose(""), "");
        // A combining mark with nothing to attach to must survive, not vanish.
        assert_eq!(compose("\u{301}"), "\u{301}");
        // Text with no decomposable pair passes through.
        assert_eq!(compose("猫 と 犬"), "猫 と 犬");
    }

    /// The pattern is `[\p{N}]`, not `[\p{N}]+`: digits are separate pieces. Easy
    /// to "fix" into a run and get every number wrong.
    #[test]
    fn digits_split_one_at_a_time() {
        assert_eq!(pre_tokenize("2024"), vec!["2", "0", "2", "4"]);
        assert_eq!(pre_tokenize("a1b"), vec!["a", "1", "b"]);
    }

    #[test]
    fn letters_and_punctuation_run_but_do_not_mix() {
        assert_eq!(pre_tokenize("hello world"), vec!["hello", "world"]);
        assert_eq!(
            pre_tokenize("hello, world!"),
            vec!["hello", ",", "world", "!"]
        );
        assert_eq!(pre_tokenize("a -- b"), vec!["a", "--", "b"]);
        assert_eq!(pre_tokenize("(test)"), vec!["(", "test", ")"]);
        assert_eq!(pre_tokenize("e-mail"), vec!["e", "-", "mail"]);
    }

    #[test]
    fn contractions_are_kept_whole() {
        assert_eq!(pre_tokenize("it's"), vec!["it", "'s"]);
        assert_eq!(pre_tokenize("isn't"), vec!["isn", "'t"]);
        assert_eq!(pre_tokenize("we're"), vec!["we", "'re"]);
        assert_eq!(pre_tokenize("i've"), vec!["i", "'ve"]);
        assert_eq!(pre_tokenize("i'm"), vec!["i", "'m"]);
        assert_eq!(pre_tokenize("we'll"), vec!["we", "'ll"]);
        assert_eq!(pre_tokenize("i'd"), vec!["i", "'d"]);
    }

    /// A quote that begins no contraction still has to advance, or the scanner
    /// spins forever on it.
    #[test]
    fn a_bare_quote_terminates() {
        assert_eq!(pre_tokenize("'"), vec!["'"]);
        assert_eq!(pre_tokenize("'x"), vec!["'", "x"]);
        assert_eq!(pre_tokenize("don'"), vec!["don", "'"]);
    }

    #[test]
    fn empty_and_whitespace_only_input_produce_nothing() {
        assert!(pre_tokenize("").is_empty());
        assert!(pre_tokenize("   \t\n ").is_empty());
    }

    /// The byte table has to be a bijection: every byte gets its own symbol, or
    /// two different inputs would collide onto one token.
    #[test]
    fn the_byte_table_is_a_bijection_over_all_256_bytes() {
        let map = byte_to_unicode();
        assert_eq!(map.len(), 256, "every byte needs a symbol");

        let mut seen = std::collections::HashSet::new();
        for b in 0u16..=255 {
            let c = map[&(b as u8)];
            assert!(seen.insert(c), "byte {b} collides with an earlier one");
        }
        // Printable ASCII maps to itself, which is what makes the vocabulary
        // readable as text.
        assert_eq!(map[&b'a'], 'a');
        assert_eq!(map[&b'!'], '!');
        assert_eq!(map[&b'~'], '~');
        // Control bytes and space are lifted out of the way.
        assert!(map[&b' '] as u32 >= 256, "space must not stay a space");
        assert!(map[&0] as u32 >= 256);
    }
}
