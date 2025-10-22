use serde::{Deserialize, Serialize};
use std::collections::{HashSet, HashMap};

/// BM25 scoring parameters
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Bm25Params {
    /// Controls term frequency saturation (typically 1.2-2.0)
    /// Higher values = term frequency has more impact
    #[serde(default = "default_k1")]
    pub k1: f32,

    /// Controls length normalization (typically 0.5-0.8)
    /// 0.0 = no length normalization, 1.0 = full normalization
    #[serde(default = "default_b")]
    pub b: f32,

    /// Epsilon value to prevent log(0)
    #[serde(default = "default_epsilon")]
    pub epsilon: f32,
}

fn default_k1() -> f32 {
    1.2
}
fn default_b() -> f32 {
    0.75
}
fn default_epsilon() -> f32 {
    0.25
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self {
            k1: default_k1(),
            b: default_b(),
            epsilon: default_epsilon(),
        }
    }
}

/// BM25 index for efficient keyword search
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Bm25Index {
    /// Document frequencies for each term
    doc_frequencies: HashMap<String, usize>,

    /// Document lengths (in tokens)
    doc_lengths: Vec<usize>,

    /// Average document length
    avg_doc_length: f32,

    /// Total number of documents
    total_docs: usize,

    /// Inverted index: term -> list of (doc_id, term_frequency)
    inverted_index: HashMap<String, Vec<(usize, usize)>>,

    /// BM25 parameters
    params: Bm25Params,

    /// Token to index mapping for faster lookups
    token_to_docs: HashMap<String, HashSet<usize>>,
}

impl Bm25Index {
    pub fn new() -> Self {
        Self {
            doc_frequencies: HashMap::new(),
            doc_lengths: Vec::new(),
            avg_doc_length: 0.0,
            total_docs: 0,
            inverted_index: HashMap::new(),
            params: Bm25Params::default(),
            token_to_docs: HashMap::new()
        }
    }

    pub fn search(&self, query: &str, limit: usize) -> Vec<(usize, f32)> {
        if self.total_docs == 0 {
            return Vec::new();
        }

        let query_tokens = tokenize(query);
        if query_tokens.is_empty() {
            return Vec::new();
        }

        let mut scores: HashMap<usize, f32> = HashMap::new();

        for doc_id in 0..self.total_docs {
            let score = self.calculate_score(&query_tokens, doc_id);
            if score > 0.0 {
                scores.insert(doc_id, score);
            }
        }

        let mut results: Vec<(usize, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    /// Calculate BM25 score for a document given query terms
    fn calculate_score(&self, query_tokens: &[String], doc_id: usize) -> f32 {
        let mut score = 0.0;
        let doc_length = self.doc_lengths[doc_id] as f32;

        // Length normalization factor
        let length_norm = 1.0 - self.params.b + self.params.b * (doc_length / self.avg_doc_length);

        for term in query_tokens {
            // Get term frequency in this document
            let tf = self.get_term_frequency(term, doc_id) as f32;
            if tf == 0.0 {
                continue;
            }

            // Get document frequency
            let df = self.doc_frequencies.get(term).copied().unwrap_or(0) as f32;
            if df == 0.0 {
                continue;
            }

            // Calculate IDF component
            // Using log((N - df + 0.5) / (df + 0.5) + 1) for better scores
            let idf = ((self.total_docs as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();

            // Calculate normalized term frequency with saturation
            let normalized_tf = (tf * (self.params.k1 + 1.0)) / (tf + self.params.k1 * length_norm);

            score += idf * normalized_tf;
        }

        score
    }

    fn get_term_frequency(&self, term: &str, doc_id: usize) -> usize {
        self.inverted_index
            .get(term)
            .and_then(|postings| {
                postings
                    .iter()
                    .find(|(id, _)| *id == doc_id)
                    .map(|(_, freq)| *freq)
            })
            .unwrap_or(0)
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty() && s.len() >= 2)
        .map(|s| s.to_string())
        .collect()
}