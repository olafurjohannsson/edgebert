use std::collections::HashMap;

pub fn hybrid_search(
    keyword_results: Vec<(usize, f32)>,
    semantic_results: Vec<(usize, f32)>,
    limit: usize,
) -> Vec<(usize, f32)> {
    let mut combined_scores: HashMap<usize, f32> = HashMap::new();
    let k = 60.0;

    for (rank, (idx, _score)) in keyword_results.iter().enumerate() {
        let score = 1.0 / (k + (rank + 1) as f32);
        combined_scores
            .entry(*idx)
            .and_modify(|s| *s += score)
            .or_insert(score);
    }

    for (rank, (idx, _score)) in semantic_results.iter().enumerate() {
        let score = 1.0 / (k + (rank + 1) as f32);
        combined_scores
            .entry(*idx)
            .and_modify(|s| *s += score)
            .or_insert(score);
    }

    let mut final_results: Vec<(usize, f32)> = combined_scores.into_iter().collect();
    final_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    final_results.truncate(limit);
    final_results
}