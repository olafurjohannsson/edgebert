//! Tests for the reranker module.

use super::*;

// Unit Tests

#[test]
fn test_rerank_overrides_default() {
    let overrides = RerankOverrides::default();
    assert!(overrides.top_k.is_none());
    assert!(overrides.threshold.is_none());
    assert!(!overrides.normalize_scores);
}

#[test]
fn test_rerank_overrides_for_search() {
    let overrides = RerankOverrides::for_search();
    assert_eq!(overrides.top_k, Some(10));
    assert!(overrides.threshold.is_none());
}

#[test]
fn test_rerank_overrides_for_filtering() {
    let overrides = RerankOverrides::for_filtering(0.5);
    assert!(overrides.top_k.is_none());
    assert_eq!(overrides.threshold, Some(0.5));
}

#[test]
fn test_rerank_overrides_top_k() {
    let overrides = RerankOverrides::top_k(5);
    assert_eq!(overrides.top_k, Some(5));
}

#[test]
fn test_rerank_result_display() {
    let result = RerankResult::new(0, 0.95, "This is a test document");
    let display = format!("{}", result);
    assert!(display.contains("[0]"));
    assert!(display.contains("0.95"));
    assert!(display.contains("This is a test document"));
}

#[test]
fn test_rerank_result_display_truncation() {
    let long_doc = "A".repeat(100);
    let result = RerankResult::new(1, 0.5, long_doc);
    let display = format!("{}", result);
    assert!(display.contains("..."));
    assert!(display.len() < 100);
}

#[test]
fn test_validation_unknown_model() {
    let result = is_reranking_model("nonexistent-model-xyz");
    assert!(result.is_err());

    if let Err(RerankerError::UnknownModel(name)) = result {
        assert_eq!(name, "nonexistent-model-xyz");
    } else {
        panic!("Expected UnknownModel error");
    }
}

#[test]
fn test_preset_minilm() {
    let preset = &presets::RERANKER_MINILM_V1;
    assert_eq!(preset.name, "RERANKER_MINILM_V1");
    assert!(preset.max_seq_length > 0);
    assert!(preset.memory_mb > 0);
}

#[test]
fn test_preset_msmarco() {
    let preset = &presets::RERANKER_MSMARCO_V1;
    assert_eq!(preset.name, "RERANKER_MSMARCO_V1");
    assert!(preset.max_seq_length > 0);
}

#[test]
fn test_find_preset() {
    let preset = presets::find_preset("RERANKER_MINILM_V1");
    assert!(preset.is_some());
    assert_eq!(preset.unwrap().name, "RERANKER_MINILM_V1");

    // Case insensitive
    let preset_lower = presets::find_preset("reranker_minilm_v1");
    assert!(preset_lower.is_some());

    // Not found
    let not_found = presets::find_preset("nonexistent");
    assert!(not_found.is_none());
}

#[test]
fn test_reranker_tier_resolve() {
    let fast = RerankerTier::Fast.resolve();
    assert_eq!(fast.name, "RERANKER_MINILM_V1");

    let balanced = RerankerTier::Balanced.resolve();
    assert_eq!(balanced.name, "RERANKER_MSMARCO_V1");
}

#[test]
fn test_reranker_tier_default() {
    let default = RerankerTier::default();
    assert_eq!(default, RerankerTier::Balanced);
}

mod integration {
    use super::*;

    #[tokio::test]
    async fn test_reranker_score_single_pair() {
        let reranker = Reranker::new("minilm-l6-v2-cross-encoder")
            .await
            .expect("Failed to load reranker");

        let score = reranker
            .score("What is Rust?", "Rust is a systems programming language.")
            .await
            .expect("Failed to score");

        // Score should be a reasonable value
        assert!(score.is_finite());
    }

    #[tokio::test]
    async fn test_reranker_rerank_documents() {
        let reranker = Reranker::new("minilm-l6-v2-cross-encoder")
            .await
            .expect("Failed to load reranker");

        let query = "What is machine learning?";
        let documents = vec![
            "Machine learning is a subset of artificial intelligence.",
            "The weather today is sunny.",
            "Deep learning uses neural networks.",
            "I like pizza.",
        ];

        let results = reranker
            .rerank(query, &documents)
            .await
            .expect("Failed to rerank");

        assert_eq!(results.len(), documents.len());

        // Results should be sorted by score descending
        for i in 1..results.len() {
            assert!(results[i - 1].score >= results[i].score);
        }

        // ML-related docs should rank higher than unrelated ones
        let ml_indices: Vec<usize> = results.iter().take(2).map(|r| r.index).collect();

        // Indices 0 and 2 are ML-related
        assert!(ml_indices.contains(&0) || ml_indices.contains(&2));
    }

    #[tokio::test]
    async fn test_reranker_top_k() {
        let reranker = Reranker::new("minilm-l6-v2-cross-encoder")
            .await
            .expect("Failed to load reranker");

        let documents = vec!["doc1", "doc2", "doc3", "doc4", "doc5"];

        let results = reranker
            .rerank_top_k("query", &documents, 3)
            .await
            .expect("Failed to rerank");

        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn test_reranker_with_threshold() {
        let reranker = Reranker::new("minilm-l6-v2-cross-encoder")
            .await
            .expect("Failed to load reranker");

        let documents = vec![
            "Very relevant document about the query topic.",
            "Completely unrelated content about cooking recipes.",
        ];

        let results = reranker
            .rerank_with_threshold("programming languages", &documents, 0.0)
            .await
            .expect("Failed to rerank");

        // Some results might be filtered out
        assert!(results.len() <= documents.len());
    }

    #[tokio::test]
    async fn test_reranker_empty_documents() {
        let reranker = Reranker::new("minilm-l6-v2-cross-encoder")
            .await
            .expect("Failed to load reranker");

        let results = reranker
            .rerank("query", &[])
            .await
            .expect("Failed to rerank empty");

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_reranker_score_pairs() {
        let reranker = Reranker::new("minilm-l6-v2-cross-encoder")
            .await
            .expect("Failed to load reranker");

        let pairs = vec![
            ("What is Rust?", "Rust is a programming language."),
            ("What is Python?", "Python is a snake."),
        ];

        let scores = reranker
            .score_pairs(&pairs)
            .await
            .expect("Failed to score pairs");

        assert_eq!(scores.len(), 2);
        assert!(scores.iter().all(|s| s.is_finite()));

        // First pair should have higher score (relevant)
        assert!(scores[0] > scores[1]);
    }

    #[tokio::test]
    async fn test_reranker_builder_gpu() {
        let result = Reranker::builder("minilm-l6-v2-cross-encoder")
            .gpu()
            .build()
            .await;

        // May fail if no GPU, that's ok
        if let Ok(reranker) = result {
            assert_eq!(reranker.device(), kjarni_transformers::traits::Device::Wgpu);
        }
    }

    #[tokio::test]
    async fn test_reranker_accessors() {
        let reranker = Reranker::new("minilm-l6-v2-cross-encoder")
            .await
            .expect("Failed to load reranker");

        assert!(!reranker.model_id().is_empty());
        assert!(reranker.model_type().is_some());
        assert!(!reranker.model_name().is_empty());
        assert!(reranker.max_seq_length() > 0);
        assert!(reranker.hidden_size() > 0);
    }

    #[tokio::test]
    async fn test_convenience_rerank() {
        let documents = vec!["relevant doc", "irrelevant doc"];

        let results = rerank("minilm-l6-v2-cross-encoder", "test query", &documents)
            .await
            .expect("Failed to rerank");

        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_convenience_score() {
        let score_result = score("minilm-l6-v2-cross-encoder", "query", "document")
            .await
            .expect("Failed to score");

        assert!(score_result.is_finite());
    }
}
