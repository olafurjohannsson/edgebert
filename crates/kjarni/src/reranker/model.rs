//! Core Reranker implementation.

use std::path::Path;
use std::sync::Arc;

use kjarni_transformers::{WgpuContext, models::ModelType, traits::Device};

use crate::CrossEncoder;
use crate::common::{default_cache_dir, ensure_model_downloaded};

use super::builder::RerankerBuilder;
use super::types::{RerankOverrides, RerankResult, RerankerError, RerankerResult};
use super::validation::validate_for_reranking;

/// Turns raw cross-encoder logits into the scores a caller sees, then applies the
/// threshold in that same space.
///
/// Separated from `rerank_with_config` so the transform can be tested without a
/// model: it is the whole of the behaviour change, and the part most likely to
/// break someone quietly.
///
/// A cross-encoder emits a logit, which for ms-marco runs from about -11 to +11
/// and means nothing to a caller. Squashing it to a probability is what
/// `return_raw_scores: false` has always promised and never done, and it is safe
/// to change: sigmoid is monotonic, so every ranking is unchanged and only the
/// printed numbers move.
///
/// The threshold is compared *after* the transform, so it is always read on
/// whichever scale the scores are on. Comparing a 0..1 threshold against logits
/// would silently pass almost everything.
fn score_and_filter(
    ranked: Vec<(usize, f32)>,
    overrides: &RerankOverrides,
) -> impl Iterator<Item = (usize, f32)> + '_ {
    let raw = overrides.return_raw_scores;
    ranked
        .into_iter()
        .map(move |(index, logit)| (index, if raw { logit } else { sigmoid(logit) }))
        .filter(move |(_, score)| overrides.threshold.map(|t| *score >= t).unwrap_or(true))
}

/// Logistic squash, saturating rather than overflowing at the extremes.
///
/// `exp(-x)` for x around -100 is inf, and inf/inf is NaN, which would sort
/// unpredictably. Cross-encoder logits do not reach that far, but a sort key is
/// the wrong place to rely on that.
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// High-level text reranker using cross-encoder models.
///
/// Cross-encoders process query-document pairs together, producing
/// more accurate relevance scores than bi-encoder similarity.
pub struct Reranker {
    /// The underlying cross-encoder.
    inner: CrossEncoder,

    /// Model identifier.
    model_id: String,

    /// Model type (None if loaded from path).
    model_type: Option<ModelType>,

    /// Default overrides.
    default_overrides: RerankOverrides,

    /// Device.
    device: Device,

    /// GPU context if using GPU.
    #[allow(dead_code)]
    context: Option<Arc<WgpuContext>>,
}

impl Reranker {
    /// Create a Reranker with default settings.
    ///
    /// Uses CPU, downloads model if needed.
    pub async fn new(model: &str) -> RerankerResult<Self> {
        Self::builder(model).build().await
    }

    /// Load a reranker from a local path.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let reranker = Reranker::from_path("/models/my-cross-encoder")
    ///     .build()
    ///     .await?;
    /// ```
    pub fn from_path(path: impl Into<std::path::PathBuf>) -> RerankerBuilder {
        RerankerBuilder::new("custom").model_path(path)
    }

    /// Internal: construct from builder.
    pub(crate) async fn from_builder(builder: RerankerBuilder) -> RerankerResult<Self> {
        let device = builder.device.to_device();

        // Create GPU context if needed
        let context = if device == Device::Wgpu {
            if let Some(ctx) = builder.context {
                Some(ctx)
            } else {
                Some(
                    WgpuContext::new()
                        .await
                        .map_err(|_| RerankerError::GpuUnavailable)?,
                )
            }
        } else {
            None
        };

        let load_config = builder.load_config.map(|c| c.into_inner());

        // Determine loading strategy
        let (inner, model_type, model_id) = if let Some(model_path) = &builder.model_path {
            // Load from local path
            Self::load_from_path(
                model_path,
                device,
                context.clone(),
                load_config,
                builder.quiet,
            )
            .await?
        } else {
            // Load from registry
            Self::load_from_registry(
                &builder.model,
                builder.cache_dir.as_deref(),
                device,
                context.clone(),
                load_config,
                builder.download_policy,
                builder.quiet,
            )
            .await?
        };

        Ok(Self {
            inner,
            model_id,
            model_type,
            default_overrides: builder.overrides,
            device,
            context,
        })
    }

    /// Load reranker from registry.
    async fn load_from_registry(
        model: &str,
        cache_dir: Option<&Path>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<kjarni_transformers::models::base::ModelLoadConfig>,
        download_policy: crate::common::DownloadPolicy,
        quiet: bool,
    ) -> RerankerResult<(CrossEncoder, Option<ModelType>, String)> {
        let model_type = ModelType::resolve(model).map_err(RerankerError::UnknownModel)?;

        // Validate for reranking
        validate_for_reranking(model_type)?;

        // Ensure downloaded
        let cache_dir_path = cache_dir
            .map(|p| p.to_path_buf())
            .unwrap_or_else(default_cache_dir);

        ensure_model_downloaded(model_type, Some(&cache_dir_path), download_policy, quiet)
            .await
            .map_err(|e| match e {
                crate::common::KjarniError::ModelNotDownloaded(m) => {
                    RerankerError::ModelNotDownloaded(m)
                }
                crate::common::KjarniError::DownloadFailed { model, source } => {
                    RerankerError::DownloadFailed { model, source }
                }
                other => RerankerError::RerankingFailed(other.into()),
            })?;

        let inner = CrossEncoder::from_registry(
            model_type,
            Some(cache_dir_path),
            device,
            context,
            load_config,
        )
        .await
        .map_err(|e| RerankerError::LoadFailed {
            model: model.to_string(),
            source: e,
        })?;

        Ok((inner, Some(model_type), model.to_string()))
    }

    /// Load reranker from local path.
    async fn load_from_path(
        path: &Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<kjarni_transformers::models::base::ModelLoadConfig>,
        quiet: bool,
    ) -> RerankerResult<(CrossEncoder, Option<ModelType>, String)> {
        // Validate path exists
        if !path.exists() {
            return Err(RerankerError::ModelPathNotFound(path.display().to_string()));
        }

        // Check for required files
        let config_path = path.join("config.json");
        if !config_path.exists() {
            return Err(RerankerError::InvalidConfig(format!(
                "config.json not found in {}",
                path.display()
            )));
        }

        let tokenizer_path = path.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(RerankerError::InvalidConfig(format!(
                "tokenizer.json not found in {}",
                path.display()
            )));
        }

        if !quiet {
            eprintln!("Loading reranker from {}...", path.display());
        }

        let inner = CrossEncoder::from_pretrained(path, device, context, load_config, None)
            .map_err(|e| RerankerError::LoadFailed {
                model: path.display().to_string(),
                source: e,
            })?;

        let model_id = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "custom".to_string());

        Ok((inner, None, model_id))
    }

    /// Score a single query-document pair.
    /// Relevance of one document to one query, on the same scale as
    /// [`RerankResult::score`].
    ///
    /// 0..1 unless the reranker was built with `return_raw_scores`. This has to
    /// agree with `rerank`: the two used to disagree, so scoring a pair directly
    /// gave a logit while ranking the same pair gave a probability, and nothing
    /// said which you had.
    pub async fn score(&self, query: &str, document: &str) -> RerankerResult<f32> {
        let raw = self
            .inner
            .predict_pair(query, document)
            .await
            .map_err(RerankerError::RerankingFailed)?;
        Ok(self.present(raw))
    }

    /// Applies the same transform `rerank` applies, so every score this type
    /// hands out is on one scale.
    fn present(&self, raw: f32) -> f32 {
        if self.default_overrides.return_raw_scores {
            raw
        } else {
            sigmoid(raw)
        }
    }

    /// Score multiple query-document pairs.
    /// Score multiple query-document pairs, on the same scale as [`Self::score`].
    pub async fn score_pairs(&self, pairs: &[(&str, &str)]) -> RerankerResult<Vec<f32>> {
        let raw = self
            .inner
            .predict_pairs(pairs)
            .await
            .map_err(RerankerError::RerankingFailed)?;
        Ok(raw.into_iter().map(|r| self.present(r)).collect())
    }

    /// Rerank documents by relevance to a query.
    pub async fn rerank(
        &self,
        query: &str,
        documents: &[&str],
    ) -> RerankerResult<Vec<RerankResult>> {
        self.rerank_with_config(query, documents, &RerankOverrides::default())
            .await
    }

    /// Rerank with custom overrides.
    pub async fn rerank_with_config(
        &self,
        query: &str,
        documents: &[&str],
        overrides: &RerankOverrides,
    ) -> RerankerResult<Vec<RerankResult>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        let merged = self.merge_overrides(overrides);

        // Get raw scores
        let ranked = self
            .inner
            .rerank(query, documents)
            .await
            .map_err(RerankerError::RerankingFailed)?;

        let mut results: Vec<RerankResult> = score_and_filter(ranked, &merged)
            .map(|(index, score)| RerankResult {
                index,
                score,
                document: documents[index].to_string(),
            })
            .collect();

        // Apply top_k
        if let Some(k) = merged.top_k {
            results.truncate(k);
        }

        Ok(results)
    }

    /// Rerank and return only top-k results.
    pub async fn rerank_top_k(
        &self,
        query: &str,
        documents: &[&str],
        k: usize,
    ) -> RerankerResult<Vec<RerankResult>> {
        self.rerank_with_config(
            query,
            documents,
            &RerankOverrides {
                top_k: Some(k),
                ..Default::default()
            },
        )
        .await
    }

    /// Rerank, keeping only results at or above `threshold`.
    ///
    /// `threshold` is on the same scale as `score`: 0..1 by default, or raw
    /// logits if `return_raw_scores` is set.
    pub async fn rerank_with_threshold(
        &self,
        query: &str,
        documents: &[&str],
        threshold: f32,
    ) -> RerankerResult<Vec<RerankResult>> {
        self.rerank_with_config(
            query,
            documents,
            &RerankOverrides {
                threshold: Some(threshold),
                ..Default::default()
            },
        )
        .await
    }

    /// Rerank
    pub async fn rerank_owned(
        &self,
        query: &str,
        documents: &[String],
    ) -> RerankerResult<Vec<RerankResult>> {
        let doc_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        self.rerank(query, &doc_refs).await
    }

    /// Get the model identifier.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Get the model type (None if loaded from path).
    pub fn model_type(&self) -> Option<ModelType> {
        self.model_type
    }

    /// Get the model's CLI name (if from registry).
    pub fn model_name(&self) -> &str {
        self.model_type
            .map(|t| t.cli_name())
            .unwrap_or(&self.model_id)
    }

    /// Get the device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Get maximum sequence length.
    pub fn max_seq_length(&self) -> usize {
        self.inner.max_seq_length()
    }

    /// Get the hidden size.
    pub fn hidden_size(&self) -> usize {
        self.inner.hidden_size()
    }
    fn merge_overrides(&self, runtime: &RerankOverrides) -> RerankOverrides {
        RerankOverrides {
            top_k: runtime.top_k.or(self.default_overrides.top_k),
            threshold: runtime.threshold.or(self.default_overrides.threshold),
            return_raw_scores: runtime.return_raw_scores
                || self.default_overrides.return_raw_scores,
            batch_size: runtime.batch_size.or(self.default_overrides.batch_size),
        }
    }
}

#[cfg(test)]
mod scoring_tests {
    use super::*;

    fn overrides(raw: bool, threshold: Option<f32>) -> RerankOverrides {
        RerankOverrides {
            top_k: None,
            threshold,
            return_raw_scores: raw,
            batch_size: None,
        }
    }

    /// Values taken from a real ms-marco run: these are the logits the quickstart
    /// printed before the squash, and what a caller sees now.
    #[test]
    fn sigmoid_matches_the_reference() {
        // Computed as `1/(1+exp(-x))` in float32, not written from memory.
        for (logit, want) in [
            (0.0f32, 0.500_000_00f32),
            (1.3282, 0.790_542_72),
            (-10.5874, 0.000_025_231),
            (-11.0939, 0.000_015_205),
            (2.0, 0.880_797_03),
            (-2.0, 0.119_202_93),
        ] {
            let got = sigmoid(logit);
            assert!(
                (got - want).abs() < 1e-5,
                "sigmoid({logit}) = {got}, expected {want}"
            );
        }
    }

    /// The reason `sigmoid` is written with two branches rather than the obvious
    /// one-liner. `exp(-x)` for a large negative x is `inf`, and `inf / inf` is
    /// `NaN`, which sorts unpredictably and would scramble a ranking rather than
    /// merely mis-scale it. Anyone "simplifying" this should fail here.
    #[test]
    fn sigmoid_saturates_instead_of_producing_nan() {
        for x in [
            -1000.0f32,
            -100.0,
            -50.0,
            50.0,
            100.0,
            1000.0,
            f32::MIN,
            f32::MAX,
        ] {
            let got = sigmoid(x);
            assert!(got.is_finite(), "sigmoid({x}) was {got}");
            assert!((0.0..=1.0).contains(&got), "sigmoid({x}) = {got} left 0..1");
        }
        assert_eq!(sigmoid(-1000.0), 0.0);
        assert_eq!(sigmoid(1000.0), 1.0);
    }

    /// The claim that made this change safe to ship: squashing cannot reorder
    /// results, so every existing ranking is preserved and only the numbers move.
    #[test]
    fn squashing_never_changes_the_ranking() {
        let logits: Vec<f32> = vec![
            1.3282, -10.5874, -11.0939, 0.0, 7.5, -3.25, 2.0, -0.001, 11.0, -11.0,
        ];

        let mut by_logit: Vec<usize> = (0..logits.len()).collect();
        by_logit.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));

        let squashed: Vec<f32> = logits.iter().map(|l| sigmoid(*l)).collect();
        let mut by_score: Vec<usize> = (0..squashed.len()).collect();
        by_score.sort_by(|a, b| squashed[*b].total_cmp(&squashed[*a]));

        assert_eq!(
            by_logit, by_score,
            "sigmoid reordered results; the transform must be monotonic"
        );
    }

    #[test]
    fn scores_are_probabilities_by_default() {
        let ranked = vec![(0, 1.3282f32), (1, -10.5874), (2, -11.0939)];
        let out: Vec<(usize, f32)> = score_and_filter(ranked, &overrides(false, None)).collect();

        assert_eq!(out.len(), 3, "no threshold means nothing is dropped");
        assert!((out[0].1 - 0.790_542_72).abs() < 1e-6);
        for (_, s) in &out {
            assert!((0.0..=1.0).contains(s), "score {s} is not a probability");
        }
    }

    #[test]
    fn raw_scores_are_passed_through_untouched() {
        let ranked = vec![(0, 1.3282f32), (1, -10.5874)];
        let out: Vec<(usize, f32)> = score_and_filter(ranked, &overrides(true, None)).collect();
        assert_eq!(out, vec![(0, 1.3282), (1, -10.5874)]);
    }

    /// The half most likely to break someone quietly. A threshold has to be read
    /// on whichever scale the scores are on: 0.5 against probabilities keeps one
    /// document here, while 0.5 against logits would keep the same one for an
    /// entirely different reason, and 0.5 applied to the wrong scale would let
    /// everything through.
    #[test]
    fn threshold_is_read_on_the_same_scale_as_the_scores() {
        let ranked = vec![(0, 1.3282f32), (1, -10.5874), (2, -11.0939)];

        // Probabilities: only the 0.79 document clears 0.5.
        let kept: Vec<usize> = score_and_filter(ranked.clone(), &overrides(false, Some(0.5)))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            kept,
            vec![0],
            "0.5 as a probability should keep only the match"
        );

        // Raw logits: 0.5 is a logit here, and only the 1.33 document clears it.
        let kept: Vec<usize> = score_and_filter(ranked.clone(), &overrides(true, Some(0.5)))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(kept, vec![0], "0.5 as a logit should keep only the match");

        // The failure this guards against: a probability threshold compared
        // against logits. -1.0 is below every logit here, so everything would
        // survive, which is what a mis-scaled threshold looks like in production.
        let kept: Vec<usize> = score_and_filter(ranked, &overrides(true, Some(-1.0)))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(kept, vec![0], "only one logit is above -1.0");
    }

    #[test]
    fn a_threshold_above_everything_returns_nothing() {
        let ranked = vec![(0, 1.3282f32), (1, -10.5874)];
        let out: Vec<(usize, f32)> =
            score_and_filter(ranked, &overrides(false, Some(0.999))).collect();
        assert!(
            out.is_empty(),
            "nothing should clear a 0.999 probability here"
        );
    }
}
