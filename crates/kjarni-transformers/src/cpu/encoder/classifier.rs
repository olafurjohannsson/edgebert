//! Concrete classification heads for encoder models.

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use crate::gpu::{GpuFrameContext, GpuTensor};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use crate::gpu_ops::primitives::add::GpuAdd;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use crate::gpu_ops::primitives::layout::slice::GpuSlice;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use crate::gpu_ops::primitives::{linear::GpuLinearLayer, tanh::GpuTanh};
use crate::linear_layer::LinearLayer;
use crate::models::base::ModelLoadConfig;
use crate::weights::ModelWeights;
use crate::{PoolingStrategy, last_token_pool};
use anyhow::{Result, anyhow};
use ndarray::{Array2, Array3, s};

/// Layout for sequence classification head.
#[derive(Debug, Clone)]
pub struct ClassificationHeadLayout {
    /// Pre-classifier projection (hidden_size -> hidden_size)
    pub pre_classifier_weight: Option<String>,
    pub pre_classifier_bias: Option<String>,

    /// Final classifier (hidden_size -> num_labels)
    pub classifier_weight: String,
    pub classifier_bias: Option<String>,
}

/// Activation function for the classification head.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HeadActivation {
    #[default]
    Tanh,
    Relu,
    Gelu,
    None,
}

/// CPU classification head supporting multiple architectures.
pub struct CpuSequenceClassificationHead {
    /// Pooler layer (BERT-style, uses Tanh)
    pooler: Option<LinearLayer>,
    /// Pre-classifier layer (DistilBERT-style)
    pre_classifier: Option<LinearLayer>,
    /// Final classifier
    classifier: LinearLayer,
    /// Activation after pooler/pre_classifier
    activation: HeadActivation,
    pooling_strategy: PoolingStrategy,
    /// Label names (optional)
    labels: Option<Vec<String>>,
}

impl CpuSequenceClassificationHead {
    /// Create a new classification head.
    pub fn new(pooler: Option<LinearLayer>, classifier: LinearLayer) -> Result<Self> {
        Self::with_config(
            pooler,
            None,
            classifier,
            HeadActivation::Tanh,
            PoolingStrategy::Cls,
            None,
        )
    }

    /// Create with full configuration.
    pub fn with_config(
        pooler: Option<LinearLayer>,
        pre_classifier: Option<LinearLayer>,
        classifier: LinearLayer,
        activation: HeadActivation,
        pooling_strategy: PoolingStrategy,
        labels: Option<Vec<String>>,
    ) -> Result<Self> {
        // Validate dimensions
        if let Some(ref p) = pooler
            && p.out_features() != classifier.in_features()
        {
            return Err(anyhow!(
                "Pooler output ({}) != Classifier input ({})",
                p.out_features(),
                classifier.in_features()
            ));
        }
        if let Some(ref pc) = pre_classifier
            && pc.out_features() != classifier.in_features()
        {
            return Err(anyhow!(
                "Pre-classifier output ({}) != Classifier input ({})",
                pc.out_features(),
                classifier.in_features()
            ));
        }

        Ok(Self {
            pooler,
            pre_classifier,
            classifier,
            activation,
            pooling_strategy,
            labels,
        })
    }

    pub fn from_weights(
        weights: &ModelWeights,
        load_config: &ModelLoadConfig,
        labels: Option<Vec<String>>,
    ) -> Result<Self> {
        let pooling_strategy = if weights.config_json().contains("\"model_type\": \"bart\"") {
            PoolingStrategy::LastToken
        } else {
            PoolingStrategy::Cls
        };
        if weights.contains("classification_head.dense.weight") {
            let pre_classifier = LinearLayer::builder(weights, "classification_head.dense.weight")
                .with_optional_bias(Some("classification_head.dense.bias"))
                .with_target_dtype(load_config.target_dtype)
                .build()?;
            let classifier = LinearLayer::builder(weights, "classification_head.out_proj.weight")
                .with_optional_bias(Some("classification_head.out_proj.bias"))
                .with_target_dtype(load_config.target_dtype)
                .build()?;
            return Self::with_config(
                None,
                Some(pre_classifier),
                classifier,
                HeadActivation::Tanh,
                pooling_strategy,
                labels,
            );
        }

        if weights.contains("classifier.dense.weight") {
            let pre_classifier = LinearLayer::builder(weights, "classifier.dense.weight")
                .with_optional_bias(Some("classifier.dense.bias"))
                .with_target_dtype(load_config.target_dtype)
                .build()?;
            let classifier = LinearLayer::builder(weights, "classifier.out_proj.weight")
                .with_optional_bias(Some("classifier.out_proj.bias"))
                .with_target_dtype(load_config.target_dtype)
                .build()?;
            return Self::with_config(
                None,
                Some(pre_classifier),
                classifier,
                HeadActivation::Tanh,
                pooling_strategy,
                labels,
            );
        }

        if weights.contains("pre_classifier.weight") {
            let pre_classifier = LinearLayer::builder(weights, "pre_classifier.weight")
                .with_optional_bias(Some("pre_classifier.bias"))
                .with_target_dtype(load_config.target_dtype)
                .build()?;
            let classifier = LinearLayer::builder(weights, "classifier.weight")
                .with_optional_bias(Some("classifier.bias"))
                .with_target_dtype(load_config.target_dtype)
                .build()?;
            return Self::with_config(
                None,
                Some(pre_classifier),
                classifier,
                HeadActivation::Relu,
                pooling_strategy,
                labels,
            );
        }

        if weights.contains("bert.pooler.dense.weight") {
            let pooler = LinearLayer::builder(weights, "bert.pooler.dense.weight")
                .with_optional_bias(Some("bert.pooler.dense.bias"))
                .with_target_dtype(load_config.target_dtype)
                .build()?;
            let classifier = LinearLayer::builder(weights, "classifier.weight")
                .with_optional_bias(Some("classifier.bias"))
                .with_target_dtype(load_config.target_dtype)
                .build()?;
            return Self::with_config(
                Some(pooler),
                None,
                classifier,
                HeadActivation::Tanh,
                pooling_strategy,
                labels,
            );
        }

        if weights.contains("classifier.weight") {
            let classifier = LinearLayer::builder(weights, "classifier.weight")
                .with_optional_bias(Some("classifier.bias"))
                .with_target_dtype(load_config.target_dtype)
                .build()?;
            return Self::with_config(
                None,
                None,
                classifier,
                HeadActivation::None,
                pooling_strategy,
                labels,
            );
        }

        Err(anyhow!(
            "Could not auto-detect a valid classification head structure from the model weights. Checked for 'classification_head.dense', 'classifier.dense', 'pre_classifier', 'bert.pooler', and 'classifier' weights."
        ))
    }

    /// Forward pass: hidden_states -> logits.
    pub fn forward(
        &self,
        encoder_hidden_states: &Array3<f32>,
        attention_mask: Option<&Array2<f32>>,
    ) -> Result<Array2<f32>> {
        let (batch, seq_len, _hidden_size) = encoder_hidden_states.dim();
        if batch == 0 || seq_len == 0 {
            return Ok(Array2::<f32>::zeros((batch, self.num_classes())));
        }
        let sequence_embedding = match self.pooling_strategy {
            PoolingStrategy::Cls => encoder_hidden_states.slice(s![.., 0, ..]).to_owned(),
            PoolingStrategy::LastToken => {
                if let Some(mask) = attention_mask {
                    last_token_pool(encoder_hidden_states, mask)?
                } else if seq_len == 1 {
                    encoder_hidden_states.slice(s![.., 0, ..]).to_owned()
                } else {
                    return Err(anyhow!(
                        "LastToken pooling requires an attention mask for seq_len > 1"
                    ));
                }
            }
            _ => {
                return Err(anyhow!(
                    "Unsupported pooling strategy for classification head: {:?}",
                    self.pooling_strategy
                ));
            }
        };

        let features = if let Some(ref pooler) = self.pooler {
            let mut pooled = pooler.matmul(&sequence_embedding.view());
            self.apply_activation_inplace(&mut pooled);
            pooled
        } else if let Some(ref pre_classifier) = self.pre_classifier {
            let mut pre_out = pre_classifier.matmul(&sequence_embedding.view());
            self.apply_activation_inplace(&mut pre_out);
            pre_out
        } else {
            sequence_embedding
        };

        let logits = self.classifier.matmul(&features.view());

        Ok(logits)
    }

    fn apply_activation_inplace(&self, x: &mut Array2<f32>) {
        match self.activation {
            HeadActivation::Tanh => x.mapv_inplace(f32::tanh),
            HeadActivation::Relu => x.mapv_inplace(|v| v.max(0.0)),
            HeadActivation::Gelu => x.mapv_inplace(|v| {
                let sqrt_2_over_pi = 0.7978845608;
                0.5 * v * (1.0 + (sqrt_2_over_pi * (v + 0.044715 * v.powi(3))).tanh())
            }),
            HeadActivation::None => {}
        }
    }

    pub fn num_classes(&self) -> usize {
        self.classifier.out_features()
    }

    pub fn labels(&self) -> Option<&[String]> {
        self.labels.as_deref()
    }
}

/// A GPU-accelerated head for sequence classification tasks.
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
pub struct GpuSequenceClassificationHead {
    // Kernels for operations
    slicer: GpuSlice,
    linear: GpuLinearLayer,
    add_bias: GpuAdd,
    tanh: GpuTanh,
    pooler_weight: Option<GpuTensor>,
    pooler_bias: Option<GpuTensor>,
    classifier_weight: GpuTensor,
    classifier_bias: GpuTensor,
}

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
impl GpuSequenceClassificationHead {
    pub fn new(
        context: &std::sync::Arc<crate::WgpuContext>,
        pooler_weight: Option<GpuTensor>,
        pooler_bias: Option<GpuTensor>,
        classifier_weight: GpuTensor,
        classifier_bias: GpuTensor,
    ) -> Result<Self> {
        if pooler_weight.is_some() != pooler_bias.is_some() {
            return Err(anyhow!(
                "GPU Pooler weight and bias must both be Some or None."
            ));
        }
        Ok(Self {
            slicer: GpuSlice::new(context),
            linear: GpuLinearLayer::new(context),
            add_bias: GpuAdd::new(context),
            tanh: GpuTanh::new(context),
            pooler_weight,
            pooler_bias,
            classifier_weight,
            classifier_bias,
        })
    }

    pub fn forward(
        &self,
        frame: &mut GpuFrameContext,
        encoder_hidden_states: &GpuTensor,
    ) -> Result<GpuTensor> {
        let (batch, seq_len, hidden) = encoder_hidden_states.dims3();
        let (encoder_cmd, pool) = frame.resources();

        // Slice the CLS token
        let hidden_states_4d = encoder_hidden_states.view(vec![batch, 1, seq_len, hidden]);
        let cls_embedding_4d = pool.get(vec![batch, 1, 1, hidden]);
        self.slicer.encode(
            encoder_cmd,
            &hidden_states_4d,
            &cls_embedding_4d,
            &[0, 0, 0, 0],
        );
        let cls_embedding_2d = cls_embedding_4d.view(vec![batch, hidden]);

        // pooler layer, if it exists.
        let pooled_output =
            if let (Some(weight), Some(bias)) = (&self.pooler_weight, &self.pooler_bias) {
                let pooled_linear_out = pool.get(vec![batch, weight.shape()[0]]);
                self.linear
                    .encode(encoder_cmd, &cls_embedding_2d, weight, &pooled_linear_out);

                let pooled_with_bias = pool.get(pooled_linear_out.shape().to_vec());
                self.add_bias.encode_broadcast_row(
                    encoder_cmd,
                    &pooled_linear_out,
                    bias,
                    &pooled_with_bias,
                );

                self.tanh.encode_inplace(encoder_cmd, &pooled_with_bias);

                pooled_with_bias
            } else {
                cls_embedding_2d
            };

        // final classifier linear layer.
        let logits_linear_out = pool.get(vec![batch, self.classifier_weight.shape()[0]]);
        self.linear.encode(
            encoder_cmd,
            &pooled_output,
            &self.classifier_weight,
            &logits_linear_out,
        );

        let final_logits = pool.get(logits_linear_out.shape().to_vec());
        self.add_bias.encode_broadcast_row(
            encoder_cmd,
            &logits_linear_out,
            &self.classifier_bias,
            &final_logits,
        );

        Ok(final_logits)
    }

    pub fn num_classes(&self) -> usize {
        self.classifier_bias.shape()[0]
    }
}

#[cfg(test)]
mod classification_head_tests {
    use super::*;
    use crate::linear_layer::LinearLayer;
    use ndarray::{Array1, Array2, Array3, array};

    use crate::weights::ModelWeights;
    use anyhow::Result;
    use safetensors::tensor::{Dtype, TensorView};
    use std::collections::HashMap;
    use tempfile::TempDir;

    fn create_test_weights(
        weights_map: HashMap<String, Vec<f32>>,
        shapes: HashMap<String, Vec<usize>>,
    ) -> Result<(ModelWeights, TempDir)> {
        let dir = tempfile::tempdir()?;
        let stored_data: Vec<(String, Vec<usize>, Vec<u8>)> = weights_map
            .into_iter()
            .map(|(k, v)| {
                let shape = shapes.get(&k).unwrap().clone();
                let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                (k, shape, bytes)
            })
            .collect();

        let mut tensors = HashMap::new();
        for (k, shape, bytes) in &stored_data {
            tensors.insert(
                k.clone(),
                TensorView::new(Dtype::F32, shape.clone(), bytes)?,
            );
        }

        let file_path = dir.path().join("model.safetensors");
        safetensors::serialize_to_file(&tensors, &None, &file_path)?;

        // Dummy config.json to satisfy ModelWeights loader if needed
        std::fs::write(dir.path().join("config.json"), "{}")?;

        let weights = ModelWeights::new(dir.path())?;
        Ok((weights, dir))
    }

    // BART logic: LastToken Pooling (input is already EOS) -> Dense -> Tanh -> Classifier
    #[test]
    fn test_bart_head_golden() -> Result<()> {
        let mut w = HashMap::new();
        let mut s = HashMap::new();

        // Load Golden Data
        let bart_dense_weight_data = vec![
            0.257632, -0.220689, -0.096931, 0.234684, -0.470718, 0.299859, -0.102863, 0.254372,
            0.069508, -0.061222, 0.138680, 0.024666, 0.182614, -0.194851, -0.036454, -0.045014,
            0.072472, -0.001997, 0.437083, 0.155595, -0.186203, -0.301981, -0.083808, -0.215670,
            -0.160224, 0.023941, 0.298064, 0.271768, -0.488775, 0.309960, 0.139682, 0.474278,
        ];
        w.insert(
            "classification_head.dense.weight".into(),
            bart_dense_weight_data,
        );
        s.insert("classification_head.dense.weight".into(), vec![8, 4]);

        let bart_dense_bias_data = vec![
            0.010000, 0.010000, 0.010000, 0.010000, 0.010000, 0.010000, 0.010000, 0.010000,
        ];
        w.insert(
            "classification_head.dense.bias".into(),
            bart_dense_bias_data,
        );
        s.insert("classification_head.dense.bias".into(), vec![8]);

        let bart_out_proj_weight_data = vec![
            0.233366, -0.322136, -0.336162, -0.170530, 0.310459, -0.058891, 0.151307, -0.164300,
            0.346913, -0.149588, 0.265138, 0.004187, -0.186258, 0.181732, -0.187673, 0.103982,
        ];
        w.insert(
            "classification_head.out_proj.weight".into(),
            bart_out_proj_weight_data,
        );
        s.insert("classification_head.out_proj.weight".into(), vec![2, 8]);

        let bart_out_proj_bias_data = vec![0.010000, 0.010000];
        w.insert(
            "classification_head.out_proj.bias".into(),
            bart_out_proj_bias_data,
        );
        s.insert("classification_head.out_proj.bias".into(), vec![2]);

        let (weights, _tmp) = create_test_weights(w, s)?;

        //  Configure Head
        let pre_classifier = LinearLayer::builder(&weights, "classification_head.dense.weight")
            .with_optional_bias(Some("classification_head.dense.bias"))
            .build()?;

        let classifier = LinearLayer::builder(&weights, "classification_head.out_proj.weight")
            .with_optional_bias(Some("classification_head.out_proj.bias"))
            .build()?;

        let head = CpuSequenceClassificationHead::with_config(
            None,
            Some(pre_classifier),
            classifier,
            HeadActivation::Tanh,
            PoolingStrategy::LastToken,
            None,
        )?;

        //  Inputs
        let input_data = vec![0.336690, 0.128809, 0.234462, 0.230333];
        let input = Array3::from_shape_vec((1, 1, 4), input_data)?;
        let output = head.forward(&input, None)?;

        //  Validation
        let golden_data = vec![0.103089, -0.002410];
        let golden = Array2::from_shape_vec((1, 2), golden_data)?;

        let diff = (&output - &golden).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!("BART Head Max Diff: {}", max_diff);
        assert!(max_diff < 1e-5);

        Ok(())
    }

    // RoBERTa logic: CLS Pooling (Index 0) -> Dense -> Tanh -> Classifier
    #[test]
    fn test_roberta_head_golden() -> Result<()> {
        let mut w = HashMap::new();
        let mut s = HashMap::new();

        // Load Weights
        let roberta_dense_weight_data = vec![
            0.114695, -0.118987, 0.137114, -0.025539, 0.213594, 0.119036, -0.057467, -0.404231,
            0.114157, -0.442672, 0.065714, 0.033229, -0.109949, 0.408848, 0.033368, 0.207339,
            0.211597, -0.294966, -0.192217, 0.480859, -0.489739, -0.033959, -0.039625, 0.354655,
            -0.047535, 0.131659, -0.024002, -0.279973, -0.283395, -0.242917, -0.454202, -0.324491,
        ];
        w.insert("classifier.dense.weight".into(), roberta_dense_weight_data);
        s.insert("classifier.dense.weight".into(), vec![8, 4]);

        let roberta_dense_bias_data = vec![
            0.010000, 0.010000, 0.010000, 0.010000, 0.010000, 0.010000, 0.010000, 0.010000,
        ];
        w.insert("classifier.dense.bias".into(), roberta_dense_bias_data);
        s.insert("classifier.dense.bias".into(), vec![8]);

        let roberta_out_proj_weight_data = vec![
            0.083208, 0.232690, 0.017417, -0.162065, 0.155381, -0.135719, -0.078319, -0.193821,
            -0.111017, -0.327602, 0.150858, 0.137474, 0.070228, 0.173588, 0.149847, 0.015642,
        ];
        w.insert(
            "classifier.out_proj.weight".into(),
            roberta_out_proj_weight_data,
        );
        s.insert("classifier.out_proj.weight".into(), vec![2, 8]);

        let roberta_out_proj_bias_data = vec![0.010000, 0.010000];
        w.insert(
            "classifier.out_proj.bias".into(),
            roberta_out_proj_bias_data,
        );
        s.insert("classifier.out_proj.bias".into(), vec![2]);

        let (weights, _tmp) = create_test_weights(w, s)?;

        // Configure Head
        let pre_classifier = LinearLayer::builder(&weights, "classifier.dense.weight")
            .with_optional_bias(Some("classifier.dense.bias"))
            .build()?;
        let classifier = LinearLayer::builder(&weights, "classifier.out_proj.weight")
            .with_optional_bias(Some("classifier.out_proj.bias"))
            .build()?;

        let head = CpuSequenceClassificationHead::with_config(
            None,
            Some(pre_classifier),
            classifier,
            HeadActivation::Tanh,
            PoolingStrategy::Cls,
            None,
        )?;

        // Input
        let input_data = vec![
            0.325502, -0.479145, 1.379008, 2.528557, 0.410742, -0.988007, -0.908073, 0.542274,
            0.110255, -2.259010, 0.606700, -0.138310,
        ];
        let input = Array3::from_shape_vec((1, 3, 4), input_data)?;

        // Run
        let output = head.forward(&input, None)?;

        // Validation
        let golden_data = vec![0.066929, 0.402067];
        let golden = Array2::from_shape_vec((1, 2), golden_data)?;

        let diff = (&output - &golden).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!("RoBERTa Head Max Diff: {}", max_diff);
        assert!(max_diff < 1e-5);
        Ok(())
    }

    // BERT logic: CLS Pooling (Index 0) -> Pooler (Dense+Tanh) -> Classifier
    #[test]
    fn test_bert_head_golden() -> Result<()> {
        let mut w = HashMap::new();
        let mut s = HashMap::new();

        // Load Weights
        let bert_pooler_weight_data = vec![
            -0.495736, -0.394431, -0.214158, -0.473045, -0.028386, -0.439884, 0.271866, 0.243699,
            0.094421, 0.387857, -0.048969, 0.299491, -0.350159, -0.098531, -0.445823, -0.040595,
        ];
        w.insert("bert.pooler.dense.weight".into(), bert_pooler_weight_data);
        s.insert("bert.pooler.dense.weight".into(), vec![4, 4]);

        let bert_pooler_bias_data = vec![0.010000, 0.010000, 0.010000, 0.010000];
        w.insert("bert.pooler.dense.bias".into(), bert_pooler_bias_data);
        s.insert("bert.pooler.dense.bias".into(), vec![4]);

        let bert_classifier_weight_data = vec![
            -0.324395, 0.449183, 0.347319, 0.374856, 0.148310, -0.285242, 0.449305, -0.487891,
        ];
        w.insert("classifier.weight".into(), bert_classifier_weight_data);
        s.insert("classifier.weight".into(), vec![2, 4]);

        let bert_classifier_bias_data = vec![0.010000, 0.010000];
        w.insert("classifier.bias".into(), bert_classifier_bias_data);
        s.insert("classifier.bias".into(), vec![2]);

        let (weights, _tmp) = create_test_weights(w, s)?;

        // Configure Head
        let pooler = LinearLayer::builder(&weights, "bert.pooler.dense.weight")
            .with_optional_bias(Some("bert.pooler.dense.bias"))
            .build()?;
        let classifier = LinearLayer::builder(&weights, "classifier.weight")
            .with_optional_bias(Some("classifier.bias"))
            .build()?;

        let head = CpuSequenceClassificationHead::with_config(
            Some(pooler), // BERT uses pooler
            None,
            classifier,
            HeadActivation::Tanh, // BERT Pooler uses Tanh internally, head does not add extra
            PoolingStrategy::Cls,
            None,
        )?;

        // Input
        let input_data = vec![
            -0.476570, 0.248044, 0.155925, -0.160735, 0.417234, 1.000363, 0.600762, 0.109795,
            -0.841132, -0.290797, -0.147981, -0.926648,
        ];
        let input = Array3::from_shape_vec((1, 3, 4), input_data)?;

        // Run
        let output = head.forward(&input, None)?;

        // Validation
        let golden_data = vec![-0.052821, 0.020354];
        let golden = Array2::from_shape_vec((1, 2), golden_data)?;

        let diff = (&output - &golden).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!("BERT Head Max Diff: {}", max_diff);
        assert!(max_diff < 1e-5);
        Ok(())
    }

    #[test]
    fn test_distilbert_head_golden() -> Result<()> {
        let mut w = HashMap::new();
        let mut s = HashMap::new();

        let distil_pre_classifier_weight_data = vec![
            0.059639, 0.059092, -0.408543, -0.289997, -0.492805, -0.461038, 0.492895, 0.413113,
            0.118579, 0.474389, -0.181088, -0.285172, 0.426260, -0.026483, 0.094946, 0.295581,
            0.263484, -0.286297, -0.193368, -0.461447, 0.022029, -0.179258, 0.107378, 0.023262,
            0.426268, 0.043063, 0.250572, 0.157776, 0.443614, -0.207854, 0.417489, 0.358624,
        ];
        w.insert(
            "pre_classifier.weight".into(),
            distil_pre_classifier_weight_data,
        );
        s.insert("pre_classifier.weight".into(), vec![8, 4]);

        let distil_pre_classifier_bias_data = vec![
            0.010000, 0.010000, 0.010000, 0.010000, 0.010000, 0.010000, 0.010000, 0.010000,
        ];
        w.insert(
            "pre_classifier.bias".into(),
            distil_pre_classifier_bias_data,
        );
        s.insert("pre_classifier.bias".into(), vec![8]);

        let distil_classifier_weight_data = vec![
            -0.141110, 0.216720, 0.326859, -0.199907, -0.137904, 0.241867, 0.285077, 0.093267,
            0.108571, 0.261865, 0.309848, -0.223606, -0.145429, 0.297681, 0.190961, -0.165891,
        ];
        w.insert("classifier.weight".into(), distil_classifier_weight_data);
        s.insert("classifier.weight".into(), vec![2, 8]);

        let distil_classifier_bias_data = vec![0.010000, 0.010000];
        w.insert("classifier.bias".into(), distil_classifier_bias_data);
        s.insert("classifier.bias".into(), vec![2]);

        let (weights, _tmp) = create_test_weights(w, s)?;

        let pre_classifier = LinearLayer::builder(&weights, "pre_classifier.weight")
            .with_optional_bias(Some("pre_classifier.bias"))
            .build()?;
        let classifier = LinearLayer::builder(&weights, "classifier.weight")
            .with_optional_bias(Some("classifier.bias"))
            .build()?;

        let head = CpuSequenceClassificationHead::with_config(
            None,
            Some(pre_classifier),
            classifier,
            HeadActivation::Relu,
            PoolingStrategy::Cls,
            None,
        )?;

        let input_data = vec![
            0.336690, 0.128809, 0.234462, 0.230333, -1.122856, -0.186328, 2.208201, -0.637997,
            0.461657, 0.267351, 0.534905, 0.809357,
        ];
        let input = Array3::from_shape_vec((1, 3, 4), input_data)?;
        let output = head.forward(&input, None)?;

        let golden_data = vec![0.070543, -0.038873];
        let golden = Array2::from_shape_vec((1, 2), golden_data)?;

        let diff = (&output - &golden).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));
        println!("DistilBERT Head Max Diff: {}", max_diff);
        assert!(max_diff < 1e-5);
        Ok(())
    }

    #[test]
    fn test_simple_head_golden() -> Result<()> {
        let mut w = HashMap::new();
        let mut s = HashMap::new();

        let simple_classifier_weight_data = vec![
            0.330252, -0.373889, 0.407470, 0.319927, 0.420103, -0.383355, -0.335608, 0.237919,
        ];
        w.insert("classifier.weight".into(), simple_classifier_weight_data);
        s.insert("classifier.weight".into(), vec![2, 4]);

        let simple_classifier_bias_data = vec![0.010000, 0.010000];
        w.insert("classifier.bias".into(), simple_classifier_bias_data);
        s.insert("classifier.bias".into(), vec![2]);

        let (weights, _tmp) = create_test_weights(w, s)?;

        let classifier = LinearLayer::builder(&weights, "classifier.weight")
            .with_optional_bias(Some("classifier.bias"))
            .build()?;

        let head = CpuSequenceClassificationHead::with_config(
            None,
            None,
            classifier,
            HeadActivation::None,
            PoolingStrategy::Cls,
            None,
        )?;

        // UPDATE: Use correct inputs from Python output
        let input_data = vec![
            -0.728938, -0.913169, -0.051331, 0.107147, 1.133505, -0.271534, 0.309822, 0.470191,
            -1.204516, 0.533298, 1.263937, 0.337583,
        ];
        let input = Array3::from_shape_vec((1, 3, 4), input_data)?;
        let output = head.forward(&input, None)?;

        // UPDATE: Use correct logits
        let golden_data = vec![0.124054, 0.096558];
        let golden = Array2::from_shape_vec((1, 2), golden_data)?;

        let diff = (&output - &golden).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));
        println!("Simple Head Max Diff: {}", max_diff);
        assert!(max_diff < 1e-5);
        Ok(())
    }

    #[test]
    fn test_gelu_head_golden() -> Result<()> {
        let mut w = HashMap::new();
        let mut s = HashMap::new();

        let gelu_dense_weight_data = vec![
            0.072198, 0.053876, 0.486830, 0.108030, -0.265344, -0.050821, 0.174336, 0.248007,
            0.060109, -0.332623, -0.166670, -0.035248, 0.133244, 0.269246, -0.285298, 0.281496,
            0.364409, 0.105173, -0.135026, -0.246423, -0.335796, -0.216680, -0.114181, 0.333748,
            0.117322, -0.107669, -0.312197, 0.337465, -0.289097, -0.071846, -0.002587, -0.465979,
        ];
        w.insert("dense.weight".into(), gelu_dense_weight_data);
        s.insert("dense.weight".into(), vec![8, 4]);

        let gelu_dense_bias_data = vec![
            0.010000, 0.010000, 0.010000, 0.010000, 0.010000, 0.010000, 0.010000, 0.010000,
        ];
        w.insert("dense.bias".into(), gelu_dense_bias_data);
        s.insert("dense.bias".into(), vec![8]);

        let gelu_out_proj_weight_data = vec![
            0.271316, -0.286595, 0.197593, 0.139298, -0.124084, -0.183458, 0.270408, -0.144778,
            0.314997, 0.001186, 0.333142, 0.246243, -0.102061, -0.059122, 0.014213, 0.224869,
        ];
        w.insert("out_proj.weight".into(), gelu_out_proj_weight_data);
        s.insert("out_proj.weight".into(), vec![2, 8]);

        let gelu_out_proj_bias_data = vec![0.010000, 0.010000];
        w.insert("out_proj.bias".into(), gelu_out_proj_bias_data);
        s.insert("out_proj.bias".into(), vec![2]);

        let (weights, _tmp) = create_test_weights(w, s)?;

        let dense = LinearLayer::builder(&weights, "dense.weight")
            .with_optional_bias(Some("dense.bias"))
            .build()?;
        let out_proj = LinearLayer::builder(&weights, "out_proj.weight")
            .with_optional_bias(Some("out_proj.bias"))
            .build()?;

        let head = CpuSequenceClassificationHead::with_config(
            None,
            Some(dense),
            out_proj,
            HeadActivation::Gelu,
            PoolingStrategy::Cls,
            None,
        )?;

        let input_data = vec![
            -0.197415, 1.942783, -1.401702, -0.762557, 0.631213, -0.899135, -0.557793, 0.690719,
            0.222459, -0.666225, 0.684642, 0.574002,
        ];
        let input = Array3::from_shape_vec((1, 3, 4), input_data)?;
        let output = head.forward(&input, None)?;

        let golden_data = vec![0.004101, 0.051191];
        let golden = Array2::from_shape_vec((1, 2), golden_data)?;

        let diff = (&output - &golden).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));
        println!("GELU Head Max Diff: {}", max_diff);
        assert!(max_diff < 1e-5);
        Ok(())
    }

    fn make_linear_layer(in_features: usize, out_features: usize) -> LinearLayer {
        let mut weight = Array2::<f32>::zeros((out_features, in_features));
        // Fill diagonal-ish to make it act like identity
        for i in 0..out_features.min(in_features) {
            weight[[i, i]] = 1.0;
        }
        let bias = Array1::<f32>::zeros(out_features);
        LinearLayer::new_f32(weight, bias)
    }

    #[test]
    fn test_cpu_classification_head_no_pooler() {
        // Input: batch=2, seq_len=3, hidden_size=3
        let hidden_states = array![
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
        ];
        let classifier = make_linear_layer(3, 2); // 3 hidden -> 2 classes
        let head = CpuSequenceClassificationHead::new(None, classifier).unwrap();

        let logits = head.forward(&hidden_states, None).unwrap();

        // Should be [batch, num_classes] = [2,2]
        assert_eq!(logits.shape(), &[2, 2]);

        assert_eq!(logits[[0, 0]], 1.0); // corresponds to first hidden dim
        assert_eq!(logits[[0, 1]], 2.0);
        assert_eq!(logits[[1, 0]], 10.0);
        assert_eq!(logits[[1, 1]], 11.0);

        assert_eq!(head.num_classes(), 2);
    }

    #[test]
    fn test_cpu_classification_head_with_pooler() {
        // Input: batch=2, seq_len=3, hidden_size=3
        let hidden_states = array![
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
        ];

        // Pooler: maps hidden_size->4
        let pooler = make_linear_layer(3, 4);

        // Classifier: maps 4->2 classes
        let classifier = make_linear_layer(4, 2);

        let head = CpuSequenceClassificationHead::new(Some(pooler), classifier).unwrap();

        let logits = head.forward(&hidden_states, None).unwrap();
        assert_eq!(logits.shape(), &[2, 2]);

        for b in 0..2 {
            for c in 0..2 {
                assert!(logits[[b, c]].is_finite());
            }
        }

        assert_eq!(head.num_classes(), 2);
    }

    #[test]
    fn test_cpu_classification_head_pooler_dimension_mismatch() {
        let pooler = make_linear_layer(3, 5);
        let classifier = make_linear_layer(4, 2); // mismatch: pooler out 5 != classifier in 4

        let err = CpuSequenceClassificationHead::new(Some(pooler), classifier);
        assert!(err.is_err());
    }

    #[test]
    fn test_forward_empty_batch() {
        // batch=0, seq_len=3, hidden_size=3
        let hidden_states = Array3::<f32>::zeros((0, 3, 3));
        let classifier = make_linear_layer(3, 2);
        let head = CpuSequenceClassificationHead::new(None, classifier).unwrap();

        let logits = head.forward(&hidden_states, None).unwrap();
        assert_eq!(logits.shape(), &[0, 2]);
    }
}
