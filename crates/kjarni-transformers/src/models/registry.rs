//! Pretrained model registry with metadata and download utilities.

use crate::utils::levenshtein;
use anyhow::Result;
#[cfg(not(target_arch = "wasm32"))]
use anyhow::anyhow;
use std::path::{Path, PathBuf};
use strum_macros::EnumIter;

/// Model weight storage format for loading and inference.
pub enum WeightsFormat {
    /// SafeTensors format
    SafeTensors,

    /// GGUF format with quantization
    GGUF,
}

/// Defines the model architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelArchitecture {
    /// Standard Llama architecture
    Llama,

    /// Qwen2 family with bias terms in attention projections.
    Qwen2,

    /// Mistral family with sliding window attention.
    Mistral,

    /// Phi-3 family with LongRoPE scaling.
    Phi3,

    /// BERT family with absolute positional embeddings.
    Bert,

    /// MPNet with relative attention bias.
    Mpnet,

    /// NomicBERT with rotary position embeddings.
    NomicBert,

    /// T5/FLAN family with relative positional buckets.
    T5,

    /// BART family with learned positional embeddings.
    Bart,

    /// GPT family (legacy).
    GPT,

    /// Whisper family for speech-to-text.
    Whisper,

    /// CLIP family: a ViT image tower and a text tower projected into one space.
    ///
    /// The image tower is an ordinary transformer encoder over patches, so it runs
    /// the same block stack as BERT does over tokens. What differs is the front:
    /// pixels become patches through one strided projection rather than ids
    /// through an embedding table.
    Clip,
}

impl ModelArchitecture {
    /// Returns a human-readable display name for the architecture.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Llama => "Llama (Standard)",
            Self::Qwen2 => "Qwen2 (Biased)",
            Self::Mistral => "Mistral (SWA)",
            Self::Phi3 => "Phi-3 (LongRoPE)",
            Self::Clip => "CLIP (ViT + text)",
            Self::Bert => "BERT",
            Self::Mpnet => "Mpnet",
            Self::NomicBert => "Nomic-BERT",
            Self::T5 => "T5",
            Self::Bart => "BART",
            Self::GPT => "GPT",
            Self::Whisper => "Whisper (ASR)",
        }
    }

    /// Returns architecture.
    pub fn category(&self) -> &'static str {
        match self {
            // Decoders (LLMs)
            Self::Llama | Self::Qwen2 | Self::Mistral | Self::Phi3 | Self::GPT => "decoder",

            // Encoders (Embeddings/Classifiers)
            Self::Bert | Self::NomicBert | Self::Mpnet | Self::Clip => "encoder",

            // Seq2Seq
            Self::T5 | Self::Bart | Self::Whisper => "encoder-decoder",
        }
    }
}

/// Defines the primary intended use case for a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelTask {
    /// Vector embedding
    Embedding,

    /// Reranking
    ReRanking,

    /// Text -> Class label
    Classification,

    /// Interactive chat and instruction following
    Chat,

    /// Deep reasoning and logical inference.
    Reasoning,

    /// Detecting positive/negative sentiment in text
    SentimentAnalysis,

    /// Classifying text into arbitrary user-defined labels
    ZeroShotClassification,

    /// Translation, summarization, and text-to-text generation
    Seq2Seq,

    /// General text generation
    Generation,

    /// Long text -> Concise summary.
    Summarization,

    /// Text in language A -> Text in language B.
    Translation,

    /// Speech-to-text transcription.
    SpeechToText,

    /// General text-to-text transformation.
    TextToText,

    /// Image -> vector, in a space shared with text embeddings.
    ///
    /// Distinct from `Embedding` because the input is pixels: it needs image
    /// preprocessing rather than tokenisation, and the two cannot be swapped.
    ImageEmbedding,
}

/// The curated list of pretrained models supported by Kjarni.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumIter)]
pub enum ModelType {
    MiniLML6V2,
    NomicEmbedText,
    BgeM3,
    MiniLML6V2CrossEncoder,
    MpnetBaseV2,
    DistilBertBaseCased,

    DistilBertSST2,
    TwitterRobertaSentiment,
    BertMultilingualSentiment,
    ToxicBertMultilingual,
    RobertaGoEmotions,
    DistilRobertaEmotion,

    Qwen2_5_0_5B_Instruct,
    Qwen2_5_1_5B_Instruct,
    Llama3_2_1B_Instruct,
    Llama3_2_3B_Instruct,
    Phi3_5_Mini_Instruct,
    Mistral7B_v0_3_Instruct,
    Llama3_1_8B_Instruct,
    DeepSeek_R1_Distill_Llama_8B,
    FlanT5Base,
    FlanT5Large,
    DistilBartCnn,
    BartLargeCnn,
    WhisperSmall,
    WhisperLargeV3,
    DistilGpt2,
    Gpt2,

    // Vision
    ClipVitBase32,
}

/// Download URLs for all required model files.
#[derive(Debug, Clone)]
pub struct ModelPaths {
    /// URL to SafeTensors weights file or index.
    ///
    /// Points to either a single `model.safetensors` file or a
    /// `model.safetensors.index.json` for sharded models.
    pub weights_url: &'static str,

    /// URL to tokenizer configuration.
    ///
    /// `None` for models that take no text at all: a vision encoder embeds pixels,
    /// so there is no vocabulary to fetch. Every text model has one.
    pub tokenizer_url: Option<&'static str>,

    /// URL to model configuration.
    ///
    /// Always required. Contains hyperparameters like hidden size, number of layers, etc.
    pub config_url: &'static str,

    /// Optional URL to quantized GGUF file.
    pub gguf_url: Option<&'static str>,

    /// Optional URL to `preprocessor_config.json`.
    ///
    /// Image models need it and text models do not. It carries the resize, crop,
    /// mean and std the weights were trained against, and getting any of them
    /// wrong produces embeddings that look plausible while quietly ranking badly,
    /// so these are read rather than hardcoded per model.
    pub preprocessor_url: Option<&'static str>,
}

/// Complete metadata for a pretrained model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// The model's structural architecture family.
    pub architecture: ModelArchitecture,
    /// The model's primary intended use case.
    pub task: ModelTask,
    /// Download URLs for all model files.
    pub paths: ModelPaths,
    /// Human-readable description of the model's capabilities.
    pub description: &'static str,
    /// Approximate disk size in megabytes (SafeTensors format).
    pub size_mb: usize,
    /// Number of parameters in millions.
    pub params_millions: usize,
}

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

impl ModelType {
    pub fn architecture(&self) -> ModelArchitecture {
        self.info().architecture
    }
    pub fn is_llama_model(&self) -> bool {
        matches!(self.architecture(), ModelArchitecture::Llama)
    }
    pub fn is_gpt2_model(&self) -> bool {
        matches!(self, Self::DistilGpt2 | Self::Gpt2)
    }
    pub fn is_qwen_model(&self) -> bool {
        matches!(self.architecture(), ModelArchitecture::Qwen2)
    }
    pub fn is_phi_model(&self) -> bool {
        matches!(self.architecture(), ModelArchitecture::Phi3)
    }
    /// Get the CLI-friendly slug (e.g., "llama3.2-1b")
    pub fn cli_name(&self) -> &'static str {
        match self {
            // Embeddings
            Self::MiniLML6V2 => "minilm-l6-v2",
            Self::NomicEmbedText => "nomic-embed-text",
            Self::BgeM3 => "bge-m3",
            Self::MpnetBaseV2 => "mpnet-base-v2",
            Self::DistilBertBaseCased => "distilbert-base",

            // Reranker
            Self::MiniLML6V2CrossEncoder => "minilm-l6-v2-cross-encoder",

            // Classifiers

            // Sentiment
            Self::DistilBertSST2 => "distilbert-sentiment",
            Self::TwitterRobertaSentiment => "roberta-sentiment",
            Self::BertMultilingualSentiment => "bert-sentiment-multilingual",

            // Emotion
            Self::RobertaGoEmotions => "roberta-emotions",
            Self::DistilRobertaEmotion => "distilroberta-emotion",

            // Toxicity
            Self::ToxicBertMultilingual => "toxic-bert",

            // Edge LLMs
            Self::Qwen2_5_0_5B_Instruct => "qwen2.5-0.5b-instruct",
            Self::Qwen2_5_1_5B_Instruct => "qwen2.5-1.5b-instruct",
            Self::Llama3_2_1B_Instruct => "llama3.2-1b-instruct",
            Self::Llama3_2_3B_Instruct => "llama3.2-3b-instruct",
            Self::Phi3_5_Mini_Instruct => "phi3.5-mini-instruct",

            // Workhorse LLMs
            Self::Mistral7B_v0_3_Instruct => "mistral-7b-instruct",
            Self::Llama3_1_8B_Instruct => "llama3.1-8b-instruct",
            Self::DeepSeek_R1_Distill_Llama_8B => "deepseek-r1-8b",

            // Seq2Seq
            Self::FlanT5Base => "flan-t5-base",
            Self::FlanT5Large => "flan-t5-large",
            Self::DistilBartCnn => "distilbart-cnn",
            Self::BartLargeCnn => "bart-large-cnn",
            Self::WhisperSmall => "whisper-small",
            Self::WhisperLargeV3 => "whisper-large-v3",

            // Legacy
            Self::DistilGpt2 => "distilgpt2",
            Self::Gpt2 => "gpt2",

            // Vision
            Self::ClipVitBase32 => "clip-vit-base-32",
        }
    }

    pub fn display_group(&self) -> &'static str {
        match self.info().task {
            // Generation
            ModelTask::Chat | ModelTask::Reasoning => "LLM (Decoder)",

            //Translation/Summary
            ModelTask::Seq2Seq
            | ModelTask::Summarization
            | ModelTask::Translation
            | ModelTask::TextToText
            | ModelTask::SpeechToText => "Seq2Seq",

            // Vector embedding
            ModelTask::Embedding => "Embedding",

            // Same index, different input: these vectors share a space with text
            // embeddings, so they are grouped apart only by what goes in.
            ModelTask::ImageEmbedding => "Image Embedding",

            //  Reranking
            ModelTask::ReRanking => "Re-Ranker",

            // Classification
            ModelTask::SentimentAnalysis
            | ModelTask::ZeroShotClassification
            | ModelTask::Classification => "Classifier",

            ModelTask::Generation => "Generation (Decoder)",
        }
    }

    pub fn is_instruct_model(&self) -> bool {
        matches!(
            self.info().task,
            ModelTask::Chat | ModelTask::Reasoning | ModelTask::Seq2Seq
        )
    }

    pub fn info(&self) -> ModelInfo {
        match self {
            Self::MiniLML6V2 => ModelInfo {
                architecture: ModelArchitecture::Bert,
                task: ModelTask::Embedding,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors",
                    tokenizer_url: Some(
                        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json",
                    gguf_url: None,
                    preprocessor_url: None,
                },
                description: "Fastest sentence embedding model. Ideal for basic RAG.",
                size_mb: 90,
                params_millions: 22,
            },

            Self::MiniLML6V2CrossEncoder => ModelInfo {
                architecture: ModelArchitecture::Bert,
                task: ModelTask::ReRanking,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/model.safetensors",
                    tokenizer_url: Some(
                        "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/config.json",
                    gguf_url: None,
                    preprocessor_url: None,
                },
                description: "Cross-encoder for passage reranking. Use for search result reordering, NOT sentiment.",
                size_mb: 90,
                params_millions: 22,
            },

            Self::MpnetBaseV2 => ModelInfo {
                architecture: ModelArchitecture::Mpnet,
                task: ModelTask::Embedding,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/model.safetensors",
                    tokenizer_url: Some(
                        "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/config.json",
                    gguf_url: None,
                    preprocessor_url: None,
                },
                description: "High-quality sentence embedding model.",
                size_mb: 420,
                params_millions: 110,
            },

            Self::DistilBertBaseCased => ModelInfo {
                architecture: ModelArchitecture::Bert,
                task: ModelTask::Embedding,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/model.safetensors",
                    tokenizer_url: Some(
                        "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/config.json",
                    gguf_url: None,
                    preprocessor_url: None,
                },
                description: "Lightweight DistilBERT for question answering.",
                size_mb: 260,
                params_millions: 66,
            },

            Self::NomicEmbedText => ModelInfo {
                architecture: ModelArchitecture::NomicBert,
                task: ModelTask::Embedding,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/model.safetensors",
                    tokenizer_url: Some(
                        "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/config.json",
                    gguf_url: None,
                    preprocessor_url: None,
                },
                description: "Modern standard for RAG. 8192 context length, matryoshka embeddings.",
                size_mb: 550,
                params_millions: 137,
            },
            Self::BgeM3 => ModelInfo {
                architecture: ModelArchitecture::Bert,
                task: ModelTask::Embedding,
                paths: ModelPaths {
                    // BAAI publish bge-m3 only as pytorch_model.bin, so the
                    // safetensors URL on their repo 404s. This mirror carries the
                    // converted weights and the same tokenizer and config.
                    weights_url: "https://huggingface.co/olafuraron/bge-m3-safetensors/resolve/main/model.safetensors",
                    tokenizer_url: Some(
                        "https://huggingface.co/olafuraron/bge-m3-safetensors/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/olafuraron/bge-m3-safetensors/resolve/main/config.json",
                    gguf_url: None,
                    preprocessor_url: None,
                },
                description: "Massive multilingual embedding model. State of the art for diverse languages.",
                size_mb: 2200,
                params_millions: 567,
            },
            Self::DistilBertSST2 => ModelInfo {
                architecture: ModelArchitecture::Bert,
                task: ModelTask::SentimentAnalysis,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/model.safetensors",
                    tokenizer_url: Some(
                        "https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/config.json",
                    gguf_url: None,
                    preprocessor_url: None,
                },
                description: "Fast binary sentiment (positive/negative). Best for simple yes/no sentiment.",
                size_mb: 268,
                params_millions: 66,
            },
            Self::TwitterRobertaSentiment => ModelInfo {
                architecture: ModelArchitecture::Bert,
                task: ModelTask::SentimentAnalysis,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/olafuraron/twitter-roberta-base-sentiment-latest-safetensors/resolve/main/model.safetensors",
                    tokenizer_url: Some(
                        "https://huggingface.co/olafuraron/twitter-roberta-base-sentiment-latest-safetensors/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/olafuraron/twitter-roberta-base-sentiment-latest-safetensors/resolve/main/config.json",
                    gguf_url: None,
                    preprocessor_url: None,
                },
                description: "3-class sentiment (negative/neutral/positive). Optimized for social media text.",
                size_mb: 499,
                params_millions: 125,
            },
            Self::BertMultilingualSentiment => ModelInfo {
                architecture: ModelArchitecture::Bert,
                task: ModelTask::SentimentAnalysis,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/olafuraron/bert-base-multilingual-uncased-sentiment-safetensors/resolve/main/model.safetensors",
                    tokenizer_url: Some(
                        "https://huggingface.co/olafuraron/bert-base-multilingual-uncased-sentiment-safetensors/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/olafuraron/bert-base-multilingual-uncased-sentiment-safetensors/resolve/main/config.json",
                    gguf_url: None,
                    preprocessor_url: None,
                },
                description: "5-star sentiment (1-5). Multilingual: EN, DE, FR, ES, IT, NL.",
                size_mb: 681,
                params_millions: 168,
            },
            Self::RobertaGoEmotions => ModelInfo {
                architecture: ModelArchitecture::Bert,
                task: ModelTask::Classification,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/SamLowe/roberta-base-go_emotions/resolve/main/model.safetensors",
                    tokenizer_url: Some(
                        "https://huggingface.co/SamLowe/roberta-base-go_emotions/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/SamLowe/roberta-base-go_emotions/resolve/main/config.json",
                    gguf_url: None,
                    preprocessor_url: None,
                },
                description: "28 emotion labels (multi-label). Detects nuanced emotions like admiration, amusement, anger, etc.",
                size_mb: 499,
                params_millions: 125,
            },
            Self::DistilRobertaEmotion => ModelInfo {
                architecture: ModelArchitecture::Bert,
                task: ModelTask::Classification,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/olafuraron/emotion-english-distilroberta-base-safetensors/resolve/main/model.safetensors",
                    tokenizer_url: Some(
                        "https://huggingface.co/olafuraron/emotion-english-distilroberta-base-safetensors/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/olafuraron/emotion-english-distilroberta-base-safetensors/resolve/main/config.json",
                    gguf_url: None,
                    preprocessor_url: None,
                },
                description: "7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise.",
                size_mb: 329,
                params_millions: 82,
            },
            Self::ToxicBertMultilingual => ModelInfo {
                architecture: ModelArchitecture::Bert,
                task: ModelTask::Classification,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/olafuraron/toxic-bert-safetensors/resolve/main/model.safetensors",
                    tokenizer_url: Some(
                        "https://huggingface.co/olafuraron/toxic-bert-safetensors/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/olafuraron/toxic-bert-safetensors/resolve/main/config.json",
                    gguf_url: None,
                    preprocessor_url: None,
                },
                description: "Toxic comment classifier. Detects: toxic, severe_toxic, obscene, threat, insult, identity_hate.",
                size_mb: 438,
                params_millions: 110,
            },
            Self::Qwen2_5_0_5B_Instruct => ModelInfo {
                architecture: ModelArchitecture::Qwen2,
                task: ModelTask::Chat,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors",
                    tokenizer_url: Some(
                        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/config.json",
                    gguf_url: Some(
                        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
                    ),
                    preprocessor_url: None,
                },
                description: "Tiny logic engine. Perfect for structured output and sanity checks.",
                size_mb: 990,
                params_millions: 490,
            },
            Self::Qwen2_5_1_5B_Instruct => ModelInfo {
                architecture: ModelArchitecture::Qwen2,
                task: ModelTask::Chat,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/model.safetensors",
                    tokenizer_url: Some(
                        "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/config.json",
                    gguf_url: Some(
                        "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
                    ),
                    preprocessor_url: None,
                },
                description: "Balanced edge model. Good reasoning in a small package.",
                size_mb: 3100,
                params_millions: 1540,
            },
            Self::Llama3_2_1B_Instruct => ModelInfo {
                architecture: ModelArchitecture::Llama,
                task: ModelTask::Chat,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/resolve/main/model.safetensors",
                    tokenizer_url: Some(
                        "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/resolve/main/config.json",
                    gguf_url: Some(
                        "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                    ),
                    preprocessor_url: None,
                },
                description: "Official Meta edge model. Very fast, good general chat.",
                size_mb: 2500,
                params_millions: 1230,
            },
            Self::Llama3_2_3B_Instruct => ModelInfo {
                architecture: ModelArchitecture::Llama,
                task: ModelTask::Chat,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/resolve/main/model.safetensors.index.json",
                    tokenizer_url: Some(
                        "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/resolve/main/config.json",
                    gguf_url: Some(
                        "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                    ),
                    preprocessor_url: None,
                },
                description: "The 3B standard. Excellent balance of speed and coherence.",
                size_mb: 6500,
                params_millions: 3210,
            },
            Self::Phi3_5_Mini_Instruct => ModelInfo {
                architecture: ModelArchitecture::Phi3,
                task: ModelTask::Reasoning,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/microsoft/Phi-3.5-mini-instruct/resolve/main/model.safetensors.index.json",
                    tokenizer_url: Some(
                        "https://huggingface.co/microsoft/Phi-3.5-mini-instruct/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/microsoft/Phi-3.5-mini-instruct/resolve/main/config.json",
                    gguf_url: Some(
                        "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf",
                    ),
                    preprocessor_url: None,
                },
                description: "Microsoft's 3.8B reasoning champion. Punches way above its weight.",
                size_mb: 7500,
                params_millions: 3800,
            },
            Self::Mistral7B_v0_3_Instruct => ModelInfo {
                architecture: ModelArchitecture::Mistral,
                task: ModelTask::Chat,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/model.safetensors.index.json",
                    tokenizer_url: Some(
                        "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/config.json",
                    gguf_url: Some(
                        "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
                    ),
                    preprocessor_url: None,
                },
                description: "Mistral v0.3. Extremely reliable 7B model for all tasks.",
                size_mb: 14500,
                params_millions: 7240,
            },
            Self::Llama3_1_8B_Instruct => ModelInfo {
                architecture: ModelArchitecture::Llama,
                task: ModelTask::Chat,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/model.safetensors.index.json",
                    tokenizer_url: Some(
                        "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/config.json",
                    gguf_url: Some(
                        "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                    ),
                    preprocessor_url: None,
                },
                description: "The open source standard. Robust, smart, and safe.",
                size_mb: 16000,
                params_millions: 8030,
            },
            Self::DeepSeek_R1_Distill_Llama_8B => ModelInfo {
                architecture: ModelArchitecture::Llama,
                task: ModelTask::Reasoning,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/resolve/main/model.safetensors.index.json",
                    tokenizer_url: Some(
                        "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/resolve/main/config.json",
                    gguf_url: Some(
                        "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
                    ),
                    preprocessor_url: None,
                },
                description: "State-of-the-Art reasoning distilled from DeepSeek R1.",
                size_mb: 16000,
                params_millions: 8030,
            },
            Self::FlanT5Base => ModelInfo {
                architecture: ModelArchitecture::T5,
                task: ModelTask::Seq2Seq,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/google/flan-t5-base/resolve/main/model.safetensors",
                    tokenizer_url: Some(
                        "https://huggingface.co/google/flan-t5-base/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/google/flan-t5-base/resolve/main/config.json",
                    gguf_url: None,
                    preprocessor_url: None,
                },
                description: "General purpose instruction follower (Text-to-Text).",
                size_mb: 990,
                params_millions: 250,
            },
            Self::FlanT5Large => ModelInfo {
                architecture: ModelArchitecture::T5,
                task: ModelTask::Seq2Seq,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/google/flan-t5-large/resolve/main/model.safetensors",
                    tokenizer_url: Some(
                        "https://huggingface.co/google/flan-t5-large/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/google/flan-t5-large/resolve/main/config.json",
                    gguf_url: None,
                    preprocessor_url: None,
                },
                description: "Powerful instruction follower. Great for translation and summarization.",
                size_mb: 3000,
                params_millions: 780,
            },

            Self::BartLargeCnn => ModelInfo {
                architecture: ModelArchitecture::Bart,
                task: ModelTask::Seq2Seq,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/facebook/bart-large-cnn/resolve/main/model.safetensors",
                    tokenizer_url: Some(
                        "https://huggingface.co/facebook/bart-large-cnn/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/facebook/bart-large-cnn/resolve/main/config.json",
                    gguf_url: None,
                    preprocessor_url: None,
                },
                description: "BART large fine-tuned for summarization.",
                size_mb: 1600,
                params_millions: 406,
            },

            Self::DistilBartCnn => ModelInfo {
                architecture: ModelArchitecture::Bart,
                task: ModelTask::Seq2Seq,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/olafuraron/distilbart-cnn-12-6/resolve/main/model.safetensors",
                    tokenizer_url: Some(
                        "https://huggingface.co/olafuraron/distilbart-cnn-12-6/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/olafuraron/distilbart-cnn-12-6/resolve/main/config.json",
                    gguf_url: None,
                    preprocessor_url: None,
                },
                description: "Distilled BART for fast summarization.",
                size_mb: 1000,
                params_millions: 306,
            },

            Self::WhisperSmall => ModelInfo {
                architecture: ModelArchitecture::Whisper,
                task: ModelTask::SpeechToText,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/openai/whisper-small/resolve/main/model.safetensors",
                    tokenizer_url: Some(
                        "https://huggingface.co/openai/whisper-small/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/openai/whisper-small/resolve/main/config.json",
                    gguf_url: None,
                    preprocessor_url: None,
                },
                description: "OpenAI Whisper small for speech-to-text transcription.",
                size_mb: 1500,
                params_millions: 244,
            },

            Self::WhisperLargeV3 => ModelInfo {
                architecture: ModelArchitecture::Whisper,
                task: ModelTask::SpeechToText,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/openai/whisper-large-v3/resolve/main/model.safetensors",
                    tokenizer_url: Some(
                        "https://huggingface.co/openai/whisper-large-v3/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/openai/whisper-large-v3/resolve/main/config.json",
                    gguf_url: None,
                    preprocessor_url: None,
                },
                description: "OpenAI Whisper large v3 for high-accuracy speech-to-text transcription.",
                size_mb: 7700,
                params_millions: 1550,
            },

            Self::DistilGpt2 => ModelInfo {
                architecture: ModelArchitecture::GPT,
                task: ModelTask::Generation,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/distilgpt2/resolve/main/model.safetensors",
                    tokenizer_url: Some(
                        "https://huggingface.co/distilgpt2/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/distilgpt2/resolve/main/config.json",
                    gguf_url: None,
                    preprocessor_url: None,
                },
                description: "Distilled GPT-2 for lightweight text generation.",
                size_mb: 319,
                params_millions: 82,
            },

            Self::Gpt2 => ModelInfo {
                architecture: ModelArchitecture::GPT,
                task: ModelTask::Generation,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/gpt2/resolve/main/model.safetensors",
                    tokenizer_url: Some("https://huggingface.co/gpt2/resolve/main/tokenizer.json"),
                    config_url: "https://huggingface.co/gpt2/resolve/main/config.json",
                    gguf_url: None,
                    preprocessor_url: None,
                },
                description: "GPT-2 small: general-purpose text generator.",
                size_mb: 548,
                params_millions: 117,
            },

            // LAION's B/32 rather than `openai/clip-vit-base-patch32`, which ships
            // only `pytorch_model.bin` and no safetensors at all: the loader here
            // cannot read a pickle. Same architecture and size, trained on LAION-2B,
            // and it uses plain GELU where OpenAI's needs QuickGELU, so it runs on
            // the activations this engine already has.
            //
            // 32x32 patches over a 224px image is 49 patches plus one CLS: 50
            // positions, so the image tower is cheaper than a short sentence
            // through a text encoder.
            Self::ClipVitBase32 => ModelInfo {
                architecture: ModelArchitecture::Clip,
                task: ModelTask::ImageEmbedding,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/model.safetensors",
                    // Present because CLIP has a text tower too: the whole point is
                    // that a caption and a photo land in the same 512-d space.
                    tokenizer_url: Some(
                        "https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/tokenizer.json",
                    ),
                    config_url: "https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/config.json",
                    gguf_url: None,
                    preprocessor_url: Some(
                        "https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/preprocessor_config.json",
                    ),
                },
                description: "CLIP ViT-B/32: images and text in one embedding space. Search photos by description.",
                size_mb: 605,
                params_millions: 151,
            },
        }
    }

    pub fn resolve(name: &str) -> Result<ModelType, String> {
        if let Some(m) = Self::from_cli_name(name) {
            return Ok(m);
        }

        // Try substring match first
        let all_names: Vec<&str> = ModelType::all().map(|m| m.cli_name()).collect();
        let substring_matches: Vec<&str> = all_names
            .iter()
            .filter(|n| n.contains(&name.to_lowercase()))
            .copied()
            .collect();

        if !substring_matches.is_empty() {
            return Err(format!(
                "Unknown model '{name}'. Did you mean: {}?",
                substring_matches.join(", ")
            ));
        }

        // Fall back to Levenshtein
        let suggestions = Self::find_similar(name);
        if suggestions.is_empty() {
            Err(format!("Unknown model '{name}'"))
        } else {
            let names: Vec<&str> = suggestions.iter().map(|(n, _)| n.as_str()).collect();
            Err(format!(
                "Unknown model '{name}'. Did you mean: {}?",
                names.join(", ")
            ))
        }
    }

    pub fn from_cli_name(name: &str) -> Option<ModelType> {
        use strum::IntoEnumIterator;
        let normalized = name.to_lowercase();

        // Try CLI names first
        if let Some(m) = ModelType::iter().find(|m| m.cli_name() == normalized) {
            return Some(m);
        }

        // Try HuggingFace aliases
        match normalized.as_str() {
            "all-minilm-l6-v2" | "sentence-transformers/all-minilm-l6-v2" => Some(Self::MiniLML6V2),
            "all-mpnet-base-v2" | "sentence-transformers/all-mpnet-base-v2" => {
                Some(Self::MpnetBaseV2)
            }
            "ms-marco-minilm-l-6-v2" | "cross-encoder/ms-marco-minilm-l-6-v2" => {
                Some(Self::MiniLML6V2CrossEncoder)
            }
            "nomic-embed-text-v1.5" | "nomic-ai/nomic-embed-text-v1.5" => {
                Some(Self::NomicEmbedText)
            }
            "bge-m3" | "baai/bge-m3" => Some(Self::BgeM3),
            "distilbert-base-uncased-finetuned-sst-2-english" => Some(Self::DistilBertSST2),
            "twitter-roberta-base-sentiment-latest" => Some(Self::TwitterRobertaSentiment),
            "bert-base-multilingual-uncased-sentiment"
            | "bert-base-multilingual-uncased-sentiment-safetensors" => {
                Some(Self::BertMultilingualSentiment)
            }
            "toxic-bert" | "toxic-bert-safetensors" | "unitary/toxic-bert" => {
                Some(Self::ToxicBertMultilingual)
            }
            "roberta-base-go_emotions" | "samlowe/roberta-base-go_emotions" => {
                Some(Self::RobertaGoEmotions)
            }
            "emotion-english-distilroberta-base" => Some(Self::DistilRobertaEmotion),

            "distilbart-cnn" | "olafuraron/distilbart-cnn-12-6" | "distilbart-cnn-12-6" => {
                Some(Self::DistilBartCnn)
            }
            "bart-large-cnn" | "facebook/bart-large-cnn" => Some(Self::BartLargeCnn),
            "whisper-small" | "openai/whisper-small" => Some(Self::WhisperSmall),
            "whisper-large-v3" | "openai/whisper-large-v3" => Some(Self::WhisperLargeV3),
            "distilgpt2" | "distilgpt2/resolve/main/model.safetensors" => Some(Self::DistilGpt2),
            "gpt2" | "gpt2/resolve/main/model.safetensors" => Some(Self::Gpt2),

            // Former CLI names, kept resolving so existing scripts do not break.
            // These three lacked the `-instruct` suffix their variants have.
            "qwen2.5-1.5b" => Some(Self::Qwen2_5_1_5B_Instruct),
            "phi3.5-mini" => Some(Self::Phi3_5_Mini_Instruct),
            "mistral-7b" => Some(Self::Mistral7B_v0_3_Instruct),

            _ => None,
        }
    }

    pub fn all() -> impl Iterator<Item = ModelType> {
        use strum::IntoEnumIterator;
        ModelType::iter()
    }

    pub fn find_similar(query: &str) -> Vec<(String, f32)> {
        let all_names: Vec<&str> = ModelType::all().map(|m| m.cli_name()).collect();
        levenshtein::find_similar(query, &all_names, 3, 0.4)
    }

    /// Get the local cache directory for this model
    pub fn cache_dir(&self, base_dir: &Path) -> PathBuf {
        base_dir.join(self.repo_id().replace('/', "_"))
    }

    /// Check if this model is downloaded in the given cache directory
    pub fn is_downloaded(&self, base_dir: &Path) -> bool {
        let model_dir = self.cache_dir(base_dir);

        // Check for essential files
        let config_exists = model_dir.join("config.json").exists();
        let tokenizer_exists = model_dir.join("tokenizer.json").exists();

        // Weights can be either single file or sharded
        let weights_exist = model_dir.join("model.safetensors").exists()
            || model_dir.join("model.safetensors.index.json").exists();

        config_exists && tokenizer_exists && weights_exist
    }

    pub fn search(query: &str) -> Vec<(ModelType, f32)> {
        let query_lower = query.to_lowercase();
        let mut matches: Vec<(ModelType, f32)> = ModelType::all()
            .filter_map(|m| {
                let name = m.cli_name().to_lowercase();
                let desc = m.info().description.to_lowercase();
                let name_sim = levenshtein::similarity(&query_lower, &name);
                // Boost score if substring match found
                let contains_bonus = if name.contains(&query_lower) {
                    0.5
                } else if desc.contains(&query_lower) {
                    0.3
                } else {
                    0.0
                };
                let score = name_sim + contains_bonus;
                if score > 0.3 { Some((m, score)) } else { None }
            })
            .collect();
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        matches
    }

    pub fn repo_id(&self) -> String {
        let url = self.info().paths.weights_url;
        let parts: Vec<&str> = url.split('/').collect();
        // Heuristic: https://huggingface.co/{ORG}/{REPO}/resolve/...
        if parts.len() >= 5 {
            format!("{}/{}", parts[3], parts[4])
        } else {
            "unknown/unknown".to_string()
        }
    }
}

// Download Utilities

/// Downloads all required files for a model to the specified directory.
#[cfg(not(target_arch = "wasm32"))]
pub async fn download_model_files(
    model_dir: &Path,
    paths: &ModelPaths,
    format: WeightsFormat,
    quiet: bool,
) -> Result<PathBuf> {
    tokio::fs::create_dir_all(model_dir).await?;

    // A vision encoder has no vocabulary, so this is absent rather than empty.
    if let Some(url) = paths.tokenizer_url {
        download_file(model_dir, "tokenizer.json", url, quiet).await?;
    }
    download_file(model_dir, "config.json", paths.config_url, quiet).await?;
    download_sentence_bert_config(model_dir, paths.config_url).await;

    // Image models carry their resize, crop, mean and std here. Fetched eagerly
    // because the loader cannot guess them and a wrong guess degrades retrieval
    // silently rather than failing.
    if let Some(url) = paths.preprocessor_url {
        download_file(model_dir, "preprocessor_config.json", url, quiet).await?;
    }

    let use_gguf = matches!(format, WeightsFormat::GGUF) && paths.gguf_url.is_some();

    if use_gguf {
        let url = paths.gguf_url.unwrap();
        download_file(model_dir, "model.gguf", url, quiet).await?;
        Ok(model_dir.join("model.gguf"))
    } else {
        if matches!(format, WeightsFormat::GGUF) {
            eprintln!("  GGUF not available, falling back to SafeTensors.");
        }

        if paths.weights_url.ends_with(".index.json") {
            download_sharded_weights(model_dir, paths.weights_url, quiet).await?;
            Ok(model_dir.join("model.safetensors.index.json"))
        } else {
            download_file(model_dir, "model.safetensors", paths.weights_url, quiet).await?;
            Ok(model_dir.join("model.safetensors"))
        }
    }
}

/// Fetches `sentence_bert_config.json` beside the model config, if it exists.
///
/// This is the file the encoder loader reads to decide where to truncate. Only
/// sentence-transformers exports carry it, so a 404 is the normal outcome for most
/// models and is not an error. Its URL is derived from `config_url` rather than
/// stored per model, so every registry entry gets the behaviour without thirty new
/// string literals to keep in sync.
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code, reason = "reachable only on some targets")]
async fn download_sentence_bert_config(model_dir: &Path, config_url: &str) {
    sentence_bert_config_with(model_dir, config_url, |url| async move {
        download_file(model_dir, SENTENCE_BERT_CONFIG, &url, true).await
    })
    .await
}

#[cfg(not(target_arch = "wasm32"))]
const SENTENCE_BERT_CONFIG: &str = "sentence_bert_config.json";

/// Marks a model directory as known not to have `sentence_bert_config.json`.
///
/// Hidden so it does not look like part of the model, and named after the file it
/// speaks for rather than being a generic "negative cache", because it only ever
/// covers this one optional file.
#[cfg(not(target_arch = "wasm32"))]
const SENTENCE_BERT_ABSENT_MARKER: &str = ".sentence_bert_config.absent";

/// The decision half of fetching the optional config, with the fetch injected.
///
/// Split out so the three behaviours that matter can be tested without a network:
/// that a recorded absence stops the request happening at all, that only a
/// definitive answer is recorded, and that a transient failure records nothing.
///
/// `load_from_registry` calls into here on every load, not only when the model is
/// missing, and `download_file` short-circuits on files that exist. An optional
/// file that legitimately does not exist therefore had nothing to short-circuit
/// on and was re-requested every time, at a cost of twenty seconds of retry
/// backoff per load. The marker is what turns that into a one-off.
#[cfg(not(target_arch = "wasm32"))]
async fn sentence_bert_config_with<F, Fut>(model_dir: &Path, config_url: &str, fetch: F)
where
    F: FnOnce(String) -> Fut,
    Fut: std::future::Future<Output = Result<()>>,
{
    let Some(base) = config_url.strip_suffix("config.json") else {
        return;
    };

    let marker = model_dir.join(SENTENCE_BERT_ABSENT_MARKER);
    if marker.exists() {
        return;
    }

    // Deliberately not fatal: absence is the normal case for anything that is not
    // a sentence-transformers export, and a network blip here must not fail a
    // download whose required files already succeeded.
    match fetch(format!("{base}{SENTENCE_BERT_CONFIG}")).await {
        Ok(()) => {}
        Err(e) if is_permanent(&e) => {
            log::debug!("no {SENTENCE_BERT_CONFIG} for this model ({e})");
            // Only a definitive answer is recorded. A timeout, a 5xx, a 429 or a
            // dropped connection leaves no marker, so a model that does have the
            // file still picks it up on a later load rather than being written off
            // because the CDN had a bad minute.
            let _ = tokio::fs::write(&marker, b"the server returned 4xx for this file\n").await;
        }
        Err(e) => log::debug!("could not fetch {SENTENCE_BERT_CONFIG} ({e})"),
    }
}

#[cfg(target_arch = "wasm32")]
#[allow(
    dead_code,
    reason = "the caller is native-only; this keeps the signature honest"
)]
async fn download_sentence_bert_config(_model_dir: &Path, _config_url: &str) {}

/// A download that failed with a status the server will keep returning.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
struct PermanentIfClientError {
    filename: String,
    status: reqwest::StatusCode,
}

#[cfg(not(target_arch = "wasm32"))]
impl std::fmt::Display for PermanentIfClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Failed to download {}: HTTP {}",
            self.filename, self.status
        )
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl std::error::Error for PermanentIfClientError {}

/// Whether re-sending the same request could ever give a different answer.
///
/// Most 4xx says the request itself is wrong: the file is not there, or we may
/// not have it. 5xx and transport errors are worth retrying, which is what the
/// backoff was written for.
///
/// Two 4xx codes are the exception and they matter more than the rule. 429 is
/// rate limiting and 408 is a server-side timeout: both mean "ask again later",
/// and both come back from a busy CDN under exactly the conditions a retry loop
/// exists for. Treating them as final would abandon a download that was about to
/// succeed, and for an optional file it would write the "absent" marker for one
/// the server actually has, making a transient hiccup permanent.
#[cfg(not(target_arch = "wasm32"))]
fn is_permanent(e: &anyhow::Error) -> bool {
    e.downcast_ref::<PermanentIfClientError>().is_some_and(|p| {
        p.status.is_client_error()
            && p.status != reqwest::StatusCode::TOO_MANY_REQUESTS
            && p.status != reqwest::StatusCode::REQUEST_TIMEOUT
    })
}

#[cfg(not(target_arch = "wasm32"))]
async fn download_file(model_dir: &Path, filename: &str, url: &str, quiet: bool) -> Result<()> {
    download_file_with(model_dir, filename, url, quiet, &RETRY).await
}

/// As `download_file`, with the retry schedule supplied.
///
/// Only tests pass anything but [`RETRY`]; the parameter exists so the loop can be
/// exercised without waiting out twenty seconds of real backoff.
#[cfg(not(target_arch = "wasm32"))]
async fn download_file_with(
    model_dir: &Path,
    filename: &str,
    url: &str,
    quiet: bool,
    retry: &RetryPolicy,
) -> Result<()> {
    let local_path = model_dir.join(filename);
    if local_path.exists() {
        return Ok(());
    }

    // Downloaded to a .part file and renamed once complete. `local_path.exists()`
    // above is the only "already downloaded" check there is, so a file appearing
    // at the final name before it is whole would be taken for a finished
    // download on the next run and loaded as a truncated tensor.
    let part_path = local_path.with_extension(format!(
        "{}.part",
        local_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
    ));

    // Multi-gigabyte shards over a CDN drop mid-transfer often enough that a
    // single attempt is not a download strategy: this one died at 38% of 4.9GB.
    // Each retry resumes from what is already on disk instead of starting over.
    // Timeouts, because the default is none: a connection that stalls rather than
    // drops would hang here forever, and the retry loop would never run. No total
    // timeout, since a legitimate multi-gigabyte shard can take many minutes; the
    // read timeout is what distinguishes slow from dead.
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        .read_timeout(std::time::Duration::from_secs(60))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    for attempt in 1..=retry.attempts {
        match download_to_part(&client, url, filename, &part_path, quiet).await {
            Ok(()) => break,
            // A 4xx is the server's final answer. Retrying it cannot change the
            // outcome and costs 20 seconds of backoff to learn nothing: a missing
            // optional file used to spend exactly that on every single load.
            Err(e) if is_permanent(&e) => return Err(e),
            Err(e) if attempt < retry.attempts => {
                if !quiet {
                    eprintln!(
                        "    {filename} interrupted ({e}); retry {attempt}/{}",
                        retry.attempts
                    );
                }
                tokio::time::sleep(retry.backoff(attempt)).await;
            }
            Err(e) => return Err(e),
        }
    }

    tokio::fs::rename(&part_path, &local_path).await?;
    Ok(())
}

/// How hard to retry a download, and how long to wait between attempts.
///
/// A policy rather than two constants inline, because the schedule is the
/// expensive part: five attempts at 2, 4, 6 and 8 seconds is twenty seconds spent
/// before giving up, which is what made a single missing optional file cost
/// twenty seconds on every model load. Naming it also lets a test drive the loop
/// in milliseconds instead of waiting out the real backoff.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, Copy)]
struct RetryPolicy {
    attempts: u32,
    /// Multiplied by the attempt number, so waits grow linearly.
    step: std::time::Duration,
}

#[cfg(not(target_arch = "wasm32"))]
impl RetryPolicy {
    fn backoff(&self, attempt: u32) -> std::time::Duration {
        self.step * attempt
    }
}

/// Multi-gigabyte shards over a CDN drop mid-transfer often enough that a single
/// attempt is not a download strategy: one died at 38% of 4.9GB. Each retry
/// resumes from what is already on disk rather than starting over.
#[cfg(not(target_arch = "wasm32"))]
static RETRY: RetryPolicy = RetryPolicy {
    attempts: 5,
    step: std::time::Duration::from_secs(2),
};

/// Fetches `url` into `part_path`, continuing from whatever is already there.
#[cfg(not(target_arch = "wasm32"))]
async fn download_to_part(
    client: &reqwest::Client,
    url: &str,
    filename: &str,
    part_path: &Path,
    quiet: bool,
) -> Result<()> {
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;

    let mut have = tokio::fs::metadata(part_path)
        .await
        .map(|m| m.len())
        .unwrap_or(0);

    let mut req = client.get(url);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        req = req.header("Authorization", format!("Bearer {}", token));
    }
    if have > 0 {
        req = req.header("Range", format!("bytes={have}-"));
    }

    let response = req.send().await?;
    if !response.status().is_success() {
        let status = response.status();
        return Err(anyhow!(PermanentIfClientError {
            filename: filename.to_string(),
            status,
        }));
    }

    // A server that ignores Range replies 200 with the whole file, so what is on
    // disk is not a prefix of what is arriving and must be discarded.
    if have > 0 && response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        have = 0;
    }

    // content_length() describes this response, which on a resumed request is
    // only the remaining bytes.
    let total = response.content_length().map(|len| len + have);

    let mut file = if have > 0 {
        tokio::fs::OpenOptions::new()
            .append(true)
            .open(part_path)
            .await?
    } else {
        tokio::fs::File::create(part_path).await?
    };

    let mut stream = response.bytes_stream();
    let mut downloaded = have;
    let mut last_report = have;

    // Streamed rather than collected: a 7B model ships ~5GB shards, and
    // buffering a whole shard in memory before writing it needed as much free
    // RAM as the shard was large.
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;

        // Without this the process looks hung for minutes at a time.
        if !quiet && downloaded - last_report >= 64 * 1024 * 1024 {
            last_report = downloaded;
            match total {
                Some(total) if total > 0 => eprintln!(
                    "    {} {:.0}% ({:.1} / {:.1} GB)",
                    filename,
                    (downloaded as f64 / total as f64) * 100.0,
                    downloaded as f64 / 1e9,
                    total as f64 / 1e9,
                ),
                _ => eprintln!("    {} {:.1} GB", filename, downloaded as f64 / 1e9),
            }
        }
    }

    file.flush().await?;

    // The .part file is left in place on purpose: the next attempt resumes from
    // it rather than re-fetching the whole shard.
    if let Some(total) = total
        && downloaded != total
    {
        return Err(anyhow!(
            "incomplete download of {}: got {} of {} bytes",
            filename,
            downloaded,
            total
        ));
    }

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
async fn download_sharded_weights(model_dir: &Path, index_url: &str, quiet: bool) -> Result<()> {
    download_file(model_dir, "model.safetensors.index.json", index_url, quiet).await?;

    // Parse index
    let index_path = model_dir.join("model.safetensors.index.json");
    let content = tokio::fs::read_to_string(index_path).await?;
    let json: serde_json::Value = serde_json::from_str(&content)?;

    let weight_map = json["weight_map"]
        .as_object()
        .ok_or_else(|| anyhow!("Invalid index.json"))?;

    let mut shards: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    shards.sort();
    shards.dedup();

    // Download shards
    let base_url = index_url.rsplit_once('/').unwrap().0;

    for (i, shard) in shards.iter().enumerate() {
        let url = format!("{}/{}", base_url, shard);
        if !quiet {
            eprintln!("  Processing shard {}/{}...", i + 1, shards.len());
        }
        download_file(model_dir, shard, &url, quiet).await?;
    }

    Ok(())
}

/// Returns the default cache directory for Kjarni models.
#[cfg(not(target_arch = "wasm32"))]
pub fn get_default_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("KJARNI_CACHE_DIR") {
        PathBuf::from(dir)
    } else {
        dirs::cache_dir()
            .expect("No cache directory found on system")
            .join("kjarni")
    }
}

/// Returns the default cache directory for Kjarni models.
///
/// There is no filesystem cache dir on wasm32; callers there use
/// in-memory or browser-provided storage instead.
#[cfg(target_arch = "wasm32")]
pub fn get_default_cache_dir() -> PathBuf {
    PathBuf::from("/kjarni-cache")
}

/// Formats parameter count in human-readable form.
pub fn format_params(millions: usize) -> String {
    if millions >= 1000 {
        format!("{:.1}B", millions as f64 / 1000.0)
    } else {
        format!("{}M", millions)
    }
}

/// Formats file size in human-readable form.
pub fn format_size(mb: usize) -> String {
    if mb >= 1000 {
        format!("{:.1} GB", mb as f64 / 1000.0)
    } else {
        format!("{} MB", mb)
    }
}

// /get model cache dir ~/.cache/kjarni/<model_dir>
pub fn model_cache_dir(model_dir: &str) -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("kjarni")
        .join(model_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole of the download fix. A 4xx is the server's final answer, so
    /// retrying it cannot change the outcome and costs 20 seconds of backoff to
    /// learn nothing: a missing optional file used to spend exactly that on every
    /// single load. A 5xx or a dropped connection is worth retrying, which is what
    /// the backoff was written for, so misclassifying either way is expensive.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn only_client_errors_are_treated_as_permanent() {
        let permanent = [
            reqwest::StatusCode::NOT_FOUND,
            reqwest::StatusCode::FORBIDDEN,
            reqwest::StatusCode::UNAUTHORIZED,
            reqwest::StatusCode::GONE,
        ];
        for status in permanent {
            let e = anyhow!(PermanentIfClientError {
                filename: "sentence_bert_config.json".to_string(),
                status,
            });
            assert!(is_permanent(&e), "{status} should not be retried");
        }

        // Retryable: the server is having a bad minute, not answering definitively.
        let transient = [
            reqwest::StatusCode::INTERNAL_SERVER_ERROR,
            reqwest::StatusCode::BAD_GATEWAY,
            reqwest::StatusCode::SERVICE_UNAVAILABLE,
            reqwest::StatusCode::GATEWAY_TIMEOUT,
            // These two are 4xx but mean "ask again later", not "no". Marking
            // either permanent would abandon a download that was about to
            // succeed, and would write an "absent" marker for a file that exists.
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            reqwest::StatusCode::REQUEST_TIMEOUT,
        ];
        for status in transient {
            let e = anyhow!(PermanentIfClientError {
                filename: "model.safetensors".to_string(),
                status,
            });
            assert!(!is_permanent(&e), "{status} should still be retried");
        }
    }

    /// A transport failure carries no status at all and must not be mistaken for a
    /// definitive answer, or a flaky connection would permanently mark a file
    /// absent that the server has.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn errors_without_a_status_are_retryable() {
        let e = anyhow!("connection reset by peer");
        assert!(!is_permanent(&e));
        let e = anyhow!(std::io::Error::other("timed out"));
        assert!(!is_permanent(&e));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn a_permanent_error_names_the_file_and_status() {
        let e = PermanentIfClientError {
            filename: "sentence_bert_config.json".to_string(),
            status: reqwest::StatusCode::NOT_FOUND,
        };
        let msg = e.to_string();
        assert!(msg.contains("sentence_bert_config.json"), "{msg}");
        assert!(msg.contains("404"), "{msg}");
    }

    /// Renaming these was a deliberate consistency fix, and the aliases are what
    /// keeps every script, blog post and README that used the old names working.
    /// Dropping one is a silent break for anyone who wrote it down.
    #[test]
    fn former_cli_names_still_resolve() {
        for (old, expected) in [
            ("qwen2.5-1.5b", ModelType::Qwen2_5_1_5B_Instruct),
            ("phi3.5-mini", ModelType::Phi3_5_Mini_Instruct),
            ("mistral-7b", ModelType::Mistral7B_v0_3_Instruct),
        ] {
            assert_eq!(
                ModelType::from_cli_name(old),
                Some(expected),
                "the former name {old} must keep resolving"
            );
        }
    }

    /// Every `_Instruct` variant should say so in its CLI name. Three did not,
    /// which is what the rename fixed; this stops the next one drifting.
    #[test]
    fn instruct_variants_are_named_consistently() {
        for m in ModelType::all() {
            let variant = format!("{m:?}");
            if variant.ends_with("_Instruct") {
                assert!(
                    m.cli_name().contains("instruct"),
                    "{variant} is an instruct model but its CLI name is {}",
                    m.cli_name()
                );
            }
        }
    }

    /// A name that resolves to nothing is a model nobody can load.
    #[test]
    fn every_cli_name_round_trips() {
        for m in ModelType::all() {
            assert_eq!(
                ModelType::from_cli_name(m.cli_name()),
                Some(m),
                "{} does not resolve back to itself",
                m.cli_name()
            );
        }
    }

    /// CLI names are how a user addresses a model; two models sharing one means
    /// whichever `ModelType::iter()` yields first wins, silently.
    #[test]
    fn cli_names_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for m in ModelType::all() {
            assert!(
                seen.insert(m.cli_name()),
                "duplicate CLI name: {}",
                m.cli_name()
            );
        }
    }

    /// Text models need a vocabulary; a vision tower does not, which is why
    /// `tokenizer_url` became optional. Anything that is not an image model
    /// dropping its tokenizer would fail much later, at load.
    #[test]
    fn only_image_models_may_omit_a_tokenizer() {
        for m in ModelType::all() {
            let info = m.info();
            if info.paths.tokenizer_url.is_none() {
                assert_eq!(
                    info.task,
                    ModelTask::ImageEmbedding,
                    "{} has no tokenizer but is not an image model",
                    m.cli_name()
                );
            }
        }
    }

    /// The preprocessor carries resize, crop, mean and std. An image model without
    /// it would fall back to hardcoded defaults, which is the silent-wrongness
    /// this field exists to prevent.
    #[test]
    fn image_models_carry_a_preprocessor_config() {
        for m in ModelType::all() {
            let info = m.info();
            if info.task == ModelTask::ImageEmbedding {
                assert!(
                    info.paths.preprocessor_url.is_some(),
                    "{} is an image model with no preprocessor_config.json",
                    m.cli_name()
                );
            }
        }
    }

    /// Every registry entry needs somewhere to download from.
    #[test]
    fn every_model_has_weights_and_a_config() {
        for m in ModelType::all() {
            let info = m.info();
            let name = m.cli_name();
            assert!(
                !info.paths.weights_url.is_empty(),
                "{name} has no weights URL"
            );
            assert!(
                !info.paths.config_url.is_empty(),
                "{name} has no config URL"
            );
            assert!(
                info.paths.config_url.ends_with("config.json"),
                "{name}: the sentence_bert_config URL is derived by stripping \
                 'config.json' from this, so it has to end with it"
            );
        }
    }

    // ── the optional-config marker ───────────────────────────────────────
    //
    // The fetch is injected, so these assert on what the decision does rather than
    // on a network. The two that matter most are negative: that a recorded absence
    // stops the request happening at all, and that a transient failure records
    // nothing.

    #[cfg(not(target_arch = "wasm32"))]
    mod marker {
        use super::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        const CONFIG_URL: &str = "https://example.invalid/model/config.json";

        fn err(status: reqwest::StatusCode) -> anyhow::Error {
            anyhow!(PermanentIfClientError {
                filename: SENTENCE_BERT_CONFIG.to_string(),
                status,
            })
        }

        /// The performance fix itself: once absence is recorded, no request is
        /// made. Asserting the call count is the only way to state this, since the
        /// function returns nothing either way.
        #[tokio::test]
        async fn a_recorded_absence_stops_the_request_happening() {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join(SENTENCE_BERT_ABSENT_MARKER), b"x").unwrap();

            let calls = AtomicUsize::new(0);
            sentence_bert_config_with(dir.path(), CONFIG_URL, |_url| {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Ok(()) }
            })
            .await;

            assert_eq!(
                calls.load(Ordering::SeqCst),
                0,
                "a model already known to lack the file must not be asked again"
            );
        }

        /// A definitive 404 is recorded, so the next load short-circuits.
        #[tokio::test]
        async fn a_404_is_recorded_so_it_is_asked_only_once() {
            let dir = tempfile::tempdir().unwrap();
            sentence_bert_config_with(dir.path(), CONFIG_URL, |_url| async {
                Err(err(reqwest::StatusCode::NOT_FOUND))
            })
            .await;

            assert!(
                dir.path().join(SENTENCE_BERT_ABSENT_MARKER).exists(),
                "a 404 is final and should be remembered"
            );
        }

        /// The asymmetry, and the one most expensive to get wrong: a server having
        /// a bad minute must not permanently mark a file absent that it has.
        #[tokio::test]
        async fn a_transient_failure_records_nothing() {
            for status in [
                reqwest::StatusCode::INTERNAL_SERVER_ERROR,
                reqwest::StatusCode::SERVICE_UNAVAILABLE,
                reqwest::StatusCode::TOO_MANY_REQUESTS,
                reqwest::StatusCode::REQUEST_TIMEOUT,
            ] {
                let dir = tempfile::tempdir().unwrap();
                sentence_bert_config_with(dir.path(), CONFIG_URL, |_url| async move {
                    Err(err(status))
                })
                .await;

                assert!(
                    !dir.path().join(SENTENCE_BERT_ABSENT_MARKER).exists(),
                    "{status} is not a final answer and must not be recorded"
                );
            }
        }

        /// A transport error carries no status at all and is likewise not final.
        #[tokio::test]
        async fn a_dropped_connection_records_nothing() {
            let dir = tempfile::tempdir().unwrap();
            sentence_bert_config_with(dir.path(), CONFIG_URL, |_url| async {
                Err(anyhow!("connection reset by peer"))
            })
            .await;
            assert!(!dir.path().join(SENTENCE_BERT_ABSENT_MARKER).exists());
        }

        /// A model that has the file leaves no marker, or it would stop being
        /// refreshed if it were ever deleted.
        #[tokio::test]
        async fn a_successful_fetch_records_nothing() {
            let dir = tempfile::tempdir().unwrap();
            sentence_bert_config_with(dir.path(), CONFIG_URL, |_url| async { Ok(()) }).await;
            assert!(!dir.path().join(SENTENCE_BERT_ABSENT_MARKER).exists());
        }

        /// The URL is derived by stripping `config.json`, so an entry whose config
        /// URL is shaped differently must be skipped rather than fetched from a
        /// nonsense address.
        #[tokio::test]
        async fn an_unexpected_config_url_is_left_alone() {
            let dir = tempfile::tempdir().unwrap();
            let calls = AtomicUsize::new(0);
            sentence_bert_config_with(dir.path(), "https://example.invalid/cfg.yaml", |_u| {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Ok(()) }
            })
            .await;
            assert_eq!(calls.load(Ordering::SeqCst), 0);
        }

        /// The URL actually requested sits beside the config it was derived from.
        #[tokio::test]
        async fn the_fetch_url_sits_beside_the_config() {
            let dir = tempfile::tempdir().unwrap();
            let seen = std::sync::Mutex::new(String::new());
            sentence_bert_config_with(dir.path(), CONFIG_URL, |url| {
                *seen.lock().unwrap() = url;
                async { Ok(()) }
            })
            .await;
            assert_eq!(
                seen.into_inner().unwrap(),
                "https://example.invalid/model/sentence_bert_config.json"
            );
        }
    }

    // ── the retry loop ───────────────────────────────────────────────────
    //
    // Against a real socket, because this is about what `download_file` does with
    // reqwest rather than about a decision function. What is asserted is the
    // request count: the 21-second bug was a 404 being asked five times, and only
    // counting requests can state that it is now asked once.

    #[cfg(not(target_arch = "wasm32"))]
    mod retry {
        use super::*;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::io::AsyncWriteExt;
        use tokio::net::TcpListener;

        /// A server that answers everything with `status` and counts requests.
        ///
        /// Returns the port and the counter. Deliberately not a real HTTP stack:
        /// it writes a fixed response and closes, which is all the client needs to
        /// classify the answer.
        async fn counting_server(status: &'static str) -> (u16, Arc<AtomicUsize>) {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
            let port = listener.local_addr().expect("addr").port();
            let hits = Arc::new(AtomicUsize::new(0));

            let counter = hits.clone();
            tokio::spawn(async move {
                loop {
                    let Ok((mut sock, _)) = listener.accept().await else {
                        return;
                    };
                    counter.fetch_add(1, Ordering::SeqCst);
                    let body = format!(
                        "HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    );
                    let _ = sock.write_all(body.as_bytes()).await;
                    let _ = sock.shutdown().await;
                }
            });
            (port, hits)
        }

        /// Milliseconds rather than the real 2/4/6/8 second schedule, so a test
        /// that exercises every attempt still finishes instantly.
        const FAST: RetryPolicy = RetryPolicy {
            attempts: 5,
            step: std::time::Duration::from_millis(1),
        };

        async fn fetch(status: &'static str, retry: &RetryPolicy) -> (usize, bool) {
            let (port, hits) = counting_server(status).await;
            let dir = tempfile::tempdir().unwrap();
            let url = format!("http://127.0.0.1:{port}/sentence_bert_config.json");
            let ok = download_file_with(dir.path(), "f.json", &url, true, retry)
                .await
                .is_ok();
            (hits.load(Ordering::SeqCst), ok)
        }

        /// The bug. A 404 was retried five times with 2+4+6+8 seconds of backoff,
        /// which is where the twenty seconds went. It must be asked exactly once.
        #[tokio::test]
        async fn a_404_is_requested_exactly_once() {
            let (hits, ok) = fetch("404 Not Found", &FAST).await;
            assert_eq!(hits, 1, "a 404 is final and must not be retried");
            assert!(!ok, "the call still fails, it just fails immediately");
        }

        #[tokio::test]
        async fn other_client_errors_are_also_asked_once() {
            for status in ["403 Forbidden", "401 Unauthorized", "410 Gone"] {
                let (hits, _) = fetch(status, &FAST).await;
                assert_eq!(hits, 1, "{status} should not be retried");
            }
        }

        /// The other half: a server error is worth asking again, and the loop must
        /// still use its whole budget.
        #[tokio::test]
        async fn a_500_uses_every_attempt() {
            let (hits, ok) = fetch("500 Internal Server Error", &FAST).await;
            assert_eq!(hits, FAST.attempts as usize, "a 5xx should be retried");
            assert!(!ok);
        }

        /// Rate limiting is a 4xx that means "later", not "no". This is the case
        /// that was wrong when the fix first landed: treating it as final would
        /// abandon a download that was about to succeed.
        #[tokio::test]
        async fn rate_limiting_is_retried_despite_being_a_4xx() {
            let (hits, _) = fetch("429 Too Many Requests", &FAST).await;
            assert_eq!(hits, FAST.attempts as usize, "429 must keep retrying");

            let (hits, _) = fetch("408 Request Timeout", &FAST).await;
            assert_eq!(hits, FAST.attempts as usize, "408 must keep retrying");
        }

        /// An already-present file short-circuits before any socket is opened,
        /// which is what keeps a warm cache from touching the network at all.
        #[tokio::test]
        async fn an_existing_file_is_never_requested() {
            let (port, hits) = counting_server("404 Not Found").await;
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join("f.json"), b"{}").unwrap();

            let url = format!("http://127.0.0.1:{port}/f.json");
            let ok = download_file_with(dir.path(), "f.json", &url, true, &FAST)
                .await
                .is_ok();

            assert!(ok);
            assert_eq!(
                hits.load(Ordering::SeqCst),
                0,
                "a cached file needs no request"
            );
        }

        /// The schedule itself, since it is now a value rather than a literal.
        #[test]
        fn backoff_grows_with_each_attempt() {
            assert_eq!(RETRY.backoff(1), std::time::Duration::from_secs(2));
            assert_eq!(RETRY.backoff(4), std::time::Duration::from_secs(8));
            // 2+4+6+8: the twenty seconds a missing optional file used to cost on
            // every single load.
            let total: std::time::Duration = (1..RETRY.attempts).map(|a| RETRY.backoff(a)).sum();
            assert_eq!(total, std::time::Duration::from_secs(20));
        }
    }

    /// CLIP specifically, since it is the first image model and the first entry to
    /// use the two new `ModelPaths` fields.
    #[test]
    fn the_clip_entry_is_complete() {
        let info = ModelType::ClipVitBase32.info();
        assert_eq!(info.architecture, ModelArchitecture::Clip);
        assert_eq!(info.task, ModelTask::ImageEmbedding);
        assert!(info.paths.preprocessor_url.is_some());
        assert!(
            info.paths.tokenizer_url.is_some(),
            "CLIP has a text tower, so it does need a vocabulary"
        );
        assert!(
            info.paths.weights_url.ends_with(".safetensors"),
            "the loader cannot read a pickle; openai/clip-vit-base-patch32 ships \
             only pytorch_model.bin, which is why this points at LAION's export"
        );
        assert_eq!(ModelType::ClipVitBase32.cli_name(), "clip-vit-base-32");
    }
}
