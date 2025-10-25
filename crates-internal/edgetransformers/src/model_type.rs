//! model_type.rs
//! ---------------------------------------------------------------------------
//! Defines supported pretrained transformer models and their canonical
//! metadata for the inference library.
//!
//! Each [`ModelType`] provides a [`ModelInfo`] struct containing:
//! - Canonical Hugging Face URLs for model, tokenizer, and config.
//! - Architecture type (encoder, cross-encoder, decoder, encoder–decoder).
//! - Description of model purpose and strengths.
//!
//! This enables introspection (for UI / CLI tools) and validation
//! (for ensuring an encoder-only model is not used in a decoder pipeline).
//!
//! ---------------------------------------------------------------------------

/// Distinguishes the architectural type of a transformer model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelArchitecture {
    /// Encoder-only transformer (e.g., BERT, MiniLM, MPNet, DistilBERT).
    Encoder,
    /// Cross-encoder (used for reranking / pairwise scoring).
    CrossEncoder,
    /// Decoder-only (causal LM / autoregressive) transformer (e.g., GPT-2).
    Decoder,
    /// Encoder-decoder (seq2seq) model (e.g., T5, BART, MarianMT).
    EncoderDecoder,
}

/// Supported pretrained model identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelType {
    // === Sentence Embedding Models (Encoders) ===============================
    MiniLML6V2BiEncoder,
    MpnetBaseV2BiEncoder,

    // === Cross Encoders / Rerankers ========================================
    MiniLML6V2CrossEncoder,

    // === QA / Classification Models (Encoders) =============================
    DistilBertBaseCasedQA,

    // === Summarization / Seq2Seq (Encoder–Decoder) =========================
    BartLargeCnn,
    DistilBartCnn12_6,
    T5Small,
    MarianEnIs,

    // === Decoder-only / Causal LMs ========================================
    DistilGpt2,
    Gpt2Small,
    Gpt2Medium,
    Gpt2Large,
    Gpt2XL,
}

/// Canonical Hugging Face resource URLs for a model.
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub weights_url: &'static str,
    pub tokenizer_url: &'static str,
    pub config_url: &'static str,
}

/// Describes a model’s architecture, URLs, and recommended use cases.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// The model's general transformer architecture type.
    pub architecture: ModelArchitecture,
    /// Canonical download URLs for model, tokenizer, and config.
    pub paths: ModelPaths,
    /// Short summary of the model’s intended use cases and strengths.
    pub description: &'static str,
}

impl ModelType {
    /// Returns detailed metadata for this model type.
    pub fn info(&self) -> ModelInfo {
        match self {
            // === Encoders ===================================================
            ModelType::MiniLML6V2BiEncoder => ModelInfo {
                architecture: ModelArchitecture::Encoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json",
                },
                description: "Compact sentence embedding model ideal for CPU or edge inference. \
                              Excels at semantic similarity, clustering, and retrieval tasks.",
            },
            ModelType::MpnetBaseV2BiEncoder => ModelInfo {
                architecture: ModelArchitecture::Encoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/config.json",
                },
                description: "High-quality sentence embedding model. Slightly heavier than MiniLM \
                              but achieves stronger semantic performance. Recommended for GPU or high-end CPU inference.",
            },
            ModelType::DistilBertBaseCasedQA => ModelInfo {
                architecture: ModelArchitecture::Encoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/config.json",
                },
                description: "Lightweight DistilBERT fine-tuned for question answering on SQuAD. \
                              Suitable for CPU inference and low-latency QA pipelines.",
            },

            // === Cross Encoders =============================================
            ModelType::MiniLML6V2CrossEncoder => ModelInfo {
                architecture: ModelArchitecture::CrossEncoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/config.json",
                },
                description: "Compact cross-encoder trained for passage reranking (MS MARCO). \
                              Best used to rerank top candidates from a bi-encoder retrieval stage.",
            },

            // === Encoder–Decoder (Seq2Seq) ==================================
            ModelType::BartLargeCnn => ModelInfo {
                architecture: ModelArchitecture::EncoderDecoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/facebook/bart-large-cnn/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/facebook/bart-large-cnn/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/facebook/bart-large-cnn/resolve/main/config.json",
                },
                description: "Large BART model fine-tuned for news summarization. \
                              Excellent quality for abstractive summaries, requires GPU for best performance.",
            },
            ModelType::DistilBartCnn12_6 => ModelInfo {
                architecture: ModelArchitecture::EncoderDecoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/sshleifer/distilbart-cnn-12-6/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/sshleifer/distilbart-cnn-12-6/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/sshleifer/distilbart-cnn-12-6/resolve/main/config.json",
                },
                description: "Distilled version of BART-CNN. \
                              60% faster and smaller, great for summarization on limited hardware or CPU servers.",
            },
            ModelType::T5Small => ModelInfo {
                architecture: ModelArchitecture::EncoderDecoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/t5-small/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/t5-small/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/t5-small/resolve/main/config.json",
                },
                description: "Small T5 model for general-purpose seq2seq tasks like summarization, translation, or text-to-text QA. \
                              Runs efficiently on CPUs and edge devices.",
            },
            ModelType::MarianEnIs => ModelInfo {
                architecture: ModelArchitecture::EncoderDecoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/Helsinki-NLP/opus-mt-en-is/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/Helsinki-NLP/opus-mt-en-is/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/Helsinki-NLP/opus-mt-en-is/resolve/main/config.json",
                },
                description: "MarianMT model for English → Icelandic translation. \
                              Compact and accurate, suitable for on-device translation or multilingual applications.",
            },

            // === Decoder-only (Causal LMs) ==================================
            ModelType::DistilGpt2 => ModelInfo {
                architecture: ModelArchitecture::Decoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/distilgpt2/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/distilgpt2/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/distilgpt2/resolve/main/config.json",
                },
                description: "Distilled GPT-2 for lightweight text generation and chat. \
                              Great for CPU or edge inference where low latency is required.",
            },
            ModelType::Gpt2Small => ModelInfo {
                architecture: ModelArchitecture::Decoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/gpt2/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/gpt2/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/gpt2/resolve/main/config.json",
                },
                description: "GPT-2 small: good general-purpose text generator or chatbot baseline. \
                              Suitable for research, small dialogue agents, and CPU inference.",
            },
            ModelType::Gpt2Medium => ModelInfo {
                architecture: ModelArchitecture::Decoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/gpt2-medium/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/gpt2-medium/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/gpt2-medium/resolve/main/config.json",
                },
                description: "Medium GPT-2 model offering stronger coherence and context handling. \
                              Recommended for GPU inference or optimized quantized runtime.",
            },
            ModelType::Gpt2Large => ModelInfo {
                architecture: ModelArchitecture::Decoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/gpt2-large/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/gpt2-large/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/gpt2-large/resolve/main/config.json",
                },
                description: "Large GPT-2 variant for higher-quality completions and chat. \
                              Requires GPU; good middle ground between quality and runtime cost.",
            },
            ModelType::Gpt2XL => ModelInfo {
                architecture: ModelArchitecture::Decoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/gpt2-xl/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/gpt2-xl/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/gpt2-xl/resolve/main/config.json",
                },
                description: "GPT-2 XL (1.5B parameters): strong open decoder model for generation. \
                              Not suitable for edge devices; intended for high-end GPU inference.",
            },
        }
    }
}
