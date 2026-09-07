// Decoder models build for wasm; the CPU decoder path and its pipeline are
// available on every target. The encoder-decoder models (bart, t5, whisper) are
// not yet: they still go through the GPU-coupled encoder_decoder pipeline.
pub mod llama;
// GPT-2 keeps GPU types in its struct fields rather than behind the trait, so it
// needs more than cfg attributes to build for wasm. It is also not a chat model, so
// it earns nothing in a browser. Left native-only.
#[cfg(not(target_arch = "wasm32"))]
pub mod bart;
#[cfg(not(target_arch = "wasm32"))]
pub mod gpt2;
pub mod mistral;
pub mod phi;
pub mod qwen;
#[cfg(not(target_arch = "wasm32"))]
pub mod t5;
#[cfg(not(target_arch = "wasm32"))]
pub mod whisper;

#[cfg(not(target_arch = "wasm32"))]
pub mod clip;
pub mod cross_encoder;
pub mod sentence_encoder;
pub mod sequence_classifier;
