//! WASM bindings for GPT models

pub mod fetch_utils;
pub mod errors;

use wasm_bindgen::prelude::*;
use crate::{ModelType, GPTConfig, GenerationConfig, SamplingStrategy};
use crate::model::distilgpt2::DistilGPT2;
use crate::weights::ModelWeights;
use crate::tokenizer::wasm::BPETokenizer;

#[wasm_bindgen]
pub enum WasmModelType {
    DistilGPT2,
    GPT2,
}

impl From<WasmModelType> for ModelType {
    fn from(wasm_type: WasmModelType) -> Self {
        match wasm_type {
            WasmModelType::DistilGPT2 => ModelType::DistilGPT2,
            WasmModelType::GPT2 => ModelType::GPT2,
        }
    }
}

#[wasm_bindgen]
pub enum WasmSamplingStrategy {
    Greedy,
    TopK,
    TopP,
    Temperature,
}

impl From<WasmSamplingStrategy> for SamplingStrategy {
    fn from(wasm_strategy: WasmSamplingStrategy) -> Self {
        match wasm_strategy {
            WasmSamplingStrategy::Greedy => SamplingStrategy::Greedy,
            WasmSamplingStrategy::TopK => SamplingStrategy::TopK,
            WasmSamplingStrategy::TopP => SamplingStrategy::TopP,
            WasmSamplingStrategy::Temperature => SamplingStrategy::Temperature,
        }
    }
}

#[wasm_bindgen]
pub struct WasmGPT {
    model: DistilGPT2,
}

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
impl WasmGPT {
    #[wasm_bindgen(constructor)]
    pub fn new(
        weights_data: &[u8],
        config_json: &str,
        tokenizer_json: &str,
    ) -> Result<WasmGPT, JsValue> {
        let weights = ModelWeights::from_bytes(weights_data, config_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        let tokenizer = BPETokenizer::from_json_str(tokenizer_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        let config: GPTConfig = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        let model = DistilGPT2::from_weights(weights, tokenizer, config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        Ok(WasmGPT { model })
    }
    
    #[wasm_bindgen]
    pub fn generate(
        &self,
        prompt: String,
        max_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
        strategy: WasmSamplingStrategy,
    ) -> Result<String, JsValue> {
        let config = GenerationConfig {
            max_new_tokens: max_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty: 1.0,
            sampling_strategy: strategy.into(),
            ..Default::default()
        };
        
        self.model
            .generate(&prompt, &config)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
    
    #[wasm_bindgen]
    pub fn get_embeddings(&self, text: String) -> Result<Vec<f32>, JsValue> {
        self.model
            .get_embeddings(&text)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

// Helper functions for loading models
#[wasm_bindgen]
pub async fn fetch_gpt_model(model_type: WasmModelType) -> Result<WasmGPT, JsValue> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Response, Window, WorkerGlobalScope};
    
    let (weights_url, config_url, tokenizer_url) = match model_type {
        WasmModelType::DistilGPT2 => (
            "https://huggingface.co/distilbert/distilgpt2/resolve/main/model.safetensors",
            "https://huggingface.co/distilbert/distilgpt2/resolve/main/config.json",
            "https://huggingface.co/distilbert/distilgpt2/resolve/main/tokenizer.json",
        ),
        WasmModelType::GPT2 => (
            "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors",
            "https://huggingface.co/openai-community/gpt2/resolve/main/config.json",
            "https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json",
        ),
    };
    
    // Fetch all files in parallel
    let global = js_sys::global();
    
    let fetch_bytes = |url: &str| async move {
        let resp_js = if let Ok(win) = global.clone().dyn_into::<Window>() {
            JsFuture::from(win.fetch_with_str(url)).await?
        } else if let Ok(worker) = global.clone().dyn_into::<WorkerGlobalScope>() {
            JsFuture::from(worker.fetch_with_str(url)).await?
        } else {
            return Err(JsValue::from_str("Unknown global scope"));
        };
        
        let resp: Response = resp_js.dyn_into()?;
        let array_buffer = JsFuture::from(resp.array_buffer()?).await?;
        Ok::<Vec<u8>, JsValue>(js_sys::Uint8Array::new(&array_buffer).to_vec())
    };
    
    let fetch_text = |url: &str| async move {
        let resp_js = if let Ok(win) = global.clone().dyn_into::<Window>() {
            JsFuture::from(win.fetch_with_str(url)).await?
        } else if let Ok(worker) = global.clone().dyn_into::<WorkerGlobalScope>() {
            JsFuture::from(worker.fetch_with_str(url)).await?
        } else {
            return Err(JsValue::from_str("Unknown global scope"));
        };
        
        let resp: Response = resp_js.dyn_into()?;
        let text_js = JsFuture::from(resp.text()?).await?;
        text_js.as_string().ok_or_else(|| JsValue::from_str("Failed to convert text"))
    };
    
    let (weights, config, tokenizer) = futures::future::try_join3(
        fetch_bytes(weights_url),
        fetch_text(config_url),
        fetch_text(tokenizer_url),
    ).await?;
    
    WasmGPT::new(&weights, &config, &tokenizer)
}