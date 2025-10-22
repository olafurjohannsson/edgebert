//! WASM bindings for BERT models

use wasm_bindgen::prelude::*;
use crate::{ModelType, BertConfig};
use crate::model::{BertBiEncoder, BertCrossEncoder};
use crate::weights::ModelWeights;
use crate::tokenizer::wasm::WordPieceTokenizer;

#[wasm_bindgen]
pub enum WasmModelType {
    MiniLML6V2BiEncoder,
    MiniLML6V2CrossEncoder,
}

impl From<WasmModelType> for ModelType {
    fn from(wasm_type: WasmModelType) -> Self {
        match wasm_type {
            WasmModelType::MiniLML6V2BiEncoder => ModelType::MiniLML6V2BiEncoder,
            WasmModelType::MiniLML6V2CrossEncoder => ModelType::MiniLML6V2CrossEncoder,
        }
    }
}

#[wasm_bindgen]
pub struct WasmBertBiEncoder {
    inner: BertBiEncoder,
}

#[wasm_bindgen]
pub struct WasmBertCrossEncoder {
    inner: BertCrossEncoder,
}

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
impl WasmBertBiEncoder {
    #[wasm_bindgen(constructor)]
    pub fn new(
        weights_data: &[u8],
        config_json: &str,
        tokenizer_json: &str,
    ) -> Result<WasmBertBiEncoder, JsValue> {
        let weights = ModelWeights::from_bytes(weights_data, config_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        let tokenizer = WordPieceTokenizer::from_json_str(tokenizer_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        let config: BertConfig = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        let model = BertBiEncoder::from_weights(weights, tokenizer, config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        Ok(WasmBertBiEncoder { inner: model })
    }
    
    #[wasm_bindgen]
    pub fn encode(&mut self, texts: Vec<String>, normalize: bool) -> Result<Vec<f32>, JsValue> {
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = self.inner
            .encode(text_refs, normalize)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        // Flatten embeddings for WASM return
        let vector: Vec<f32> = embeddings.into_iter().flatten().collect();
        Ok(vector)
    }
}

#[wasm_bindgen]
impl WasmBertCrossEncoder {
    #[wasm_bindgen(constructor)]
    pub fn new(
        weights_data: &[u8],
        config_json: &str,
        tokenizer_json: &str,
    ) -> Result<WasmBertCrossEncoder, JsValue> {
        let weights = ModelWeights::from_bytes(weights_data, config_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        let tokenizer = WordPieceTokenizer::from_json_str(tokenizer_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        let config: BertConfig = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        let model = BertCrossEncoder::from_weights(weights, tokenizer, config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        Ok(WasmBertCrossEncoder { inner: model })
    }
    
    #[wasm_bindgen]
    pub fn score(&mut self, query: String, document: String) -> Result<f32, JsValue> {
        self.inner
            .score_pair(&query, &document)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
    
    #[wasm_bindgen]
    pub fn score_batch(
        &mut self,
        queries: Vec<String>,
        documents: Vec<String>,
    ) -> Result<Vec<f32>, JsValue> {
        if queries.len() != documents.len() {
            return Err(JsValue::from_str("queries and documents must have same length"));
        }
        
        let pairs: Vec<(&str, &str)> = queries
            .iter()
            .zip(documents.iter())
            .map(|(q, d)| (q.as_str(), d.as_str()))
            .collect();
        
        self.inner
            .score_batch(pairs)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

// Helper functions for loading models from URLs in WASM
#[wasm_bindgen]
pub async fn fetch_bytes(url: &str) -> Result<Vec<u8>, JsValue> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Response, Window, WorkerGlobalScope};
    
    let global = js_sys::global();
    
    let resp_js = if let Ok(win) = global.clone().dyn_into::<Window>() {
        JsFuture::from(win.fetch_with_str(url)).await?
    } else if let Ok(worker) = global.clone().dyn_into::<WorkerGlobalScope>() {
        JsFuture::from(worker.fetch_with_str(url)).await?
    } else {
        return Err(JsValue::from_str("Unknown global scope"));
    };
    
    let resp: Response = resp_js.dyn_into()?;
    let array_buffer = JsFuture::from(resp.array_buffer()?).await?;
    
    Ok(js_sys::Uint8Array::new(&array_buffer).to_vec())
}

#[wasm_bindgen]
pub async fn fetch_text(url: &str) -> Result<String, JsValue> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Response, Window, WorkerGlobalScope};
    
    let global = js_sys::global();
    
    let resp_js = if let Ok(win) = global.clone().dyn_into::<Window>() {
        JsFuture::from(win.fetch_with_str(url)).await?
    } else if let Ok(worker) = global.clone().dyn_into::<WorkerGlobalScope>() {
        JsFuture::from(worker.fetch_with_str(url)).await?
    } else {
        return Err(JsValue::from_str("Unknown global scope"));
    };
    
    let resp: Response = resp_js.dyn_into()?;
    let text_js = JsFuture::from(resp.text()?).await?;
    
    text_js.as_string()
        .ok_or_else(|| JsValue::from_str("Failed to convert text"))
}