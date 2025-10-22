mod types;
mod vector;
mod bm25;
mod hybrid;

use wasm_bindgen::prelude::*;
pub use types::{Chunk, SearchResult, SearchType};
use vector::VectorStore;
use bm25::Bm25Index;

#[wasm_bindgen]
pub struct EdgeRAG {
    vectors: VectorStore,
    bm25: Bm25Index,
    chunks: Vec<Chunk>,
}

#[wasm_bindgen]
impl EdgeRAG {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        Self {
            vectors: VectorStore::default(),
            bm25: Bm25Index::new(),
            chunks: Vec::new(),
        }
    }

    #[wasm_bindgen(js_name = loadVectors)]
    pub fn load_vectors(&mut self, json: &str) -> Result<(), JsValue> {
        self.vectors = serde_json::from_str(json)
            .map_err(|e| JsValue::from_str(&format!("Load vectors failed: {}", e)))?;
        web_sys::console::log_1(&format!("Loaded {} vectors", self.vectors.embeddings.len()).into());
        Ok(())
    }

    #[wasm_bindgen(js_name = loadBM25)]
    pub fn load_bm25(&mut self, json: &str) -> Result<(), JsValue> {
        self.bm25 = serde_json::from_str(json)
            .map_err(|e| JsValue::from_str(&format!("Load BM25 failed: {}", e)))?;
        Ok(())
    }

    #[wasm_bindgen(js_name = loadChunks)]
    pub fn load_chunks(&mut self, json: &str) -> Result<(), JsValue> {
        self.chunks = serde_json::from_str(json)
            .map_err(|e| JsValue::from_str(&format!("Load chunks failed: {}", e)))?;
        web_sys::console::log_1(&format!("Loaded {} chunks", self.chunks.len()).into());
        Ok(())
    }

    #[wasm_bindgen]
    pub fn search(&self, query_embedding: Vec<f32>, query_text: &str, k: usize) -> JsValue {
        let vector_results = self.vectors.search(&query_embedding, k * 2);
        let bm25_results = self.bm25.search(query_text, k * 2);
        
        let fused = hybrid::hybrid_search(vector_results, bm25_results, k);
        
        let results: Vec<SearchResult> = fused
            .into_iter()
            .filter_map(|(idx, score)| {
                self.chunks.get(idx).map(|chunk| SearchResult {
                    score,
                    chunk: chunk.clone(),
                    search_type: SearchType::Hybrid,
                })
            })
            .collect();

        serde_wasm_bindgen::to_value(&results).unwrap()
    }
}