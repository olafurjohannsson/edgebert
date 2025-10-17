export * from './edgebert.js';
import init, { WasmModel as WasmModelRaw, WasmModelType } from './edgebert.js';

const CACHE_VERSION = 'v0.3.4';
const DB_NAME = 'edgebert-cache';

async function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onerror = () => reject(req.error);
    req.onsuccess = () => resolve(req.result);
    req.onupgradeneeded = (e) => {
      e.target.result.createObjectStore('models');
    };
  });
}

async function getCached(modelType) {
  try {
    const db = await openDB();
    const key = `${CACHE_VERSION}:${modelType}`;
    return new Promise((resolve) => {
      const tx = db.transaction('models', 'readonly');
      const req = tx.objectStore('models').get(key);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => resolve(null);
    });
  } catch {
    return null;
  }
}

async function setCache(modelType, weights, config, tokenizer) {
  try {
    const db = await openDB();
    const key = `${CACHE_VERSION}:${modelType}`;
    return new Promise((resolve) => {
      const tx = db.transaction('models', 'readwrite');
      tx.objectStore('models').put({ weights, config, tokenizer }, key);
      tx.oncomplete = () => resolve();
    });
  } catch (e) {
    console.warn('Failed to cache model:', e);
  }
}

function getModelUrls(modelType) {
  switch (modelType) {
    case WasmModelType.MiniLML6V2:
      return {
        weights: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors",
        config: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json",
        tokenizer: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
      };
  }
}

// Wrapper class that adds caching
export class WasmModel extends WasmModelRaw {
  static async from_type(modelType) {
    // Check cach
    const cached = await getCached(modelType);
    if (cached) {
      console.log('EdgeBERT: Loading from cache...');
      return new WasmModel(
        new Uint8Array(cached.weights),
        cached.config,
        cached.tokenizer
      );
    }
    
    // Download files
    console.log('EdgeBERT: Downloading model...');
    const urls = getModelUrls(modelType);
    
    const [weightsResp, configResp, tokenizerResp] = await Promise.all([
      fetch(urls.weights),
      fetch(urls.config),
      fetch(urls.tokenizer),
    ]);
    
    const weights = new Uint8Array(await weightsResp.arrayBuffer());
    const config = await configResp.text();
    const tokenizer = await tokenizerResp.text();
    
    await setCache(modelType, Array.from(weights), config, tokenizer);
    return new WasmModel(weights, config, tokenizer);
  }
}

export { init, WasmModelType };
export default init;