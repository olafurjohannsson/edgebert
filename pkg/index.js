export * from './edgebert.js';
import init, { WasmModel as WasmModelRaw, WasmModelType } from './edgebert.js';

const CACHE_VERSION = 'v0.3.4';
const DB_NAME = 'edgebert-cache';

const globalContext = typeof window !== 'undefined' ? window : self;

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
    
    const data = {
      weights: weights.buffer,
      config,
      tokenizer
    };
    
    return new Promise((resolve) => {
      const tx = db.transaction('models', 'readwrite');
      tx.objectStore('models').put(data, key);
      tx.oncomplete = () => resolve();
    });
  } catch (e) {
    console.warn('Failed to cache model:', e);
  }
}

async function fetchWithProgress(url, onProgress) {
  const response = await fetch(url);
  const contentLength = response.headers.get('content-length');
  
  if (!contentLength) {
    console.warn('No content-length header, progress unavailable');
    return await response.blob();
  }
  
  const total = parseInt(contentLength, 10);
  let loaded = 0;
  
  const reader = response.body.getReader();
  const chunks = [];
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    chunks.push(value);
    loaded += value.length;
    
    if (onProgress) {
      onProgress({ loaded, total, percent: (loaded / total) * 100 });
    }
  }
  
  return new Blob(chunks);
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

export class WasmModel extends WasmModelRaw {
  static async from_type(modelType) {
    const cached = await getCached(modelType);
    if (cached) {
      console.log('EdgeBERT: Loading from cache...');
      globalContext.dispatchEvent(new CustomEvent('edgebert-progress', {
        detail: { percent: 100, loaded: 90000000, total: 90000000, fromCache: true }
      }));
      return new WasmModel(
        new Uint8Array(cached.weights),
        cached.config,
        cached.tokenizer
      );
    }
    
    console.log('EdgeBERT: Downloading model (~90MB)...');
    
    globalContext.dispatchEvent(new CustomEvent('edgebert-progress', {
      detail: { percent: 0, loaded: 0, total: 90000000 }
    }));
    
    const urls = getModelUrls(modelType);
    
    const weightsBlob = await fetchWithProgress(urls.weights, (progress) => {
      console.log('Progress:', progress); // Debug
      globalContext.dispatchEvent(new CustomEvent('edgebert-progress', {
        detail: progress
      }));
    });
    
    const weights = new Uint8Array(await weightsBlob.arrayBuffer());
    
    // Download config and tokenizer
    const [configResp, tokenizerResp] = await Promise.all([
      fetch(urls.config),
      fetch(urls.tokenizer),
    ]);
    
    const config = await configResp.text();
    const tokenizer = await tokenizerResp.text();
    
    await setCache(modelType, weights, config, tokenizer);
    
    return new WasmModel(weights, config, tokenizer);
  }
}

export { init, WasmModelType };
export default init;