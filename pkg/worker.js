import init, { WasmModel, WasmModelType } from './index.js'; 

let model = null;

// Listen for progress events and forward to main thread
self.addEventListener('edgebert-progress', (e) => {
  self.postMessage({ type: 'progress', detail: e.detail });
});

self.onmessage = async (e) => {
  const { type, data } = e.data;
  
  if (type === 'init') {
    try {
      await init();
      model = await WasmModel.from_type(WasmModelType.MiniLML6V2);
      self.postMessage({ type: 'ready' });
    } catch (error) {
      self.postMessage({ type: 'error', error: error.message });
    }
  }
  
  if (type === 'encode') {
    try {
      const { texts, normalize } = data;
      const embeddings = model.encode(texts, normalize);
      self.postMessage({ type: 'embeddings', embeddings });
    } catch (error) {
      self.postMessage({ type: 'error', error: error.message });
    }
  }
};