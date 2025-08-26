import init, { WasmModel } from './pkg/edgebert.js';

let model = null;

self.onmessage = async function(e) {
    if (e.data.type === 'init') {
        try {
            self.postMessage({ type: 'log', message: 'Initializing WASM...' });
            await init();

            self.postMessage({ type: 'log', message: 'Creating model...' });
            const { weights, config, tokenizer } = e.data;
            model = new WasmModel(new Uint8Array(weights), config, tokenizer);

            self.postMessage({ type: 'ready' });
        } catch (error) {
            self.postMessage({ type: 'error', error: error.message });
        }
    } else if (e.data.type === 'encode') {
        if (!model) {
            self.postMessage({ type: 'error', error: 'Model not initialized' });
            return;
        }

        try {
            const embeddings = model.encode(e.data.texts);
            self.postMessage({
                type: 'result',
                embeddings: Array.from(embeddings)
            });
        } catch (error) {
            self.postMessage({ type: 'error', error: error.message });
        }
    }
};