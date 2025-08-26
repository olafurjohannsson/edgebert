import init, { WasmModel, WasmModelType } from './pkg/edgebert.js';

let model = null;

self.onmessage = async function (e) {
    if (e.data.type === 'init') {
        try {
            self.postMessage({ type: 'log', message: 'Initializing WASM...' });
            await init();

            self.postMessage({ type: 'log', message: 'Loading model...' });

            model = await WasmModel.from_type(WasmModelType.MiniLML6V2);

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
            const embeddings = model.encode(e.data.texts, true);
            self.postMessage({
                type: 'result',
                embeddings: Array.from(embeddings),
            });
        } catch (error) {
            self.postMessage({ type: 'error', error: error.message });
        }
    }
};
