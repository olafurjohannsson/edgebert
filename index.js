import init, { WasmModel } from "./pkg/edgebert.js"; // wasm-pack output

// Cosine similarity (raw embeddings)
function cosineSim(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function runDemo() {
    const output = document.getElementById("output");

    // Initialize WASM module
    await init();

    // Load model files
    const weights = await fetch('minilm-l6-v2.safetensors').then(r => r.arrayBuffer());
    const config = await fetch('minilm-l6-v2_config.json').then(r => r.text());
    const tokenizer = await fetch('minilm-l6-v2_tokenizer.json').then(r => r.text());


    // Create model
    const model = new WasmModel(
        new Uint8Array(weights),
        config,
        tokenizer
    );
    console.log('creating mode', model)

    // Use the model
    const texts = ["Hello world", "How are you?"];
    console.log(texts)
    const embeddings = model.encode(texts);

    // embeddings is a flat Float32Array, reshape as needed
    const embeddingSize = 384; // for minilm-l6-v2
    const numTexts = texts.length;

    for (let i = 0; i < numTexts; i++) {
        const start = i * embeddingSize;
        const end = start + embeddingSize;
        const embedding = embeddings.slice(start, end);
        console.log(`Embedding for "${texts[i]}":`, embedding);
    }


    output.textContent += "Encoding: " + JSON.stringify(texts) + "\n\n";

    // Regular embeddings

    const sim = cosineSim(embeddings[0], embeddings[1]);
    output.textContent += `Cosine similarity: ${sim.toFixed(3)}\n\n`;

    // Normalized embeddings (cosine = dot product)
    const normEmb = await model.encode(texts, true);
    let dot = 0;
    for (let i = 0; i < normEmb[0].length; i++) {
        dot += normEmb[0][i] * normEmb[1][i];
    }
    output.textContent += `Cosine similarity (normalized): ${dot.toFixed(3)}\n`;
}

document.getElementById("run").addEventListener("click", runDemo);
