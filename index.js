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

    await init(); // load WASM
    console.log(WasmModel)
    const model = await WasmModel.from_pretrained("minilm-l6-v2");

    const texts = ["Hello world", "How are you?"];
    output.textContent += "Encoding: " + JSON.stringify(texts) + "\n\n";

    // Regular embeddings
    const embeddings = await model.encode(texts);
    const sim = cosineSim(embeddings[0], embeddings[1]);
    output.textContent += `Cosine similarity: ${sim.toFixed(3)}\n\n`;

    // Normalized embeddings (cosine = dot product)
    const normEmb = await model.encode_normalized(texts);
    let dot = 0;
    for (let i = 0; i < normEmb[0].length; i++) {
        dot += normEmb[0][i] * normEmb[1][i];
    }
    output.textContent += `Cosine similarity (normalized): ${dot.toFixed(3)}\n`;
}

document.getElementById("run").addEventListener("click", runDemo);
