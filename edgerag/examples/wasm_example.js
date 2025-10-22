// WASM example for browser/Node.js

import init, { WasmGPT, WasmBertBiEncoder, EdgeRAG } from './pkg/edgegpt.js';

async function main() {
    await init();
    
    console.log("Loading models...");
    
    // Load GPT model
    const gptModel = await WasmGPT.from_type(WasmModelType.DistilGPT2);
    
    // Load BERT model
    const bertModel = await WasmBertBiEncoder.from_type(WasmModelType.MiniLML6V2BiEncoder);
    
    // Initialize RAG
    const rag = new EdgeRAG();
    
    // Sample documents
    const documents = [
        "JavaScript is a programming language commonly used for web development.",
        "React is a JavaScript library for building user interfaces.",
        "WebAssembly allows running code written in languages like Rust in browsers.",
    ];
    
    // Generate embeddings
    console.log("Generating embeddings...");
    const embeddings = [];
    for (const doc of documents) {
        const emb = bertModel.encode([doc], true);
        embeddings.push(emb);
    }
    
    // Load into RAG system
    rag.loadVectors(JSON.stringify({ embeddings, dimension: embeddings[0].length }));
    rag.loadChunks(JSON.stringify(documents.map((doc, i) => ({
        id: `doc_${i}`,
        content: { type: "text", text: { text: doc } },
        metadata: { page_number: i }
    }))));
    
    // Query
    const query = "What is React?";
    console.log(`Query: ${query}`);
    
    const queryEmb = bertModel.encode([query], true);
    const results = rag.search(queryEmb, query, 2);
    
    console.log("Search results:", results);
    
    // Generate answer with streaming
    const context = results.map(r => r.chunk.content.text.text).join(" ");
    const prompt = `Context: ${context}\n\nQuestion: ${query}\nAnswer:`;
    
    console.log("Generating answer...");
    let generatedText = "";
    
    // Note: In real WASM implementation, you'd need to modify this
    // to support streaming callbacks
    const answer = gptModel.generate(
        prompt,
        50,  // max_tokens
        0.7, // temperature
        40,  // top_k
        0.9, // top_p
        WasmSamplingStrategy.TopKTopP
    );
    
    console.log("Answer:", answer);
}

main().catch(console.error);