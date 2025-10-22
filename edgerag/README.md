# EdgeRAG

**A pure Rust + WASM client-side RAG system with hybrid search**

[![docs.rs](https://docs.rs/edgerag/badge.svg)](https://docs.rs/edgerag)
[![Rust](https://github.com/olafurjohannsson/edgerag/actions/workflows/rust.yml/badge.svg)](https://github.com/olafurjohannsson/edgerag/actions/workflows/rust.yml)

--- 

## Overview

EdgeRAG is a lightweight Rust implementation of Retrieval-Augmented Generation focused on client-side deployment. Run complete hybrid search (BM25 + semantic) in the browser without a backend.

## Status

- ✅ BM25 keyword search
- ✅ Vector semantic search
- ✅ Hybrid search (RRF fusion)
- ✅ WASM support
- ✅ Zero backend required
- 🚧 Additional ranking algorithms coming soon

## Contributions

All contributions welcome, this is very early stages.

## Components

- Hybrid Search: Combines BM25 keyword matching with semantic vector search
- Pure Client-Side: No server needed after building index
- Cross-Platform: WebAssembly and native Rust

**Use This If You Need**
- RAG without a backend
- Offline-first search applications
- Privacy-focused document search
- Fast prototyping with pre-built indexes

**Don't Use This If You Need**
- Real-time document indexing (build indexes offline)
- LLM answer generation (use separately or call an external API)
- Multi-language support (English only currently)
- Advanced reranking (use cross-encoders separately)

## Getting Started

### 1. Build Your Index (Offline)

```rust
rustuse edgerag::prelude::*;
use edgebert::{Model, ModelType};

fn main() -> Result<()> {
    // Load your documents
    let chunks = vec![
        Chunk {
            id: "doc1".into(),
            text: "Mars is the fourth planet from the Sun".into(),
            metadata: Default::default(),
        },
        // ... more chunks
    ];
    
    // Generate embeddings
    let bert = Model::from_pretrained(ModelType::MiniLML6V2)?;
    let texts: Vec<_> = chunks.iter().map(|c| c.text.as_str()).collect();
    let vectors = bert.encode(texts, true)?;
    
    // Build BM25 index
    let bm25 = Bm25Index::build(&chunks)?;
    
    // Save to files
    std::fs::write("vectors.bin", bincode::serialize(&vectors)?)?;
    std::fs::write("bm25.json", serde_json::to_string(&bm25)?)?;
    std::fs::write("chunks.json", serde_json::to_string(&chunks)?)?;
    
    Ok(())
}
```
### 2. Use in Browser (WebAssembly)
```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

# Build
```
wasm-pack build --target web --release
```

```html
html<!DOCTYPE html>
<html>
<head>
    <title>EdgeRAG Demo</title>
</head>
<body>
<script type="module">
    import init, { EdgeRAG } from './pkg/edgerag.js';
    import initBert, { WasmModel, WasmModelType } from './pkg/edgebert.js';

    async function run() {
        // Initialize WASM modules
        await Promise.all([init(), initBert()]);
        
        // Load EdgeBERT for embeddings
        const bert = await WasmModel.from_type(WasmModelType.MiniLML6V2);
        
        // Load EdgeRAG
        const rag = new EdgeRAG();
        
        // Load pre-built indexes
        const [vectors, bm25, chunks] = await Promise.all([
            fetch('/data/vectors.bin').then(r => r.arrayBuffer()),
            fetch('/data/bm25.json').then(r => r.text()),
            fetch('/data/chunks.json').then(r => r.text()),
        ]);
        
        rag.loadVectors(new Uint8Array(vectors));
        rag.loadBM25(bm25);
        rag.loadChunks(chunks);
        
        // Search
        const query = "What is Mars?";
        const embedding = bert.encode([query], true);
        const results = rag.search(embedding, query, 5);
        
        console.log("Results:", results);
    }

    run().catch(console.error);
</script>
</body>
</html>
```

## 3. Native Rust Application

```rust

rustuse edgerag::prelude::*;
use edgebert::{Model, ModelType};

fn main() -> Result<()> {
    // Load indexes
    let vectors: Vec<Vec<f32>> = bincode::deserialize(&std::fs::read("vectors.bin")?)?;
    let bm25: Bm25Index = serde_json::from_str(&std::fs::read_to_string("bm25.json")?)?;
    let chunks: Vec<Chunk> = serde_json::from_str(&std::fs::read_to_string("chunks.json")?)?;
    
    // Create RAG engine
    let rag = EdgeRAG::from_parts(vectors, bm25, chunks)?;
    
    // Load embedding model
    let bert = Model::from_pretrained(ModelType::MiniLML6V2)?;
    
    // Search
    let query = "What is Mars?";
    let embedding = bert.encode(vec![query], true)?[0].clone();
    let results = rag.search_with_embedding(&embedding, query, 5)?;
    
    for result in results {
        println!("Score: {:.4}", result.score);
        println!("Text: {}", result.chunk.text);
        println!();
    }
    
    Ok(())
}

```
### Roadmap

- Multi-language tokenization
- Streaming index loading
- Advanced filtering options
- Query expansion
- More ranking algorithms

### License
MIT

## Related Projects

![EdgeBERT - BERT inference in Rust/WASM](https://github.com/olafurjohannsson/edgebert)
