# EdgeBERT

**A pure Rust + WASM implementation for BERT inference with minimal dependencies**

# WIP

[![crates.io]()](https://crates.io/crates/edgebert)
[![docs.rs](https://docs.rs/edgebert/badge.svg)](https://docs.rs/edgebert)
[![Build Status](https://github.com/olafurjohannssson/edgebert/workflows/CI/badge.svg)](https://github.com/your-username/edgebert/actions)

---

## Overview

EdgeBERT is a lightweight, dependency-free Rust implementation of a BERT encoder and its WordPiece tokenizer. 
This project was made because of the need for a pure Rust implementation to do inference on sentence-transformers,
mainly MiniLM without pulling in big runtimes or C/C++, Python dependencies

## Components

- Encoder: Run inference to turn text into embeddings
- WordPiece tokenization: A small tokenization implementation based on WordPiece
- Cross-Platform (WebAssembly and native)
- No Python or C/C++ dependencies except for OpenBLAS for ndarray vectorized matrix operations


## Getting Started

### 1. Native Rust Application

For server-side or desktop applications, you can use the library directly.

**`Cargo.toml`**
```toml
[dependencies]
edgebert = "0.1.0" # Replace with the actual version
anyhow = "1.0"
```

###

```rust

use edgebert::Model;

fn main() -> anyhow::Result<()> {
    let model = Model::from_pretrained("minilm-l6-v2")?;
    let texts = vec!["Hello world", "How are you?"];
    let embeddings = model.encode_normalized(texts)?;
    println!("Cosine similarity: {}", embeddings[0].iter().zip(&embeddings[1]).map(|(a,b)| a*b).sum::<f32>());
    Ok(())
}
```

```javascript

import init, { WasmModel } from "./pkg/edgebert.js";

async function runDemo() {
  await init();
  const weights = await fetch('minilm-l6-v2.safetensors').then(r => r.arrayBuffer());
  const config = await fetch('minilm-l6-v2_config.json').then(r => r.text());
  const tokenizer = await fetch('minilm-l6-v2_tokenizer.json').then(r => r.text());
  const model = new WasmModel(new Uint8Array(weights), config, tokenizer);
  const texts = ["Hello world", "How are you?"];
  const embeddings = model.encode_normalized(texts);
  console.log("Cosine similarity:", embeddings[0].reduce((s, v, i) => s + v * embeddings[1][i], 0));
}
```
