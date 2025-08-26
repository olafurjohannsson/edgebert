# EdgeBERT

**A pure Rust + WASM implementation for BERT inference with minimal dependencies**

# WIP

[![crates.io]()](https://crates.io/crates/edgebert)
[![docs.rs](https://docs.rs/edgebert/badge.svg)](https://docs.rs/edgebert)
[![Build Status](https://github.com/olafurjohannsson/edgebert/workflows/CI/badge.svg)](https://github.com/olafurjohannsson/edgebert/actions)

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
edgebert = "0.3.0"
anyhow = "1.0"
```

**`main.rs`**
```rust
use anyhow::Result;
use edgebert::{Model, ModelType};
fn main() -> Result<()> {
    let model = Model::from_pretrained(ModelType::MiniLML6V2)?;

    let texts = vec!["Hello world", "How are you"];
    let embeddings = model.encode(texts.clone(), true)?;

    for (i, embedding) in embeddings.iter().enumerate() {
        let n = embedding.len().min(10);
        println!("Text: {} == {:?}...", texts[i], &embedding[0..n]);
    }
    Ok(())
}
```

**Output:**
```
     Running `target/debug/examples/basic`
Text: Hello world == [-0.034439795, 0.030909885, 0.0066969804, 0.02608013, -0.03936993, -0.16037229, 0.06694216, -0.0065279473, -0.0474657, 0.014813968]...
Text: How are you == [-0.031447295, 0.03784213, 0.0761843, 0.045665547, -0.0012263817, -0.07488511, 0.08155286, 0.010209872, -0.11220472, 0.04075747]...
```

You can see the full example under `examples/basic.rs` - to build and run:
```bash
cargo run --example basic
```

### 2. WebAssembly
```javascript
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EdgeBERT WASM Test</title>
</head>
<body>
<script type="module">
    import init, { WasmModel, WasmModelType } from './pkg/edgebert.js';

    async function run() {
        await init();

        const model = await WasmModel.from_type(WasmModelType.MiniLML6V2);
        const texts = ["Hello world", "How are you"];
        const embeddings = model.encode(texts, true);

        console.log("First 10 values:", embeddings.slice(0, 10));
    }

    run().catch(console.error);
</script>
</body>
</html>

```

**Output:**
```
First 10 values: Float32Array(10) [-0.034439802169799805, 0.03090989589691162, 0.006696964148432016, 0.02608015574514866, -0.03936990723013878, -0.16037224233150482, 0.06694218516349792, -0.006527911406010389, -0.04746570065617561, 0.014813981018960476, buffer: ArrayBuffer(40), byteLength: 40, byteOffset: 0, length: 10, Symbol(Symbol.toStringTag): 'Float32Array']
```

You can see the full example under `examples/basic.html` - to build run `scripts/wasm-build.sh` and go into `examples/` and run a local server, `npx serve` can serve wasm.

### 3. Web Workers

You can look at `examples/worker.html` and `examples/worker.js` to see how to use web workers and web assembly, the library
handles both when window is defined, as with `basic.html` and also when it is not, web workers.

After compiling the WASM build, if you used the wasm-build.sh it should be inside examples/pkg, use npx serve
and open `localhost:3000/worker`

Clicking on generate embeddings after the model loads generates

```
Encoding texts: ["Hello world","How are you?"]
Embeddings shape: [2, 384]
'Hello world' vs 'How are you?': 0.305
First embedding norm: 1.000000
First 10 values: [-0.0344, 0.0309, 0.0067, 0.0261, -0.0394, -0.1604, 0.0669, -0.0065, -0.0475, 0.0148]
```