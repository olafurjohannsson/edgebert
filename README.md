# EdgeBERT

**A pure Rust + WASM implementation for BERT inference with minimal dependencies**

[![docs.rs](https://docs.rs/edgebert/badge.svg)](https://docs.rs/edgebert)
[![Rust](https://github.com/olafurjohannsson/edgebert/actions/workflows/rust.yml/badge.svg)](https://github.com/olafurjohannsson/edgebert/actions/workflows/rust.yml)

---

## Overview

EdgeBERT is a lightweight Rust implementation of BERT inference focused on native, edge computing and WASM deployment. Run sentence-transformers models anywhere without Python or large ML runtimes.

## Status
- ✅ MiniLM-L6-v2 inference working
- ✅ WASM support
- 🚧 Additional models coming soon

## Contributions

All contributions welcome, this is very early stages.

## Components

- Encoder: Run inference to turn text into embeddings
- WordPiece tokenization: A small tokenization implementation based on WordPiece
- Cross-Platform (WebAssembly and native)
- No Python or C/C++ dependencies except for optional feature OpenBLAS for ndarray vectorized matrix operations

**Use this if you need:**
- Embeddings in pure Rust without Python/C++ dependencies
- BERT in browsers or edge devices
- Offline RAG systems
- Small binary size over maximum speed

**Don't use this if you need:**
- Multiple model architectures (currently only MiniLM)
- GPU acceleration
- Production-grade performance (use ONNX Runtime instead)

## Getting Started

### 1. Native Rust Application

For server-side or desktop applications, you can use the library directly.

**`Cargo.toml`**
```toml
[dependencies]
edgebert = "0.3.4"
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
Text: Hello world == [-0.034439795, 0.030909885, 0.0066969804, 0.02608013, -0.03936993, -0.16037229, 0.06694216, -0.0065279473, -0.0474657, 0.014813968]...
Text: How are you == [-0.031447295, 0.03784213, 0.0761843, 0.045665547, -0.0012263817, -0.07488511, 0.08155286, 0.010209872, -0.11220472, 0.04075747]...
```

You can see the full example under `examples/basic.rs` - to build and run:
```bash
cargo run --release --example basic
```

### 2. WebAssembly

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build
./scripts/wasm-build.sh

# Serve
cd examples && npx serve
```

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


### 4. Comparison with PyTorch

Small example from pytorch to encode and show similarity

```python
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

texts = ["Hello world", "How are you"]

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(texts, convert_to_tensor=True)

for i, emb in enumerate(embeddings):
    print(f"Text: {texts[i]}")
    print("First 10 values:", emb[:10].tolist())
    print()

cos_sim = F.cosine_similarity(embeddings[0], embeddings[1], dim=0)
print(f"Cosine similarity ('{texts[0]}' vs '{texts[1]}'):", cos_sim.item())

```

**Output from Python:**
```
Text: Hello world == ['-0.0345', '0.0310', '0.0067', '0.0261', '-0.0394', '-0.1603', '0.0669', '-0.0064', '-0.0475', '0.0148']...
Text: How are you == ['-0.0314', '0.0378', '0.0763', '0.0457', '-0.0012', '-0.0748', '0.0816', '0.0102', '-0.1122', '0.0407']...

Cosine similarity ('Hello world' vs 'How are you'): 0.3624

```

**EdgeBERT**

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

    let dot: f32 = embeddings[0].iter().zip(&embeddings[1]).map(|(a, b)| a * b).sum();
    let norm_a: f32 = embeddings[0].iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_b: f32 = embeddings[1].iter().map(|v| v * v).sum::<f32>().sqrt();
    let cos_sim = dot / (norm_a * norm_b);

    println!("\nCosine similarity ('{}' vs '{}'): {:.4}", texts[0], texts[1], cos_sim);

    Ok(())
}
```

**Output**
```rust
Text: Hello world == [-0.034439795, 0.030909885, 0.0066969804, 0.02608013, -0.03936993, -0.16037229, 0.06694216, -0.0065279473, -0.0474657, 0.014813968]...
Text: How are you == [-0.031447295, 0.03784213, 0.0761843, 0.045665547, -0.0012263817, -0.07488511, 0.08155286, 0.010209872, -0.11220472, 0.04075747]...

Cosine similarity ('Hello world' vs 'How are you'): 0.3623

```

Cosine similarity has 0.3623 for Rust and Python 0.3624, acceptable with around 99.97% 
accuracy, tiny discrepancy because of floating point rounding differences.
