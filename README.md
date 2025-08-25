# EdgeBERT

**A pure Rust + WASM implementation for BERT inference with minimal dependencies**

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
