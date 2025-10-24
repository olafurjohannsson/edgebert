use anyhow::{Result, anyhow};
use edgemodels::bert::BertBiEncoder;
use edgetransformers::prelude::Device;
use edgetransformers::wgpu_context::WgpuContext;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use rand::seq::SliceRandom;

use std::time::Instant;
const USE_GPU: bool = false;
const NUM_RUNS: usize = 5;
const BATCH_SIZE: usize = 4;

/// ASYNC helper function to ensure model files are available, downloading them if necessary.
/// This logic lives in the example, NOT in the main library.
async fn ensure_model_files(repo_id: &str, local_dir: &Path) -> Result<()> {
    if !local_dir.exists() {
        println!("-> Creating cache directory: {:?}", local_dir);
        // Use tokio's async version of create_dir_all
        tokio::fs::create_dir_all(local_dir).await?;
    }

    let files_to_check = ["model.safetensors", "config.json", "tokenizer.json"];
    for filename in files_to_check {
        let local_path = local_dir.join(filename);
        if !local_path.exists() {
            println!("-> Downloading {}...", filename);
            let download_url = format!(
                "https://huggingface.co/{}/resolve/main/{}",
                repo_id, filename
            );

            // Use the ASYNC version of reqwest and .await it.
            let response = reqwest::get(&download_url).await?.error_for_status()?;

            let content = response.bytes().await?;

            // Use tokio's async version of write
            tokio::fs::write(&local_path, &content).await?;
            println!("   ... downloaded to {:?}", local_path);
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let device_type_str = if USE_GPU { "GPU (WGPU)" } else { "CPU" };
    println!("Starting BERT embedding example...");

    // --- 1. Define Model and Ensure Files are Present ---
    let model_repo = "sentence-transformers/all-MiniLM-L6-v2";

    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| anyhow!("Failed to get cache directory"))?
        .join("edgegpt");

    let model_dir = cache_dir.join(model_repo.replace('/', "_"));

    // We now .await the async helper function.
    ensure_model_files(model_repo, &model_dir).await?;
    println!("Model files are available in: {:?}", model_dir);

    let wgpu_context: Option<Arc<WgpuContext>> = if USE_GPU {
        println!("\nInitializing WGPU context...");
        let context = WgpuContext::new().await;
        println!("WGPU context initialized successfully.");
        Some(Arc::new(context))
    } else {
        None
    };
    println!("\nInitializing BertBiEncoder");
    let device = if USE_GPU { Device::Wgpu } else { Device::Cpu };
    let bi_encoder = BertBiEncoder::from_pretrained(&model_dir, device, wgpu_context)?;
    println!("Model initialized successfully.");

    // --- 3. Prepare Input Texts ---
    // A pool of sentences to choose from for each batch
    let sentence_pool = vec![
        "The quick brown fox jumps over the lazy dog.",
        "Rust is a systems programming language.",
        "WGPU provides a modern graphics and compute API.",
        "This library aims for maximum performance.",
        "Edge computing is a distributed computing paradigm.",
        "Multi-head self-attention is a key component.",
        "This is a test sentence for benchmarking.",
        "Each vector is a high-dimensional representation.",
        "The model is running on the specified device.",
        "We are measuring the mean inference time.",
        "Hello world from the edge!",
        "Tokenization splits text into smaller pieces.",
    ];

    // --- 3. Warmup Run ---
    // The first run can be slower due to cache misses, pipeline creation, etc.
    // We do one run here to warm everything up before starting the timer.
    println!("\nPerforming one warmup run...");
    let _ = bi_encoder.encode(vec!["Warmup sentence."], true).await?;
    println!("Warmup complete.");

    // --- 4. Benchmark Loop ---
    println!("\nStarting benchmark...");
    let mut durations: Vec<u128> = Vec::with_capacity(NUM_RUNS);
    let mut rng = rand::thread_rng();

    for i in 0..NUM_RUNS {
        // Create a random batch for this iteration
        let mut batch: Vec<&str> = Vec::with_capacity(BATCH_SIZE);
        for _ in 0..BATCH_SIZE {
            batch.push(sentence_pool.choose(&mut rng).unwrap());
        }

        let start_time = Instant::now();
        let _embeddings = bi_encoder.encode(batch, true).await?;
        let duration = start_time.elapsed();

        durations.push(duration.as_millis());
        print!("\rRun {}/{} complete...", i + 1, NUM_RUNS);
    }

    // --- 5. Report Results ---
    let total_time_ms: u128 = durations.iter().sum();
    let mean_time_ms = total_time_ms as f64 / NUM_RUNS as f64;
    let min_time_ms = *durations.iter().min().unwrap_or(&0);
    let max_time_ms = *durations.iter().max().unwrap_or(&0);
    let p95_index = (NUM_RUNS as f64 * 0.95) as usize;
    let mut sorted_durations = durations.clone();
    sorted_durations.sort();
    let p95_time_ms = sorted_durations.get(p95_index).unwrap_or(&0);

    println!("\n\n--- Benchmark Results for {} ---", device_type_str);
    println!("Total runs: {}", NUM_RUNS);
    println!("Sentences per batch: {}", BATCH_SIZE);
    println!("\nMean latency: {:.2} ms", mean_time_ms);
    println!("Min latency:  {} ms", min_time_ms);
    println!("Max latency:  {} ms", max_time_ms);
    println!("p95 latency:  {} ms", p95_time_ms);
    println!("------------------------------------");

    Ok(())
}
