use anyhow::{Result, anyhow};
use edgemodels::bert::SentenceEncoder;
use edgemodels::roberta::RobertaBiEncoder;
use edgemodels::roberta::RobertaConfig;
use edgetransformers::prelude::Device;
use edgetransformers::wgpu_context::WgpuContext;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use rand::seq::SliceRandom;
//First 10 dims: [-0.0123080015, -0.050815407, -0.02291113, -0.008355495, -0.0016676121, -0.0072230296, 0.004740711, 0.01285743, -0.020140342, -0.023169108]
use std::time::Instant;
const USE_GPU: bool = false;
const NUM_RUNS: usize = 1;
const BATCH_SIZE: usize = 32;

// First 10 dims: [-0.023348324, 0.09190789, 0.024982348, 0.08469894, 0.021637155, 0.012153852, 0.054321263, -0.03697413, -0.042735234, -0.018510059]
// Warmup complete

// cpu: [-0.023348212, 0.09190799, 0.024982281, 0.08469892, 0.021637136, 0.012153861, 0.054321203, -0.036974117, -0.042735167, -0.018510068]
// gpu: [-0.023348324, 0.09190789, 0.024982348, 0.08469894, 0.021637155, 0.012153852, 0.054321263, -0.03697413, -0.042735234, -0.018510059]
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
//bert-base: First 10 dims: [-0.0040795375, -0.018265525, -0.018570526, 0.007197875, 0.040293448, -0.03368905, 0.024132011, 0.0022599571, 0.0032060216, -0.040955316]
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
    let bi_encoder = SentenceEncoder::from_pretrained(&model_dir, device, wgpu_context)?;
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
    let t = bi_encoder.encode(vec!["Warmup sentence."], true).await?;
    for (i, embedding) in t.iter().enumerate() {
        let norm: f32 = embedding.iter().map(|x| x*x).sum::<f32>().sqrt();
        let n = embedding.len().min(10);
        println!("First 10 dims: {:?}", &embedding[0..n]);
    }
    println!("Warmup complete");

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
