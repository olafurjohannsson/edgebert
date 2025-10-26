use anyhow::Result;
use crate::bert::SentenceEncoder;
use edgetransformers::prelude::Device;
use edgetransformers::wgpu_context::WgpuContext;
use std::path::Path;
use tokio;
use tokio::fs;
use std::sync::Arc;

/// Helper to ensure model files exist
async fn ensure_model_files(repo_id: &str, local_dir: &Path) -> Result<()> {
    if !local_dir.exists() {
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

            let response = reqwest::get(&download_url).await?.error_for_status()?;
            let content = response.bytes().await?;
            fs::write(&local_path, &content).await?;
            println!("   ... downloaded to {:?}", local_path);
        }
    }
    Ok(())
}

fn assert_embeddings_close(cpu: &[Vec<f32>], gpu: &[Vec<f32>], tolerance: f32, name: &str) {
    assert_eq!(cpu.len(), gpu.len(), "{}: Batch size mismatch", name);
    
    for (batch_idx, (cpu_emb, gpu_emb)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert_eq!(
            cpu_emb.len(),
            gpu_emb.len(),
            "{}: Embedding dimension mismatch at batch {}",
            name,
            batch_idx
        );

        let mut max_diff = 0.0f32;
        let mut max_rel_diff = 0.0f32;
        let mut num_large_diffs = 0;

        for (i, (&cpu_val, &gpu_val)) in cpu_emb.iter().zip(gpu_emb.iter()).enumerate() {
            if cpu_val.is_nan() || gpu_val.is_nan() {
                panic!(
                    "{}: NaN at batch {} index {}: CPU={}, GPU={}",
                    name, batch_idx, i, cpu_val, gpu_val
                );
            }

            let diff = (cpu_val - gpu_val).abs();
            let rel_diff = if cpu_val.abs() > 1e-6 {
                diff / cpu_val.abs()
            } else {
                diff
            };

            max_diff = max_diff.max(diff);
            max_rel_diff = max_rel_diff.max(rel_diff);

            if diff > tolerance {
                num_large_diffs += 1;
                if num_large_diffs <= 5 {
                    println!(
                        "{}: Large diff at batch {} index {}: CPU={:.6}, GPU={:.6}, diff={:.6}",
                        name, batch_idx, i, cpu_val, gpu_val, diff
                    );
                }
            }
        }

        println!(
            "{}: Batch {} - Max abs diff: {:.6}, Max rel diff: {:.6}, Large diffs: {}/{}",
            name,
            batch_idx,
            max_diff,
            max_rel_diff,
            num_large_diffs,
            cpu_emb.len()
        );

        assert!(
            max_diff < tolerance * 10.0,
            "{}: Batch {} max difference {:.6} exceeds 10x tolerance {:.6}",
            name,
            batch_idx,
            max_diff,
            tolerance * 10.0
        );
    }
}

#[tokio::test]
async fn test_cpu_gpu_parity_single_sentence() -> Result<()> {
    println!("\n=== CPU vs GPU Parity Test: Single Sentence ===\n");

    let model_repo = "sentence-transformers/all-MiniLM-L6-v2";
    let cache_dir = dirs::cache_dir()
        .unwrap()
        .join("edgegpt")
        .join(model_repo.replace('/', "_"));

    ensure_model_files(model_repo, &cache_dir).await?;

    // Initialize both encoders
    println!("Initializing CPU encoder...");
    let cpu_encoder = SentenceEncoder::from_pretrained(&cache_dir, Device::Cpu, None)?;

    println!("Initializing GPU encoder...");
    let gpu_context = Arc::new(WgpuContext::new().await);
    let gpu_encoder = SentenceEncoder::from_pretrained(&cache_dir, Device::Wgpu, Some(gpu_context))?;

    // Test with single sentence
    let sentence = vec!["The quick brown fox jumps over the lazy dog."];

    println!("\n--- Encoding on CPU ---");
    let cpu_embeddings = cpu_encoder.encode(sentence.clone(), true).await?;
    println!("CPU output stats:");
    let cpu_norm: f32 = cpu_embeddings[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("  Norm: {:.6}", cpu_norm);
    println!("  First 10: {:?}", &cpu_embeddings[0][..10]);

    println!("\n--- Encoding on GPU ---");
    let gpu_embeddings = gpu_encoder.encode(sentence.clone(), true).await?;
    println!("GPU output stats:");
    let gpu_norm: f32 = gpu_embeddings[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("  Norm: {:.6}", gpu_norm);
    println!("  First 10: {:?}", &gpu_embeddings[0][..10]);

    // Compare
    assert_embeddings_close(&cpu_embeddings, &gpu_embeddings, 1e-3, "Single sentence");

    println!("\n✅ Single sentence test passed!\n");
    Ok(())
}

// #[tokio::test]
// async fn test_cpu_gpu_parity_batch() -> Result<()> {
//     println!("\n=== CPU vs GPU Parity Test: Batch ===\n");

//     let model_repo = "sentence-transformers/all-MiniLM-L6-v2";
//     let cache_dir = dirs::cache_dir()
//         .unwrap()
//         .join("edgegpt")
//         .join(model_repo.replace('/', "_"));

//     ensure_model_files(model_repo, &cache_dir).await?;

//     // Initialize both encoders
//     println!("Initializing CPU encoder...");
//     let cpu_encoder = SentenceEncoder::from_pretrained(&cache_dir, Device::Cpu, None)?;

//     println!("Initializing GPU encoder...");
//     let gpu_context = Arc::new(WgpuContext::new().await);
//     let gpu_encoder = SentenceEncoder::from_pretrained(&cache_dir, Device::Wgpu, Some(gpu_context))?;

//     // Test with batch of sentences
//     let sentences = vec![
//         "The quick brown fox jumps over the lazy dog.",
//         "Rust is a systems programming language.",
//         "WGPU provides a modern graphics and compute API.",
//         "This library aims for maximum performance.",
//     ];

//     println!("\n--- Encoding batch on CPU ---");
//     let cpu_embeddings = cpu_encoder.encode(sentences.clone(), true).await?;
//     println!("CPU batch encoded: {} sentences", cpu_embeddings.len());

//     println!("\n--- Encoding batch on GPU ---");
//     let gpu_embeddings = gpu_encoder.encode(sentences.clone(), true).await?;
//     println!("GPU batch encoded: {} sentences", gpu_embeddings.len());

//     // Compare
//     assert_embeddings_close(&cpu_embeddings, &gpu_embeddings, 1e-3, "Batch");

//     println!("\n✅ Batch test passed!\n");
//     Ok(())
// }

// #[tokio::test]
// async fn test_cpu_gpu_parity_varied_lengths() -> Result<()> {
//     println!("\n=== CPU vs GPU Parity Test: Varied Lengths ===\n");

//     let model_repo = "sentence-transformers/all-MiniLM-L6-v2";
//     let cache_dir = dirs::cache_dir()
//         .unwrap()
//         .join("edgegpt")
//         .join(model_repo.replace('/', "_"));

//     ensure_model_files(model_repo, &cache_dir).await?;

//     // Initialize both encoders
//     let cpu_encoder = SentenceEncoder::from_pretrained(&cache_dir, Device::Cpu, None)?;
//     let gpu_context = Arc::new(WgpuContext::new().await);
//     let gpu_encoder = SentenceEncoder::from_pretrained(&cache_dir, Device::Wgpu, Some(gpu_context))?;

//     // Test with varied length sentences
//     let sentences = vec![
//         "Hi",
//         "Hello world!",
//         "This is a medium length sentence for testing.",
//         "This is a much longer sentence that should test the model's ability to handle various input lengths and still produce consistent embeddings between CPU and GPU implementations.",
//     ];

//     println!("\n--- Encoding varied lengths on CPU ---");
//     let cpu_embeddings = cpu_encoder.encode(sentences.clone(), true).await?;

//     println!("\n--- Encoding varied lengths on GPU ---");
//     let gpu_embeddings = gpu_encoder.encode(sentences.clone(), true).await?;

//     // Compare
//     assert_embeddings_close(&cpu_embeddings, &gpu_embeddings, 1e-3, "Varied lengths");

//     println!("\n✅ Varied length test passed!\n");
//     Ok(())
// }

// #[tokio::test]
// async fn test_cpu_gpu_parity_large_batch() -> Result<()> {
//     println!("\n=== CPU vs GPU Parity Test: Large Batch ===\n");

//     let model_repo = "sentence-transformers/all-MiniLM-L6-v2";
//     let cache_dir = dirs::cache_dir()
//         .unwrap()
//         .join("edgegpt")
//         .join(model_repo.replace('/', "_"));

//     ensure_model_files(model_repo, &cache_dir).await?;

//     // Initialize both encoders
//     let cpu_encoder = BiEncoder::from_pretrained(&cache_dir, Device::Cpu, None)?;
//     let gpu_context = Arc::new(WgpuContext::new().await);
//     let gpu_encoder = BiEncoder::from_pretrained(&cache_dir, Device::Wgpu, Some(gpu_context))?;

//     // Create large batch (32 sentences)
//     let base_sentences = vec![
//         "The quick brown fox jumps over the lazy dog.",
//         "Rust is a systems programming language.",
//         "WGPU provides a modern graphics and compute API.",
//         "This library aims for maximum performance.",
//     ];
    
//     let mut sentences = Vec::new();
//     for i in 0..8 {
//         for s in &base_sentences {
//             sentences.push(format!("{} Iteration {}.", s, i));
//         }
//     }
//     let sentence_refs: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();

//     println!("\n--- Encoding large batch ({} sentences) on CPU ---", sentences.len());
//     let cpu_embeddings = cpu_encoder.encode(sentence_refs.clone(), true).await?;

//     println!("\n--- Encoding large batch ({} sentences) on GPU ---", sentences.len());
//     let gpu_embeddings = gpu_encoder.encode(sentence_refs.clone(), true).await?;

//     // Compare
//     assert_embeddings_close(&cpu_embeddings, &gpu_embeddings, 1e-3, "Large batch");

//     println!("\n✅ Large batch test passed!\n");
//     Ok(())
// }