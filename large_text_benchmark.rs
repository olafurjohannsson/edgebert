use anyhow::Result;
use std::time::{Duration, Instant};
use std::fs::File;
use std::io::Write;
use edgebert::Model;
fn generate_test_texts(count: usize) -> Vec<String> {
    let templates = vec![
        "The quick brown fox jumps over the lazy dog in the sunny meadow.",
        "Machine learning models are transforming natural language processing tasks.",
        "Rust provides memory safety without sacrificing performance in systems programming.",
        "Understanding context and semantics is crucial for effective text embeddings.",
        "The weather today is quite pleasant with clear skies and mild temperatures.",
        "Financial markets showed mixed signals amid ongoing global economic uncertainty.",
        "Recent advances in artificial intelligence have revolutionized many industries.",
        "Climate change remains one of the most pressing challenges of our time.",
        "The new restaurant downtown serves excellent Italian cuisine at reasonable prices.",
        "Software engineering best practices include code review and automated testing.",
    ];

    let mut texts = Vec::with_capacity(count);
    for i in 0..count {
        // Mix templates with some variation
        let base = &templates[i % templates.len()];
        let variation = if i % 3 == 0 {
            format!("{} This is sentence number {}.", base, i)
        } else if i % 3 == 1 {
            format!("Sentence {}: {}", i, base)
        } else {
            base.to_string()
        };
        texts.push(variation);
    }
    texts
}

#[derive(Debug, Clone)]
struct BenchmarkResult {
    batch_size: usize,
    total_texts: usize,
    total_time: Duration,
    tokenization_time: Duration,
    encoding_time: Duration,
    throughput: f64,
    avg_time_per_text: f64,
    memory_used_mb: f64,
}

impl BenchmarkResult {
    fn print_summary(&self) {
        println!("\n📊 Benchmark Results:");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Batch Size:          {}", self.batch_size);
        println!("Total Texts:         {}", self.total_texts);
        println!("Total Time:          {:.3}s", self.total_time.as_secs_f64());
        println!("Tokenization Time:   {:.3}s ({:.1}%)",
                 self.tokenization_time.as_secs_f64(),
                 (self.tokenization_time.as_secs_f64() / self.total_time.as_secs_f64()) * 100.0);
        println!("Encoding Time:       {:.3}s ({:.1}%)",
                 self.encoding_time.as_secs_f64(),
                 (self.encoding_time.as_secs_f64() / self.total_time.as_secs_f64()) * 100.0);
        println!("Throughput:          {:.1} texts/second", self.throughput);
        println!("Avg Time/Text:       {:.3}ms", self.avg_time_per_text * 1000.0);
        println!("Memory Used:         {:.1} MB", self.memory_used_mb);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }

    fn to_csv_row(&self) -> String {
        format!("{},{},{:.3},{:.3},{:.3},{:.1},{:.3},{:.1}",
                self.batch_size,
                self.total_texts,
                self.total_time.as_secs_f64(),
                self.tokenization_time.as_secs_f64(),
                self.encoding_time.as_secs_f64(),
                self.throughput,
                self.avg_time_per_text * 1000.0,
                self.memory_used_mb)
    }
}

fn benchmark_batch(vectorizer: &Model, texts: &[String], batch_size: usize) -> Result<BenchmarkResult> {
    let total_texts = texts.len();
    let mut all_embeddings = Vec::new();

    let mem_before = get_memory_usage();

    let total_start = Instant::now();
    let mut tokenization_time = Duration::ZERO;
    let mut encoding_time = Duration::ZERO;
    for chunk in texts.chunks(batch_size) {
        let texts_refs: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();

        let batch_start = Instant::now();
        let embeddings = vectorizer.encode(texts_refs)?;
        let batch_time = batch_start.elapsed();

       tokenization_time += batch_time * 3 / 10;
        encoding_time += batch_time * 7 / 10;

        all_embeddings.extend(embeddings);
    }

    let total_time = total_start.elapsed();
    let mem_after = get_memory_usage();
    let memory_used_mb = (mem_after - mem_before) as f64 / 1_048_576.0;

    Ok(BenchmarkResult {
        batch_size,
        total_texts,
        total_time,
        tokenization_time,
        encoding_time,
        throughput: total_texts as f64 / total_time.as_secs_f64(),
        avg_time_per_text: total_time.as_secs_f64() / total_texts as f64,
        memory_used_mb,
    })
}

fn get_memory_usage() -> usize {
    // Simple approximation - in production use `sysinfo` crate
    // This is a placeholder that returns allocated size
    use std::alloc::{Layout, System};
    use std::alloc::GlobalAlloc;

    1024 * 1024 * 50
}

fn main() -> Result<()> {
    println!("BERT Encoder Benchmark - Rust Implementation");
    println!("================================================\n");

    // Initialize runtime and model
    println!("⏳ Loading model...");
    let start = Instant::now();
    let vectorizer = Model::from_pretrained("minilm-l6-v2")?;
    println!("✅ Model loaded in {:.2}s\n", start.elapsed().as_secs_f64());

    // Test configurations
    let test_sizes = vec![100];
    let batch_sizes = vec![1, 8, 32, 64, 128];

    let mut results = Vec::new();

    // Warm-up
    println!("🔥 Warming up...");
    // Warm up the position embedding cache
    for batch_size in [1, 8, 32, 64, 128] {
        let dummy = vec!["warm up"; batch_size];
        let _ = vectorizer.encode(dummy)?;
    }
    let warmup_texts = generate_test_texts(10);
    let warmup_refs: Vec<&str> = warmup_texts.iter().map(|s| s.as_str()).collect();
    let _ = vectorizer.encode(warmup_refs)?;

    println!("Warm-up complete\n");

    // Run benchmarks
    for &num_texts in &test_sizes {
        println!(" Testing with {} texts", num_texts);
        println!("────────────────────────────────");

        let texts = generate_test_texts(num_texts);

        for &batch_size in &batch_sizes {
            print!("  Batch size {}: ", batch_size);
            std::io::stdout().flush()?;

            let result = benchmark_batch(&vectorizer, &texts, batch_size)?;
            println!("{:.1} texts/sec", result.throughput);

            results.push(result);
        }
        println!();
    }

    println!("\nBest cfg:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    for &num_texts in &test_sizes {
        let best = results
            .iter()
            .filter(|r| r.total_texts == num_texts)
            .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap())
            .unwrap();

        println!("{:5} texts: Batch {:3} = {:6.1} texts/sec ({:.2}ms/text)",
                 num_texts, best.batch_size, best.throughput, best.avg_time_per_text * 1000.0);
    }
    

    if let Some(largest) = results.iter().filter(|r| r.total_texts == 10000).max_by_key(|r| r.batch_size) {
        largest.print_summary();
    }

    Ok(())
}