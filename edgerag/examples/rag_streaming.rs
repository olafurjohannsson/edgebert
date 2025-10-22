//! RAG with streaming generation

use anyhow::Result;
use edgebert::{BertModel, ModelType as BertModelType};
use edgegpt::{GPTModel, ModelType as GPTModelType, GenerationConfig};
use edgegpt::generation::generate_text_streaming;
use std::io::{self, Write};

fn main() -> Result<()> {
    println!("Loading models...");
    let bert_model = BertModel::from_pretrained(BertModelType::MiniLML6V2BiEncoder)?;
    let gpt_model = match GPTModel::from_pretrained(GPTModelType::DistilGPT2)? {
        GPTModel::DistilGPT2(m) => m,
        _ => panic!("Expected DistilGPT2"),
    };
    
    // Knowledge base
    let documents = vec![
        "The Large Hadron Collider (LHC) is the world's largest particle accelerator.",
        "CERN operates the LHC near Geneva, Switzerland.",
        "The LHC discovered the Higgs boson in 2012.",
        "Particle physics studies the fundamental constituents of matter.",
    ];
    
    // Generate embeddings
    let doc_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
    let doc_embeddings = bert_model.encode(doc_refs, true)?;
    
    let query = "Tell me about the Large Hadron Collider";
    println!("\nQuery: {}", query);
    
    // Find relevant documents
    let query_emb = bert_model.encode(vec![query], true)?;
    let query_emb = &query_emb[0];
    
    let mut scores = Vec::new();
    for (idx, doc_emb) in doc_embeddings.iter().enumerate() {
        let sim = edgebert::cosine_similarity(query_emb, doc_emb);
        scores.push((idx, sim));
    }
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    // Get top 2 documents
    let context = scores.iter()
        .take(2)
        .map(|(idx, _)| documents[*idx])
        .collect::<Vec<_>>()
        .join(" ");
    
    println!("\nContext: {}", context);
    
    let prompt = format!(
        "Context: {}\n\nQuestion: {}\nDetailed Answer:",
        context, query
    );
    
    let config = GenerationConfig {
        max_new_tokens: 150,
        temperature: 0.7,
        top_k: Some(40),
        top_p: Some(0.9),
        repetition_penalty: 1.1,
        ..Default::default()
    };
    
    println!("\nGenerating answer (streaming):");
    print!("Answer: ");
    io::stdout().flush()?;
    
    let mut token_count = 0;
    let result = generate_text_streaming(
        &gpt_model.base,
        &gpt_model.tokenizer,
        &prompt,
        &config,
        Box::new(move |_token_id, token_text| {
            print!("{}", token_text);
            io::stdout().flush().unwrap();
            token_count += 1;
            true
        }),
    )?;
    
    println!("\n\nGenerated {} tokens", token_count);
    
    Ok(())
}