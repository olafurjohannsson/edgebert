use anyhow::Result;
use edgebert::{cosine_similarity, BertModel, ModelType};

fn main() -> Result<()> {
    let query = "What is the capital of France?";

    let documents = vec![
        "The Eiffel Tower is a famous landmark in Paris.",
        "Paris is the capital and most populous city of France.",
        "French cuisine is renowned for its cheeses and wines.",
        "Lyon is a major city in France, known for its historical and architectural landmarks.",
        "The Louvre Museum, located in Paris, is the world's largest art museum.",
    ];

    println!("Query: \"{}\"\n", query);

    println!("Retrieving candidates with Bi-Encoder");
    let bi_encoder = BertModel::from_pretrained(ModelType::MiniLML6V2BiEncoder)?;
    let query_embedding = bi_encoder.encode(vec![query], true)?[0].clone();
    let doc_embeddings = bi_encoder.encode(documents.clone(), true)?;

    let mut retrieval_results: Vec<(String, f32)> = documents
        .iter()
        .zip(doc_embeddings.iter())
        .map(|(doc, emb)| {
            let sim = cosine_similarity(&query_embedding, emb);
            (doc.to_string(), sim)
        })
        .collect();

    retrieval_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Initial Retrieval Results (Top 3):");
    let top_k = 3;
    let candidates: Vec<String> = retrieval_results
        .iter()
        .take(top_k)
        .enumerate()
        .map(|(i, (doc, score))| {
            println!("{}. Score: {:.4} - \"{}\"", i + 1, score, doc);
            doc.clone()
        })
        .collect();

    println!("\n Reranking top candidates with Cross-Encoder");

    let mut cross_encoder = BertModel::from_pretrained(ModelType::MiniLML6V2CrossEncoder)?;
    let pairs: Vec<(&str, &str)> = candidates.iter().map(|doc| (query, doc.as_str())).collect();
    let new_scores = cross_encoder.score_batch(pairs)?;
    let mut reranked_results: Vec<(String, f32)> =
        candidates.into_iter().zip(new_scores.into_iter()).collect();

    // Sort by the new cross-encoder score
    reranked_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Final Reranked Results:");
    for (i, (doc, score)) in reranked_results.iter().enumerate() {
        println!("{}. Score: {:.4} - \"{}\"", i + 1, score, doc);
    }

    Ok(())
}
