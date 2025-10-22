//! Basic RAG example using EdgeBERT and EdgeGPT

use anyhow::Result;
use edgebert::{Model as BertModel, ModelType as BertModelType};
use edgegpt::{GPTModel, ModelType as GPTModelType, GenerationConfig};

struct SimpleRAG {
    bert_model: BertModel,
    gpt_model: GPTModel,
    documents: Vec<String>,
    embeddings: Vec<Vec<f32>>,
}

impl SimpleRAG {
    fn new(documents: Vec<String>) -> Result<Self> {
        println!("Loading models...");
        let bert_model = BertModel::from_pretrained(BertModelType::MiniLML6V2BiEncoder)?;
        let gpt_model = GPTModel::from_pretrained(GPTModelType::DistilGPT2)?;
        
        // Generate embeddings for documents
        println!("Generating document embeddings...");
        let doc_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        let embeddings = bert_model.encode(doc_refs, true)?;
        
        Ok(Self {
            bert_model,
            gpt_model,
            documents,
            embeddings,
        })
    }
    
    fn search(&self, query: &str, top_k: usize) -> Vec<(usize, f32, &str)> {
        // Encode query
        let query_emb = self.bert_model.encode(vec![query], true).unwrap();
        let query_emb = &query_emb[0];
        
        // Calculate similarities
        let mut scores = Vec::new();
        for (idx, doc_emb) in self.embeddings.iter().enumerate() {
            let sim = edgebert::cosine_similarity(query_emb, doc_emb);
            scores.push((idx, sim, self.documents[idx].as_str()));
        }
        
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(top_k);
        scores
    }
    
    fn generate_answer(&self, query: &str, context: &str) -> Result<String> {
        let prompt = format!(
            "You are an expert assistant. Answer the question strictly based on the provided context. 
               If the answer is not in the context, say \"I don\'t know\".Context: {}\n\nQuestion: {}\nAnswer:",
            context, query
        );
        
        let config = GenerationConfig {
            max_new_tokens: 100,
            temperature: 0.2,
            ..Default::default()
        };
        
        self.gpt_model.generate(&prompt, &config)
    }
}

fn main() -> Result<()> {
    // Sample documents
    let documents = vec![
        "The Eiffel Tower is located in Paris, France. It was built in 1889.".to_string(),
        "Python is a popular programming language known for its simplicity.".to_string(),
        "Climate change is one of the biggest challenges facing humanity.".to_string(),
        "The human brain contains approximately 86 billion neurons.".to_string(),
        "Jazz music originated in New Orleans in the early 20th century.".to_string(),
    ];
    
    let rag = SimpleRAG::new(documents)?;
    
    let queries = vec![
        "Where is the Eiffel Tower?",
        "What is Python?",
        "Tell me about jazz music",
    ];
    
    for query in queries {
        println!("\n\nQuery: {}", query);
        
        // Search for relevant documents
        let results = rag.search(query, 2);
        
        println!("Top relevant documents:");
        for (idx, score, doc) in &results {
            println!("  [{}, score: {:.3}] {}", idx, score, doc);
        }
        
        // Generate answer using context
        let context = results.iter()
            .filter(|(_, score, _)| *score > 0.1) // skip low-score docs
            .map(|(_, _, doc)| *doc)
            .take(1) 
            .collect::<Vec<_>>()
            .join(" ");
        
        let answer = rag.generate_answer(query, &context)?;
        println!("Generated answer: {}", answer);
    }
    
    Ok(())
}