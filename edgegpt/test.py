import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# --- Model and text setup ---
model_id = "sshleifer/distilbart-cnn-12-6"
article_text = "The Apollo 11 mission was the first manned mission to land on the Moon. The mission, carried out by NASA, took place in July 1969. The three astronauts on board were Neil Armstrong, Buzz Aldrin, and Michael Collins. Armstrong was the first person to step onto the lunar surface, famously declaring, 'That's one small step for man, one giant leap for mankind.' The mission successfully collected lunar samples and returned the astronauts safely to Earth, marking a significant milestone in human space exploration."
torch.set_printoptions(precision=6, sci_mode=False, linewidth=120)
tokenizer = BartTokenizer.from_pretrained(model_id)
model = BartForConditionalGeneration.from_pretrained(model_id)
model.eval()

# --- Inputs and Embeddings ---
inputs = tokenizer(article_text, return_tensors="pt")
attention_mask = inputs["attention_mask"]
encoder_embeddings = model.model.encoder.embed_tokens(inputs["input_ids"]) + model.model.encoder.embed_positions(inputs["input_ids"])
encoder_embeddings = model.model.encoder.layernorm_embedding(encoder_embeddings)

# --- Process Layer 0 to get correct input for Layer 1 ---
def prepare_encoder_attention_mask(mask, input_shape, embed_dtype):
    expanded_mask = mask[:, None, None, :].expand(input_shape[0], 1, input_shape[1], -1).to(embed_dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(embed_dtype).min)

processed_attention_mask = prepare_encoder_attention_mask(
    attention_mask, encoder_embeddings.shape[:2], encoder_embeddings.dtype
)
hidden_states_final_L0 = model.model.encoder.layers[0](encoder_embeddings, attention_mask=processed_attention_mask, layer_head_mask=None)[0]

# --- Layer 1 Deep Dive ---
print("\n--- PYTHON ENCODER LAYER 1: ATTENTION DEEP DIVE ---")
input_to_L1 = hidden_states_final_L0
attn_module = model.model.encoder.layers[1].self_attn
num_heads = attn_module.num_heads
head_dim = attn_module.head_dim

# --- THE CORRECT ORDER OF OPERATIONS ---

# 1. Project Q, K
q_proj = attn_module.q_proj(input_to_L1)
k_proj = attn_module.k_proj(input_to_L1)

# 2. Scale the 3D Q-Projection
q_proj_scaled = q_proj * attn_module.scaling
print(f"\n--- B.2. Scaled Q Projection (3D) ---")
print("First vector (first 5 values):", q_proj_scaled[0, 0, :5])

# 3. Reshape the SCALED Q and unscaled K
def shape(tensor, seq_len, bsz):
    return tensor.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

q = shape(q_proj_scaled, -1, 1) # Use the scaled projection here
k = shape(k_proj, -1, 1)        # Use the unscaled projection here

# 4. Matrix Multiply (with no further scaling)
attn_weights = torch.matmul(q, k.transpose(-1, -2))
print(f"\n--- C. Attention Scores (before mask) ---")
print("First 5 scores [Head 0, Token 0]:", attn_weights[0, 0, 0, :5])

# --- Final Verification ---
final_encoder_output = model.model.encoder(inputs_embeds=encoder_embeddings, attention_mask=attention_mask).last_hidden_state
print("\n--- PYTHON ENCODER: Final Output (Target) ---")
print("First vector (first 5 values):", final_encoder_output[0, 0, :5])
print("-" * 20)