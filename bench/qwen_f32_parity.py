"""The PyTorch half of an f32 Qwen parity check.

Both sides pinned to f32 and to a neutral repetition penalty, which is the only
configuration where exact greedy token parity is meaningful:

  * Qwen ships bf16 weights, and Kjarni keeps bf16 weights with f32 activations
    while torch's bfloat16 mode makes activations bf16 too. Different pipelines,
    so they cannot agree token for token by construction.
  * generation_config.json sets repetition_penalty 1.1, and transformers applies
    it even under do_sample=False, because it is a logits processor rather than a
    sampling setting. Kjarni defaults to 1.0.

Writes the token ids, which is a sharper comparison than the decoded prose: one
flipped logit early changes every token after it, so matching text is a strong
claim and differing text on its own says little about where it went wrong.
"""
import json, os

os.environ["OMP_NUM_THREADS"] = "8"
import torch
torch.set_num_threads(8)
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "The capital of Iceland is"
N = 16

tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.float32).eval()

ids = tok(PROMPT, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=N, do_sample=False,
                         repetition_penalty=1.0, pad_token_id=tok.eos_token_id)

new = out[0][ids["input_ids"].shape[1]:].tolist()
result = {
    "dtype": "float32",
    "repetition_penalty": 1.0,
    "prompt": PROMPT,
    "token_ids": new,
    "pieces": [tok.decode([t]) for t in new],
    "text": tok.decode(new, skip_special_tokens=True),
}
open("torch_f32_parity.json", "w").write(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
