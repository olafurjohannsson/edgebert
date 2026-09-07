"""Top logits after the prompt, so a divergence can be classified.

Comparing generated text says two runs differ but not why. A near-tie between the
top two logits is precision; a large gap is a bug in attention, RoPE or the LM
head. Only the numbers distinguish those, and by the second token the histories
have already diverged, so the comparison has to be the first forward pass over
the prompt and nothing after it.
"""
import json, os

os.environ["OMP_NUM_THREADS"] = "8"
import torch
torch.set_num_threads(8)
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "The capital of Iceland is"

tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.float32).eval()

ids = tok(PROMPT, return_tensors="pt")
with torch.no_grad():
    logits = model(**ids).logits[0, -1].float()   # last position only

top = torch.topk(logits, 10)
out = {
    "prompt": PROMPT,
    "prompt_token_ids": ids["input_ids"][0].tolist(),
    "top10": [
        {"id": int(i), "piece": tok.decode([int(i)]), "logit": float(v)}
        for v, i in zip(top.values, top.indices)
    ],
    "logit_sum": float(logits.sum()),
    "logit_max": float(logits.max()),
    "logit_mean": float(logits.mean()),
    "vocab": int(logits.shape[0]),
}
open("torch_logits.json", "w").write(json.dumps(out, indent=2))
for e in out["top10"][:5]:
    print(f"  {e['logit']:9.4f}  {e['id']:>7}  {e['piece']!r}")
print(f"  gap between top two: {out['top10'][0]['logit'] - out['top10'][1]['logit']:.4f}")
