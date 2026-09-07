"""The PyTorch half of the comparison.

Deliberately uses sentence-transformers for MiniLM rather than raw transformers:
that is what people actually run, and it already does pooling and normalisation,
so the two sides are doing the same work rather than one doing less.

Threads are pinned explicitly. Both runtimes default to every logical core, and
on a hybrid CPU that means the comparison partly measures which scheduler copes
better with E-cores. Pinning makes it a comparison of inference.

    python torch_side.py --threads 8 --json out.json
"""

import argparse, json, os, sys, time


def timed(fn, runs):
    """Best of `runs` after one warm-up. Returns (best_seconds, all_seconds)."""
    fn()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times), times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    # Must be set before torch spins up its pools.
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import torch
    torch.set_num_threads(args.threads)

    import transformers
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "threads": args.threads,
        "python": sys.version.split()[0],
    }

    # ── MiniLM ────────────────────────────────────────────────────
    t = time.perf_counter()
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    out["minilm_load_s"] = time.perf_counter() - t

    corpus = [
        f"Document number {i} about refunds, delivery and account settings."
        for i in range(64)
    ]
    out["minilm"] = {}
    for n in (1, 8, 64):
        batch = corpus[:n]
        best, all_t = timed(
            lambda: st.encode(batch, batch_size=n, show_progress_bar=False,
                              convert_to_numpy=True, normalize_embeddings=True),
            args.runs,
        )
        out["minilm"][str(n)] = {"best_ms": best * 1000, "all_ms": [x * 1000 for x in all_t]}

    # The vector for a fixed sentence, so the other side can be checked against it
    # rather than both sides being timed on possibly different computations.
    ref = st.encode(["The capital of Iceland is Reykjavik."],
                    normalize_embeddings=True, convert_to_numpy=True)[0]
    out["minilm_reference_vector"] = [float(x) for x in ref[:16]]
    out["minilm_dim"] = int(ref.shape[0])

    # ── Qwen ──────────────────────────────────────────────────────
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    t = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    model.eval()
    out["qwen_load_s"] = time.perf_counter() - t

    prompt = "The capital of Iceland is"
    ids = tok(prompt, return_tensors="pt")

    def gen():
        with torch.no_grad():
            return model.generate(**ids, max_new_tokens=args.tokens,
                                  do_sample=False, pad_token_id=tok.eos_token_id)

    best, all_t = timed(gen, max(2, args.runs // 2))
    text = tok.decode(gen()[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    out["qwen"] = {
        "tokens": args.tokens,
        "best_ms": best * 1000,
        "all_ms": [x * 1000 for x in all_t],
        "tokens_per_s": args.tokens / best,
        "text": text.strip(),
    }

    print(json.dumps(out, indent=2))
    if args.json:
        open(args.json, "w").write(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
