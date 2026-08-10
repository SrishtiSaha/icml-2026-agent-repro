#!/usr/bin/env python3
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
# ]
# ///
"""Scaled GPU smoke test for the cramming mechanism.

This intentionally uses sshleifer/tiny-gpt2 so the job is cheap.  It is not used
to decide the paper's architecture-level claims; those are audited from the
paper's own source figures and table.  The smoke check verifies that the
soft-prompt cramming procedure runs end-to-end on a real decoder-only
Transformer and reports exact-match accessibility over a tiny grid.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Result:
    model: str
    device: str
    m: int
    n: int
    target: list[int]
    generated: list[int]
    success: bool
    final_loss: float


def set_seed(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cram_once(model, embeds, vocab_size: int, m: int, n: int, steps: int, device: str) -> Result:
    target = torch.randint(0, min(vocab_size, 256), (n,), device=device)
    prompt = torch.nn.Parameter(torch.randn(m, embeds.embedding_dim, device=device) * 0.02)
    opt = torch.optim.Adam([prompt], lr=0.08)
    labels = target.unsqueeze(0)

    final_loss = 0.0
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        target_emb = embeds(target).unsqueeze(0)
        inputs_embeds = torch.cat([prompt.unsqueeze(0), target_emb[:, :-1, :]], dim=1)
        out = model(inputs_embeds=inputs_embeds)
        logits = out.logits[:, -n:, :]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1))
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().cpu())

    generated = []
    with torch.no_grad():
        cur = prompt.unsqueeze(0)
        for _ in range(n):
            logits = model(inputs_embeds=cur).logits[:, -1, :]
            nxt = torch.argmax(logits, dim=-1)
            generated.append(int(nxt.item()))
            cur = torch.cat([cur, embeds(nxt).unsqueeze(1)], dim=1)

    return Result(
        model="https://huggingface.co/sshleifer/tiny-gpt2",
        device=device,
        m=m,
        n=n,
        target=[int(x) for x in target.detach().cpu().tolist()],
        generated=generated,
        success=generated == [int(x) for x in target.detach().cpu().tolist()],
        final_loss=final_loss,
    )


def main() -> None:
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(json.dumps({"cuda_available": torch.cuda.is_available(), "device": device}))
    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2").to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    embeds = model.get_input_embeddings()
    vocab_size = int(model.config.vocab_size)

    steps = int(os.environ.get("CRAM_STEPS", "80"))
    grid = [(1, 4), (1, 8), (2, 8), (3, 12)]
    results = [asdict(cram_once(model, embeds, vocab_size, m, n, steps, device)) for m, n in grid]
    print("RESULTS_JSON_START")
    print(json.dumps({"steps": steps, "results": results}, indent=2))
    print("RESULTS_JSON_END")


if __name__ == "__main__":
    main()
