import random
import re
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME_MAP = {
    "Qwen3-4B": "Qwen/Qwen2.5-4B-Instruct",
    "Qwen3-8B": "Qwen/Qwen2.5-8B-Instruct",
}


class LLMWrapper:
    def __init__(self, model_cfg, device: str = "cpu") -> None:
        model_name = MODEL_NAME_MAP.get(model_cfg.name, model_cfg.name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=".cache/")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=".cache/")
        self.model.to(device)
        self.model.eval()
        self.device = device

    def generate_batch(self, prompts: List[str], batch_size: int = 8, max_new_tokens: int = 32) -> List[str]:
        outputs = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            inputs = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).to(self.device)
            if inputs.input_ids.shape[0] > 0:
                assert inputs.input_ids.shape[0] == inputs.attention_mask.shape[0]
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            outputs.extend([self._postprocess(d) for d in decoded])
        return outputs

    def _postprocess(self, text: str) -> str:
        match = re.findall(r"\b[A-D]\b", text.strip())
        if match:
            return match[-1]
        nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
        if nums:
            return nums[-1]
        return text.strip().split()[-1] if text.strip() else ""


class PromptProposer:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.seed_prompts = [
            "You are a helpful assistant.",
            "Solve the problem carefully and answer precisely.",
            "Read the question and respond with the final answer only.",
            "Follow the instructions and provide the exact output format.",
        ]
        self.mutations = [
            "Be concise.",
            "Double-check your reasoning.",
            "Use step-by-step reasoning internally.",
            "Avoid extraneous text.",
            "Be confident in the final answer.",
        ]

    def propose(self, n: int, history: List) -> List[str]:
        if not history:
            return [self.rng.choice(self.seed_prompts) for _ in range(n)]
        top_prompts = [h.prompt for h in history[: min(len(history), 5)]]
        proposals = []
        for _ in range(n):
            base = self.rng.choice(top_prompts)
            mut = self.rng.choice(self.mutations)
            if self.rng.random() < 0.5:
                proposal = f"{base} {mut}"
            else:
                proposal = f"{mut} {base}"
            proposals.append(proposal)
        return proposals


class MetadataPredictor(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
