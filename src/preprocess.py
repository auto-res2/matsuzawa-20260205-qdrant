import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from datasets import load_dataset


@dataclass
class DatasetBundle:
    train: List[Dict]
    val: List[Dict]
    test: List[Dict]
    dataset_name: str
    metadata_dim: int

    def sample_val(self, n: int) -> List[Dict]:
        return random.sample(self.val, k=min(n, len(self.val)))

    def format_prompt(self, system_prompt: str, example: Dict) -> str:
        if self.dataset_name == "ARC":
            choices = "\n".join([f"{opt['label']}. {opt['text']}" for opt in example["choices"]])
            return (
                f"{system_prompt}\nQuestion: {example['question']}\n"
                f"Choices:\n{choices}\nAnswer with a single letter (A/B/C/D)."
            )
        return (
            f"{system_prompt}\nQuestion: {example['question']}\n"
            "Provide the final numeric answer only."
        )

    def latent_group(self, z: np.ndarray) -> np.ndarray:
        b1 = z[:, 0].astype(int)
        b2 = z[:, 1].astype(int)
        return (b1 ^ b2).astype(int)


def canonicalize_number(text: str) -> Optional[float]:
    if text is None:
        return None
    cleaned = text.replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_metadata_arc(example: Dict, length_median: int, numeral_median: int) -> np.ndarray:
    text = example["question"]
    length_bin = int(len(text.split()) >= length_median)
    numeral_count = sum(ch.isdigit() for ch in text)
    numeral_bin = int(numeral_count >= numeral_median)
    has_unit = int(any(u in text.lower() for u in ["km", "m", "kg", "cm", "mph", "hours"]))
    choice_lengths = [len(c["text"].split()) for c in example["choices"]]
    choice_var = np.var(choice_lengths)
    choice_var_bin = int(choice_var >= np.median(choice_lengths))
    return np.array([length_bin, numeral_bin, has_unit, choice_var_bin], dtype=np.int64)


def extract_metadata_gsm8k(example: Dict, length_median: int, numeral_median: int) -> np.ndarray:
    text = example["question"]
    length_bin = int(len(text.split()) >= length_median)
    numeral_count = sum(ch.isdigit() for ch in text)
    numeral_bin = int(numeral_count >= numeral_median)
    has_dollar = int("$" in text)
    return np.array([length_bin, numeral_bin, has_dollar, 0], dtype=np.int64)


def load_dataset_bundle(cfg) -> DatasetBundle:
    max_samples = cfg.dataset.get("max_samples", None)
    if cfg.dataset.name == "ARC":
        raw = load_dataset("ai2_arc", "ARC-Challenge", cache_dir=".cache/")
        data = raw["train"].shuffle(seed=cfg.training.seed)
        test = raw["test"].shuffle(seed=cfg.training.seed)
        if max_samples:
            data = data.select(range(min(len(data), max_samples)))
            test = test.select(range(min(len(test), max_samples)))
        questions = [ex["question"] for ex in data]
        length_median = int(np.median([len(q.split()) for q in questions]))
        numeral_median = int(np.median([sum(ch.isdigit() for ch in q) for q in questions]))
        processed = []
        for ex in data:
            choices = [{"label": l, "text": t} for l, t in zip(ex["choices"]["label"], ex["choices"]["text"])]
            processed.append(
                {
                    "question": ex["question"],
                    "choices": choices,
                    "label": ex["answerKey"],
                    "label_binary": int(ex["answerKey"] in ["A", "B", "C", "D"]),
                    "metadata": extract_metadata_arc(
                        {"question": ex["question"], "choices": choices},
                        length_median,
                        numeral_median,
                    ),
                }
            )
        test_processed = []
        for ex in test:
            choices = [{"label": l, "text": t} for l, t in zip(ex["choices"]["label"], ex["choices"]["text"])]
            test_processed.append(
                {
                    "question": ex["question"],
                    "choices": choices,
                    "label": ex["answerKey"],
                    "label_binary": int(ex["answerKey"] in ["A", "B", "C", "D"]),
                    "metadata": extract_metadata_arc(
                        {"question": ex["question"], "choices": choices},
                        length_median,
                        numeral_median,
                    ),
                }
            )
    else:
        raw = load_dataset("gsm8k", "main", cache_dir=".cache/")
        data = raw["train"].shuffle(seed=cfg.training.seed)
        test = raw["test"].shuffle(seed=cfg.training.seed)
        if max_samples:
            data = data.select(range(min(len(data), max_samples)))
            test = test.select(range(min(len(test), max_samples)))
        questions = [ex["question"] for ex in data]
        length_median = int(np.median([len(q.split()) for q in questions]))
        numeral_median = int(np.median([sum(ch.isdigit() for ch in q) for q in questions]))
        processed = []
        for ex in data:
            answer = ex["answer"].split("####")[-1].strip()
            processed.append(
                {
                    "question": ex["question"],
                    "label": answer,
                    "label_binary": int(canonicalize_number(answer) is not None),
                    "metadata": extract_metadata_gsm8k(
                        {"question": ex["question"]},
                        length_median,
                        numeral_median,
                    ),
                }
            )
        test_processed = []
        for ex in test:
            answer = ex["answer"].split("####")[-1].strip()
            test_processed.append(
                {
                    "question": ex["question"],
                    "label": answer,
                    "label_binary": int(canonicalize_number(answer) is not None),
                    "metadata": extract_metadata_gsm8k(
                        {"question": ex["question"]},
                        length_median,
                        numeral_median,
                    ),
                }
            )

    n = len(processed)
    n_train = int(n * cfg.dataset.split_ratios.train)
    n_val = int(n * cfg.dataset.split_ratios.val)
    train = processed[:n_train]
    val = processed[n_train : n_train + n_val]
    test = test_processed

    metadata_dim = len(train[0]["metadata"]) if train else 0
    return DatasetBundle(train=train, val=val, test=test, dataset_name=cfg.dataset.name, metadata_dim=metadata_dim)


def make_slices(z: np.ndarray, slice_family: str) -> List[np.ndarray]:
    slices = [np.ones(z.shape[0], dtype=bool)]
    for j in range(z.shape[1]):
        slices.append(z[:, j] == 0)
        slices.append(z[:, j] == 1)
    if slice_family == "single_feature+depth2_tree":
        for j in range(z.shape[1]):
            for k in range(j + 1, z.shape[1]):
                slices.append((z[:, j] == 1) & (z[:, k] == 1))
                slices.append((z[:, j] == 0) & (z[:, k] == 1))
                slices.append((z[:, j] == 1) & (z[:, k] == 0))
                slices.append((z[:, j] == 0) & (z[:, k] == 0))
    return slices
