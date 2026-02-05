import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import torch
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from .model import LLMWrapper, MetadataPredictor, PromptProposer
from .preprocess import DatasetBundle, load_dataset_bundle, make_slices


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class CandidateResult:
    prompt: str
    score: float
    metrics: Dict[str, float]


def lcb_hoeffding(mean: float, n: int, delta: float) -> float:
    rad = math.sqrt(math.log(2.0 / delta) / (2.0 * max(int(n), 1)))
    return float(mean - rad)


def lcb_normal_approx(mean: float, n: int, delta: float) -> float:
    if n <= 1:
        return float(mean - 1.0)
    z = abs(torch.distributions.Normal(0, 1).icdf(torch.tensor(delta / 2))).item()
    se = math.sqrt(max(mean * (1.0 - mean), 1e-6) / n)
    return float(mean - z * se)


def lcb_empirical_bernstein(mean: float, var: float, n: int, delta: float) -> float:
    if n <= 1:
        return float(mean - 1.0)
    rad = math.sqrt(2.0 * var * math.log(3.0 / delta) / n) + 3.0 * math.log(3.0 / delta) / n
    return float(mean - rad)


def score_candidate_savr(
    y: np.ndarray,
    z: np.ndarray,
    delta_total: float,
    slice_family: str,
    cs_type: str,
) -> float:
    slices = make_slices(z, slice_family=slice_family)
    if not slices:
        return -1e9
    lcb_vals = []
    for mask in slices:
        n = int(mask.sum())
        if n == 0:
            continue
        mean = float(y[mask].mean())
        if cs_type == "stitched_hoeffding":
            lcb = lcb_hoeffding(mean, n, delta_total / max(len(slices), 1))
        else:
            var = float(y[mask].var())
            lcb = lcb_empirical_bernstein(mean, var, n, delta_total / max(len(slices), 1))
        lcb_vals.append(lcb)
    return float(min(lcb_vals)) if lcb_vals else -1e9


def score_candidate_naive(y: np.ndarray, delta: float, lcb_type: str) -> float:
    mean = float(y.mean())
    n = int(y.shape[0])
    if lcb_type == "normal_approx":
        return lcb_normal_approx(mean, n, delta)
    return lcb_hoeffding(mean, n, delta)


def evaluate_candidate_race(
    model: LLMWrapper,
    dataset: DatasetBundle,
    prompt: str,
    n0: int,
    n1: int,
    stage2: bool,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    val_samples = dataset.sample_val(n0)
    prompts = [dataset.format_prompt(prompt, ex) for ex in val_samples]
    labels = [ex["label"] for ex in val_samples]
    preds = model.generate_batch(prompts, batch_size=batch_size)
    assert len(preds) == len(labels), "Prediction/label length mismatch"
    y = np.array([int(p == y) for p, y in zip(preds, labels)], dtype=np.int64)
    z = np.stack([ex["metadata"] for ex in val_samples], axis=0)
    used = n0
    if stage2 and n1 > 0:
        val_samples2 = dataset.sample_val(n1)
        prompts2 = [dataset.format_prompt(prompt, ex) for ex in val_samples2]
        labels2 = [ex["label"] for ex in val_samples2]
        preds2 = model.generate_batch(prompts2, batch_size=batch_size)
        y2 = np.array([int(p == y) for p, y in zip(preds2, labels2)], dtype=np.int64)
        z2 = np.stack([ex["metadata"] for ex in val_samples2], axis=0)
        y = np.concatenate([y, y2])
        z = np.concatenate([z, z2])
        used += n1
    return y, z, used


def train_search(cfg: DictConfig, dataset: DatasetBundle, model: LLMWrapper) -> Dict[str, Any]:
    proposer = PromptProposer(seed=cfg.training.seed)
    history: List[CandidateResult] = []
    label_budget_used = 0
    best_scores = []

    for it in range(cfg.optimization.iterations):
        candidates = proposer.propose(cfg.optimization.candidates_per_iter, history)
        scored: List[CandidateResult] = []

        stage1_results = []
        for prompt in candidates:
            y, z, used = evaluate_candidate_race(
                model,
                dataset,
                prompt,
                cfg.optimization.race_n0,
                0,
                False,
                cfg.training.batch_size,
            )
            label_budget_used += used
            if "SAVR" in cfg.method:
                score = score_candidate_savr(
                    y,
                    z,
                    cfg.optimization.delta_total,
                    cfg.optimization.slice_family,
                    cfg.optimization.cs_type,
                )
            else:
                score = score_candidate_naive(
                    y,
                    cfg.optimization.delta,
                    cfg.optimization.lcb_type,
                )
            stage1_results.append((prompt, y, z, score))

        stage1_results.sort(key=lambda x: x[3], reverse=True)

        for rank, (prompt, y, z, _) in enumerate(stage1_results):
            stage2 = rank < cfg.optimization.promote_M
            y_final, z_final, used = evaluate_candidate_race(
                model,
                dataset,
                prompt,
                0,
                cfg.optimization.race_n1,
                stage2,
                cfg.training.batch_size,
            )
            if stage2:
                y_all = np.concatenate([y, y_final]) if y_final.size else y
                z_all = np.concatenate([z, z_final]) if z_final.size else z
                label_budget_used += used
            else:
                y_all, z_all = y, z
            if "SAVR" in cfg.method:
                score = score_candidate_savr(
                    y_all,
                    z_all,
                    cfg.optimization.delta_total,
                    cfg.optimization.slice_family,
                    cfg.optimization.cs_type,
                )
            else:
                score = score_candidate_naive(
                    y_all,
                    cfg.optimization.delta,
                    cfg.optimization.lcb_type,
                )
            scored.append(CandidateResult(prompt=prompt, score=score, metrics={"val_score": score}))

        scored.sort(key=lambda r: r.score, reverse=True)
        history.extend(scored)
        history = history[: cfg.optimization.top_k]

        best = history[0]
        best_scores.append(best.score)
        if cfg.wandb.mode != "disabled":
            wandb.log(
                {
                    "iteration": it,
                    "best_val_score": best.score,
                    "label_budget_used": label_budget_used,
                },
                step=it,
            )

    best_prompt = history[0].prompt
    return {
        "best_prompt": best_prompt,
        "label_budget_used": label_budget_used,
        "best_val_score": max(best_scores) if best_scores else float("-inf"),
    }


def evaluate_on_test(cfg: DictConfig, dataset: DatasetBundle, model: LLMWrapper, prompt: str) -> Dict[str, float]:
    test_samples = dataset.test
    prompts = [dataset.format_prompt(prompt, ex) for ex in test_samples]
    labels = [ex["label"] for ex in test_samples]
    preds = model.generate_batch(prompts, batch_size=cfg.training.batch_size)
    y = np.array([int(p == y) for p, y in zip(preds, labels)], dtype=np.int64)
    z = np.stack([ex["metadata"] for ex in test_samples], axis=0)
    g = dataset.latent_group(z)
    a0 = float(y[g == 0].mean()) if np.any(g == 0) else 0.0
    a1 = float(y[g == 1].mean()) if np.any(g == 1) else 0.0
    worst = float(min(a0, a1))
    avg = float(y.mean())
    gap = float(abs(a0 - a1))

    return {
        "worst_group_accuracy": worst,
        "average_test_accuracy": avg,
        "group_gap": gap,
        "group_acc_0": a0,
        "group_acc_1": a1,
    }


def empirical_lcb_coverage(cfg: DictConfig, dataset: DatasetBundle, model: LLMWrapper, prompt: str) -> float:
    violations = 0
    total = 0
    for _ in range(cfg.evaluation.coverage_checks):
        samples = dataset.sample_val(cfg.evaluation.coverage_batch)
        prompts = [dataset.format_prompt(prompt, ex) for ex in samples]
        labels = [ex["label"] for ex in samples]
        preds = model.generate_batch(prompts, batch_size=cfg.training.batch_size)
        y = np.array([int(p == y) for p, y in zip(preds, labels)], dtype=np.int64)
        z = np.stack([ex["metadata"] for ex in samples], axis=0)
        slices = make_slices(z, slice_family=cfg.optimization.slice_family)
        for mask in slices:
            n = int(mask.sum())
            if n == 0:
                continue
            mean = float(y[mask].mean())
            if "SAVR" in cfg.method:
                lcb = lcb_hoeffding(mean, n, cfg.optimization.delta_total)
            else:
                lcb = lcb_hoeffding(mean, n, cfg.optimization.delta)
            true_acc = mean
            total += 1
            if true_acc < lcb:
                violations += 1
    return float(violations / max(total, 1))


def train_metadata_predictor(cfg: DictConfig, dataset: DatasetBundle, device: str) -> None:
    if cfg.training.eval_only:
        return
    model = MetadataPredictor(input_dim=dataset.metadata_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    x = torch.tensor(np.stack([ex["metadata"] for ex in dataset.train], axis=0), dtype=torch.float32).to(device)
    y = torch.tensor([ex["label_binary"] for ex in dataset.train], dtype=torch.float32).to(device)

    assert x.shape[0] == y.shape[0], "Batch-start shape mismatch"

    model.train()
    for epoch in range(cfg.training.epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x).squeeze(-1)
        loss = loss_fn(logits, y)
        aux_grads = torch.autograd.grad(loss, model.parameters(), create_graph=False, retain_graph=True)
        grad_norm = sum(g.detach().abs().sum() for g in aux_grads)
        loss.backward()

        grads_exist = [p.grad is not None for p in model.parameters()]
        assert all(grads_exist), "Gradients missing before optimizer.step"
        non_zero = [torch.any(p.grad != 0).item() for p in model.parameters() if p.grad is not None]
        assert any(non_zero), "All gradients are zero before optimizer.step"

        optimizer.step()
        if cfg.wandb.mode != "disabled":
            wandb.log({"aux_loss": loss.item(), "aux_grad_norm": grad_norm.item()}, step=epoch)


def optuna_search(cfg: DictConfig, dataset: DatasetBundle, model: LLMWrapper) -> Dict[str, Any]:
    def objective(trial: optuna.Trial) -> float:
        cfg_trial = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_trial.wandb.mode = "disabled"
        for space in cfg_trial.optuna.search_spaces:
            name = space.param_name
            if space.distribution_type == "int":
                val = trial.suggest_int(name, int(space.low), int(space.high))
            elif space.distribution_type == "loguniform":
                val = trial.suggest_float(name, float(space.low), float(space.high), log=True)
            else:
                val = trial.suggest_categorical(name, list(space.choices))
            cfg_trial.optimization[name] = val
        results = train_search(cfg_trial, dataset, model)
        return float(results["best_val_score"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=cfg.optuna.n_trials)
    best_params = study.best_params
    for k, v in best_params.items():
        cfg.optimization[k] = v
    return best_params


def run(cfg: DictConfig) -> None:
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.optimization.iterations = min(cfg.optimization.iterations, 1)
        cfg.optimization.candidates_per_iter = min(cfg.optimization.candidates_per_iter, 2)
        cfg.optimization.race_n0 = min(cfg.optimization.race_n0, 4)
        cfg.optimization.race_n1 = min(cfg.optimization.race_n1, 8)
        if "dataset" in cfg:
            cfg.dataset.max_samples = min(cfg.dataset.get("max_samples", 128), 64)

    set_seed(cfg.training.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_dataset_bundle(cfg)
    model = LLMWrapper(cfg.model, device=device)

    assert model.tokenizer.pad_token_id is not None, "Tokenizer pad_token_id must be set"
    assert model.model.config.vocab_size > 0, "Model vocab size invalid"

    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
        )
        print(wandb.run.url)

    train_metadata_predictor(cfg, dataset, device)

    if cfg.optuna.n_trials > 0:
        best_params = optuna_search(cfg, dataset, model)
        if cfg.wandb.mode != "disabled":
            wandb.summary["optuna_best_params"] = best_params

    results = train_search(cfg, dataset, model)
    metrics = evaluate_on_test(cfg, dataset, model, results["best_prompt"])
    metrics["label_budget_used"] = results["label_budget_used"]
    metrics["anytime_valid_lcb_coverage"] = empirical_lcb_coverage(cfg, dataset, model, results["best_prompt"])

    if cfg.wandb.mode != "disabled":
        wandb.log(metrics)
        for k, v in metrics.items():
            wandb.summary[k] = v
        wandb.finish()


if __name__ == "__main__":
    from hydra import main as hydra_main

    @hydra_main(config_path="../config", config_name="config", version_base=None)
    def _main(cfg: DictConfig) -> None:
        run(cfg)

    _main()
