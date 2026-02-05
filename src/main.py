import subprocess
import sys

from hydra import main as hydra_main
from omegaconf import DictConfig


@hydra_main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"

    cmd = [
        sys.executable,
        "-m",
        "src.train",
        f"run={cfg.run.run_id}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
