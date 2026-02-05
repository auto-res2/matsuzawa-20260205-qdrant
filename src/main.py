from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf

from .train import run


@hydra_main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"

    # Merge the run configuration into the main config if it exists
    if hasattr(cfg, 'run') and cfg.run is not None:
        # Override config values with run-specific settings
        for key in ['model', 'dataset', 'training', 'optuna']:
            if key in cfg.run:
                cfg[key] = OmegaConf.merge(cfg.get(key, {}), cfg.run[key])
        if 'method' in cfg.run:
            cfg['method'] = cfg.run.method

    # Call the train run function directly instead of subprocess
    run(cfg)


if __name__ == "__main__":
    main()
