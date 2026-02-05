import json
import yaml
from pathlib import Path
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
    run_config_loaded = False
    if hasattr(cfg, 'run') and cfg.run is not None and not isinstance(cfg.run, str):
        # Override config values with run-specific settings from Hydra
        for key in ['model', 'dataset', 'training', 'optuna']:
            if key in cfg.run:
                cfg[key] = OmegaConf.merge(cfg.get(key, {}), cfg.run[key])
        if 'method' in cfg.run:
            cfg['method'] = cfg.run.method
        run_config_loaded = True

    # If run config is not loaded from Hydra, try loading from research_history.json
    if not run_config_loaded:
        research_history_path = Path(".research/research_history.json")
        run_id = None

        # Get run_id from command line args or cfg
        if hasattr(cfg, 'run'):
            if isinstance(cfg.run, str):
                run_id = cfg.run
            elif cfg.run is None:
                # Check if run was passed as a command line override
                import sys
                for arg in sys.argv:
                    if arg.startswith('run='):
                        run_id = arg.split('=', 1)[1]
                        break

        if research_history_path.exists() and run_id:
            with open(research_history_path, 'r') as f:
                research_data = json.load(f)

            # Search for run config in research_history.json
            run_config_yaml = None
            if 'experiment_code' in research_data:
                exp_code = research_data['experiment_code']
                if 'run_configs' in exp_code and run_id in exp_code['run_configs']:
                    run_config_yaml = exp_code['run_configs'][run_id]

            # Parse and merge the run config if found
            if run_config_yaml:
                run_config = yaml.safe_load(run_config_yaml)
                for key in ['model', 'dataset', 'training', 'optuna']:
                    if key in run_config:
                        cfg[key] = OmegaConf.merge(cfg.get(key, {}), OmegaConf.create(run_config[key]))
                if 'method' in run_config:
                    cfg['method'] = run_config['method']

    # Call the train run function directly instead of subprocess
    run(cfg)


if __name__ == "__main__":
    main()
