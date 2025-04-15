import argparse
import yaml
import os
from .experiment import Experiment


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run a tinylang experiment')
    parser.add_argument('config', type=str, help='Path to config YAML file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--wandb', action='store_true', help='Use wandb')
    args = parser.parse_args()

    # get parent of folder that config is in
    filename = os.path.basename(args.config)
    parent_dir = "experiments/"

    # path of config after configs/
    config_path = args.config.split("configs/")[1].split(".")[0]

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        config["training"]["log_dir"] = os.path.join(parent_dir, f"logs/{config_path}")
    
    # Create and run experiment
    config["training"]["verbose"] = args.verbose
    config["training"]["wandb"] = args.wandb
    experiment = Experiment.from_config(config)
    experiment.train()


if __name__ == '__main__':
    main() 