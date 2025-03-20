import argparse
import yaml
import os
from .experiment import Experiment


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run a tinylang experiment')
    parser.add_argument('config', type=str, help='Path to config YAML file')
    args = parser.parse_args()

    # get parent of folder that config is in
    filename = os.path.basename(args.config)
    config_dir = os.path.dirname(args.config)
    parent_dir = os.path.dirname(config_dir)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        config["training"]["log_dir"] = os.path.join(parent_dir, f"logs/{filename}")
    
    # Create and run experiment
    experiment = Experiment.from_config(config)
    experiment.train()


if __name__ == '__main__':
    main() 