import argparse
import yaml
import os
from .experiment import Experiment


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run a tinylang experiment')
    parser.add_argument('config', type=str, help='Path to config YAML file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    # get parent of folder that config is in
    filename = os.path.basename(args.config)
    config_dir = os.path.dirname(args.config)
    parent_dir = os.path.dirname(config_dir)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        config["training"]["log_dir"] = os.path.join(parent_dir, f"logs/{filename.split('.')[0]}")
    
    # Create and run experiment
    config["training"]["verbose"] = args.verbose
    experiment = Experiment.from_config(config)
    experiment.train()


if __name__ == '__main__':
    main() 