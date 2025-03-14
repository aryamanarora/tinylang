import argparse
import yaml
from .experiment import Experiment


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run a tinylang experiment')
    parser.add_argument('config', type=str, help='Path to config YAML file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create and run experiment
    experiment = Experiment.from_config(config)
    experiment.train()


if __name__ == '__main__':
    main() 