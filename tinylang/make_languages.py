from tinylang.language.pcfg import Language
import os
import numpy as np
import torch
import random
import argparse
import yaml
import glob

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_language(config_path, output_name):
    # fix all seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    config = load_config(config_path)
    language = Language.from_config(config)
    language.prepare_sets(
        train_set_size=100032,
        eval_set_size=1024,
    )
    print(output_name)
    for k, v in language.stats["test"].items():
        if isinstance(v, list):
            print(f"{k:>40}: {np.mean(v):.5f}, {np.min(v):.5f}, {np.max(v):.5f}")
    language.save(f"../languages/{config_path.split('/')[-2]}/{output_name}.pkl")

def main():
    os.makedirs("../languages", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="pcfg_vary_rhs")
    args = parser.parse_args()

    config_dir = f"../languages/{args.config}"
    for config_path in sorted(glob.glob(f"{config_dir}/*.yaml")):
        name = os.path.basename(config_path).split('.')[0]
        generate_language(config_path, name)

if __name__ == "__main__":
    main()