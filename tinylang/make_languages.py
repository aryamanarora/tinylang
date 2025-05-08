from tinylang.language.pcfg import Language
import os
import numpy as np
import torch
import random
import argparse

# fix all seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# function to make languages
def pcfg_vary_max_rhs_len():
    # this is the easy config
    config = {
        "class": "PCFG",
        "config": {
            "head_position": "right",
            "mask_nonquery": True,
            "max_depth": 10,
            "max_rhs_len": 5,
            "max_rules_per_nt": 5,
            "no_child_queries": True,
            "no_sibling_queries": True,
            "num_nonterminals": 40,
            "num_terminals": 20,
            "train_test_split": 0.0,
            "transparent_nonterminals": False,
            "unambiguous_queries": True,
            "prepare_train_set": True,
        },
    }

    for name, max_rhs_len in [("easy", 5), ("medium", 10), ("hard", 15)]:
        config["config"]["max_rhs_len"] = max_rhs_len
        language = Language.from_config(config)
        language.prepare_sets(
            train_set_size=100032,
            eval_set_size=1024,
        )
        for k, v in language.stats["test"].items():
            if isinstance(v, list):
                print(f"{k:>40}: {np.mean(v):.5f}, {np.min(v):.5f}, {np.max(v):.5f}")
        language.save(f"../languages/pcfg_{name}.pkl")


def ar_vary_num_kv():
    config = {
        "class": "AR",
        "config": {
            "num_kv": 8192,
            "max_length": 8,
            "min_length": 8,
            "query_type": "key",
            "mask_nonquery": True,
            "prepare_train_set": True,
        },
    }

    lengths = [8, 16, 32, 64, 128, 256, 512, 1024]
    for length in lengths:
        config["config"]["max_length"] = length
        config["config"]["min_length"] = length
        language = Language.from_config(config)
        language.prepare_sets(
            train_set_size=100032,
            eval_set_size=1024,
        )
        # print(language.get_train_step(0, 1))
        for k, v in language.stats["test"].items():
            if isinstance(v, list):
                print(f"{k:>40}: {np.mean(v):.5f}, {np.min(v):.5f}, {np.max(v):.5f}")
        language.save(f"../languages/ar_{length}.pkl")


# mapping
CONFIG_MAPPING = {
    "pcfg_vary_max_rhs_len": pcfg_vary_max_rhs_len,
    "ar_vary_num_kv": ar_vary_num_kv,
}


def main():
    os.makedirs("../languages", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="pcfg_vary_max_rhs_len")
    args = parser.parse_args()

    CONFIG_MAPPING[args.config]()


if __name__ == "__main__":
    main()