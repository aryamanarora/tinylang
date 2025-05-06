from tinylang.language.pcfg import Language
import os
import numpy as np
import torch
import random

# fix all seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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

os.makedirs("../languages", exist_ok=True)

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