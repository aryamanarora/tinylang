import yaml
from copy import deepcopy

# one less, two more
lang_ablations = {
    "max_rhs_len": [5, 20, 40],
    "num_nonterminals": [20, 80, 160],
    "num_terminals": [10, 40, 80],
    "max_rules_per_nt": [5, 20, 0],
    "max_depth": [5, 20, 40],
}

lr_ablations = {
    "lr": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
}

with open("template.yaml", "r") as f:
    config = yaml.safe_load(f)

flops_unit = None

for lang_ablation in lang_ablations:
    for lang_ablation_value in lang_ablations[lang_ablation]:
        for lr in lr_ablations["lr"]:
            new_config = deepcopy(config)
            new_config["language"]["config"][lang_ablation] = lang_ablation_value
            new_config["training"]["lr"] = lr
            # save config
            name = f"{lang_ablation}___{lang_ablation_value}___{lr:.0e}"
            with open(f"./{name}.yaml", "w") as f:
                yaml.dump(new_config, f)
