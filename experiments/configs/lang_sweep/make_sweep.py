import yaml
from copy import deepcopy

ablations = {
    "num_terminals": [20, 40, 80],
    "num_nonterminals": [10, 20, 40],
    "max_rhs_len": [5, 10, 20],
    "max_rules_per_nt": [5, 10, 20],
    "max_depth": [10, 20, 40],
}

with open("template.yaml", "r") as f:
    config = yaml.safe_load(f)

for ablation in ablations:
    for value in ablations[ablation][1:]:
        new_config = deepcopy(config)
        new_config["language"]["config"][ablation] = value
        with open(f"./{ablation}_{value}.yaml", "w") as f:
            yaml.dump(new_config, f)
