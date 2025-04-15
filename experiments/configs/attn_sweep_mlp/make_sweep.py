import yaml
from copy import deepcopy

ablations = {
    "n_head": [1],
    "n_layer": [1, 2, 3, 4],
    "n_embd": [16, 32, 64, 128],
}

with open("template.yaml", "r") as f:
    config = yaml.safe_load(f)

for n_head in ablations["n_head"]:
    for n_layer in ablations["n_layer"]:
        for n_embd in ablations["n_embd"]:
            new_config = deepcopy(config)
            new_config["model"]["config"]["n_head"] = n_head
            new_config["model"]["config"]["n_layer"] = n_layer
            new_config["model"]["config"]["n_embd"] = n_embd
            with open(f"./{n_layer}_{n_head}_{n_embd}.yaml", "w") as f:
                yaml.dump(new_config, f)
