import yaml
from copy import deepcopy

ablations = {
    "lr": [1e-5, 8e-5, 1e-4, 8e-4, 1e-3],
}

with open("template.yaml", "r") as f:
    config = yaml.safe_load(f)

for lr in ablations["lr"]:
    new_config = deepcopy(config)
    new_config["training"]["lr"] = lr
    name = f"{lr:.0e}_gpt2"
    with open(f"./{name}.yaml", "w") as f:
        yaml.dump(new_config, f)
