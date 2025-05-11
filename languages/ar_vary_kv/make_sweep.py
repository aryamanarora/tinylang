import yaml
from copy import deepcopy

with open("template.yaml", "r") as f:
    config = yaml.safe_load(f)

for length in [8, 16, 32, 64, 128, 256, 512, 1024]:
    new_config = deepcopy(config)
    new_config["config"]["max_length"] = length
    new_config["config"]["min_length"] = length
    with open(f"./ar_{length}.yaml", "w") as f:
        yaml.dump(new_config, f)