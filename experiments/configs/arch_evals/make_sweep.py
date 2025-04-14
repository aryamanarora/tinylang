import yaml
from copy import deepcopy

ablations = {
    "mixer_type": ["attention", "hyena", "rwkv", "base_conv", "h3", "based", "mamba"],
    "n_embd": [64, 128, 256, 512],
}

with open("template.yaml", "r") as f:
    config = yaml.safe_load(f)

for mixer_type in ablations["mixer_type"]:
    for n_embd in ablations["n_embd"]:
        new_config = deepcopy(config)
        new_config["model"]["config"]["mixer_type"] = mixer_type
        new_config["model"]["config"]["n_embd"] = n_embd
        with open(f"./{mixer_type}_{n_embd}.yaml", "w") as f:
            yaml.dump(new_config, f)
