import yaml
from copy import deepcopy

ablations = {
    "lr": [1e-5, 8e-5, 1e-4, 8e-4, 1e-3],
    "state_mixer_type": ["mlp", "glu", None],
    "bias": [True, False],
}

with open("template.yaml", "r") as f:
    config = yaml.safe_load(f)

for lr in ablations["lr"]:
    for state_mixer_type in ablations["state_mixer_type"]:
        for bias in ablations["bias"]:
            new_config = deepcopy(config)
            new_config["training"]["lr"] = lr
            new_config["model"]["config"]["state_mixer_type"] = state_mixer_type
            new_config["model"]["config"]["bias"] = bias
            name = f"{state_mixer_type}_{lr:.0e}_{'bias' if bias else 'no_bias'}"
            with open(f"./{name}.yaml", "w") as f:
                yaml.dump(new_config, f)
