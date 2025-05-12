import yaml
from copy import deepcopy

ablations = {
    "mixer_type": ["attention", "hyena", "base_conv", "h3", "based", "mamba"],
    "n_embd": [16, 32, 64, 128, 256],
    "lr": [1e-2, 3e-2],
}

with open("template.yaml", "r") as f:
    config = yaml.safe_load(f)

flops_unit = None

for mixer_type in ablations["mixer_type"]:
    for n_embd in ablations["n_embd"]:
        for lr in ablations["lr"]:
            new_config = deepcopy(config)
            new_config["model"]["config"]["mixer_type"] = mixer_type
            new_config["model"]["config"]["n_embd"] = n_embd
            if mixer_type == "mamba":
                new_config["model"]["config"]["state_mixer_type"] = None
            new_config["training"]["lr"] = lr
            # save config
            name = f"{mixer_type}___{n_embd}___{lr:.0e}"
            with open(f"./{name}.yaml", "w") as f:
                yaml.dump(new_config, f)
