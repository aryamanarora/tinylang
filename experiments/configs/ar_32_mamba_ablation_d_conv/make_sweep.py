import yaml
from copy import deepcopy

ablations = {
    "mixer_type": ["mamba"],
    "n_embd": [16, 32, 64, 128, 256],
    "lr": [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    "d_conv": [2],
}

with open("template.yaml", "r") as f:
    config = yaml.safe_load(f)

flops_unit = None

for d_conv in ablations["d_conv"]:
    for n_embd in ablations["n_embd"]:
        for lr in ablations["lr"]:
            new_config = deepcopy(config)
            new_config["model"]["config"]["d_conv"] = d_conv
            new_config["model"]["config"]["n_embd"] = n_embd
            new_config["model"]["config"]["state_mixer_type"] = None
            new_config["training"]["lr"] = lr
            # save config
            name = f"mamba_{d_conv}___{n_embd}___{lr:.0e}"
            with open(f"./{name}.yaml", "w") as f:
                yaml.dump(new_config, f)
