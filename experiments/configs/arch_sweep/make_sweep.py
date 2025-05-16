import yaml
from copy import deepcopy
from tinylang.model.zoology import Zoology
from tinylang.language import PCFG
import calflops
import torch.profiler
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
ablations = {
    "mixer_type": ["attention", "hyena", "base_conv", "h3", "based"],
    "n_embd": [64, 128, 256, 512],
    "lr": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
    # "lr": [1e-5],
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
            new_config["training"]["lr"] = lr
            # need to init the model to check flops
            vocab_size = PCFG(**new_config["language"]["config"]).vocab_size
            model = Zoology(vocab_size=vocab_size, device=device, **new_config["model"]["config"])
            batch_size = new_config["training"]["train_batch_size"]
            max_pos = new_config["model"]["config"]["n_positions"]
            random_input = torch.randint(0, vocab_size, (batch_size, max_pos)).to(device)
            # flops, macs, params = calflops.calculate_flops(model.model, kwargs={"input_ids": random_input}, output_as_string=False, print_results=False)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
                with_flops=True,  # ← enable FLOP count
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                with torch.no_grad():
                    model.model(input_ids=random_input)
            total_flops = sum(
                e.flops for e in prof.key_averages() if e.flops is not None
            )
            if flops_unit is None:
                flops_unit = total_flops
            factor = flops_unit / total_flops
            new_config["training"]["num_train_steps"] = int(new_config["training"]["num_train_steps"] * factor)
            # save config
            name = f"{mixer_type}_{n_embd}_{lr:.0e}"
            print(name, total_flops, new_config["training"]["num_train_steps"])
            with open(f"./{name}.yaml", "w") as f:
                yaml.dump(new_config, f)
            del model
            torch.cuda.empty_cache()
