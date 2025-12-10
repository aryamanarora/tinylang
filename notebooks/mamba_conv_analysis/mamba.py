from tinylang.experiment import Experiment
from tinylang.model import LanguageModel

import yaml
import pyvene as pv
import os
import pandas as pd
import torch
import numpy as np
import plotnine as p9
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# update pyvene hooks to include mamba-internal components
pv.type_to_module_mapping[LanguageModel].update({
    "conv_input": ("backbone.layers[%s].sequence_mixer.conv1d_input", 
                        pv.models.constants.CONST_INPUT_HOOK),
    "conv_output": ("backbone.layers[%s].sequence_mixer.conv1d_output", 
                        pv.models.constants.CONST_INPUT_HOOK),
    "in_proj_input": ("backbone.layers[%s].sequence_mixer.in_proj", 
                        pv.models.constants.CONST_INPUT_HOOK),
    "in_proj_output": ("backbone.layers[%s].sequence_mixer.in_proj", 
                        pv.models.constants.CONST_OUTPUT_HOOK),
    "out_proj_input": ("backbone.layers[%s].sequence_mixer.out_proj", 
                        pv.models.constants.CONST_INPUT_HOOK),
    "out_proj_output": ("backbone.layers[%s].sequence_mixer.out_proj", 
                        pv.models.constants.CONST_OUTPUT_HOOK),
    "act_input": ("backbone.layers[%s].sequence_mixer.act_input", 
                        pv.models.constants.CONST_INPUT_HOOK),
    "act_output": ("backbone.layers[%s].sequence_mixer.act_output", 
                        pv.models.constants.CONST_INPUT_HOOK),
})

# metric fields
FIELDS = ["layer", "type", "corrupted", "restored", "component", "metric"]


def analyse_mamba_checkpoint(
    name: str="ar_32_nope/mamba___256___1e-03",
):
    print(f"Analysing {name}")
    # paths
    config_path = f"/nlp/scr/aryaman/tinylang/experiments/configs/{name}.yaml"
    model_path = f"/nlp/scr/aryaman/tinylang/experiments/logs/{name}/model.pt"

    # load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        if config["model"]["config"]["mixer_type"].startswith("mamba"):
            if config["model"]["config"]["d_conv"] == 0:
                print(f"Skipping {name} because d_conv is 0")
                return pd.DataFrame()
            config["model"]["config"]["mixer_type"] = "mamba_no_causal_conv1d" # swap to vanilla pytorch mamba
        config["training"]["log_dir"] = "."

    # load model
    exp = Experiment.from_config(
        config=config
    )
    exp.model.load(model_path)
    exp.verbose = False
    exp.model.model.to(device)
    exp.model.model.eval()

    # update probing schemas to include all positions
    for batch_idx in range(len(exp.language.evalsets["test"]["probing_schemas"])):
        max_pos = exp.language.evalsets["test"]["probing_schemas"][batch_idx]['queries']['target_item']['pos']
        target_item_orig_pos = exp.language.evalsets["test"]["probing_schemas"][batch_idx]['queries']['target_item_orig']['pos']
        exp.language.evalsets["test"]["probing_schemas"][batch_idx]['queries'].update({
            f"target_item_orig__+{i}": {
                "pos": target_item_orig_pos + i,
                "target_distribution": None,
            }
            for i in range(1, 3)
        })

    # eval on all data
    full_eval_length = len(exp.language.stats["test"]["query_orig_target_orig_dist"])
    # exp.training_config.num_eval_steps = int(full_eval_length / exp.training_config.eval_batch_size)
    exp.training_config.num_eval_steps = 1

    # add components to analyse
    # only interested in what conv is doing
    exp.model.components = ["conv_input", "conv_output"]

    # run eval step
    exp.eval_step(0, split="test", evaluators=exp.evaluators["test"])

    # get all eval stats
    all_vals = []
    for key, val in exp.evaluators["test"][1].all_eval_stats[0].items():
        all_vals.append((val, key))

    # for val, key in all_vals:
    #     if 'restored_prob' not in key or not key.split('.')[0].isdigit():
    #         continue
    #     print(f"{key:>80}: {sum(val) / len(val):>6.3f} ({len(val)})")

    data = []
    for val, key in all_vals:
        mapping = dict(zip(FIELDS, key.split(".")))
        mapping["value"] = sum(val) / len(val)
        data.append(mapping)
    df = pd.DataFrame(data)
    print(df[(df.metric == "restored_prob") & (~df.layer.isin(["original", "corrupted"]))].sort_values("value", ascending=False).head(10))
    return df.assign(
        config=name,
        lr=config["training"]["lr"],
        n_embd=config["model"]["config"]["n_embd"],
        n_layer=config["model"]["config"]["n_layer"],
        d_conv=config["model"]["config"].get("d_conv", None),
        mixer_type=config["model"]["config"].get("mixer_type", None),
    )


def ar_32_main():
    all_dfs = []
    for config in ["ar_32_nope", "ar_32_more_lr"]:
        for mamba_checkpoint in sorted(list(glob.glob(f"/nlp/scr/aryaman/tinylang/experiments/configs/{config}/mamba___*___*.yaml"))):
            mamba_name = mamba_checkpoint.split("/")[-1].split(".")[0]
            df = analyse_mamba_checkpoint(config + "/" + mamba_name)
            all_dfs.append(df)
    all_dfs = pd.concat(all_dfs)
    all_dfs.to_csv("mamba_ablation.csv", index=False)


def mamba_conv_ablation():
    all_dfs = []
    for config in ["ar_32_mamba_ablation"]:
        for mamba_checkpoint in sorted(list(glob.glob(f"/nlp/scr/aryaman/tinylang/experiments/configs/{config}/mamba*___*.yaml"))):
            mamba_name = mamba_checkpoint.split("/")[-1].split(".")[0]
            try:
                df = analyse_mamba_checkpoint(config + "/" + mamba_name)
                all_dfs.append(df)
            except Exception as e:
                print(f"Error analysing {mamba_name}: {e}")
                continue
    all_dfs = pd.concat(all_dfs)
    all_dfs.to_csv("mamba_conv_ablation.csv", index=False)


if __name__ == "__main__":
    # ar_32_main()
    mamba_conv_ablation()