from .eval import Evaluator
from tinylang.model import Model
from tinylang.language import Language
import torch
import plotnine as p9
from collections import defaultdict
import torch.nn as nn
import pyvene as pv
import random
import numpy as np


class InterchangeEvaluator(Evaluator):
    def __str__(self):
        return "InterchangeEvaluator"

    def __init__(self, run_every_n_steps: int):
        super().__init__(run_every_n_steps)
        self.do_batching = False

    @torch.no_grad()
    def eval(self, model: Model, language: Language, inputs: dict, outputs: dict, step: int):
        probing_schemas = inputs["probing_schemas"]
        hidden_states = outputs["hidden_states"]
        types = set([x["type"] for x in probing_schemas])
        

        for layer in range(1, len(hidden_states)):
            true_layer = layer - 1
            config = [
                {"layer": true_layer, "component": "value_output", "group_key": 0, "unit": "pos"},
                {"layer": true_layer, "component": "query_output", "group_key": 1, "unit": "pos"},
                {"layer": true_layer, "component": "key_output", "group_key": 1, "unit": "pos"},
            ]
            for i in range(true_layer + 1, len(hidden_states) - 1):
                config.extend([
                    {"layer": i, "component": "query_output", "group_key": 1, "unit": "pos"},
                    {"layer": i, "component": "key_output", "group_key": 1, "unit": "pos"},
                ])
            num_restores = len(config) - 1
            pv_config = pv.IntervenableConfig(config)
            pv_gpt2 = pv.IntervenableModel(pv_config, model=model.model)
            pv_gpt2.disable_model_gradients()
            for t in types:
                for label_type in probing_schemas[0]["target_distributions"]:
                    for query in probing_schemas[0]["queries"]:
                        for batch_idx in range(len(hidden_states[layer])):
                            if t != probing_schemas[batch_idx]["type"]:
                                continue
                            query_pos = probing_schemas[batch_idx]["queries"][query]["pos"]
                            target_item_orig_pos = probing_schemas[batch_idx]["queries"]["target_item_orig"]["pos"]
                            target_item_pos = probing_schemas[batch_idx]["queries"]["target_item"]["pos"]
                            input_ids = inputs["input_ids"][batch_idx]
                            target_item_orig = input_ids[target_item_orig_pos]
                            target_item_new = random.randint(language.TERMINAL_START, language.QUERY_START - 1)
                            intervened_input_ids = input_ids.clone()
                            intervened_input_ids[target_item_orig_pos] = target_item_new
                            intervened_input_ids[target_item_pos] = target_item_orig

                            base_inputs = {"input_ids": input_ids.unsqueeze(0)}
                            pos = list(range(base_inputs["input_ids"].shape[-1]))
                            _, intervened_outputs = pv_gpt2(
                                base=base_inputs,
                                sources=[
                                    {"input_ids": intervened_input_ids.unsqueeze(0)},
                                    base_inputs,
                                ],
                                unit_locations={"sources->base": ((
                                    [[[query_pos],],] + [[pos]]*num_restores,
                                    [[[query_pos],],] + [[pos]]*num_restores,
                                ))}, 
                            )
                            
                            intervened_logits = intervened_outputs["logits"].cpu()
                            output_logits = outputs["logits"][batch_idx].cpu()
                            intervened_probs = torch.log_softmax(intervened_logits.squeeze(0)[target_item_pos - 1], dim=-1)
                            output_probs = torch.log_softmax(output_logits[target_item_pos - 1], dim=-1)
                            kl_div = torch.nn.functional.kl_div(intervened_probs, output_probs, log_target=True, reduction="batchmean").item()
                            intervened_prob = intervened_probs.exp()[target_item_new]
                            old_prob = output_probs.exp()[target_item_new]
                            intervened_base_prob = intervened_probs.exp()[target_item_orig]
                            old_base_prob = output_probs.exp()[target_item_orig]

                            log_odds_ratio = old_base_prob.log() - intervened_base_prob.log() + intervened_prob.log() - old_prob.log()

                            label = f"{layer - 1}.{t}.{label_type}.{query}"
                            self.all_eval_stats[step][f"{label}.kl_div"].append(kl_div)
                            self.all_eval_stats[step][f"{label}.prob_diff"].append((intervened_prob - old_prob).item())
                            self.all_eval_stats[step][f"{label}.log_odds_ratio"].append(log_odds_ratio.item())

            # deregister intervention
            pv_gpt2.enable_model_gradients()
            pv_gpt2._cleanup_states()
                            

    def post_eval(self, step: int):
        for ending in ["kl_div", "prob_diff", "log_odds_ratio"]:
            top = [(np.mean(v), k) for k, v in self.all_eval_stats[step].items() if k.endswith(ending)]
            for v, k in sorted(top):
                print(f"{k:>80}: {v:.12f}")
            print('------')


    def plot(self, log_dir: str):
        df = self.df
        df = df.groupby(["step", "variable"]).mean().reset_index()
        df["layer"] = df["variable"].str.split(".").str[0]
        df["type"] = df["variable"].str.split(".").str[1]
        df["label_type"] = df["variable"].str.split(".").str[2]
        df["query"] = df["variable"].str.split(".").str[3]
        df["variable"] = df["variable"].str.split(".").str[4]

        # make plot
        for type in df["type"].unique():
            for variable in df["variable"].unique():
                df_subset = df[df["type"] == type]
                df_subset = df_subset[df_subset["variable"] == variable]
                plot = (
                    p9.ggplot(df_subset, p9.aes(x="step", y="value", color="query"))
                    + p9.geom_line()
                    + p9.facet_grid("label_type~layer")
                )
                plot.save(f"{log_dir}/{str(self)}.{type}.{variable}.png")
            