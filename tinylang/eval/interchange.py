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
        
        for component in ["attention_input", "attention_output"]:
            for layer in range(1, len(hidden_states)):
                
                true_layer = layer - 1
                config = {"layer": true_layer, "component": component, "unit": "pos"}
                pv_config = pv.IntervenableConfig(config)
                pv_gpt2 = pv.IntervenableModel(pv_config, model=model.model)
                pv_gpt2.disable_model_gradients()

                for t in types:
                    for label_type in probing_schemas[0]["target_distributions"]:
                        for query in probing_schemas[0]["queries"]:
                            for batch_idx in range(len(hidden_states[layer])):
                                if t != probing_schemas[batch_idx]["type"]:
                                    continue
                                input_ids = inputs["input_ids"][batch_idx]
                                divider_pos = probing_schemas[batch_idx]["queries"]["divider"]["pos"]
                                
                                # how we will create the source sequence
                                pos_to_change = probing_schemas[batch_idx]["queries"][label_type]["pos"]
                                orig_value = probing_schemas[batch_idx]["target_distributions"][label_type]
                                orig_token_value = input_ids[pos_to_change]
                                # corrupted_value = random.choice([x for x in input_ids[1:divider_pos] if x != orig_token_value])
                                corrupted_value = random.randint(language.TERMINAL_START, language.QUERY_START - 1)

                                # create the source sequence
                                corrupted_input_ids = inputs["input_ids"][batch_idx].clone()
                                corrupted_input_ids[pos_to_change] = corrupted_value
                                base_inputs = {"input_ids": input_ids.unsqueeze(0)}
                                corrupted_inputs = {"input_ids": corrupted_input_ids.unsqueeze(0)}

                                # if label_type == "query_item_orig":
                                #     print(language.prettify(inputs["input_ids"][batch_idx], probing_schemas[batch_idx])[0])
                                #     print(language.prettify(corrupted_input_ids))
                                #     input()

                                # do the intervention at the query position
                                pos_to_intervene = probing_schemas[batch_idx]["queries"][query]["pos"]
                                corrupted_outputs, intervened_outputs = pv_gpt2(
                                    base=corrupted_inputs,
                                    sources=[base_inputs],
                                    unit_locations={"sources->base": int(pos_to_intervene)},
                                    output_original_output=True
                                )

                                # compute metrics
                                pos_to_check = probing_schemas[batch_idx]["queries"]["target_item"]["pos"] - 1
                                orig_output = inputs["input_ids"][batch_idx][pos_to_check + 1]
                                intervened_logits = intervened_outputs["logits"].cpu()
                                original_logits = outputs["logits"][batch_idx].cpu()
                                corrupted_logits = corrupted_outputs["logits"].cpu()

                                intervened_probs = torch.log_softmax(intervened_logits.squeeze(0)[pos_to_check], dim=-1)
                                corrupted_probs = torch.log_softmax(corrupted_logits.squeeze(0)[pos_to_check], dim=-1)

                                # # print top 5 logits for each
                                # intervened_top_5 = intervened_logits.squeeze(0)[pos_to_check].topk(5)
                                # corrupted_top_5 = corrupted_logits.squeeze(0)[pos_to_check].topk(5)
                                # original_top_5 = original_logits[pos_to_check].topk(5)

                                # if step > 1500 and batch_idx == 0:
                                #     print(f'---\n{label_type}')
                                #     print(language.prettify(inputs["input_ids"][batch_idx]))
                                #     print(language.prettify(corrupted_input_ids))
                                #     for l in [intervened_top_5, corrupted_top_5, original_top_5]:
                                #         for i in range(5):
                                #             print(f"{language.id_to_token[l.indices[i].item()]:>5}: {l.values[i]:.5f}")
                                #         print('---')
                                #     input()

                                kl_div = torch.nn.functional.kl_div(intervened_probs, corrupted_probs, log_target=True, reduction="batchmean")
                                intervened_prob = intervened_probs.exp()[orig_output]
                                corrupted_prob = corrupted_probs.exp()[orig_output]
                                original_prob = torch.log_softmax(original_logits[pos_to_check], dim=-1).exp()[orig_output]
                                # percent_restored = (intervened_prob - corrupted_prob) / (original_prob - corrupted_prob)

                                label = f"{layer}.{t}.{label_type}.{query}.{component}"
                                label_original = f"original.{t}.{label_type}.{query}.{component}"
                                label_corrupted = f"corrupted.{t}.{label_type}.{query}.{component}"
                                self.all_eval_stats[step][f"{label}.kl_div"].append(kl_div.item())
                                self.all_eval_stats[step][f"{label}.restored_prob"].append(intervened_prob.item())
                                self.all_eval_stats[step][f"{label_corrupted}.restored_prob"].append(corrupted_prob.item())
                                self.all_eval_stats[step][f"{label_original}.restored_prob"].append(original_prob.item())
                                # self.all_eval_stats[step][f"{label}.percent_restored"].append(percent_restored.item())

                # deregister intervention
                pv_gpt2._cleanup_states()
                torch.cuda.empty_cache()
                            

    def post_eval(self, step: int):
        for ending in ["kl_div", "restored_prob"]:
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
        df["component"] = df["variable"].str.split(".").str[4]
        df["variable"] = df["variable"].str.split(".").str[5]

        # make plot
        for component in df["component"].unique():
            for type in df["type"].unique():
                for variable in df["variable"].unique():
                    df_subset = df[df["type"] == type]
                    df_subset = df_subset[df_subset["component"] == component]
                    df_subset = df_subset[df_subset["variable"] == variable]
                    plot = (
                        p9.ggplot(df_subset, p9.aes(x="step", y="value", color="query"))
                        + p9.geom_line()
                        + p9.facet_grid("label_type~layer")
                    )
                    plot.save(f"{log_dir}/{str(self)}.{type}.{component}.{variable}.png")
            