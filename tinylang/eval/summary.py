from .eval import Evaluator
from tinylang.model import Model
import numpy as np
import torch
from collections import defaultdict

class SummaryEvaluator(Evaluator):
    def __str__(self):
        return "SummaryEvaluator"
    
    def __init__(self, run_every_n_steps: int):
        super().__init__(run_every_n_steps)

    def eval(self, model: Model, inputs: dict, outputs: dict, step: int):
        loss, logits = outputs["loss"], outputs["logits"]
        self.all_eval_stats[step]["loss"] = loss.item()

        # compute kl divs
        for i in range(len(inputs["probing_schemas"])):
            type = inputs["probing_schemas"][i]["type"]
            toks = inputs["strs"][i].split(" ")
            for q, info in inputs["probing_schemas"][i]["queries"].items():
                true_probs = info["target_distribution"]
                if true_probs is None:
                    continue
                target_pos = info["pos"]
                target_val = inputs["labels"][i][target_pos + 1]
                pred_logits = logits[i, target_pos]
                pred_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1).to("cpu")
                kl_divs = torch.nn.functional.kl_div(pred_probs, true_probs, reduction="none", log_target=False).sum(dim=-1)
                self.all_eval_stats[step][f"{type}.{q}.kl_div"].append(kl_divs.item())
                self.all_eval_stats[step][f"{q}.kl_div"].append(kl_divs.item())
                self.all_eval_stats[step][f"{type}.{q}.pred_prob"].append(pred_probs[target_val].exp().item())
                self.all_eval_stats[step][f"{q}.pred_prob"].append(pred_probs[target_val].exp().item())