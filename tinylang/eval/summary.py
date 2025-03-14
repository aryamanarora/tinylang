from .eval import Evaluator
from tinylang.model import Model
import numpy as np
import torch


class SummaryEvaluator(Evaluator):
    def __str__(self):
        return "SummaryEvaluator"
    
    def __init__(self, run_every_n_steps: int):
        super().__init__(run_every_n_steps)
        self.agg_funcs = {
            "loss": lambda x: np.mean(x).item(),
            "kl_div": lambda x: np.mean(x).item(),
            "pred_prob": lambda x: np.mean(x).item(),
        }

    def eval(self, model: Model, inputs: dict, outputs: dict):
        loss, logits = outputs["loss"], outputs["logits"]
        result = {"loss": loss.item()}

        # compute kl divs
        target_pos = [x["target_pos"] for x in inputs["probing_schemas"]]
        true_probs = torch.stack([x["target_distribution"] for x in inputs["probing_schemas"]])
        pred_logits = logits[torch.arange(logits.size(0)), target_pos]
        pred_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
        kl_divs = torch.nn.functional.kl_div(pred_probs, true_probs, reduction="none", log_target=False).sum(dim=-1)

        # return loss and kl div
        result["kl_div"] = kl_divs.mean().item()
        for i in range(len(kl_divs)):
            target_val = inputs["labels"][i][target_pos[i] + 1]
            type = inputs["probing_schemas"][i]["type"]
            if type not in result:
                result[f"{type}.kl_div"] = []
                result[f"{type}.pred_prob"] = []
            result[f"{type}.kl_div"].append(kl_divs[i].item())
            result[f"{type}.pred_prob"].append(pred_probs[i][target_val].exp().item())
        return result

