from .eval import Evaluator
from tinylang.model import Model
from tinylang.language import Language
import torch
from collections import defaultdict
import pandas as pd
import plotnine as p9
import os


class LCAEvaluator(Evaluator):
    def __str__(self):
        return "LCAEvaluator"
    
    def __init__(self, run_every_n_steps: int):
        super().__init__(run_every_n_steps)
        self.last_step_weights = None
        self.per_weight_attribs = defaultdict(list)

    def eval(self, model: Model, language: Language, inputs: dict, outputs: dict, step: int):
        """Run per-example backward passes and compute simple weight attributions."""
        model_module = getattr(model, "model", model)

        # Keep CPU copies of current and previous weights to compare across steps.
        prev_weights = self.last_step_weights
        current_weights = {k: v.detach().cpu().clone() for k, v in model_module.state_dict().items()}
        self.last_step_weights = current_weights

        # Need a previous snapshot to form weight deltas.
        if prev_weights is None:
            return

        weight_deltas = {k: current_weights[k] - prev_weights[k] for k in current_weights if k in prev_weights}
        device = next(model_module.parameters()).device
        self.per_weight_attribs[step] = []

        # No batching: run backward one example at a time.
        for idx in range(inputs["input_ids"].size(0)):
            model_module.zero_grad(set_to_none=True)
            with torch.enable_grad():
                input_ids = inputs["input_ids"][idx : idx + 1].to(device)
                labels = inputs["labels"][idx : idx + 1].to(device)
                loss = model.step(input_ids, labels)["loss"]
            loss.backward()

            example_attribs = {}
            for name, param in model_module.named_parameters():
                if param.grad is None or name not in weight_deltas:
                    continue
                grad_cpu = param.grad.detach().cpu()
                example_attribs[name] = weight_deltas[name] * grad_cpu
            self.per_weight_attribs[step].append(example_attribs)

    def plot(self, log_dir: str):
        """Plot attribution trajectories for the first example only."""
        records = []
        for step, example_list in self.per_weight_attribs.items():
            if not example_list:
                continue
            first_example = example_list[0]
            for name, attrib in first_example.items():
                records.append(
                    {
                        "step": step,
                        "weight": name,
                        "value": attrib.sum().item(),
                    }
                )

        if not records:
            return

        df = pd.DataFrame(records)
        top_weights = (
            df.groupby("weight")["value"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .index
        )
        df_top = df[df["weight"].isin(top_weights)]

        plot = (
            p9.ggplot(df_top, p9.aes(x="step", y="value", color="weight"))
            + p9.geom_line()
            + p9.labs(title=f"{str(self)} Top 10 Attributions (example 0)")
        )
        os.makedirs(log_dir, exist_ok=True)
        plot.save(os.path.join(log_dir, f"{str(self)}.top_attribs.png"))
