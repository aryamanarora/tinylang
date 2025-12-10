from .eval import Evaluator
from tinylang.model import Model
from tinylang.language import Language
import torch
from collections import defaultdict
from itertools import product
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from labellines import labelLines


class LCAEvaluator(Evaluator):
    def __str__(self):
        return "LCAEvaluator"
    
    def __init__(self, run_every_n_steps: int):
        super().__init__(run_every_n_steps)
        self.last_step_weights = None
        self.per_weight_attribs = defaultdict(list)
        self.examples_by_step = {}

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

        # Track a readable example for logging later.
        self.examples_by_step[step] = self._format_example(language, inputs)

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

    def _format_example(self, language: Language, inputs: dict) -> dict:
        """Return both raw token ids and pretty text for example 0 if possible."""
        tokens = inputs["input_ids"][0].detach().cpu().tolist()
        pad_token = getattr(language, "PAD", None)
        if pad_token is not None:
            while tokens and tokens[-1] == pad_token:
                tokens.pop()
        if "strs" in inputs and inputs["strs"]:
            pretty = inputs["strs"][0]
        elif hasattr(language, "prettify"):
            pretty = language.prettify(tokens)
        else:
            pretty = " ".join(str(tok) for tok in tokens)
        return {"pretty": pretty, "tokens": tokens}

    def plot(self, log_dir: str):
        """Plot attribution trajectories for the first example only."""
        records = []
        for step, example_list in self.per_weight_attribs.items():
            if not example_list:
                continue
            first_example = example_list[0]
            example_info = self.examples_by_step.get(step)
            if example_info is not None:
                print(f"{str(self)} step {step} example 0 pretty: {example_info['pretty']}")
                print(f"{str(self)} step {step} example 0 tokens: {example_info['tokens']}")
            for name, attrib in first_example.items():
                # if "wte" in name:
                #     continue
                if attrib.numel() == 0:
                    continue
                # records.append(
                #     {
                #         "step": step,
                #         "weight": name,
                #         "value": attrib.sum().item(),
                #     }
                # )
                indices = (
                    [()]
                    if attrib.dim() == 0
                    else product(*(range(dim) for dim in attrib.shape))
                )
                for idx in indices:
                    idx_suffix = "".join(f"[{i}]" for i in idx)
                    value = attrib.item() if not idx else attrib[idx].item()
                    records.append(
                        [
                            step,
                            name + idx_suffix,
                            name,
                            value,
                        ]
                    )

        if not records:
            return

        df = pd.DataFrame(records, columns=["step", "weight", "module", "value"])
        weight_totals = df.groupby("weight")["value"].sum()
        weight_totals = weight_totals.reindex(weight_totals.abs().sort_values(ascending=False).index)
        if weight_totals.empty:
            return

        top_500_weights = weight_totals.head(500).index.tolist()

        module_top_weights = {}
        for module, module_df in df.groupby("module"):
            weight_scores = (
                module_df.groupby("weight")["value"]
                .sum()
                .abs()
                .sort_values(ascending=False)
            )
            module_top_weights[module] = weight_scores.head(2).index.tolist()

        selected_weights = set(top_500_weights)
        for weights in module_top_weights.values():
            selected_weights.update(weights)

        df_top = df[df["weight"].isin(selected_weights)].copy()
        if df_top.empty:
            return

        module_order = (
            df.groupby("module")["value"]
            .sum()
            .abs()
            .sort_values(ascending=False)
            .index
            .tolist()
        )
        if not module_order:
            return

        highlight_weights_order = []
        for module in module_order:
            for weight in module_top_weights.get(module, []):
                if weight not in highlight_weights_order:
                    highlight_weights_order.append(weight)

        color_lookup = {}
        if highlight_weights_order:
            cmap = plt.get_cmap("tab20", len(highlight_weights_order))
            for idx, weight in enumerate(highlight_weights_order):
                color_lookup[weight] = cmap(idx)

        y_min_global = df_top["value"].min()
        y_max_global = df_top["value"].max()
        if y_min_global == y_max_global:
            y_min_global -= 1.0
            y_max_global += 1.0
        else:
            padding = 0.05 * (y_max_global - y_min_global)
            y_min_global -= padding
            y_max_global += padding

        n_modules = len(module_order)
        ncols = min(4, n_modules)
        nrows = math.ceil(n_modules / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6 * ncols, 4.5 * nrows),
            sharex=False,
            sharey=True,
        )

        if hasattr(axes, "flat"):
            axes_iter = list(axes.flat)
        else:
            axes_iter = [axes]

        for ax, module in zip(axes_iter, module_order):
            module_df = df_top[df_top["module"] == module].sort_values("step")
            if module_df.empty:
                ax.set_visible(False)
                continue

            top_lines = []

            for weight, weight_df in module_df.groupby("weight"):
                weight_df = weight_df.sort_values("step")
                color = color_lookup.get(weight, "black")
                is_top = weight in color_lookup
                line, = ax.plot(
                    weight_df["step"],
                    weight_df["value"],
                    color=color,
                    linewidth=1.6 if is_top else 0.8,
                    alpha=1.0 if is_top else 0.35,
                    zorder=3 if is_top else 1,
                    label=weight if is_top else "_nolegend_",
                )
                if is_top:
                    top_lines.append(line)

            if top_lines:
                labelLines(top_lines, align=True, fontsize=8, zorder=5)

            ax.set_title(module)
            ax.set_xlabel("Step")
            ax.set_ylabel("Attribution Value")
            ax.set_ylim(y_min_global, y_max_global)
            ax.grid(True, linestyle="--", alpha=0.25)

        for ax in axes_iter[len(module_order) :]:
            ax.set_visible(False)
            ax.set_ylim(y_min_global, y_max_global)

        fig.suptitle(
            f"{str(self)} Top {len(top_500_weights)} Attributions (example 0)",
            fontsize=16,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        os.makedirs(log_dir, exist_ok=True)
        fig.savefig(os.path.join(log_dir, f"{str(self)}.top_attribs.png"), dpi=200)
        plt.close(fig)
