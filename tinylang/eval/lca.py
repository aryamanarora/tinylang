from .eval import Evaluator
from tinylang.model import Model
from tinylang.language import Language
import torch
import numpy as np
from collections import defaultdict
from itertools import product
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from labellines import labelLines
from tqdm import tqdm


class LCAEvaluator(Evaluator):
    def __str__(self):
        return "LCAEvaluator"

    def __init__(self, run_every_n_steps: int):
        super().__init__(run_every_n_steps)
        self.last_step_weights = None
        self.last_step_activations = None  # {example_idx: {layer_name: (seq_len, d_model)}}
        self.per_weight_attribs = defaultdict(list)
        self.per_activation_attribs = defaultdict(list)  # step -> list of {layer_name: grad * delta}
        self.examples_by_step = {}

    def eval(self, model: Model, language: Language, inputs: dict, outputs: dict, step: int):
        """Run per-example backward passes and compute simple weight attributions."""
        model_module = getattr(model, "model", model)

        # Keep CPU copies of current and previous weights to compare across steps.
        prev_weights = self.last_step_weights
        current_weights = {k: v.detach().clone() for k, v in model_module.state_dict().items()}
        self.last_step_weights = current_weights

        # Need a previous snapshot to form weight deltas.
        if prev_weights is None:
            return

        weight_deltas = {k: current_weights[k] - prev_weights[k] for k in current_weights if k in prev_weights}
        device = next(model_module.parameters()).device
        prev_step = max(list(self.per_weight_attribs.keys()) or [None])
        self.per_weight_attribs[step] = []

        # Track a readable example for logging later.
        self.examples_by_step[step] = self._format_example(language, inputs)

        # Find layers to hook for activation gradients
        layers_to_hook = []
        backbone = getattr(model_module, "backbone", None)
        if backbone is not None and hasattr(backbone, "layers"):
            for i, layer in enumerate(backbone.layers):
                layers_to_hook.append((f"layer_{i}", layer))

        prev_activations = self.last_step_activations
        current_activations = {}  # {example_idx: {layer_name: tensor}}
        self.per_activation_attribs[step] = []

        # No batching: run backward one example at a time.
        for idx in tqdm(range(inputs["input_ids"].size(0)), desc=f"Running backward passes for step {step}"):
            model_module.zero_grad(set_to_none=True)

            # Register hooks to capture activations (forward) and gradients (backward)
            activations = {}
            activation_grads = {}
            handles = []

            def make_fwd_hook(name):
                def hook(module, input, output):
                    activations[name] = output.detach().cpu()
                return hook

            def make_bwd_hook(name):
                def hook(module, grad_input, grad_output):
                    activation_grads[name] = grad_output[0].detach().cpu()
                return hook

            for name, layer in layers_to_hook:
                handles.append(layer.register_forward_hook(make_fwd_hook(name)))
                handles.append(layer.register_full_backward_hook(make_bwd_hook(name)))

            with torch.enable_grad():
                input_ids = inputs["input_ids"][idx : idx + 1].to(device)
                labels = inputs["labels"][idx : idx + 1].to(device)
                loss = model.step(input_ids, labels)["loss"]
            loss.backward()

            # Remove hooks
            for handle in handles:
                handle.remove()

            # Store current activations for next step's delta
            current_activations[idx] = activations

            # Compute activation attributions: grad * delta (cumulative)
            activation_attribs = {}
            if prev_activations is not None and idx in prev_activations:
                for name in activations:
                    if name in activation_grads and name in prev_activations[idx]:
                        delta = activations[name] - prev_activations[idx][name]
                        activation_attribs[name] = activation_grads[name] * delta
                        # Accumulate from previous step
                        if prev_step is not None and self.per_activation_attribs[prev_step]:
                            prev_attrib = self.per_activation_attribs[prev_step][idx].get(name, 0)
                            activation_attribs[name] = activation_attribs[name] + prev_attrib
            self.per_activation_attribs[step].append(activation_attribs)

            # Compute weight attributions: grad * delta_weight
            example_attribs = {}
            for name, param in model_module.named_parameters():
                if param.grad is None or name not in weight_deltas:
                    continue
                grad_detach = param.grad.detach()
                example_attribs[name] = (weight_deltas[name] * grad_detach)  # .sum()
                if len(example_attribs[name].shape) > 1:
                    for dim in range(example_attribs[name].dim()):
                        if example_attribs[name].shape[dim] == model.config.d_model:
                            example_attribs[name] = example_attribs[name].sum(dim=dim, keepdim=True)
                if prev_step is not None:
                    example_attribs[name] += self.per_weight_attribs[prev_step][idx].get(name, 0)
            self.per_weight_attribs[step].append(example_attribs)

        # Save activations for next step
        self.last_step_activations = current_activations

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

        last_step = df["step"].max()
        module_order = (
            df[df["step"] == last_step]
            .groupby("module")["value"]
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
                # Strip module prefix from label since it's in the subplot title
                short_label = weight[len(module):] if weight.startswith(module) else weight
                line, = ax.plot(
                    weight_df["step"],
                    weight_df["value"],
                    color=color,
                    linewidth=1.6 if is_top else 0.8,
                    alpha=1.0 if is_top else 0.35,
                    zorder=3 if is_top else 1,
                    label=short_label if is_top else "_nolegend_",
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

        # Also plot activation gradients
        self._plot_activation_grads(log_dir)

    def _plot_activation_grads(self, log_dir: str):
        """Plot activation attribution (grad * delta) heatmaps across layers and positions."""
        if not self.per_activation_attribs:
            return

        steps = sorted(self.per_activation_attribs.keys())
        if not steps:
            return

        # Get layer names from the first non-empty example
        layer_names = None
        for step in steps:
            examples = self.per_activation_attribs[step]
            if examples and examples[0]:
                layer_names = sorted(examples[0].keys())
                break
        if not layer_names:
            return

        # Create subfolder for per-step plots
        frames_dir = os.path.join(log_dir, f"{str(self)}_activation_frames")
        os.makedirs(frames_dir, exist_ok=True)
        frame_paths = []

        # Compute global vmax across all steps for consistent colorbar
        global_vmax = 0.0
        for step in steps:
            example_attribs = self.per_activation_attribs[step]
            if not example_attribs or not example_attribs[0]:
                continue
            attribs = example_attribs[0]
            for layer_name in layer_names:
                if layer_name not in attribs:
                    continue
                a = attribs[layer_name]
                if a.dim() == 3:
                    a = a.squeeze(0)
                pos_attrib = a.sum(dim=-1)
                global_vmax = max(global_vmax, np.abs(pos_attrib.numpy()).max())

        # For each step, plot a heatmap of attribution sums (layers x positions)
        for step in steps:
            example_attribs = self.per_activation_attribs[step]
            if not example_attribs or not example_attribs[0]:
                continue

            # Use first example
            attribs = example_attribs[0]
            example_info = self.examples_by_step.get(step)

            # Build matrix: (n_layers, seq_len) with attribution sum per position
            seq_len = None
            attrib_sums = []
            for layer_name in layer_names:
                if layer_name not in attribs:
                    continue
                a = attribs[layer_name]  # (1, seq_len, d_model)
                if a.dim() == 3:
                    a = a.squeeze(0)  # (seq_len, d_model)
                # Sum across d_model to get per-position attribution
                pos_attrib = a.sum(dim=-1)  # (seq_len,)
                attrib_sums.append(pos_attrib.numpy())
                seq_len = pos_attrib.shape[0]

            if not attrib_sums:
                continue

            attrib_matrix = np.stack(attrib_sums, axis=0)  # (n_layers, seq_len)

            fig, ax = plt.subplots(figsize=(max(12, seq_len * 0.3), len(layer_names) * 0.5 + 2))
            # Use diverging colormap with global vmax for consistent scale
            im = ax.imshow(attrib_matrix, aspect="auto", cmap="RdBu_r", vmin=-global_vmax, vmax=global_vmax)
            ax.set_xlabel("Position")
            ax.set_ylabel("Layer")
            ax.set_yticks(range(len(layer_names)))
            ax.set_yticklabels(layer_names)

            # Add token labels if available
            if example_info and "tokens" in example_info:
                tokens = example_info["tokens"]
                if len(tokens) <= seq_len:
                    ax.set_xticks(range(len(tokens)))
                    ax.set_xticklabels(tokens, rotation=90, fontsize=8)

            plt.colorbar(im, ax=ax, label="Attribution (grad × Δact)")
            ax.set_title(f"{str(self)} Activation Attributions (step {step}, example 0)")
            fig.tight_layout()

            frame_path = os.path.join(frames_dir, f"step_{step:06d}.png")
            fig.savefig(frame_path, dpi=150)
            frame_paths.append(frame_path)
            plt.close(fig)

        # Create GIF from frames
        if frame_paths:
            try:
                from PIL import Image
                frames = [Image.open(fp) for fp in frame_paths]
                gif_path = os.path.join(log_dir, f"{str(self)}.activation_attribs.gif")
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=500,  # ms per frame
                    loop=0,
                )
            except ImportError:
                pass  # PIL not available, skip GIF creation

        # Also plot total attribution trajectory over steps
        records = []
        for step in steps:
            example_attribs = self.per_activation_attribs[step]
            if not example_attribs or not example_attribs[0]:
                continue
            attribs = example_attribs[0]
            for layer_name in layer_names:
                if layer_name not in attribs:
                    continue
                a = attribs[layer_name]
                if a.dim() == 3:
                    a = a.squeeze(0)
                total = a.sum().item()
                records.append({"step": step, "layer": layer_name, "value": total})

        if not records:
            return

        df = pd.DataFrame(records)
        fig, ax = plt.subplots(figsize=(10, 6))
        for layer_name in layer_names:
            layer_df = df[df["layer"] == layer_name].sort_values("step")
            ax.plot(layer_df["step"], layer_df["value"], label=layer_name)

        ax.set_xlabel("Step")
        ax.set_ylabel("Total Attribution")
        ax.set_title(f"{str(self)} Activation Attributions Over Training")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(log_dir, f"{str(self)}.activation_attrib_trajectory.png"), dpi=200)
        plt.close(fig)
