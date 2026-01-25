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

    def __init__(self, run_every_n_steps: int, interpolation_steps: int = 1):
        super().__init__(run_every_n_steps)
        self.interpolation_steps = interpolation_steps
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

        # Find layers to hook for activation attributions
        layers_to_hook = []
        backbone = getattr(model_module, "backbone", None)
        if backbone is not None:
            # Hook embeddings (input to layer 0)
            if hasattr(backbone, "embeddings"):
                layers_to_hook.append(("embedding", backbone.embeddings))
            # Hook each transformer/mamba layer
            if hasattr(backbone, "layers"):
                for i, layer in enumerate(backbone.layers):
                    layers_to_hook.append((f"layer_{i}", layer))

        # Keep track of current activations for next step's delta (like weights)
        prev_activations = self.last_step_activations
        current_activations = {}  # {example_idx: {layer_name: tensor}}
        self.per_activation_attribs[step] = []

        n_interp = self.interpolation_steps

        # No batching: run backward one example at a time.
        for idx in tqdm(range(inputs["input_ids"].size(0)), desc=f"Running backward passes for step {step}"):
            input_ids = inputs["input_ids"][idx : idx + 1].to(device)
            labels = inputs["labels"][idx : idx + 1].to(device)

            # Accumulators for integrated gradients (on CPU)
            accumulated_weight_grads = {k: torch.zeros_like(v, device="cpu") for k, v in weight_deltas.items()}
            accumulated_activation_attribs = {}  # sum of (delta_act * grad) at each step
            prev_interp_activations = prev_activations[idx] if (prev_activations is not None and idx in prev_activations) else None

            # Interpolate from prev_weights to current_weights
            for interp_idx in range(n_interp):
                alpha = (interp_idx + 1) / n_interp  # use right endpoint of each interval

                # Set model weights to interpolated values
                interpolated_state = {}
                for k in prev_weights:
                    if k in weight_deltas:
                        interpolated_state[k] = (prev_weights[k] + alpha * weight_deltas[k]).to(device)
                    else:
                        interpolated_state[k] = current_weights[k].to(device)
                model_module.load_state_dict(interpolated_state)

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
                    loss = model.step(input_ids, labels)["loss"]
                loss.backward()

                # Remove hooks
                for handle in handles:
                    handle.remove()

                # Accumulate weight gradients
                for name, param in model_module.named_parameters():
                    if param.grad is not None and name in accumulated_weight_grads:
                        accumulated_weight_grads[name] += param.grad.detach().cpu()

                # Compute activation attributions for this interpolation step: delta_act * grad
                for name in activation_grads:
                    if prev_interp_activations is not None and name in prev_interp_activations:
                        activation_delta = activations[name] - prev_interp_activations[name]
                        step_attrib = activation_delta * activation_grads[name]
                        if name not in accumulated_activation_attribs:
                            accumulated_activation_attribs[name] = torch.zeros_like(step_attrib)
                        accumulated_activation_attribs[name] += step_attrib

                # Current activations become previous for next interpolation step
                prev_interp_activations = activations

            # Restore current weights
            model_module.load_state_dict({k: v.to(device) for k, v in current_weights.items()})

            # Store final activations (at alpha=1) for next eval step's delta
            # prev_interp_activations now holds the final activations after the loop
            current_activations[idx] = prev_interp_activations if prev_interp_activations is not None else {}

            # Compute activation attributions: already accumulated as sum of (delta_act * grad) per interpolation step
            example_activation_attribs = {}
            for name in accumulated_activation_attribs:
                example_activation_attribs[name] = accumulated_activation_attribs[name]
                # Sum over d_model like weights do
                if len(example_activation_attribs[name].shape) > 1:
                    for dim in range(example_activation_attribs[name].dim()):
                        if example_activation_attribs[name].shape[dim] == model.config.d_model:
                            example_activation_attribs[name] = example_activation_attribs[name].sum(dim=dim, keepdim=True)
                # Accumulate from previous eval step
                if prev_step is not None and self.per_activation_attribs[prev_step]:
                    example_activation_attribs[name] = example_activation_attribs[name] + self.per_activation_attribs[prev_step][idx].get(name, 0)
            self.per_activation_attribs[step].append(example_activation_attribs)

            # Compute weight attributions: mean(grads) * delta
            example_weight_attribs = {}
            for name in accumulated_weight_grads:
                if name not in weight_deltas:
                    continue
                mean_grad = accumulated_weight_grads[name] / n_interp
                example_weight_attribs[name] = weight_deltas[name].cpu() * mean_grad
                # Sum over d_model
                if len(example_weight_attribs[name].shape) > 1:
                    for dim in range(example_weight_attribs[name].dim()):
                        if example_weight_attribs[name].shape[dim] == model.config.d_model:
                            example_weight_attribs[name] = example_weight_attribs[name].sum(dim=dim, keepdim=True)
                # Accumulate from previous step
                if prev_step is not None:
                    example_weight_attribs[name] = example_weight_attribs[name] + self.per_weight_attribs[prev_step][idx].get(name, 0)
            self.per_weight_attribs[step].append(example_weight_attribs)

        # Save activations for next step (like last_step_weights)
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

        # Get layer names and number of examples from the first non-empty step
        layer_names = None
        n_examples = 0
        for step in steps:
            examples = self.per_activation_attribs[step]
            if examples and examples[0]:
                layer_names = sorted(examples[0].keys())
                n_examples = len(examples)
                break
        if not layer_names or n_examples == 0:
            return

        # Create subfolder for per-step plots
        frames_dir = os.path.join(log_dir, f"{str(self)}_activation_frames")
        os.makedirs(frames_dir, exist_ok=True)
        frame_paths = []

        # Compute global vmax across all steps and all examples for consistent colorbar
        global_vmax = 0.0
        for step in steps:
            example_attribs = self.per_activation_attribs[step]
            if not example_attribs:
                continue
            for ex_idx, attribs in enumerate(example_attribs):
                if not attribs:
                    continue
                for layer_name in layer_names:
                    if layer_name not in attribs:
                        continue
                    a = attribs[layer_name]
                    a = a.squeeze()
                    if a.dim() > 1:
                        a = a.squeeze(-1)
                    global_vmax = max(global_vmax, np.abs(a.numpy()).max())

        # For each step, plot a faceted heatmap (one column per example)
        for step in steps:
            example_attribs = self.per_activation_attribs[step]
            if not example_attribs:
                continue

            # Determine seq_len from first valid example
            seq_len = None
            for attribs in example_attribs:
                if attribs and layer_names[0] in attribs:
                    a = attribs[layer_names[0]].squeeze()
                    if a.dim() > 1:
                        a = a.squeeze(-1)
                    seq_len = a.shape[0]
                    break
            if seq_len is None:
                continue

            # Create figure with one subplot per example
            fig, axes = plt.subplots(
                1, n_examples,
                figsize=(max(6, seq_len * 0.2) * n_examples, len(layer_names) * 0.5 + 2),
                squeeze=False
            )
            axes = axes[0]  # flatten to 1D

            for ex_idx, (ax, attribs) in enumerate(zip(axes, example_attribs)):
                if not attribs:
                    ax.set_visible(False)
                    continue

                # Build matrix: (n_layers, seq_len) with attribution per position
                attrib_values = []
                for layer_name in layer_names:
                    if layer_name not in attribs:
                        attrib_values.append(np.zeros(seq_len))
                        continue
                    a = attribs[layer_name]
                    a = a.squeeze()
                    if a.dim() > 1:
                        a = a.squeeze(-1)
                    attrib_values.append(a.numpy())

                attrib_matrix = np.stack(attrib_values, axis=0)

                im = ax.imshow(attrib_matrix, aspect="auto", cmap="RdBu_r", vmin=-global_vmax, vmax=global_vmax)
                ax.set_xlabel("Position")
                if ex_idx == 0:
                    ax.set_ylabel("Layer")
                    ax.set_yticks(range(len(layer_names)))
                    ax.set_yticklabels(layer_names)
                else:
                    ax.set_yticks([])
                ax.set_title(f"Example {ex_idx}")

            # Add single colorbar
            fig.colorbar(im, ax=axes, label="Attribution (grad × Δact)", shrink=0.8)
            fig.suptitle(f"{str(self)} Activation Attributions (step {step})", fontsize=14)
            fig.tight_layout(rect=[0, 0, 0.95, 0.95])

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

        # Also plot total attribution trajectory over steps, faceted by example
        records = []
        for step in steps:
            example_attribs = self.per_activation_attribs[step]
            if not example_attribs:
                continue
            for ex_idx, attribs in enumerate(example_attribs):
                if not attribs:
                    continue
                for layer_name in layer_names:
                    if layer_name not in attribs:
                        continue
                    a = attribs[layer_name]
                    total = a.sum().item()
                    records.append({"step": step, "layer": layer_name, "example": ex_idx, "value": total})

        if not records:
            return

        df = pd.DataFrame(records)

        # Create faceted plot with one subplot per example
        ncols = min(4, n_examples)
        nrows = math.ceil(n_examples / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        axes_flat = axes.flat

        for ex_idx in range(n_examples):
            ax = axes_flat[ex_idx]
            ex_df = df[df["example"] == ex_idx]
            for layer_name in layer_names:
                layer_df = ex_df[ex_df["layer"] == layer_name].sort_values("step")
                if not layer_df.empty:
                    ax.plot(layer_df["step"], layer_df["value"], label=layer_name)

            ax.set_xlabel("Step")
            ax.set_ylabel("Total Attribution")
            ax.set_title(f"Example {ex_idx}")
            ax.legend(loc="best", fontsize=7)
            ax.grid(True, linestyle="--", alpha=0.25)

        # Hide unused axes
        for ax in axes_flat[n_examples:]:
            ax.set_visible(False)

        fig.suptitle(f"{str(self)} Activation Attributions Over Training", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(log_dir, f"{str(self)}.activation_attrib_trajectory.png"), dpi=200)
        plt.close(fig)
