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
from plotnine import (
    ggplot, aes, geom_line, geom_text, facet_wrap,
    scale_color_brewer, theme_minimal, theme, element_text,
    labs, ggsave, scale_linetype_manual
)
import warnings
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
        self.losses_by_step = defaultdict(list)  # step -> list of loss values per example
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
            # Still compute and store initial losses for baseline
            device = next(model_module.parameters()).device
            for idx in range(inputs["input_ids"].size(0)):
                input_ids = inputs["input_ids"][idx : idx + 1].to(device)
                labels = inputs["labels"][idx : idx + 1].to(device)
                with torch.no_grad():
                    loss = model.step(input_ids, labels)["loss"].item()
                self.losses_by_step[step].append(loss)
            return

        weight_deltas = {k: current_weights[k] - prev_weights[k] for k in current_weights if k in prev_weights}
        device = next(model_module.parameters()).device
        prev_step = max(list(self.per_weight_attribs.keys()) or [None])
        self.per_weight_attribs[step] = []

        # Track a readable example for logging later.
        self.examples_by_step[step] = self._format_example(language, inputs)

        # Find layers to hook for activation attributions
        # Each entry is (name, module, capture_input) where capture_input indicates if we want input instead of output
        layers_to_hook = []
        backbone = getattr(model_module, "backbone", None)
        if backbone is not None:
            # Hook embeddings (input to layer 0)
            if hasattr(backbone, "embeddings"):
                layers_to_hook.append(("embedding", backbone.embeddings, False))
            # Hook each transformer/mamba layer and its submodules
            if hasattr(backbone, "layers"):
                for i, layer in enumerate(backbone.layers):
                    layers_to_hook.append((f"layer_{i}", layer, False))
                    # Hook sequence_mixer input and output
                    if hasattr(layer, "sequence_mixer"):
                        layers_to_hook.append((f"layer_{i}.seq_mix_in", layer.sequence_mixer, True))
                        layers_to_hook.append((f"layer_{i}.seq_mix_out", layer.sequence_mixer, False))
                    # Hook state_mixer input and output
                    if hasattr(layer, "state_mixer"):
                        layers_to_hook.append((f"layer_{i}.state_mix_in", layer.state_mixer, True))
                        layers_to_hook.append((f"layer_{i}.state_mix_out", layer.state_mixer, False))

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

                def make_fwd_hook(name, capture_input):
                    def hook(module, inp, output):
                        if capture_input:
                            # inp is a tuple, take first element
                            activations[name] = inp[0].detach().cpu() if isinstance(inp, tuple) else inp.detach().cpu()
                        else:
                            # output might be a tuple (e.g., for some layers)
                            out = output[0] if isinstance(output, tuple) else output
                            activations[name] = out.detach().cpu()
                    return hook

                def make_bwd_hook(name, capture_input):
                    def hook(module, grad_input, grad_output):
                        if capture_input:
                            # grad_input corresponds to input gradients
                            gi = grad_input[0] if isinstance(grad_input, tuple) and grad_input[0] is not None else grad_input
                            if gi is not None:
                                activation_grads[name] = gi.detach().cpu()
                        else:
                            # grad_output corresponds to output gradients
                            go = grad_output[0] if isinstance(grad_output, tuple) else grad_output
                            if go is not None:
                                activation_grads[name] = go.detach().cpu()
                    return hook

                for name, layer, capture_input in layers_to_hook:
                    handles.append(layer.register_forward_hook(make_fwd_hook(name, capture_input)))
                    handles.append(layer.register_full_backward_hook(make_bwd_hook(name, capture_input)))

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

            # Compute and store loss at current weights
            with torch.no_grad():
                loss_at_current = model.step(input_ids, labels)["loss"].item()
            self.losses_by_step[step].append(loss_at_current)

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

            # Log trajectory stats to all_eval_stats for this example
            # Activation attribution sums per layer
            for name, attrib in example_activation_attribs.items():
                self.all_eval_stats[step][f"act_attrib.{name}"].append(attrib.sum().item())

            # Total weight attribution sum
            weight_total = sum(w.sum().item() for w in example_weight_attribs.values())
            self.all_eval_stats[step]["weight_attrib.total"].append(weight_total)

            # Loss at current step
            self.all_eval_stats[step]["loss"].append(self.losses_by_step[step][idx])

        # Save activations for next step (like last_step_weights)
        self.last_step_activations = current_activations

    def _format_example(self, language: Language, inputs: dict) -> list[dict]:
        """Return both raw token ids and pretty text for all examples."""
        results = []
        pad_token = getattr(language, "PAD", None)
        for ex_idx in range(inputs["input_ids"].size(0)):
            tokens = inputs["input_ids"][ex_idx].detach().cpu().tolist()
            if pad_token is not None:
                while tokens and tokens[-1] == pad_token:
                    tokens.pop()
            if "strs" in inputs and inputs["strs"] and ex_idx < len(inputs["strs"]):
                pretty = inputs["strs"][ex_idx]
            elif hasattr(language, "prettify"):
                pretty = language.prettify(tokens)
            else:
                pretty = " ".join(str(tok) for tok in tokens)
            results.append({"pretty": pretty, "tokens": tokens})
        return results

    def _sort_layer_names(self, names: list[str]) -> list[str]:
        """Sort layer names in model forward-pass order."""
        def sort_key(name):
            # embedding comes first
            if name == "embedding":
                return (-1, 0, 0)
            # Parse layer_N or layer_N.suffix
            parts = name.split(".")
            if parts[0].startswith("layer_"):
                layer_num = int(parts[0].split("_")[1])
                if len(parts) == 1:
                    # layer_N (block output) - comes first for this layer
                    return (layer_num, 0, 0)
                else:
                    # layer_N.suffix - order by suffix type
                    suffix_order = {
                        "seq_mix_in": 1,
                        "seq_mix_out": 2,
                        "state_mix_in": 3,
                        "state_mix_out": 4,
                    }
                    suffix = parts[1]
                    return (layer_num, 1, suffix_order.get(suffix, 99))
            # Unknown format - put at end
            return (9999, 0, 0)

        return sorted(names, key=sort_key)

    def plot(self, log_dir: str):
        """Plot attribution trajectories for the first example only."""
        records = []
        for step, example_list in self.per_weight_attribs.items():
            if not example_list:
                continue
            first_example = example_list[0]
            example_info_list = self.examples_by_step.get(step, [])
            if example_info_list:
                print(f"{str(self)} step {step} example 0 pretty: {example_info_list[0]['pretty']}")
                print(f"{str(self)} step {step} example 0 tokens: {example_info_list[0]['tokens']}")
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

        # Collect highlighted weights across all modules
        highlight_weights_set = set()
        for weights in module_top_weights.values():
            highlight_weights_set.update(weights)

        # Mark highlighted weights in dataframe
        df_top["is_highlight"] = df_top["weight"].isin(highlight_weights_set)

        # Create short labels (strip module prefix)
        df_top["short_label"] = df_top.apply(
            lambda row: row["weight"][len(row["module"]):] if row["weight"].startswith(row["module"]) else row["weight"],
            axis=1
        )

        # Get labels for endpoints (last step, highlighted only)
        last_step = df_top["step"].max()
        df_labels = df_top[(df_top["step"] == last_step) & df_top["is_highlight"]].copy()

        # Build plot with plotnine
        n_modules = len(module_order)
        ncols = min(4, n_modules)
        nrows = math.ceil(n_modules / ncols)

        p = (
            ggplot(df_top, aes(x="step", y="value", group="weight", color="weight"))
            # Background lines (non-highlighted)
            + geom_line(
                data=df_top[~df_top["is_highlight"]],
                alpha=0.3,
                size=0.5,
                color="gray",
                show_legend=False
            )
            # Highlighted lines
            + geom_line(
                data=df_top[df_top["is_highlight"]],
                size=1.2,
                show_legend=False
            )
            # Labels at endpoints
            + geom_text(
                data=df_labels,
                mapping=aes(label="short_label"),
                ha="left",
                size=7,
                nudge_x=0.5,
                show_legend=False
            )
            + facet_wrap("~module", ncol=ncols, scales="free_x")
            + scale_color_brewer(type="qual", palette="Set1", guide=False)
            + labs(
                x="Step",
                y="Attribution Value",
                title=f"{str(self)} Top {len(top_500_weights)} Attributions (example 0)"
            )
            + theme_minimal()
            + theme(
                figure_size=(6 * ncols, 4.5 * nrows),
                strip_text=element_text(size=10, weight="bold"),
            )
        )

        os.makedirs(log_dir, exist_ok=True)
        ggsave(p, os.path.join(log_dir, f"{str(self)}.top_attribs.png"), dpi=200)

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
                layer_names = self._sort_layer_names(list(examples[0].keys()))
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

            # Get example info for token labels
            example_info = self.examples_by_step.get(step, [])

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

                # Add token labels on x-axis for this example
                if ex_idx < len(example_info) and "tokens" in example_info[ex_idx]:
                    tokens = example_info[ex_idx]["tokens"]
                    if len(tokens) <= seq_len:
                        ax.set_xticks(range(len(tokens)))
                        ax.set_xticklabels(tokens, rotation=90, fontsize=6)

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

        # Get baseline losses from first step
        all_loss_steps = sorted(self.losses_by_step.keys())
        baseline_losses = {}
        if all_loss_steps:
            first_loss_step = all_loss_steps[0]
            for ex_idx, loss in enumerate(self.losses_by_step[first_loss_step]):
                baseline_losses[ex_idx] = loss

        for step in steps:
            # Activation attributions per layer
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
                    records.append({"step": step, "layer": layer_name, "example": ex_idx, "value": total, "type": "activation"})

            # Weight attributions (sum over all weights)
            if step in self.per_weight_attribs:
                weight_example_attribs = self.per_weight_attribs[step]
                for ex_idx, w_attribs in enumerate(weight_example_attribs):
                    if not w_attribs:
                        continue
                    weight_total = sum(w.sum().item() for w in w_attribs.values())
                    records.append({"step": step, "layer": "weights (total)", "example": ex_idx, "value": weight_total, "type": "weight"})

            # Loss difference from step 0
            if step in self.losses_by_step:
                for ex_idx, loss in enumerate(self.losses_by_step[step]):
                    if ex_idx in baseline_losses:
                        loss_diff = loss - baseline_losses[ex_idx]
                        records.append({"step": step, "layer": "Δloss (actual)", "example": ex_idx, "value": loss_diff, "type": "loss"})

        if not records:
            return

        df = pd.DataFrame(records)

        # Assign linetypes: activation layers get solid, weight/loss get special styles
        df["linetype"] = df["type"].map({
            "activation": "solid",
            "weight": "dashed",
            "loss": "dotted"
        })

        # Create faceted plot with plotnine
        ncols = min(4, n_examples)
        p = (
            ggplot(df, aes(x="step", y="value", color="layer", linetype="linetype"))
            + geom_line(size=0.8)
            + facet_wrap("~example", ncol=ncols, labeller="label_both")
            + scale_color_brewer(type="qual", palette="Set1")
            + scale_linetype_manual(values={"solid": "solid", "dashed": "dashed", "dotted": "dotted"}, guide=False)
            + labs(
                x="Step",
                y="Total Attribution / Δloss",
                color="Layer",
                title=f"{str(self)} Attributions vs Actual Loss Change"
            )
            + theme_minimal()
            + theme(
                figure_size=(5 * ncols, 4 * math.ceil(n_examples / ncols)),
                legend_position="right",
                legend_text=element_text(size=7),
                strip_text=element_text(size=10),
            )
        )

        os.makedirs(log_dir, exist_ok=True)
        ggsave(p, os.path.join(log_dir, f"{str(self)}.activation_attrib_trajectory.png"), dpi=200)
