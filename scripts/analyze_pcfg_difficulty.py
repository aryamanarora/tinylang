"""Compare attention performance on pcfg_easy vs pcfg_vary_difficulty_random_5_20."""

import glob
import os
import pandas as pd
import plotnine as p9
from tqdm import tqdm

p9.theme_set(
    p9.theme_bw(base_size=10) +
    p9.theme(
        figure_size=(2.5, 2.5),
        axis_title=p9.element_text(size=10),
        axis_text=p9.element_text(size=8),
        legend_position="bottom",
        legend_text=p9.element_text(size=8),
        legend_title=p9.element_text(size=9),
    )
)

arch_names = {
    "attention": "Attention",
}

def read_df(dirs, filter_steps=5000):
    all_files = []
    for d in dirs:
        all_files.extend(list(glob.glob(d)))
    dfs = []
    for file in tqdm(all_files):
        dirname = os.path.dirname(file)
        evaluator = file.split("/")[-1].split(".")[0]
        split = dirname.split("/")[-1]
        arch, dim, lr = dirname.split("/")[-2].split(".")[0].split("___")
        log = pd.read_csv(file)
        log = log[(log["step"] % filter_steps) == 0]
        log["identifier"] = file
        log["arch"] = arch_names.get(arch, arch)
        log["dim"] = int(dim)
        log["lr"] = float(lr)
        log["evaluator"] = evaluator
        log["step_rel"] = log["step"] / log["step"].max()
        log["layers"] = 2
        log["dataset"] = dirname.split("/")[-3]
        log["split"] = split
        log = log.groupby(["variable", "step", "evaluator", "step_rel", "identifier", "arch", "dim", "lr", "layers", "dataset", "split"]).mean().reset_index()
        dfs.append(log)
    print(f"Loaded {len(dfs)} files")
    df = pd.concat(dfs)
    df = df.groupby(["variable", "step", "evaluator", "step_rel", "identifier", "arch", "dim", "lr", "layers", "dataset", "split"]).mean().reset_index()
    return df

# Load data
dirs = [
    "experiments/logs/pcfg_easy/attention**/test/SummaryEvaluator.csv",
    "experiments/logs/pcfg_vary_difficulty_random_5_20/attention**/test/SummaryEvaluator.csv",
]
df = read_df(dirs)

# Final accuracy: best LR per (dim, dataset)
subset_df = df[(df["variable"] == "query_item.argmax") & (df["step_rel"] == 1.0)]
subset_df["dataset"] = subset_df["dataset"].map({
    "pcfg_easy": "PCFG (informative NT)",
    "pcfg_vary_difficulty_random_5_20": "PCFG (uninformative NT)",
})
subset_df_best = subset_df[["dim", "dataset", "value"]].groupby(["dim", "dataset"]).max().reset_index()

os.makedirs("scripts/figs", exist_ok=True)

# Plot 1: Best accuracy vs dim, comparing the two datasets
subset_df_best["clean"] = subset_df_best["value"].apply(lambda x: f"{x * 100:.1f}")
plot = (
    p9.ggplot(subset_df_best, p9.aes(x="dim", y="value", group="dataset", color="dataset", shape="dataset")) +
    p9.geom_line(size=1) +
    p9.geom_point(size=3, stroke=0, alpha=0.9) +
    p9.geom_point(fill="none", stroke=0.5, size=3, color="#4f4f4f") +
    p9.scale_x_log10(breaks=[16, 32, 64, 128, 256]) +
    p9.scale_y_continuous(limits=(0, 1.05)) +
    p9.labs(y="Accuracy", x="Model dimension", color="Dataset", shape="Dataset") +
    p9.scale_color_brewer(type='qual', palette='Set2') +
    p9.theme(figure_size=(4, 3))
)
plot.save("scripts/figs/pcfg_difficulty_best.pdf", dpi=300, limitsize=False)
plot.save("scripts/figs/pcfg_difficulty_best.png", dpi=300, limitsize=False)
print("Saved pcfg_difficulty_best")

# Plot 2: Heatmap of all LR x dim combinations, faceted by dataset
from mizani.formatters import scientific_format

subset_df_temp = subset_df.copy()
subset_df_temp["lr"] = pd.Categorical(subset_df_temp["lr"])
subset_df_temp["dim"] = pd.Categorical(subset_df_temp["dim"])
subset_df_temp["value"] *= 100
plot = (
    p9.ggplot(subset_df_temp, p9.aes(y="lr", x="dim", alpha="value", fill="dataset", label="value")) +
    p9.geom_tile() +
    p9.geom_text(size=6, format_string="{:.1f}%", alpha=1.0) +
    p9.facet_wrap("~dataset", nrow=2) +
    p9.scale_y_discrete(labels=scientific_format(), expand=[0, 0]) +
    p9.scale_x_discrete(expand=[0, 0]) +
    p9.scale_color_brewer(type='qual', palette='Set2') +
    p9.labs(x="Model dimension", y="LR") +
    p9.theme(
        figure_size=(5, 5),
        legend_position="none",
        panel_grid=p9.element_blank(),
    )
)
plot.save("scripts/figs/pcfg_difficulty_heatmap.pdf", dpi=300, limitsize=False)
plot.save("scripts/figs/pcfg_difficulty_heatmap.png", dpi=300, limitsize=False)
print("Saved pcfg_difficulty_heatmap")

# Plot 3: Training curves (accuracy over steps) for best LR per dim
# Find best LR for each (dim, dataset)
best_lr = subset_df.groupby(["dim", "dataset"]).apply(lambda g: g.loc[g["value"].idxmax(), "lr"]).reset_index(name="lr")
df_curves = df[(df["variable"] == "query_item.argmax")].copy()
df_curves["dataset"] = df_curves["dataset"].map({
    "pcfg_easy": "PCFG (informative NT)",
    "pcfg_vary_difficulty_random_5_20": "PCFG (uninformative NT)",
})
df_curves = df_curves.merge(best_lr, on=["dim", "dataset", "lr"])

plot = (
    p9.ggplot(df_curves, p9.aes(x="step", y="value", color="dataset", group="dataset")) +
    p9.geom_line(size=0.8) +
    p9.facet_wrap("~dim", nrow=1, labeller="label_both") +
    p9.scale_y_continuous(limits=(0, 1.05)) +
    p9.labs(y="Accuracy", x="Step", color="Dataset") +
    p9.scale_color_brewer(type='qual', palette='Set2') +
    p9.theme(figure_size=(8, 2.5))
)
plot.save("scripts/figs/pcfg_difficulty_curves.pdf", dpi=300, limitsize=False)
plot.save("scripts/figs/pcfg_difficulty_curves.png", dpi=300, limitsize=False)
print("Saved pcfg_difficulty_curves")

print("\nDone! All plots saved to scripts/figs/")
