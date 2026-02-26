"""Compare all architectures on informative vs uninformative NT PCFG (2L)."""

import glob
import os
import pandas as pd
import plotnine as p9
from tqdm import tqdm
from mizani.formatters import scientific_format

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
    "hyena": "Hyena",
    "base_conv": "BaseConv",
    "h3": "H3",
    "based": "Based",
    "mamba": "Mamba",
    "delta_net": "DeltaNet",
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
        log["dataset"] = dirname.split("/")[-3]
        log["split"] = split
        log = log.groupby(["variable", "step", "evaluator", "step_rel", "identifier", "arch", "dim", "lr", "dataset", "split"]).mean().reset_index()
        dfs.append(log)
    print(f"Loaded {len(dfs)} files")
    df = pd.concat(dfs)
    df = df.groupby(["variable", "step", "evaluator", "step_rel", "identifier", "arch", "dim", "lr", "dataset", "split"]).mean().reset_index()
    return df

# Load data — all arches for both informative and uninformative
dirs = [
    "experiments/logs/pcfg_easy/**/test/SummaryEvaluator.csv",
    "experiments/logs/pcfg_uninformative_5_20/**/test/SummaryEvaluator.csv",
]
df = read_df(dirs)

dataset_map = {
    "pcfg_easy": "Informative NT",
    "pcfg_uninformative_5_20": "Uninformative NT",
}

# Final accuracy: best LR per (dim, arch, dataset)
subset_df = df[(df["variable"] == "query_item.argmax") & (df["step_rel"] == 1.0)].copy()
subset_df["dataset"] = subset_df["dataset"].map(dataset_map)
subset_df_best = subset_df[["dim", "arch", "dataset", "value"]].groupby(["dim", "arch", "dataset"]).max().reset_index()

os.makedirs("scripts/figs", exist_ok=True)

# Plot 1: Best accuracy vs dim, by arch, faceted by dataset
plot = (
    p9.ggplot(subset_df_best, p9.aes(x="dim", y="value", group="arch", color="arch", shape="arch")) +
    p9.geom_line(size=0.8) +
    p9.geom_point(size=2.5, stroke=0, alpha=0.9) +
    p9.geom_point(fill="none", stroke=0.5, size=2.5, color="#4f4f4f") +
    p9.scale_x_log10(breaks=[16, 32, 64, 128, 256]) +
    p9.scale_y_continuous(limits=(0, 1.05)) +
    p9.facet_wrap("~dataset", nrow=1) +
    p9.labs(y="Accuracy", x="Model dimension", color="Architecture", shape="Architecture") +
    p9.scale_color_brewer(type='qual', palette='Set2') +
    p9.theme(figure_size=(8, 3.5)) +
    p9.guides(color=p9.guide_legend(nrow=1))
)
plot.save("scripts/figs/pcfg_arch_best.pdf", dpi=300, limitsize=False)
plot.save("scripts/figs/pcfg_arch_best.png", dpi=300, limitsize=False)
print("Saved pcfg_arch_best")

# Plot 2: Heatmap of LR x dim, faceted by arch, for uninformative only
subset_uninf = subset_df[subset_df["dataset"] == "Uninformative NT"].copy()
subset_uninf["lr"] = pd.Categorical(subset_uninf["lr"])
subset_uninf["dim"] = pd.Categorical(subset_uninf["dim"])
subset_uninf["value"] *= 100
plot = (
    p9.ggplot(subset_uninf, p9.aes(y="lr", x="dim", alpha="value", fill="arch", label="value")) +
    p9.geom_tile() +
    p9.geom_text(size=4, format_string="{:.1f}%", alpha=1.0) +
    p9.facet_wrap("~arch", nrow=2) +
    p9.scale_y_discrete(labels=scientific_format(), expand=[0, 0]) +
    p9.scale_x_discrete(expand=[0, 0]) +
    p9.labs(x="Model dimension", y="LR", title="Uninformative NT PCFG") +
    p9.theme(
        figure_size=(14, 6),
        legend_position="none",
        panel_grid=p9.element_blank(),
    )
)
plot.save("scripts/figs/pcfg_arch_heatmap.pdf", dpi=300, limitsize=False)
plot.save("scripts/figs/pcfg_arch_heatmap.png", dpi=300, limitsize=False)
print("Saved pcfg_arch_heatmap")

# Plot 3: Side-by-side heatmap — informative vs uninformative, faceted by arch
subset_both = subset_df.copy()
subset_both["lr"] = pd.Categorical(subset_both["lr"])
subset_both["dim"] = pd.Categorical(subset_both["dim"])
subset_both["value"] *= 100
subset_both["label"] = subset_both["arch"] + "\n" + subset_both["dataset"]
plot = (
    p9.ggplot(subset_both, p9.aes(y="lr", x="dim", alpha="value", fill="dataset", label="value")) +
    p9.geom_tile() +
    p9.geom_text(size=3.5, format_string="{:.0f}%", alpha=1.0) +
    p9.facet_grid("dataset~arch") +
    p9.scale_y_discrete(labels=scientific_format(), expand=[0, 0]) +
    p9.scale_x_discrete(expand=[0, 0]) +
    p9.labs(x="Model dimension", y="LR") +
    p9.theme(
        figure_size=(18, 5),
        legend_position="none",
        panel_grid=p9.element_blank(),
    )
)
plot.save("scripts/figs/pcfg_arch_heatmap_both.pdf", dpi=300, limitsize=False)
plot.save("scripts/figs/pcfg_arch_heatmap_both.png", dpi=300, limitsize=False)
print("Saved pcfg_arch_heatmap_both")

print("\nDone!")
