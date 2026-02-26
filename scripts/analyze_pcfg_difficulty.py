"""Compare attention performance on pcfg_easy vs pcfg_vary_difficulty_random_5_20, 1L and 2L."""

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

# Load data
dirs = [
    "experiments/logs/pcfg_easy/attention**/test/SummaryEvaluator.csv",
    "experiments/logs/pcfg_easy_1L/attention**/test/SummaryEvaluator.csv",
    "experiments/logs/pcfg_vary_difficulty_random_5_20/attention**/test/SummaryEvaluator.csv",
    "experiments/logs/pcfg_vary_difficulty_random_5_20_1L/attention**/test/SummaryEvaluator.csv",
    "experiments/logs/pcfg_vary_difficulty_random_5_20_3L/attention**/test/SummaryEvaluator.csv",
]
df = read_df(dirs)

dataset_map = {
    "pcfg_easy": "Informative NT, 2L",
    "pcfg_easy_1L": "Informative NT, 1L",
    "pcfg_vary_difficulty_random_5_20": "Uninformative NT, 2L",
    "pcfg_vary_difficulty_random_5_20_1L": "Uninformative NT, 1L",
    "pcfg_vary_difficulty_random_5_20_3L": "Uninformative NT, 3L",
}

# Final accuracy: best LR per (dim, dataset)
subset_df = df[(df["variable"] == "query_item.argmax") & (df["step_rel"] == 1.0)].copy()
subset_df["dataset"] = subset_df["dataset"].map(dataset_map)
subset_df_best = subset_df[["dim", "dataset", "value"]].groupby(["dim", "dataset"]).max().reset_index()

os.makedirs("scripts/figs", exist_ok=True)

# Plot 1: Best accuracy vs dim, comparing all 4 conditions
subset_df_best["clean"] = subset_df_best["value"].apply(lambda x: f"{x * 100:.1f}")
plot = (
    p9.ggplot(subset_df_best, p9.aes(x="dim", y="value", group="dataset", color="dataset", shape="dataset")) +
    p9.geom_line(size=1) +
    p9.geom_point(size=3, stroke=0, alpha=0.9) +
    p9.geom_point(fill="none", stroke=0.5, size=3, color="#4f4f4f") +
    p9.scale_x_log10(breaks=[16, 32, 64, 128, 256]) +
    p9.scale_y_continuous(limits=(0, 1.05)) +
    p9.labs(y="Accuracy", x="Model dimension", color="Setting", shape="Setting") +
    p9.scale_color_brewer(type='qual', palette='Set2') +
    p9.theme(figure_size=(5, 3.5)) +
    p9.guides(color=p9.guide_legend(nrow=2))
)
plot.save("scripts/figs/pcfg_difficulty_best.pdf", dpi=300, limitsize=False)
plot.save("scripts/figs/pcfg_difficulty_best.png", dpi=300, limitsize=False)
print("Saved pcfg_difficulty_best")

# Plot 2: Heatmap of all LR x dim combinations, faceted by dataset
subset_df_temp = subset_df.copy()
subset_df_temp["lr"] = pd.Categorical(subset_df_temp["lr"])
subset_df_temp["dim"] = pd.Categorical(subset_df_temp["dim"])
subset_df_temp["value"] *= 100
subset_df_temp["dataset"] = pd.Categorical(
    subset_df_temp["dataset"],
    categories=["Informative NT, 2L", "Informative NT, 1L", "Uninformative NT, 3L", "Uninformative NT, 2L", "Uninformative NT, 1L"],
    ordered=True,
)
plot = (
    p9.ggplot(subset_df_temp, p9.aes(y="lr", x="dim", alpha="value", fill="dataset", label="value")) +
    p9.geom_tile() +
    p9.geom_text(size=5, format_string="{:.1f}%", alpha=1.0) +
    p9.facet_wrap("~dataset", nrow=2) +
    p9.scale_y_discrete(labels=scientific_format(), expand=[0, 0]) +
    p9.scale_x_discrete(expand=[0, 0]) +
    p9.scale_color_brewer(type='qual', palette='Set2') +
    p9.labs(x="Model dimension", y="LR") +
    p9.theme(
        figure_size=(8, 6),
        legend_position="none",
        panel_grid=p9.element_blank(),
    )
)
plot.save("scripts/figs/pcfg_difficulty_heatmap.pdf", dpi=300, limitsize=False)
plot.save("scripts/figs/pcfg_difficulty_heatmap.png", dpi=300, limitsize=False)
print("Saved pcfg_difficulty_heatmap")

# Plot 3: Training curves (accuracy over steps) for best LR per dim
best_lr = subset_df.groupby(["dim", "dataset"]).apply(
    lambda g: g.loc[g["value"].idxmax(), "lr"], include_groups=False
).reset_index(name="lr")
df_curves = df[(df["variable"] == "query_item.argmax")].copy()
df_curves["dataset"] = df_curves["dataset"].map(dataset_map)
df_curves = df_curves.merge(best_lr, on=["dim", "dataset", "lr"])

plot = (
    p9.ggplot(df_curves, p9.aes(x="step", y="value", color="dataset", group="dataset")) +
    p9.geom_line(size=0.8) +
    p9.facet_wrap("~dim", nrow=1, labeller="label_both") +
    p9.scale_y_continuous(limits=(0, 1.05)) +
    p9.labs(y="Accuracy", x="Step", color="Setting") +
    p9.scale_color_brewer(type='qual', palette='Set2') +
    p9.theme(figure_size=(10, 3)) +
    p9.guides(color=p9.guide_legend(nrow=2))
)
plot.save("scripts/figs/pcfg_difficulty_curves.pdf", dpi=300, limitsize=False)
plot.save("scripts/figs/pcfg_difficulty_curves.png", dpi=300, limitsize=False)
print("Saved pcfg_difficulty_curves")

###############################################################################
# Interchange Intervention Analysis
###############################################################################

positions = ["target_item_orig", "query_item_orig", "query_item", "divider"]

def get_attrib_df(df, df_int, corruption="query_item_orig", components="block_input", prefix="PARENT", max_layers=2):
    if isinstance(components, str):
        components = [components]

    subset_df = df[(df["variable"].isin(["query_item.argmax", "query_item.pred_prob"])) & (df["step_rel"] == 1.0)]
    subset_df_int = df_int[(df_int["step_rel"] == 1.0)]
    subset_df_int = pd.concat([subset_df_int, subset_df])
    subset_df_int = subset_df_int[["dim", "arch", "lr", "variable", "value", "dataset", "split"]].pivot(index=["dim", "arch", "lr", "dataset", "split"], columns="variable").reset_index()
    subset_df_int.columns = [col[1] if col[1] != '' else col[0] for col in subset_df_int.columns]

    for component in components:
        for layer in range(max_layers):
            for position in positions:
                metric = f"{layer}.{prefix}.{corruption}.{position}.{component}.restored_prob"
                if metric not in subset_df_int.columns:
                    continue
                corrupted = f"corrupted.{prefix}.{corruption}.{position}.{component}.restored_prob"
                diff = f"{layer}.{prefix}.{corruption}.{position}.{component}.prob_diff"
                original = f"original.{prefix}.{corruption}.{position}.{component}.restored_prob"
                label = f"Association @ {layer}, {position}" if len(components) == 1 else f"Association @ {layer}, {position}, {component}"
                subset_df_int[label] = (subset_df_int[diff]) / (subset_df_int[original] - (subset_df_int[metric] - subset_df_int[diff]))

        for layer in range(max_layers):
            for position in positions:
                metric = f"{layer}.{prefix}.{corruption}.{position}.{component}.restored_prob"
                if metric not in subset_df_int.columns:
                    continue
                corrupted = f"corrupted.{prefix}.{corruption}.{position}.{component}.restored_prob"
                diff = f"{layer}.{prefix}.{corruption}.{position}.{component}.prob_diff"
                original = f"original.{prefix}.{corruption}.{position}.{component}.restored_prob"
                kl_div = f"{layer}.{prefix}.{corruption}.{position}.{component}.kl_div"
                suffix = f"{layer}, {position}" if len(components) == 1 else f"{layer}, {position}, {component}"
                subset_df_int = subset_df_int.rename(columns={
                    metric: f"Restored @ {suffix}",
                    corrupted: f"Corrupted @ {suffix}",
                    diff: f"Diff @ {suffix}",
                    original: f"Original @ {suffix}",
                    kl_div: f"KL @ {suffix}",
                })

    subset_df_int = subset_df_int.rename(columns={
        "query_item.argmax": "Accuracy",
        "query_item.pred_prob": "Likelihood",
    })
    return subset_df_int

# Load interchange data
int_dirs = [
    "experiments/logs/pcfg_easy/attention**/test/InterchangeEvaluator.csv",
    "experiments/logs/pcfg_easy_1L/attention**/test/InterchangeEvaluator.csv",
    "experiments/logs/pcfg_vary_difficulty_random_5_20/attention**/test/InterchangeEvaluator.csv",
    "experiments/logs/pcfg_vary_difficulty_random_5_20_1L/attention**/test/InterchangeEvaluator.csv",
    "experiments/logs/pcfg_vary_difficulty_random_5_20_3L/attention**/test/InterchangeEvaluator.csv",
]
df_int = read_df(int_dirs)

# Need to reload summary df without dataset mapping for joining
df_summary = df.copy()

# Get attribution df
# For 2L models use max_layers=2, for 1L use max_layers=1
# Process them separately and combine
results = []
for ds_key, ds_label in dataset_map.items():
    n_layers = 1 if "1L" in ds_label else 3 if "3L" in ds_label else 2
    df_sub = df_summary[df_summary["dataset"] == ds_key]
    df_int_sub = df_int[df_int["dataset"] == ds_key]
    if len(df_int_sub) == 0:
        print(f"No interchange data for {ds_key}, skipping")
        continue
    try:
        attrib = get_attrib_df(df_sub, df_int_sub, corruption="query_item_orig", components="block_input", prefix="PARENT", max_layers=n_layers)
        attrib["dataset"] = ds_label
        results.append(attrib)
    except Exception as e:
        print(f"Error processing {ds_key}: {e}")

subset_df_int = pd.concat(results).reset_index(drop=True)

# Plot 4: Restoration heatmap (like cell 39 in Sweep.ipynb)
# For best models per (dim, dataset), show Original, Corrupted, Restored @ Key/Value/Query
mapping = {}
for col in subset_df_int.columns:
    if col.startswith("Original @ 0, query_item_orig"):
        mapping[col] = "Original"
    elif col.startswith("Corrupted @ 0, query_item_orig"):
        mapping[col] = "Corrupted Key"
    elif col.startswith("Restored @ 1, query_item_orig"):
        mapping[col] = "Restored @ Key"
    elif col.startswith("Restored @ 1, target_item_orig"):
        mapping[col] = "Restored @ Value"
    elif col.startswith("Restored @ 1, query_item") and "orig" not in col:
        mapping[col] = "Restored @ Query"
    elif col.startswith("Restored @ 0, query_item_orig"):
        mapping["r0_key"] = col  # for 1L models, layer 0
    elif col.startswith("Restored @ 0, target_item_orig"):
        mapping["r0_val"] = col
    elif col.startswith("Restored @ 0, query_item") and "orig" not in col:
        mapping["r0_query"] = col

# Use a simpler approach: just pick the available columns
avail_mapping = {}
for col in subset_df_int.columns:
    if "Original @ 0, query_item_orig" == col:
        avail_mapping[col] = "Original"
    elif "Corrupted @ 0, query_item_orig" == col:
        avail_mapping[col] = "Corrupted Key"

# For restored, use highest available layer
for layer in [1, 0]:
    key = f"Restored @ {layer}, query_item_orig"
    if key in subset_df_int.columns and "Restored @ Key" not in avail_mapping.values():
        avail_mapping[key] = "Restored @ Key"
    key = f"Restored @ {layer}, target_item_orig"
    if key in subset_df_int.columns and "Restored @ Value" not in avail_mapping.values():
        avail_mapping[key] = "Restored @ Value"
    key = f"Restored @ {layer}, query_item"
    if key in subset_df_int.columns and "Restored @ Query" not in avail_mapping.values():
        avail_mapping[key] = "Restored @ Query"

print(f"Available mapping: {avail_mapping}")

subset_df_int_best = subset_df_int.iloc[
    subset_df_int.groupby(["dim", "dataset"])["Accuracy"].idxmax().dropna()
].reset_index(drop=True)

subset_df_temp = subset_df_int_best[["dim", "dataset", "lr"] + list(avail_mapping.keys())].copy()
subset_df_temp.dropna(inplace=True)
subset_df_temp = subset_df_temp.melt(
    id_vars=["dim", "dataset", "lr"],
    value_vars=list(avail_mapping.keys()),
    var_name="metric",
    value_name="value",
)
subset_df_temp["metric"] = subset_df_temp["metric"].map(avail_mapping)
subset_df_temp["metric"] = pd.Categorical(
    subset_df_temp["metric"],
    categories=["Original", "Corrupted Key", "Restored @ Key", "Restored @ Value", "Restored @ Query"],
    ordered=True,
)
subset_df_temp["dim"] = pd.Categorical(subset_df_temp["dim"])
subset_df_temp["dataset"] = pd.Categorical(
    subset_df_temp["dataset"],
    categories=["Informative NT, 2L", "Informative NT, 1L", "Uninformative NT, 3L", "Uninformative NT, 2L", "Uninformative NT, 1L"],
    ordered=True,
)
subset_df_temp["alpha"] = subset_df_temp.apply(lambda x: max(0, (x["value"] - 0.5)) / (1 - 0.5) if x["value"] > 0.5 else 0, axis=1)
subset_df_temp["value"] = subset_df_temp["value"] * 100

plot = (
    p9.ggplot(subset_df_temp, p9.aes(y="dataset", x="dim", alpha="alpha", fill="dataset", label="value"))
    + p9.geom_tile()
    + p9.geom_text(size=5, format_string="{:.1f}", alpha=1.0)
    + p9.facet_wrap("~metric", nrow=1)
    + p9.scale_color_brewer(type='qual', palette='Set2')
    + p9.scale_x_discrete(expand=[0, 0])
    + p9.scale_y_discrete(expand=[0, 0])
    + p9.labs(x="Model dimension", y=False)
    + p9.theme(
        figure_size=(10, 3),
        legend_position="none",
        panel_grid=p9.element_blank(),
    )
)
plot.save("scripts/figs/pcfg_difficulty_restore.pdf", dpi=300, limitsize=False)
plot.save("scripts/figs/pcfg_difficulty_restore.png", dpi=300, limitsize=False)
print("Saved pcfg_difficulty_restore")

# Plot 5: Association boxplots (like cell 40)
assoc_cols = [col for col in subset_df_int.columns if col.startswith("Association @")]
if assoc_cols:
    subset_df_int_plot = subset_df_int.copy()
    # Filter to models where corruption actually hurt
    orig_col = "Original @ 0, target_item_orig"
    corr_col = "Corrupted @ 0, target_item_orig"
    if orig_col in subset_df_int_plot.columns and corr_col in subset_df_int_plot.columns:
        subset_df_int_plot = subset_df_int_plot[
            (subset_df_int_plot[orig_col] - subset_df_int_plot[corr_col]) > 0.1
        ]

    assoc_mapping = {}
    for col in assoc_cols:
        if "query_item_orig" in col:
            assoc_mapping[col] = col.replace("query_item_orig", "Key")
        elif "target_item_orig" in col:
            assoc_mapping[col] = col.replace("target_item_orig", "Value")
        elif "query_item" in col:
            assoc_mapping[col] = col.replace("query_item", "Query")

    subset_df_int_plot = subset_df_int_plot[["dim", "dataset"] + list(assoc_mapping.keys())]
    subset_df_int_plot = subset_df_int_plot.melt(id_vars=["dim", "dataset"])
    subset_df_int_plot["variable"] = subset_df_int_plot["variable"].map(assoc_mapping)
    subset_df_int_plot["dataset"] = pd.Categorical(
        subset_df_int_plot["dataset"],
        categories=["Informative NT, 2L", "Informative NT, 1L", "Uninformative NT, 3L", "Uninformative NT, 2L", "Uninformative NT, 1L"],
        ordered=True,
    )

    plot = (
        p9.ggplot(subset_df_int_plot, p9.aes(x="dataset", y="value", fill="dataset"))
        + p9.facet_wrap("~variable", nrow=1)
        + p9.geom_boxplot(outlier_alpha=0.3, color="#4f4f4f")
        + p9.labs(y="Association", x=False)
        + p9.scale_fill_brewer(type='qual', palette='Set2')
        + p9.geom_hline(yintercept=1, linetype="dashed", color="#4f4f4f")
        + p9.geom_hline(yintercept=0, linetype="dashed", color="#4f4f4f")
        + p9.theme(
            figure_size=(12, 3.5),
            legend_position="none",
            axis_text_x=p9.element_text(angle=45, hjust=1),
        )
    )
    plot.save("scripts/figs/pcfg_difficulty_association.pdf", dpi=300, limitsize=False)
    plot.save("scripts/figs/pcfg_difficulty_association.png", dpi=300, limitsize=False)
    print("Saved pcfg_difficulty_association")

print("\nDone! All plots saved to scripts/figs/")
