import pandas as pd
import plotnine as p9
from mizani.formatters import scientific_format

p9.theme_set(
    p9.theme_bw(base_size=10) +
    p9.theme(
        text=p9.element_text(family="P052", color="#000"),
        figure_size=(2.5, 2.5),
        axis_title=p9.element_text(size=10),
        axis_text=p9.element_text(size=8),
        legend_position="bottom",
        legend_text=p9.element_text(size=8),
        legend_title=p9.element_text(size=9),
        panel_grid_major=p9.element_line(size=0.3, color="#dddddd"),
        panel_grid_minor=p9.element_blank(),
        legend_justification_bottom=1,
        strip_background=p9.element_blank(),
        legend_margin=0,
    )
)

df = pd.read_csv("mamba_ablation.csv")
df["location"] = df.apply(lambda row: f"{row['component']}, {row['layer']}, {row['restored']}", axis=1)
df["architecture"] = "Mamba"
df.lr = pd.Categorical(df.lr)
df.n_embd = pd.Categorical(df.n_embd)

# locations
locations = {
    "conv_output, original, target_item_orig": "Original",
    "conv_output, corrupted, target_item_orig": "Corrupted",
    "conv_output, 0, target_item_orig": "Restored @ 0, Value",
    "conv_output, 0, target_item_orig__+1": "Restored @ 0, Next Key",
    "conv_output, 1, target_item_orig": "Restored @ 1, Value",
}

# plot the value vs the lr
subset_df_temp = df[
    (df.metric == "restored_prob") &
    (df.location.isin(list(locations.keys()))) &
    (df.corrupted == "query_item_orig")
    # (df.layer.isin(["0", "1"])) &
    # (df.component == "conv_output")
]
subset_df_temp.location = subset_df_temp.location.apply(lambda x: locations[x])
subset_df_temp.location = pd.Categorical(subset_df_temp.location, categories=list(locations.values()), ordered=True)
subset_df_temp["label"] = subset_df_temp.value.apply(lambda x: f"{x * 100:.1f}")
subset_df_temp.dropna(inplace=True)
# subset_df_temp.corrupted = susbet_df_temp.corrupted.map({
#     "query_item_orig": "Key",
#     "target_item_orig": "Value",
# })
print(subset_df_temp.value.max(), subset_df_temp.value.min())
plot = (
    p9.ggplot(subset_df_temp, p9.aes(y="lr", x="n_embd", label="label", alpha="value")) +
    p9.geom_tile(p9.aes(alpha="value")) +
    p9.geom_text(subset_df_temp[subset_df_temp.value > 0.5], size=6, alpha=1.0, color="white") +
    p9.geom_text(subset_df_temp[subset_df_temp.value <= 0.5], size=6, alpha=1.0, color="black") +
    p9.facet_grid("~location") +
    p9.scale_y_discrete(labels=scientific_format(), expand=[0, 0]) +
    p9.scale_x_discrete(expand=[0, 0]) +
    p9.labs(x="Model dimension", y="LR") +
    p9.theme(
        figure_size=(6, 1.75),
        legend_position="none",
        panel_grid=p9.element_blank(),
    )
)
plot.save(f"mamba_ablation.pdf", dpi=300)