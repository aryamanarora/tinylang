"""Heatmap of yield length vs answer position in the yield, comparing configs."""

import numpy as np
import pandas as pd
import plotnine as p9
import os

import sys
sys.path.insert(0, ".")
from tinylang.language.pcfg import PCFG

p9.theme_set(
    p9.theme_bw(base_size=10) +
    p9.theme(
        figure_size=(2.5, 2.5),
        axis_title=p9.element_text(size=10),
        axis_text=p9.element_text(size=8),
        legend_position="bottom",
    )
)

configs = {
    "rhs=5": dict(max_rhs_len=5),
    "rhs=10": dict(max_rhs_len=10),
}

all_rows = []
for config_name, overrides in configs.items():
    print(f"\n{'='*60}")
    print(f"Config: {config_name}")

    kwargs = dict(
        num_terminals=20,
        num_nonterminals=40,
        max_rhs_len=5,
        max_rules_per_nt=5,
        max_depth=10,
        head_position="right",
        mask_nonquery=True,
        no_child_queries=True,
        no_sibling_queries=True,
        uninformative_nonterminals=True,
        unambiguous_queries=True,
        prepare_train_set=True,
    )
    kwargs.update(overrides)
    lang = PCFG(**kwargs)

    for _ in range(20000):
        tokens, probing_schema = lang.sample(split="train")
        divider_pos = probing_schema["queries"]["divider"]["pos"]
        t_orig_pos = probing_schema["queries"]["target_item_orig"]["pos"]
        yield_len = divider_pos - 1
        ans_rel = (t_orig_pos - 1) / (yield_len - 1) if yield_len > 1 else 0.5
        all_rows.append({
            "config": config_name,
            "yield_len": yield_len,
            "ans_pos": t_orig_pos - 1,
            "ans_rel": ans_rel,
        })

    subset = [r for r in all_rows if r["config"] == config_name]
    sub_df = pd.DataFrame(subset)
    print(f"  Mean relative answer position: {sub_df['ans_rel'].mean():.3f}")
    print(f"  Fraction at rightmost: {(sub_df['ans_pos'] == sub_df['yield_len'] - 1).mean():.3f}")
    print(f"  Fraction in last 10%: {(sub_df['ans_rel'] >= 0.9).mean():.3f}")
    print(f"  Mean yield length: {sub_df['yield_len'].mean():.1f}")

df = pd.DataFrame(all_rows)
os.makedirs("scripts/figs", exist_ok=True)

# Combined histogram of relative answer positions
plot = (
    p9.ggplot(df, p9.aes(x="ans_rel", fill="config")) +
    p9.geom_histogram(bins=30, alpha=0.7, position="dodge") +
    p9.labs(x="Relative answer position (0=left, 1=right)", y="Count",
            title="Answer position distribution", fill="Config") +
    p9.theme(figure_size=(6, 3))
)
plot.save("scripts/figs/pcfg_anspos_hist_compare.png", dpi=300, limitsize=False)
print("\nSaved pcfg_anspos_hist_compare")

# Faceted heatmap: yield_len vs ans_pos
heatmap_df = df.groupby(["config", "yield_len", "ans_pos"]).size().reset_index(name="count")
plot = (
    p9.ggplot(heatmap_df, p9.aes(x="ans_pos", y="yield_len", fill="count")) +
    p9.geom_tile() +
    p9.scale_fill_cmap("viridis") +
    p9.facet_wrap("~config", nrow=1) +
    p9.labs(x="Answer position in yield (0-indexed)", y="Yield length", fill="Count",
            title="Yield length vs answer position") +
    p9.theme(figure_size=(12, 5))
)
plot.save("scripts/figs/pcfg_length_vs_anspos_compare.png", dpi=300, limitsize=False)
print("Saved pcfg_length_vs_anspos_compare")

print("\nDone!")
