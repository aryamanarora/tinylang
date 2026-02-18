"""Heatmap of yield length vs answer position in the yield."""

import numpy as np
import pandas as pd
import plotnine as p9

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

lang = PCFG(
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

rows = []
for _ in range(20000):
    tokens, probing_schema = lang.sample(split="train")
    divider_pos = probing_schema["queries"]["divider"]["pos"]
    t_orig_pos = probing_schema["queries"]["target_item_orig"]["pos"]
    yield_len = divider_pos - 1  # tokens[1] to tokens[divider_pos-1]
    # answer position as fraction of yield length (0 = leftmost, 1 = rightmost)
    ans_rel = (t_orig_pos - 1) / (yield_len - 1) if yield_len > 1 else 0.5
    rows.append({
        "yield_len": yield_len,
        "ans_pos": t_orig_pos - 1,  # 0-indexed within yield
        "ans_rel": ans_rel,
    })

df = pd.DataFrame(rows)

import os
os.makedirs("scripts/figs", exist_ok=True)

# Heatmap: yield_len vs ans_pos (absolute)
# Bin yield lengths
df["len_bin"] = pd.cut(df["yield_len"], bins=range(0, df["yield_len"].max() + 5, 5), right=False)
heatmap_df = df.groupby(["yield_len", "ans_pos"]).size().reset_index(name="count")

plot = (
    p9.ggplot(heatmap_df, p9.aes(x="ans_pos", y="yield_len", fill="count")) +
    p9.geom_tile() +
    p9.scale_fill_cmap("viridis") +
    p9.labs(x="Answer position in yield (0-indexed)", y="Yield length", fill="Count",
            title="Yield length vs answer position") +
    p9.theme(figure_size=(8, 6))
)
plot.save("scripts/figs/pcfg_length_vs_anspos.png", dpi=300, limitsize=False)
print("Saved pcfg_length_vs_anspos")

# Heatmap: yield_len vs relative answer position
df["ans_rel_bin"] = pd.cut(df["ans_rel"], bins=np.linspace(0, 1, 21))
heatmap_rel = df.groupby(["yield_len", "ans_rel_bin"]).size().reset_index(name="count")
heatmap_rel["ans_rel_mid"] = heatmap_rel["ans_rel_bin"].apply(lambda x: x.mid if pd.notna(x) else None)
heatmap_rel = heatmap_rel.dropna()

plot = (
    p9.ggplot(heatmap_rel, p9.aes(x="ans_rel_mid", y="yield_len", fill="count")) +
    p9.geom_tile() +
    p9.scale_fill_cmap("viridis") +
    p9.labs(x="Relative answer position (0=left, 1=right)", y="Yield length", fill="Count",
            title="Yield length vs relative answer position") +
    p9.theme(figure_size=(8, 6))
)
plot.save("scripts/figs/pcfg_length_vs_anspos_rel.png", dpi=300, limitsize=False)
print("Saved pcfg_length_vs_anspos_rel")

# Also just a histogram of relative answer positions
plot = (
    p9.ggplot(df, p9.aes(x="ans_rel")) +
    p9.geom_histogram(bins=30, fill="#66c2a5") +
    p9.labs(x="Relative answer position (0=left, 1=right)", y="Count",
            title="Distribution of answer positions") +
    p9.theme(figure_size=(6, 3))
)
plot.save("scripts/figs/pcfg_anspos_hist.png", dpi=300, limitsize=False)
print("Saved pcfg_anspos_hist")

# Stats
print(f"\nMean relative answer position: {df['ans_rel'].mean():.3f}")
print(f"Fraction of answers at rightmost position: {(df['ans_pos'] == df['yield_len'] - 1).mean():.3f}")
print(f"Fraction of answers in last 10% of yield: {(df['ans_rel'] >= 0.9).mean():.3f}")
print(f"Mean yield length: {df['yield_len'].mean():.1f}")

print("\nDone!")
