"""Analyze what a 1L attention model actually predicts on uninformative NT PCFG."""

import torch
import numpy as np
import os
import yaml
from collections import Counter
import pandas as pd
import plotnine as p9

import sys
sys.path.insert(0, ".")
from tinylang.language.language import Language
from tinylang.model.model import Model

p9.theme_set(
    p9.theme_bw(base_size=10) +
    p9.theme(
        figure_size=(2.5, 2.5),
        axis_title=p9.element_text(size=10),
        axis_text=p9.element_text(size=8),
        legend_position="bottom",
    )
)

# Pick the best 1L model (dim=128 or 256, best LR)
# Let's try a few and pick whichever exists
candidates = [
    ("experiments/logs/pcfg_vary_difficulty_random_5_20_1L/attention___128___3e-04", 128),
    ("experiments/logs/pcfg_vary_difficulty_random_5_20_1L/attention___128___1e-03", 128),
    ("experiments/logs/pcfg_vary_difficulty_random_5_20_1L/attention___256___3e-04", 256),
    ("experiments/logs/pcfg_vary_difficulty_random_5_20_1L/attention___64___3e-04", 64),
]

log_dir = None
for path, dim in candidates:
    if os.path.exists(os.path.join(path, "model.pt")):
        log_dir = path
        model_dim = dim
        break

if log_dir is None:
    print("No model found! Trying all 1L dirs...")
    import glob
    dirs = glob.glob("experiments/logs/pcfg_vary_difficulty_random_5_20_1L/attention*/model.pt")
    if dirs:
        log_dir = os.path.dirname(dirs[0])
        model_dim = int(log_dir.split("___")[1])
    else:
        raise FileNotFoundError("No trained 1L models found")

print(f"Using model from: {log_dir} (dim={model_dim})")

# Load language
lang = Language.load(os.path.join(log_dir, "language.pkl"))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model config and model
config_path = f"experiments/configs/pcfg_vary_difficulty_random_5_20_1L/attention___{model_dim}___3e-04.yaml"
if not os.path.exists(config_path):
    # find matching config
    import glob
    lr_str = log_dir.split("___")[-1]
    config_path = f"experiments/configs/pcfg_vary_difficulty_random_5_20_1L/attention___{model_dim}___{lr_str}.yaml"

with open(config_path) as f:
    config = yaml.safe_load(f)

config["model"]["config"]["vocab_size"] = lang.vocab_size
config["model"]["config"]["device"] = device
model = Model.from_config(config["model"])
model.load(os.path.join(log_dir, "model.pt"))
model.model.eval()

print(f"Vocab size: {lang.vocab_size}")
print(f"Terminals: {lang.num_terminals}")

# Run inference on many samples
n_samples = 2000
results = []

with torch.no_grad():
    for _ in range(n_samples):
        tokens, probing_schema = lang.sample(split="train")

        # Get positions
        q_pos = probing_schema["queries"]["query_item"]["pos"]  # query in suffix
        t_pos = probing_schema["queries"]["target_item"]["pos"]  # target position (what we predict)
        q_orig_pos = probing_schema["queries"]["query_item_orig"]["pos"]  # query in tree yield
        t_orig_pos = probing_schema["queries"]["target_item_orig"]["pos"]  # answer in tree yield
        divider_pos = probing_schema["queries"]["divider"]["pos"]

        # True answer
        true_answer = int(tokens[t_pos])

        # Token at query_orig + 1 (the "copy next" heuristic)
        next_tok = int(tokens[q_orig_pos + 1]) if q_orig_pos + 1 < len(tokens) else -1

        # Rightmost terminal in yield
        rightmost_tok = int(tokens[divider_pos - 1])

        # Query token
        query_tok = int(tokens[q_pos])

        # Run model
        input_ids = torch.tensor([tokens[:t_pos]], device=device)
        logits = model.model(input_ids)[0, -1]  # logits at position before target
        pred = logits.argmax().item()
        pred_prob = torch.softmax(logits, dim=0)[true_answer].item()

        # Top-5 predictions
        top5 = logits.topk(5)
        top5_tokens = top5.indices.tolist()

        # Distance from query_orig to target_orig
        distance = t_orig_pos - q_orig_pos

        # Sequence length (tree yield only)
        yield_len = divider_pos - 1  # BOS at 0, yield from 1 to divider-1

        results.append({
            "true_answer": true_answer,
            "pred": pred,
            "correct": pred == true_answer,
            "next_tok": next_tok,
            "pred_is_next": pred == next_tok,
            "rightmost_tok": rightmost_tok,
            "pred_is_rightmost": pred == rightmost_tok,
            "query_tok": query_tok,
            "distance": distance,
            "yield_len": yield_len,
            "pred_prob": pred_prob,
            "true_answer_name": lang.id_to_token.get(true_answer, "?"),
            "pred_name": lang.id_to_token.get(pred, "?"),
            "next_tok_name": lang.id_to_token.get(next_tok, "?"),
            "q_orig_pos": q_orig_pos,
            "t_orig_pos": t_orig_pos,
        })

df = pd.DataFrame(results)

os.makedirs("scripts/figs", exist_ok=True)

# Basic stats
print(f"\n=== Model accuracy: {df['correct'].mean():.3f} ===")
print(f"Pred matches 'copy next token': {df['pred_is_next'].mean():.3f}")
print(f"Pred matches 'copy rightmost': {df['pred_is_rightmost'].mean():.3f}")

# When model is correct, is it because it copied next token?
correct_df = df[df["correct"]]
print(f"\nWhen correct ({len(correct_df)}/{len(df)}):")
print(f"  True answer IS next token: {(correct_df['true_answer'] == correct_df['next_tok']).mean():.3f}")
print(f"  True answer IS rightmost: {(correct_df['true_answer'] == correct_df['rightmost_tok']).mean():.3f}")

# When model is wrong, what does it predict?
wrong_df = df[~df["correct"]]
print(f"\nWhen wrong ({len(wrong_df)}/{len(df)}):")
print(f"  Pred IS next token: {wrong_df['pred_is_next'].mean():.3f}")
print(f"  Pred IS rightmost: {wrong_df['pred_is_rightmost'].mean():.3f}")

# Accuracy by distance
acc_by_dist = df.groupby("distance")["correct"].agg(["mean", "count"]).reset_index()
acc_by_dist.columns = ["distance", "accuracy", "count"]
print(f"\nAccuracy by distance (query_orig → target_orig):")
print(acc_by_dist.to_string(index=False))

# Plot: accuracy by distance
plot = (
    p9.ggplot(acc_by_dist[acc_by_dist["count"] >= 10], p9.aes(x="distance", y="accuracy")) +
    p9.geom_col(fill="#8da0cb") +
    p9.geom_text(p9.aes(label="count"), va="bottom", size=7) +
    p9.scale_y_continuous(limits=(0, 1.05)) +
    p9.labs(x="Distance (target_orig - query_orig)", y="Accuracy",
            title="1L model accuracy by query-answer distance") +
    p9.theme(figure_size=(8, 3))
)
plot.save("scripts/figs/pcfg_1L_acc_by_distance.png", dpi=300, limitsize=False)
print("Saved pcfg_1L_acc_by_distance")

# For each position offset from query_orig, how often does the model predict
# the token at that offset?
offsets = list(range(0, 15))
offset_matches = []
for _, row in df.iterrows():
    for offset in offsets:
        pos = int(row["q_orig_pos"]) + offset
        if pos >= int(row["yield_len"]) + 1 or pos < 1:  # +1 for BOS
            continue
        # We need the actual token at that position - we don't have it stored
        # Skip this analysis, do it differently

# Instead: check if prediction equals any token in a window around query_orig
# We need to re-run with token storage
print("\n=== Re-running with token storage for position analysis ===")
position_pred_matches = Counter()  # offset -> count of times pred matches token at that offset
position_total = Counter()

with torch.no_grad():
    for _ in range(5000):
        tokens, probing_schema = lang.sample(split="train")
        q_orig_pos = probing_schema["queries"]["query_item_orig"]["pos"]
        t_pos = probing_schema["queries"]["target_item"]["pos"]
        divider_pos = probing_schema["queries"]["divider"]["pos"]

        input_ids = torch.tensor([tokens[:t_pos]], device=device)
        logits = model.model(input_ids)[0, -1]
        pred = logits.argmax().item()

        # Check each offset from query_orig
        for offset in range(-5, 20):
            pos = q_orig_pos + offset
            if pos < 1 or pos >= divider_pos:
                continue
            position_total[offset] += 1
            if int(tokens[pos]) == pred:
                position_pred_matches[offset] += 1

# Plot: how often does pred match token at each offset
offset_df = pd.DataFrame([
    {"offset": k, "match_rate": position_pred_matches[k] / position_total[k], "count": position_total[k]}
    for k in sorted(position_total.keys())
    if position_total[k] >= 50
])
print("\nPred matches token at offset from query_orig:")
print(offset_df.to_string(index=False))

plot = (
    p9.ggplot(offset_df, p9.aes(x="offset", y="match_rate")) +
    p9.geom_col(fill="#fc8d62") +
    p9.geom_vline(xintercept=0, linetype="dashed", color="blue", size=0.5) +
    p9.labs(x="Offset from query_orig position", y="P(pred = token at offset)",
            title="What position is the 1L model copying from?") +
    p9.scale_y_continuous(limits=(0, 1.0)) +
    p9.theme(figure_size=(8, 3))
)
plot.save("scripts/figs/pcfg_1L_copy_position.png", dpi=300, limitsize=False)
print("Saved pcfg_1L_copy_position")

print("\nDone!")
