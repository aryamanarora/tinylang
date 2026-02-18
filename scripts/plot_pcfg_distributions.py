"""Plot distribution over query tokens and answer tokens for uninformative NT PCFG."""

import numpy as np
import pandas as pd
import plotnine as p9
from collections import Counter

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

# Create the language with same config as pcfg_5_20.yaml
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

# Sample many examples
n_samples = 5000
query_tokens = []
answer_tokens = []
query_answer_pairs = []

for _ in range(n_samples):
    tokens, probing_schema = lang.sample(split="train")

    query_pos = probing_schema["queries"]["query_item"]["pos"]
    target_pos = probing_schema["queries"]["target_item"]["pos"]

    query_tok = lang.id_to_token[int(tokens[query_pos])]
    answer_tok = lang.id_to_token[int(tokens[target_pos])]

    query_tokens.append(query_tok)
    answer_tokens.append(answer_tok)
    query_answer_pairs.append((query_tok, answer_tok))

import os
os.makedirs("scripts/figs", exist_ok=True)

# Plot 1: Distribution over answer tokens
answer_counts = Counter(answer_tokens)
answer_df = pd.DataFrame({"token": list(answer_counts.keys()), "count": list(answer_counts.values())})
answer_df["freq"] = answer_df["count"] / answer_df["count"].sum()
answer_df = answer_df.sort_values("token")

plot = (
    p9.ggplot(answer_df, p9.aes(x="token", y="freq")) +
    p9.geom_col(fill="#66c2a5") +
    p9.labs(x="Answer token", y="Frequency", title="Answer distribution") +
    p9.theme(figure_size=(6, 3), axis_text_x=p9.element_text(rotation=45, hjust=1))
)
plot.save("scripts/figs/pcfg_answer_dist.png", dpi=300, limitsize=False)
print("Saved pcfg_answer_dist")

# Plot 2: Distribution over query tokens
query_counts = Counter(query_tokens)
query_df = pd.DataFrame({"token": list(query_counts.keys()), "count": list(query_counts.values())})
query_df["freq"] = query_df["count"] / query_df["count"].sum()
query_df = query_df.sort_values("token")

plot = (
    p9.ggplot(query_df, p9.aes(x="token", y="freq")) +
    p9.geom_col(fill="#fc8d62") +
    p9.labs(x="Query token", y="Frequency", title="Query distribution") +
    p9.theme(figure_size=(6, 3), axis_text_x=p9.element_text(rotation=45, hjust=1))
)
plot.save("scripts/figs/pcfg_query_dist.png", dpi=300, limitsize=False)
print("Saved pcfg_query_dist")

# Plot 3: P(answer | query) heatmap
pair_counts = Counter(query_answer_pairs)
pair_df = pd.DataFrame([(q, a, c) for (q, a), c in pair_counts.items()], columns=["query", "answer", "count"])
# normalize per query
pair_df["total"] = pair_df.groupby("query")["count"].transform("sum")
pair_df["prob"] = pair_df["count"] / pair_df["total"]

plot = (
    p9.ggplot(pair_df, p9.aes(x="answer", y="query", fill="prob")) +
    p9.geom_tile() +
    p9.scale_fill_cmap("viridis") +
    p9.labs(x="Answer", y="Query", fill="P(ans|query)", title="P(answer | query)") +
    p9.theme(
        figure_size=(7, 7),
        axis_text_x=p9.element_text(rotation=45, hjust=1),
    )
)
plot.save("scripts/figs/pcfg_query_answer_heatmap.png", dpi=300, limitsize=False)
print("Saved pcfg_query_answer_heatmap")

# Print some stats
print(f"\nNum unique answers: {len(answer_counts)}")
print(f"Max answer freq: {max(answer_counts.values()) / n_samples:.3f}")
print(f"Min answer freq: {min(answer_counts.values()) / n_samples:.3f}")
print(f"Entropy of answer dist: {-sum(f * np.log2(f) for f in answer_df['freq']):.2f} bits (max = {np.log2(20):.2f})")

# What accuracy would "always guess most common answer" get?
print(f"Majority baseline: {max(answer_counts.values()) / n_samples:.3f}")

# What accuracy would "guess most common answer per query" get?
correct = 0
for q_tok in query_tokens:
    # most common answer for this query
    best_answer = Counter([a for (q, a) in query_answer_pairs if q == q_tok]).most_common(1)[0][0]
    # but we need per-sample, so just count
per_query_best = {}
for q, a in query_answer_pairs:
    if q not in per_query_best:
        per_query_best[q] = Counter()
    per_query_best[q][a] += 1

correct = sum(per_query_best[q].most_common(1)[0][1] for q in query_tokens if q in per_query_best) / n_samples
# wait, that double-counts. let me redo
correct = 0
for q_tok, a_tok in query_answer_pairs:
    best = per_query_best[q_tok].most_common(1)[0][0]
    if a_tok == best:
        correct += 1
print(f"Per-query majority baseline: {correct / n_samples:.3f}")

# Plot 4: Distance between query_item_orig and target_item_orig
distances = []
answer_is_right = []
for _ in range(10000):
    tokens, probing_schema = lang.sample(split="train")
    q_pos = probing_schema["queries"]["query_item_orig"]["pos"]
    t_pos = probing_schema["queries"]["target_item_orig"]["pos"]
    distances.append(t_pos - q_pos)
    answer_is_right.append(t_pos > q_pos)

dist_df = pd.DataFrame({"distance": distances})
print(f"\nAnswer is to the RIGHT of query: {sum(answer_is_right)/len(answer_is_right)*100:.1f}%")
print(f"Mean distance: {np.mean(distances):.1f}")
print(f"Median distance: {np.median(distances):.1f}")

# histogram
plot = (
    p9.ggplot(dist_df, p9.aes(x="distance")) +
    p9.geom_histogram(bins=50, fill="#8da0cb") +
    p9.geom_vline(xintercept=0, linetype="dashed", color="red") +
    p9.labs(x="Distance (target_pos - query_pos)", y="Count", title="Query-to-answer distance") +
    p9.theme(figure_size=(6, 3))
)
plot.save("scripts/figs/pcfg_distance_dist.png", dpi=300, limitsize=False)
print("Saved pcfg_distance_dist")

# What accuracy would "copy token at position query+1" get?
copy_right_correct = 0
for _ in range(10000):
    tokens, probing_schema = lang.sample(split="train")
    q_pos = probing_schema["queries"]["query_item_orig"]["pos"]
    t_pos = probing_schema["queries"]["target_item_orig"]["pos"]
    target_tok = int(tokens[t_pos])
    # try copying from query+1
    if q_pos + 1 < len(tokens):
        if int(tokens[q_pos + 1]) == target_tok:
            copy_right_correct += 1
print(f"'Copy token at query+1' accuracy: {copy_right_correct/10000:.3f}")

# What about "copy rightmost terminal in sequence"?
rightmost_correct = 0
for _ in range(10000):
    tokens, probing_schema = lang.sample(split="train")
    t_pos = probing_schema["queries"]["target_item_orig"]["pos"]
    target_tok = int(tokens[t_pos])
    # find rightmost terminal (before EOS)
    divider_pos = probing_schema["queries"]["divider"]["pos"]
    rightmost_tok = int(tokens[divider_pos - 1])
    if rightmost_tok == target_tok:
        rightmost_correct += 1
print(f"'Copy rightmost terminal' accuracy: {rightmost_correct/10000:.3f}")

# What about bigram statistics? P(next_token | token) from the tree yield
# If the query token is X, and in tree yields X is typically followed by Y,
# then predicting Y gets you accuracy without needing to locate X.
bigram_counts = Counter()  # (token, next_token) -> count
token_counts = Counter()   # token -> count as non-final position
for _ in range(10000):
    tokens, probing_schema = lang.sample(split="train")
    divider_pos = probing_schema["queries"]["divider"]["pos"]
    # tree yield is tokens[1:divider_pos] (skip BOS, stop before EOS)
    yield_tokens = [int(tokens[i]) for i in range(1, divider_pos)]
    for i in range(len(yield_tokens) - 1):
        bigram_counts[(yield_tokens[i], yield_tokens[i+1])] += 1
        token_counts[yield_tokens[i]] += 1

# For each token, what's the most likely next token?
best_next = {}
for tok in token_counts:
    candidates = {next_tok: bigram_counts[(tok, next_tok)] for next_tok in range(lang.vocab_size) if (tok, next_tok) in bigram_counts}
    if candidates:
        best_next[tok] = max(candidates, key=candidates.get)

# Now test: if model just predicts best_next[query_token], what accuracy?
bigram_correct = 0
bigram_total = 0
for _ in range(10000):
    tokens, probing_schema = lang.sample(split="train")
    q_pos = probing_schema["queries"]["query_item"]["pos"]
    t_pos = probing_schema["queries"]["target_item"]["pos"]
    query_tok = int(tokens[q_pos])
    target_tok = int(tokens[t_pos])
    if query_tok in best_next:
        if best_next[query_tok] == target_tok:
            bigram_correct += 1
    bigram_total += 1
print(f"Bigram heuristic (predict most common next token for query type): {bigram_correct/bigram_total:.3f}")

# Also: full conditional P(answer | query, using bigram)
# i.e., predict with the full bigram distribution, what's the expected accuracy?
bigram_prob_correct = 0
for _ in range(10000):
    tokens, probing_schema = lang.sample(split="train")
    q_pos = probing_schema["queries"]["query_item"]["pos"]
    t_pos = probing_schema["queries"]["target_item"]["pos"]
    query_tok = int(tokens[q_pos])
    target_tok = int(tokens[t_pos])
    if query_tok in token_counts:
        prob = bigram_counts.get((query_tok, target_tok), 0) / token_counts[query_tok]
        bigram_prob_correct += prob
print(f"Bigram expected accuracy (P(answer|query) from bigram): {bigram_prob_correct/10000:.3f}")

print("\nDone!")
