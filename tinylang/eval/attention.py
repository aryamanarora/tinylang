from .eval import Evaluator
from tinylang.model import Model
from collections import defaultdict
import numpy as np
import plotnine as p9
import os
import pandas as pd


class AttentionEvaluator(Evaluator):
    def __str__(self):
        return "AttentionEvaluator"

    def __init__(self, run_every_n_steps: int):
        super().__init__(run_every_n_steps)

    def eval(self, model: Model, inputs: dict, outputs: dict, step: int):
        probing_schemas = inputs["probing_schemas"]
        attentions = outputs["attentions"]

        # for i in range(attentions[0].shape[0]):
        #     type = probing_schemas[i]["type"]
        #     # highlight query_pos and key_pos in toks
        #     if type == "CHILD":
        #         print(inputs["strs_pretty"][i][0])
        #         print(inputs["strs_pretty"][i][1])
        #         print(inputs["strs"][i])
        #         input()

        for layer in range(len(attentions)):
            for head in range(attentions[layer].shape[1]):
                for i in range(attentions[layer].shape[0]):
                    for q, info in probing_schemas[i]["queries"].items():
                        attention = attentions[layer][i, head]
                        schema = probing_schemas[i]
                        query_pos = info["pos"]
                        type = schema["type"]
                        for k, key_pos in schema["keys"].items():
                            attn_score = 0.0
                            if isinstance(key_pos, list):
                                for key_pos_i in key_pos:
                                    attn_score += attention[query_pos, key_pos_i].item()
                            else:
                                attn_score = attention[query_pos, key_pos].item()
                            self.all_eval_stats[step][f"{layer}.{head}.{type}.{q}.{k}.attn"].append(attn_score)


    def plot(self, log_dir: str):
        """Plot the attention scores."""

        # split the column name into layer, head, type, token
        df = self.df
        df = df.groupby(["step", "variable"]).mean().reset_index()
        df["head"] = df["variable"].str.split(".").str[0:2].apply(lambda x: ".".join(x))
        df["type"] = df["variable"].str.split(".").str[2]
        df["query"] = df["variable"].str.split(".").str[3]
        df["key"] = df["variable"].str.split(".").str[4]
        df = df.drop(columns=["variable"])

        # make plot
        for query in df["query"].unique():
            df_subset = df[df["query"] == query]
            plot = (
                p9.ggplot(df_subset, p9.aes(x="step", y="value", color="head"))
                + p9.geom_line()
                + p9.facet_wrap("~type + key")
            )
            plot.save(os.path.join(log_dir, f"{str(self)}.per_key.{query}.png"))

            plot = (
                p9.ggplot(df_subset[~df_subset["key"].str.startswith("_")], p9.aes(x="step", y="value", fill="key"))
                + p9.geom_area(position="stack")
                + p9.facet_wrap("~type + head")
            )
            plot.save(os.path.join(log_dir, f"{str(self)}.per_head.{query}.png"))

        for key in df["key"].unique():
            df_subset = df[df["key"] == key]
            plot = (
                p9.ggplot(df_subset, p9.aes(x="step", y="value", color="head"))
                + p9.geom_line()
                + p9.facet_wrap("~type + query")
            )
            plot.save(os.path.join(log_dir, f"{str(self)}.per_query.{key}.png"))
