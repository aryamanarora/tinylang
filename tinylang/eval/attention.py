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
        self.agg_funcs = {
            "attn": lambda x: np.mean(x).item(),
        }

    def eval(self, model: Model, inputs: dict, outputs: dict):
        probing_schemas = inputs["probing_schemas"]
        attentions = outputs["attentions"]
        result = defaultdict(list)

        for layer in range(len(attentions)):
            for head in range(attentions[layer].shape[1]):
                for i in range(attentions[layer].shape[0]):
                    attention = attentions[layer][i, head]
                    schema = probing_schemas[i]
                    query_pos = schema["target_pos"]
                    type = schema["type"]
                    for token, key_pos in schema["tokens"].items():
                        attn_score = attention[query_pos, key_pos]
                        result[f"{layer}.{head}.{type}.{token}.attn"].append(attn_score)

        return result


    def plot(self, df: pd.DataFrame, log_dir: str):
        """Plot the attention scores."""

        # split the column name into layer, head, type, token
        df = df.melt(id_vars=["step"])
        df["layer"] = df["variable"].str.split(".").str[0]
        df["head"] = df["variable"].str.split(".").str[1]
        df["type"] = df["variable"].str.split(".").str[2]
        df["token"] = df["variable"].str.split(".").str[3]
        df = df.drop(columns=["variable"])

        # make plot
        plot = (
            p9.ggplot(df, p9.aes(x="step", y="value", color="layer", linetype="head"))
            + p9.geom_line()
            + p9.facet_wrap("~type + token")
        )
        plot.save(os.path.join(log_dir, f"{str(self)}.png"))
