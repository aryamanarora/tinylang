from abc import ABC, abstractmethod
from tinylang.model import Model
import torch
import importlib
from typing import Any
import pandas as pd
import plotnine as p9
import os
from collections import defaultdict
def flatten_list(inputs: list[list[Any]]) -> list[Any]:
    if not isinstance(inputs[0], list):
        return inputs
    return [item for sublist in inputs for item in sublist]


class Evaluator(ABC):
    def __init__(
        self,
        run_every_n_steps: int,
    ):
        self.run_every_n_steps = run_every_n_steps
        self.all_eval_stats = defaultdict(lambda: defaultdict(list))
        self.agg_funcs = {}

    @abstractmethod
    def __str__(self):
        pass

    @classmethod
    def from_config(cls, config: dict):
        """Load an evaluator from a config dict.
        
        Args:
            config: Dict containing:
                - class: Name of the evaluator class to instantiate
                - config: Dict of config parameters for the evaluator
        """
        # Get the class from the config
        evaluator_class = getattr(importlib.import_module("tinylang.eval"), config["class"])
        # Create an instance with the config
        evaluator = evaluator_class(**config["config"])
        return evaluator
    
    @abstractmethod
    def eval(self, model: Model, inputs: dict, outputs: dict, step: int):
        pass

    def get_agg_func(self, stat: str):
        return self.agg_funcs[stat.split(".")[-1]]
    
    def prepare_plot(self):
        rows = []
        for step, stats in self.all_eval_stats.items():
            for k, v in stats.items():
                if isinstance(v, list):
                    for val in v:
                        rows.append({"variable": k, "value": val, "step": step})
                else:
                    rows.append({"variable": k, "value": v, "step": step})
        self.df = pd.DataFrame(rows)

    def plot(self, log_dir: str):
        """Default plot method for all evaluators."""
        assert self.df is not None, "Please call prepare_plot() first"
        df = self.df
        for col in df["variable"].unique():
            # make sure type is numeric
            if col == "step":
                continue
            df_subset = df[df["variable"] == col].drop(columns=["variable"])
            if not pd.api.types.is_numeric_dtype(df_subset["value"]):
                continue
            df_subset = df_subset.dropna().groupby("step").mean().reset_index()
            plot = p9.ggplot(df_subset, p9.aes(x="step", y="value")) + p9.geom_line()
            plot.save(os.path.join(log_dir, f"{str(self)}.{col}.png"))
    
    def post_eval(self, step: int):
        pass