from abc import ABC, abstractmethod
from tinylang.model import Model
import torch
import importlib
from typing import Any

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
    def eval(self, model: Model, inputs: dict):
        pass

    def aggregate(self, stats: list[dict]):
        result = {
            k: self.agg_funcs[k.split(".")[-1]](flatten_list([stat.get(k, []) for stat in stats]))
            for k in stats[0]
        }
        return result
