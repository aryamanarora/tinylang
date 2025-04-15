from abc import ABC, abstractmethod
import importlib
import torch
from typing import Any


class Model(ABC):
    def __init__(self):
        return
    
    @classmethod
    def from_config(cls, config: dict):
        """Load a model from a config dict.
        
        Args:
            config: Dict containing:
                - class: Name of the model class to instantiate
                - config: Dict of config parameters for the model
        """
        # Get the class from the config
        model_class = getattr(importlib.import_module("tinylang.model"), config["class"])
        # Create an instance with the config
        model = model_class(**config["config"])
        model.config_dict = config
        return model
    
    @abstractmethod
    def step(self, input_ids: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a single step.
        
        Args:
            input_ids: The input tokens
            labels: The labels

        Returns:
            A tuple containing the logits and the loss
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save the model to a file."""
        pass