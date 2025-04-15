from abc import ABC, abstractmethod
import importlib
import torch
import pickle

class Language(ABC):
    def __init__(self):
        return
    
    @classmethod
    def from_config(cls, config: dict):
        """Load a language from a config dict.
        
        Args:
            config: Dict containing:
                - class: Name of the language class to instantiate
                - config: Dict of config parameters for the language
        """
        # Get the class from the config
        language_class = getattr(importlib.import_module("tinylang.language"), config["class"])
        # Create an instance with the config
        language = language_class(**config["config"])
        language.config_dict = config
        return language
    
    @abstractmethod
    def prepare_sets(self, train_batch_size: int, eval_batch_size: int, num_train_steps: int, num_eval_steps: int):
        """Prepare the train and eval sets."""
        pass
    
    @abstractmethod
    def get_train_step(self, step: int, batch_size: int) -> dict:
        """Get a train step.
        
        Args:
            step: The current step in the training loop
            batch_size: The number of samples to generate
        Returns:
            A dict containing any data to be used for training
        """
        pass
    
    @abstractmethod
    def get_eval_step(self, step: int, batch_size: int) -> dict:
        """Get an eval step.
        
        Args:
            step: The current step in the training loop
            batch_size: The number of samples to generate
        Returns:
            A dict containing any data to be used for eval
        """
        pass


    def save(self, path: str):
        """Save the language to a file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        """Load the language from a file."""
        with open(path, "rb") as f:
            return pickle.load(f)