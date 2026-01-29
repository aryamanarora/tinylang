from abc import ABC
import importlib
import pickle
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Language(ABC):
    # Default special token IDs (subclasses can override)
    PAD = 0
    BOS = 1
    EOS = 2

    def __init__(self):
        self.config_dict = {}
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
    
    def prepare_sets(self, train_set_size: int, eval_set_size: int):
        """Prepare the train and eval sets."""
        # train set?
        if self.prepare_train_set:
            self.train_set = defaultdict(list)
            for _ in tqdm(range(train_set_size), desc="Preparing train set"):
                tok, probing_schema = self.sample(split="train", return_stats=False)
                self.train_set["toks"].append(tok)
                self.train_set["probing_schemas"].append(probing_schema)

        # we keep separate dev and test sets, unless theres no train/test split
        self.evalsets = {"dev": {}, "test": {}}
        if len(self.prohibited_pairs) == 0:
            del self.evalsets["dev"]
        self.stats = {}

        # generate eval sets
        for split in self.evalsets.keys():
            self.evalsets[split]["toks"], self.evalsets[split]["probing_schemas"] = [], []
            self.stats[split] = defaultdict(list)
            for _ in range(eval_set_size):
                tok, probing_schema, stats = self.sample(split=split, return_stats=True)
                self.evalsets[split]["toks"].append(tok)
                self.evalsets[split]["probing_schemas"].append(probing_schema)
                for key in stats.keys():
                    self.stats[split][key].append(stats[key])
            
            # log means to config dict
            for key, value in self.stats[split].items():
                self.config_dict[f"summary/{split}/{key}"] = np.mean(value).item()


    def get_train_step(self, step: int, batch_size: int, verbose: bool = False) -> dict:
        if self.prepare_train_set:
            batch_start, batch_end = step * batch_size, min(len(self.train_set["toks"]), (step + 1) * batch_size)
            return self.batchify(
                self.train_set["toks"][batch_start:batch_end],
                self.train_set["probing_schemas"][batch_start:batch_end],
                verbose=verbose,
            )
        else:
            toks, probing_schemas = [], []
            for _ in range(batch_size):
                tok, probing_schema = self.sample(split="train")
                toks.append(tok)
                probing_schemas.append(probing_schema)

            return self.batchify(toks, probing_schemas, verbose=verbose)


    def get_eval_step(self, step: int, batch_size: int, split: str="test") -> dict:
        """Get an eval step."""
        batch_start, batch_end = step * batch_size, min(len(self.evalsets[split]["toks"]), (step + 1) * batch_size)
        return self.batchify(
            self.evalsets[split]["toks"][batch_start:batch_end],
            self.evalsets[split]["probing_schemas"][batch_start:batch_end],
            verbose=True,
        )


    def batchify(self, toks: list[list], probing_schemas: list[dict], verbose: bool = False) -> dict:
        """Convert a list of token sequences into a batched dict for training/eval."""
        tokens = [torch.tensor(tok) for tok in toks]
        strs = [self.prettify(tok) for tok in toks]

        # pad sequences and create labels (replace PAD with -100)
        tokens_padded = torch.nn.utils.rnn.pad_sequence(
            tokens, batch_first=True, padding_value=self.PAD
        ).to(DEVICE)
        labels = tokens_padded.clone().to(DEVICE)
        labels[labels == self.PAD] = -100

        if self.mask_nonquery:
            for i in range(len(tokens)):
                target_token = probing_schemas[i]["queries"]["target_item"]["pos"]
                labels[i, :target_token] = -100
                labels[i, target_token + 1:] = -100

        ret = {
            "input_ids": tokens_padded,
            "labels": labels,
            "strs": strs,
            "probing_schemas": probing_schemas,
        }
        return self._batchify_extras(ret, tokens, probing_schemas, verbose)

    def _batchify_extras(self, ret: dict, tokens: list, probing_schemas: list[dict], verbose: bool) -> dict:
        """Hook for subclasses to add extra fields to batchify output."""
        return ret


    def save(self, path: str):
        """Save the language to a file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        """Load the language from a file."""
        with open(path, "rb") as f:
            return pickle.load(f)