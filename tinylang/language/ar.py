from .language import Language
import random
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AR(Language):
    def __init__(
        self,
        num_kv: int,
        max_length: int,
        min_length: int=2,
        query_type: str="key",
        mask_nonquery: bool=False,
        prepare_train_set: bool=False,
        train_test_split: float=0.0,
    ):
        super().__init__()
        assert num_kv % 2 == 0
        assert query_type in ["key", "value"]
        self.num_kv = num_kv
        self.max_length = max_length
        self.min_length = min_length
        self.query_type = query_type
        self.mask_nonquery = mask_nonquery

        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.KEY_START = 3
        self.VALUE_START = 3 + num_kv // 2
        self.keys = [f"k{i}" for i in range(num_kv // 2)]
        self.values = [f"v{i}" for i in range(num_kv // 2)]
        self.num_keys = num_kv // 2
        self.num_values = num_kv // 2
        self.id_to_token = {
            self.PAD: "<pad>",
            self.BOS: "<bos>",
            self.EOS: "<eos>",
        }
        for i in range(num_kv // 2):
            self.id_to_token[i + self.KEY_START] = self.keys[i]
            self.id_to_token[i + self.KEY_START + num_kv // 2] = self.values[i]
        self.id_to_token[-100] = "<pad>"
        self.vocab_size = len(self.id_to_token)
        self.TERMINAL_START = self.KEY_START if self.query_type == "key" else self.KEY_START + num_kv // 2
        self.QUERY_START = self.TERMINAL_START + num_kv // 2
        self.prepare_train_set = prepare_train_set

        # are any pairs prohibited in the train set?
        if train_test_split > 0.0:
            self.blacklist_matrix = np.random.rand(self.num_keys, self.num_values)
            # set train_test_split % of matrix to 1
            self.blacklist_matrix = (self.blacklist_matrix < train_test_split).astype(int)
            self.prohibited_pairs = set([(i, j) for i in range(self.num_keys) for j in range(self.num_values) if self.blacklist_matrix[i, j] == 1])
        else:
            self.prohibited_pairs = set()

    
    def sample(self, split: str="test", return_stats: bool=False):
        """Generate a random sentence from the AR."""
        num_sample = random.randint(self.min_length // 2, self.max_length // 2)
        keys = np.random.choice(self.num_keys, size=num_sample, replace=False)
        values = np.random.choice(self.num_values, size=num_sample, replace=False)
        
        # construct KV pairs
        sentence = [self.BOS]
        for i in range(num_sample):
            sentence.append(keys[i] + self.KEY_START)
            sentence.append(values[i] + self.VALUE_START)
        sentence.append(self.EOS)

        # get eligible pairs
        if self.prohibited_pairs is not None:
            eligible_pairs = []
            for i in range(num_sample):
                if (keys[i], values[i]) not in self.prohibited_pairs:
                    eligible_pairs.append(i)
            q_pos = random.choice(eligible_pairs)
        else:
            q_pos = np.random.randint(0, num_sample)

        # set up query
        if self.query_type == "key":
            query = keys[q_pos]
            answer = values[q_pos]
            q, a = q_pos * 2, q_pos * 2 + 1
        else:
            query = values[q_pos]
            answer = keys[q_pos]
            q, a = q_pos * 2 + 1, q_pos * 2

        # add query and answer to sentence
        query += self.KEY_START
        answer += self.VALUE_START
        sentence.extend([query, answer])
        sentence = sentence + [self.EOS]

        # make probing schema
        probing_schema = {
            "type": "KEY" if self.query_type == "key" else "VALUE",
            "keys": {
                "query_item_orig": [q + 1],
                "target_item_orig": [a + 1],
                "query_item": [len(sentence) - 3],
                "target_item": [len(sentence) - 2],
                "divider": [len(sentence) - 4],
                "bos": [0],
            },
            "queries": {
                "query_item_orig": {
                    "pos": q + 1,
                    "target_distribution": None,
                },
                "target_item_orig": {
                    "pos": a + 1,
                    "target_distribution": None,
                },
                "query_item": {
                    "pos": len(sentence) - 3,
                    "target_distribution": True,
                },
                "target_item": {
                    "pos": len(sentence) - 2,
                    "target_distribution": None,
                },
                "divider": {
                    "pos": len(sentence) - 4,
                    "target_distribution": None,
                },
            },
            "target_distributions": {
                "query_item_orig": query - self.KEY_START,
                "target_item_orig": answer - self.KEY_START - len(self.keys),
            }
        }

        if return_stats:
            stats = {
                "doc_length": len(sentence),
                "num_pairs": num_sample,
                "query_orig_target_orig_dist": np.abs(probing_schema["queries"]["query_item_orig"]["pos"] - probing_schema["queries"]["target_item_orig"]["pos"]),
                "query_query_orig_dist": np.abs(probing_schema["queries"]["query_item_orig"]["pos"] - probing_schema["queries"]["query_item"]["pos"]),
                "query_target_orig_dist": np.abs(probing_schema["queries"]["target_item_orig"]["pos"] - probing_schema["queries"]["query_item"]["pos"]),
                "eligible_pairs": len(eligible_pairs),
            }
            return sentence, probing_schema, stats
        return sentence, probing_schema
    
    def prettify(self, toks: list[int]) -> str:
        return " ".join([self.id_to_token[int(tok)] for tok in toks])
    
    def batchify(self, toks: list[list], probing_schemas: list[dict], verbose: bool=False) -> dict:
        tokens = [torch.tensor(tok) for tok in toks]
        strs = [self.prettify(tok) for tok in toks]

        # labels replace PAD with -100
        tokens_padded = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=self.PAD).to(DEVICE)
        labels = tokens_padded.clone().to(DEVICE)
        labels[labels == self.PAD] = -100
        
        if self.mask_nonquery:
            for i in range(len(tokens)):
                target_token = probing_schemas[i]["queries"]["target_item"]["pos"]
                # mask all except query token
                labels[i, :target_token] = -100
                labels[i, target_token + 1:] = -100

        ret = {
            "input_ids": tokens_padded,
            "labels": labels,
            "strs": strs,
            "probing_schemas": probing_schemas,
        }
        return ret