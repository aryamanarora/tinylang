from .language import Language
import random
import numpy as np
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AR(Language):
    def __init__(
        self,
        num_kv: int,
        max_length: int,
        min_length: int=2,
        query_type: str="key",
        mask_nonquery: bool=False,
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
        self.keys = [f"k{i}" for i in range(num_kv // 2)]
        self.values = [f"v{i}" for i in range(num_kv // 2)]
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
    

    def prepare_sets(self, train_batch_size: int, eval_batch_size: int, num_train_steps: int, num_eval_steps: int):
        """Prepare the train and eval sets."""
        # we ignore train steps since we are generating on the fly
        self.eval_toks, self.eval_probing_schemas = [], []
        for _ in range(num_eval_steps * eval_batch_size):
            tok, probing_schema = self.sample()
            self.eval_toks.append(tok)
            self.eval_probing_schemas.append(probing_schema)
    
    def sample(self):
        """Generate a random sentence from the AR."""
        num_sample = random.randint(self.min_length // 2, self.max_length // 2)
        keys = list(range(self.num_kv // 2))
        values = list(range(self.num_kv // 2, self.num_kv))
        random.shuffle(keys)
        random.shuffle(values)

        # construct KV pairs
        sentence = [self.BOS]
        for i in range(num_sample):
            sentence.append(keys[i] + self.KEY_START)
            sentence.append(values[i] + self.KEY_START)
        sentence.append(self.EOS)

        # set up query
        q_pos = random.randint(0, num_sample - 1)
        if self.query_type == "key":
            query = keys[q_pos]
            answer = values[q_pos]
            q, a = q_pos * 2, q_pos * 2 + 1
        else:
            query = values[q_pos]
            answer = keys[q_pos]
            q, a = q_pos * 2 + 1, q_pos * 2
        query += self.KEY_START
        answer += self.KEY_START
        sentence.extend([query, answer])
        sentence = sentence + [self.EOS]
        # print(self.prettify(sentence))
        # input()

        # get the target distribution
        target_distribution = np.zeros(self.vocab_size)
        target_distribution[answer] += 1

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
                    "target_distribution": torch.tensor(target_distribution),
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

    
    def get_train_step(self, step: int, batch_size: int, verbose: bool = False) -> dict:
        toks, probing_schemas = [], []
        for _ in range(batch_size):
            tok, probing_schema = self.sample()
            toks.append(tok)
            probing_schemas.append(probing_schema)

        return self.batchify(toks, probing_schemas, verbose=verbose)


    def get_eval_step(self, step: int, batch_size: int) -> dict:
        """Get an eval step."""
        batch_start, batch_end = step * batch_size, min(len(self.eval_toks), (step + 1) * batch_size)
        return self.batchify(
            self.eval_toks[batch_start:batch_end],
            self.eval_probing_schemas[batch_start:batch_end],
            verbose=True,
        )