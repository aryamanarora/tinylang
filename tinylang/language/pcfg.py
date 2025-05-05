from .language import Language
from collections import namedtuple, defaultdict
import numpy as np
from enum import IntEnum
import torch
import termcolor


COLORS = ["red", "green", "blue", "yellow", "magenta", "cyan", "light_red", "light_green", "light_blue", "light_yellow", "light_magenta", "light_cyan"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_RECURSION_DEPTH = 20
MAX_LENGTH = 1024


class QueryType(IntEnum):
    PARENT = 0
    CHILD = 1
    SIBLING = 2

Node = namedtuple("Node", ["label", "id", "head_id", "depth"])

class PCFG(Language):
    def __init__(
        self,
        num_terminals: int,
        num_nonterminals: int,
        max_rhs_len: int,
        max_rules_per_nt: int,
        max_depth: int,
        head_position: str="left",
        mask_nonquery: bool=False,
        no_sibling_queries: bool=False,
        no_child_queries: bool=False,
        max_length: int=1024,
        train_test_split: float=0.0,
        tts_temp: float=0.0,
        transparent_nonterminals: bool=False,
        unambiguous_queries: bool=False,
        sample_first: str="target",
    ):
        super().__init__()
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.TERMINAL_START = 3
        self.QUERY_START = self.TERMINAL_START + num_terminals
        if transparent_nonterminals:
            self.QUERY_START = self.QUERY_START + num_nonterminals
        self.vocab_size = self.QUERY_START + max(QueryType).value + 1
        self.id_to_token = {
            self.PAD: "<pad>",
            self.BOS: "<bos>",
            self.EOS: "<eos>",
            self.TERMINAL_START: "<t>",
            self.QUERY_START: "<q>",
        }
        for i in range(num_terminals):
            self.id_to_token[self.TERMINAL_START + i] = f"t{i}"
        for i in range(len(QueryType)):
            self.id_to_token[self.QUERY_START + i] = f"q{QueryType(i).name}"

        self.num_terminals = num_terminals
        self.num_nonterminals = num_nonterminals
        self.max_rhs_len = max_rhs_len
        self.max_rules_per_nt = max_rules_per_nt
        self.max_depth = max_depth
        self.head_position = head_position
        self.mask_nonquery = mask_nonquery
        self.max_length = max_length

        # which queries are disabled?
        self.no_sibling_queries = no_sibling_queries
        self.no_child_queries = no_child_queries
        self.acceptable_query_types = [QueryType.PARENT]
        if not self.no_child_queries:
            self.acceptable_query_types.append(QueryType.CHILD)
        if not self.no_sibling_queries:
            self.acceptable_query_types.append(QueryType.SIBLING)
        self.unambiguous_queries = unambiguous_queries
        self.sample_first = sample_first

        # are any pairs prohibited in the train set?
        if train_test_split > 0.0:
            self.blacklist_matrix = np.random.rand(self.num_terminals, self.num_terminals)
            # set train_test_split % of matrix to 1
            self.blacklist_matrix = (self.blacklist_matrix < train_test_split).astype(int)
            self.prohibited_pairs = set([(i, j) for i in range(self.num_terminals) for j in range(self.num_terminals) if self.blacklist_matrix[i, j] == 1])
        else:
            self.prohibited_pairs = set()
        
        self.tts_temp = tts_temp
        
        # make the terminals and nonterminals
        self.terminals = [f"t{i}" for i in range(num_terminals)]
        self.nonterminals = [f"nt{i}" for i in range(num_nonterminals)]

        # each nonterminal has a max depth
        # its immediate children must have a depth less than this
        self.max_depths = {nt: np.random.randint(1, max_depth + 1) for nt in self.nonterminals} if max_depth > 0 else None

        # make the rules
        # each nonterminal has a list of production rules
        self.rules = defaultdict(list)
        for nt in self.nonterminals:
            num_rules = np.random.randint(1, max_rules_per_nt + 1)
            # select acceptable nonterminals
            nonterminals_subset = [x for x in self.nonterminals if self.max_depths[x] > self.max_depths[nt]] if max_depth > 0 else self.nonterminals
            for _ in range(num_rules):
                lhs = nt
                rhs = np.random.choice(nonterminals_subset + self.terminals, size=np.random.randint(1, max_rhs_len), replace=True)
                self.rules[lhs].append(rhs)
        
        # for each nt, set the probability of its rules
        self.rule_probs = {}
        for nt in self.nonterminals:
            num_rules = len(self.rules[nt])
            self.rule_probs[nt] = np.random.dirichlet(np.ones(num_rules))
        
        # each nt has a distribution over terminals as its "head". this makes the PCFG transparent?
        self.head_probs = {}
        for nt_idx, nt in enumerate(self.nonterminals):
            if not transparent_nonterminals:
                self.head_probs[nt] = np.random.dirichlet(np.ones(num_terminals))
            else:
                self.head_probs[nt] = np.zeros(num_terminals + num_nonterminals)
                self.head_probs[nt][num_terminals + nt_idx] = 1

        # modifications to make transparent
        if transparent_nonterminals:
            for nt_idx in range(num_nonterminals):
                self.terminals.append(f"t{nt_idx + num_terminals}")
                self.id_to_token[self.TERMINAL_START + num_terminals + nt_idx] = f"t{nt_idx + num_terminals}"
            self.num_terminals = len(self.terminals)
        
        # finally, the start symbol has a distribution over nonterminals as its head
        self.start_probs = np.ones(num_nonterminals) / num_nonterminals
    

    def prepare_sets(self, train_batch_size: int, eval_batch_size: int, num_train_steps: int, num_eval_steps: int):
        """Prepare the train and eval sets."""
        # we ignore train steps since we are generating on the fly
        self.evalsets = {"dev": {}, "test": {}}
        if len(self.prohibited_pairs) == 0:
            del self.evalsets["dev"]
        self.stats = {}
        for split in self.evalsets.keys():
            self.evalsets[split]["toks"], self.evalsets[split]["probing_schemas"] = [], []
            self.stats[split] = defaultdict(list)
            for _ in range(num_eval_steps * eval_batch_size):
                tok, probing_schema, stats = self.sample(split=split, return_stats=True)
                self.evalsets[split]["toks"].append(tok)
                self.evalsets[split]["probing_schemas"].append(probing_schema)
                for key in stats.keys():
                    self.stats[split][key].append(stats[key])
            
            # log means to config dict
            for key, value in self.stats[split].items():
                self.config_dict[f"summary/{split}/{key}"] = np.mean(value).item()

    
    def prettify(self, toks: list[int], probing_schema: dict | None = None) -> str:
        """Prettify a tokenized sentence."""
        if probing_schema is None:
            return " ".join([self.id_to_token[int(tok)] for tok in toks])
        else:
            query_toks = [self.id_to_token[int(tok)] for tok in toks]
            key_toks = [self.id_to_token[int(tok)] for tok in toks]
            query_result, key_result = "", ""
            for i, query in enumerate(probing_schema["queries"]):
                # choose color based on query, suitable for terminal colors
                color = COLORS[i % len(COLORS)]
                pos = probing_schema["queries"][query]["pos"]
                query_toks[pos] = termcolor.colored(query_toks[pos], color)
                query_result += termcolor.colored(query, color) + "\n"
            query_result = " ".join(query_toks) + "\n" + query_result
            for i, key in enumerate([k for k in probing_schema["keys"] if k[0] != "_"]):
                color = COLORS[i % len(COLORS)]
                for pos in probing_schema["keys"][key]:
                    key_toks[pos] = termcolor.colored(key_toks[pos], color)
                key_result += termcolor.colored(key, color) + "\n"
            key_result = " ".join(key_toks) + "\n" + key_result
            return query_result, key_result

    def _sample(self):
        """Generate a random sentence from the PCFG."""
        # the start symbol is a random nonterminal
        # (nt, id, head_id)
        stack = [Node(str(np.random.choice(self.nonterminals, p=self.start_probs)), 0, None, 0)]
        next_id = 1
        generated = []
        while len(stack) != 0 and len(stack) < self.max_length:
            # check if we've reached the max length
            if len(stack) >= self.max_length:
                raise RecursionError("MAX_RECURSION_DEPTH or MAX_LENGTH reached")
            
            # pop a node from the stack, we will expand it
            cur_node = stack.pop()
            nt, id, head_id, depth = cur_node

            # if it's a terminal, add it to the generated list and continue
            if nt in self.terminals:
                generated.append(Node(nt, id, head_id, depth))
            # otherwise, it's a nonterminal and we expand it
            else:
                # head ids
                this_head_id = next_id
                next_id += 1

                # sample a random rule
                rhs = np.random.choice(len(self.rules[nt]), p=self.rule_probs[nt])
                rhs = self.rules[nt][rhs]

                # first right head
                if self.head_position == "right":
                    head = np.random.choice(len(self.terminals), p=self.head_probs[nt])
                    stack.append(Node(str(self.terminals[head]), this_head_id, head_id, depth + 1))
                
                # children from rhs
                for i in range(len(rhs) - 1, -1, -1):
                    stack.append(Node(str(rhs[i]), next_id, this_head_id, depth + 1))
                    next_id += 1

                # the left head
                if self.head_position == "left":
                    head = np.random.choice(len(self.terminals), p=self.head_probs[nt])
                    stack.append(Node(str(self.terminals[head]), this_head_id, head_id, depth + 1))

        # convert ids to ints and return
        for i in range(len(generated)):
            generated[i] = Node(int(generated[i].label[1:]), generated[i].id, generated[i].head_id, generated[i].depth)
        return generated
                
    def _sample2(self):
        """Generate a random sentence from the PCFG."""
        # the start symbol is a random nonterminal
        # (nt, id, head_id)
        generated = [Node(str(np.random.choice(self.nonterminals, p=self.start_probs)), 0, None, 0)]
        next_id = 1
        done = False
        depth = 0
        while not done:
            generated_new = []
            done_new = True
            for nt, id, head_id, depth in generated:
                if nt in self.terminals:
                    generated_new.append(Node(nt, id, head_id, depth))
                else:
                    done_new = False
                    this_head_id = next_id
                    next_id += 1
                    # first sample the head (placed on the left)
                    if self.head_position == "left":
                        head = np.random.choice(len(self.terminals), p=self.head_probs[nt])
                        generated_new.append(Node(str(self.terminals[head]), this_head_id, head_id, depth + 1))
                    
                    # then sample the rhs
                    rhs = np.random.choice(len(self.rules[nt]), p=self.rule_probs[nt])
                    for child in self.rules[nt][rhs]:
                        generated_new.append(Node(str(child), next_id, this_head_id, depth + 1))
                        next_id += 1
                        
                    if self.head_position == "right":
                        head = np.random.choice(len(self.terminals), p=self.head_probs[nt])
                        generated_new.append(Node(str(self.terminals[head]), this_head_id, head_id, depth + 1))

            generated = generated_new
            done = done_new
            depth += 1
            if depth > MAX_RECURSION_DEPTH or len(generated) > MAX_LENGTH:
                raise RecursionError("MAX_RECURSION_DEPTH or MAX_LENGTH reached")

        for i in range(len(generated)):
            generated[i] = Node(int(generated[i].label[1:]), generated[i].id, generated[i].head_id, generated[i].depth)
        return generated
    
    def is_relation(self, sentence: list[Node], relation: QueryType, i: int, j: int) -> bool:
        """Check if the relation holds between the nodes at positions i and j."""
        if relation == QueryType.PARENT:
            return sentence[i].head_id == sentence[j].id
        elif relation == QueryType.CHILD:
            return sentence[i].id == sentence[j].head_id
        elif relation == QueryType.SIBLING:
            return sentence[i].head_id == sentence[j].head_id and i != j
        else:
            raise ValueError(f"Invalid query type: {relation}")

    def get_eligible_pairs(self, sentence: list[Node], split: str="train") -> tuple[dict, dict, dict]:
        """Get the eligible (query, target) pairs for the sentence based on our constraints."""
        eligible_pairs = {}

        # we will construct a list of eligible (query, target) pairs for each query type
        for query_type in self.acceptable_query_types:
            eligible_pairs[query_type] = []

            # traverse all pairs
            for target in range(len(sentence)):
                for query in range(len(sentence)):
                    # first check train/test split
                    if len(self.prohibited_pairs) > 0:
                        if split in ["train", "dev"]:
                            if (sentence[target].label, sentence[query].label) in self.prohibited_pairs:
                                # some pairs are 'rare'
                                if np.random.uniform() > self.tts_temp:
                                    continue
                        elif split == "test":
                            if (sentence[target].label, sentence[query].label) not in self.prohibited_pairs:
                                continue
                    
                    # second check if it satisfies the relation
                    if self.is_relation(sentence, query_type, query, target):
                        eligible_pairs[query_type].append((query, target))
            
        # now filter for rightmost instance of each child, for the queries
        if self.unambiguous_queries:
            # get rightmost instance of each terminal
            rightmost_types = dict()
            for i, node in enumerate(sentence):
                rightmost_types[node.label] = max(rightmost_types.get(node.label, 0), i)
            
            # query must be the rightmost instance of its type
            # TODO: still ambiguous for sibling queries!
            for query_type in self.acceptable_query_types:
                eligible_pairs[query_type] = [
                    (query, target) for query, target in eligible_pairs[query_type] if rightmost_types[sentence[query].label] == query
                ]
        
        # now we have a list of eligible (query, target) pairs for each query type
        # let's also pass the eligible queries/targets for sampling
        eligible_queries = {}
        eligible_targets = {}
        for query_type in self.acceptable_query_types:
            if len(eligible_pairs[query_type]) == 0:
                del eligible_pairs[query_type]
                continue
            eligible_queries[query_type] = list(set([query for query, _ in eligible_pairs[query_type]]))
            eligible_targets[query_type] = list(set([target for _, target in eligible_pairs[query_type]]))
        
        return eligible_pairs, eligible_queries, eligible_targets

    def sample(self, split: str="train", return_stats: bool=False, return_sentence: bool=False):
        """Generate a document from the PCFG, i.e. a sentence, a query, and a response."""
        sentence = None
        while sentence is None:
            # generate a sentence
            try:
                sentence = self._sample()
                if len(sentence) < 3 or len(sentence) > self.max_length:
                    sentence = None
                    continue
            except RecursionError:
                sentence = None
                continue
            
            # get eligible (query, target) pairs; if no eligible pairs, try again
            eligible_pairs, eligible_queries, eligible_targets = self.get_eligible_pairs(sentence, split)
            if len(eligible_targets) == 0:
                sentence = None

        # first sample query or target depending on self.sample_first
        query_type = np.random.choice(list(eligible_pairs.keys()))
        if self.sample_first == "query":
            query_item = np.random.choice(eligible_queries[query_type])
            target_item = np.random.choice([t for q, t in eligible_pairs[query_type] if q == query_item])
        elif self.sample_first == "target":
            target_item = np.random.choice(eligible_targets[query_type])
            query_item = np.random.choice([q for q, t in eligible_pairs[query_type] if t == target_item])
        else:
            raise NotImplementedError(f"sample_first={self.sample_first} not implemented")

        # construct the sentence
        tokens = [self.BOS] + [x.label + self.TERMINAL_START for x in sentence]

        # get positions of target and query items
        target_item_pos = 1 + target_item
        query_item_pos = 1 + query_item

        # get target item token
        target_item = sentence[target_item].label + self.TERMINAL_START

        # get the target distribution
        target_distribution = np.zeros(self.vocab_size)
        all_target_pos = []
        for item in range(len(sentence)):
            if (query_item, item) not in eligible_pairs[query_type]:
                continue
            tok = sentence[item].label + self.TERMINAL_START
            target_distribution[tok] += 1
            all_target_pos.append(1 + item)
        target_distribution = target_distribution / np.sum(target_distribution)

        # get query item token
        query_item = sentence[query_item].label + self.TERMINAL_START

        # generate the overall sequence to train on
        if len(self.acceptable_query_types) == 1:
            query = [query_item, target_item]
        else:
            query = [query_item, query_type + self.QUERY_START, target_item]
        tokens = tokens + [self.EOS] + query + [self.EOS]

        # generate probing schema
        probing_schema = {
            "type": QueryType(query_type).name,
            "keys": {
                "query_item": [len(tokens) - 1 - len(query)],
                "query_type": [len(tokens) - 1 - len(query) + 1] if len(self.acceptable_query_types) != 1 else [0],
                "target_item": [len(tokens) - 1 - len(query) + (2 if len(self.acceptable_query_types) != 1 else 1)],
                "target_item_orig": [target_item_pos],
                "query_item_orig": [query_item_pos],
                "divider": [len(tokens) - 1 - len(query) - 1],
                "all_target_items_orig": all_target_pos,
                "bos": [0],
            },
            "queries": {
                "query_item": {
                    "pos": len(tokens) - 1 - len(query),
                    "target_distribution": None if len(self.acceptable_query_types) != 1 else torch.tensor(target_distribution),
                },
                "query_type": {
                    "pos": len(tokens) - 1 - len(query) + 1,
                    "target_distribution": torch.tensor(target_distribution),
                },
                "target_item": {
                    "pos": len(tokens) - 1 - len(query) + (2 if len(self.acceptable_query_types) != 1 else 1),
                    "target_distribution": None,
                },
                "query_item_orig": {
                    "pos": query_item_pos,
                    "target_distribution": None,
                },
                "divider": {
                    "pos": len(tokens) - 1 - len(query) - 1,
                    "target_distribution": None,
                },
                "target_item_orig": {
                    "pos": target_item_pos,
                    "target_distribution": None,
                },
            },
            "target_distributions": {
                "target_item_orig": int(self.id_to_token[target_item][1:]),
                "query_item_orig": int(self.id_to_token[query_item][1:]),
            }
        }

        if len(self.acceptable_query_types) == 1:
            del probing_schema["queries"]["query_type"]
        
        # return stats
        if return_stats:
            all_heads = set([x.head_id for x in sentence if x.head_id is not None])
            stats = {
                "doc_length": len(sentence),
                "query_orig_target_orig_dist": np.abs(probing_schema["queries"]["query_item_orig"]["pos"] - probing_schema["queries"]["target_item_orig"]["pos"]),
                "query_query_orig_dist": np.abs(probing_schema["queries"]["query_item_orig"]["pos"] - probing_schema["queries"]["query_item"]["pos"]),
                "query_target_orig_dist": np.abs(probing_schema["queries"]["target_item_orig"]["pos"] - probing_schema["queries"]["query_item"]["pos"]),
                "branching_factor": (len(sentence) - 1) / len(all_heads),
                "percent_non_unique": (len(sentence) - len(set([x.label for x in sentence]))) / len(sentence),
                "depth": max([x.depth for x in sentence]),
                "eligible_pairs": len(eligible_pairs[query_type]),
                "eligible_queries": len(eligible_queries[query_type]),
                "eligible_targets": len(eligible_targets[query_type]),
            }
            if return_sentence:
                stats["sentence"] = sentence
            return tokens, probing_schema, stats
        return tokens, probing_schema
    
    def batchify(self, toks: list[list], probing_schemas: list[dict], verbose: bool=False) -> dict:
        tokens = [torch.tensor(tok) for tok in toks]
        strs = [self.prettify(tok) for tok in toks]

        # pad with self.PAD to max sequence length and stack with numpy
        # this is probably inefficient af
        # labels replace PAD with -100
        tokens_padded = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=self.PAD).to(DEVICE)
        labels = tokens_padded.clone().to(DEVICE)
        labels[labels == self.PAD] = -100

        if self.mask_nonquery:
            for i in range(len(tokens)):
                length = len(tokens[i])
                target_token = length - 2
                # mask all except query token
                labels[i, :target_token] = -100
                labels[i, target_token + 1:] = -100

        ret = {
            "input_ids": tokens_padded,
            "labels": labels,
            "strs": strs,
            "probing_schemas": probing_schemas,
        }
        if verbose:
            ret["strs_pretty"] = [self.prettify(toks, probing_schema) for toks, probing_schema in zip(tokens, probing_schemas)]
        return ret

    
    def get_train_step(self, step: int, batch_size: int, verbose: bool = False) -> dict:
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
