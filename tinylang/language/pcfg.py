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

Node = namedtuple("Node", ["label", "id", "head_id"])

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
            no_start_queries: bool=True,
        ):
        super().__init__()
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.TERMINAL_START = 3
        self.QUERY_START = self.TERMINAL_START + num_terminals
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
        print(self.id_to_token, self.vocab_size)

        self.num_terminals = num_terminals
        self.num_nonterminals = num_nonterminals
        self.max_rhs_len = max_rhs_len
        self.max_rules_per_nt = max_rules_per_nt
        self.max_depth = max_depth
        self.head_position = head_position
        self.mask_nonquery = mask_nonquery
        self.no_sibling_queries = no_sibling_queries
        self.no_child_queries = no_child_queries
        self.no_start_queries = no_start_queries
        
        # make the terminals and nonterminals
        self.terminals = [f"t{i}" for i in range(num_terminals)]
        self.nonterminals = [f"nt{i}" for i in range(num_nonterminals)]

        # each nonterminal has a max depth
        # its immediate children must have a depth less than this
        self.max_depths = {nt: np.random.randint(1, max_depth) for nt in self.nonterminals} if max_depth > 0 else None

        # make the rules
        # each nonterminal has a list of production rules
        self.rules = defaultdict(list)
        for nt in self.nonterminals:
            num_rules = np.random.randint(1, max_rules_per_nt)
            # select acceptable nonterminals
            nonterminals_subset = [x for x in self.nonterminals if self.max_depths[x] > self.max_depths[nt]] if self.max_depths else self.nonterminals
            for _ in range(num_rules):
                lhs = nt
                rhs = np.random.choice(nonterminals_subset + self.terminals, size=np.random.randint(1, max_rhs_len), replace=False)
                self.rules[lhs].append(rhs)
        
        # for each nt, set the probability of its rules
        self.rule_probs = {}
        for nt in self.nonterminals:
            num_rules = len(self.rules[nt])
            self.rule_probs[nt] = np.random.dirichlet(np.ones(num_rules))
        
        # each nt has a distribution over terminals as its "head". this makes the PCFG transparent?
        self.head_probs = {}
        for nt in self.nonterminals:
            self.head_probs[nt] = np.random.dirichlet(np.ones(num_terminals))
        
        # finally, the start symbol has a distribution over nonterminals as its head
        self.start_probs = np.ones(num_nonterminals) / num_nonterminals
    

    def prepare_sets(self, train_batch_size: int, eval_batch_size: int, num_train_steps: int, num_eval_steps: int):
        """Prepare the train and eval sets."""
        # we ignore train steps since we are generating on the fly
        self.eval_toks, self.eval_probing_schemas = [], []
        for _ in range(num_eval_steps * eval_batch_size):
            tok, probing_schema = self.sample()
            self.eval_toks.append(tok)
            self.eval_probing_schemas.append(probing_schema)

    
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
        generated = [Node(str(np.random.choice(self.nonterminals, p=self.start_probs)), 0, None)]
        next_id = 1
        done = False
        depth = 0
        while not done:
            generated_new = []
            done_new = True
            for nt, id, head_id in generated:
                if nt in self.terminals:
                    generated_new.append(Node(nt, id, head_id))
                else:
                    done_new = False
                    this_head_id = next_id
                    next_id += 1
                    # first sample the head (placed on the left)
                    if self.head_position == "left":
                        head = np.random.choice(len(self.terminals), p=self.head_probs[nt])
                        generated_new.append(Node(str(self.terminals[head]), this_head_id, head_id))
                    
                    # then sample the rhs
                    rhs = np.random.choice(len(self.rules[nt]), p=self.rule_probs[nt])
                    for child in self.rules[nt][rhs]:
                        generated_new.append(Node(str(child), next_id, this_head_id))
                        next_id += 1
                        
                    if self.head_position == "right":
                        head = np.random.choice(len(self.terminals), p=self.head_probs[nt])
                        generated_new.append(Node(str(self.terminals[head]), this_head_id, head_id))

            generated = generated_new
            done = done_new
            depth += 1
            if depth > MAX_RECURSION_DEPTH or len(generated) > MAX_LENGTH:
                raise RecursionError("MAX_RECURSION_DEPTH or MAX_LENGTH reached")

        return generated
    
    def sample(self):
        """Generate a document from the PCFG, i.e. a sentence, a query, and a response."""
        sentence = None
        while sentence is None:
            try:
                sentence = self._sample()
                if len(sentence) < 3:
                    sentence = None
                    continue
            except RecursionError:
                sentence = None
                continue

            # check if any eligible positions are left
            eligible_pos = list(range(len(sentence)))
            if self.no_child_queries and self.no_sibling_queries: # must be a parent query
                if self.head_position == "left":
                    if self.no_start_queries:
                        eligible_pos = [i for i in eligible_pos[1:] if sentence[i].head_id != sentence[0].id]
                    else:
                        eligible_pos = eligible_pos[1:]
                elif self.head_position == "right":
                    if self.no_start_queries:
                        eligible_pos = [i for i in eligible_pos[:-1] if sentence[i].head_id != sentence[-1].id]
                    else:
                        eligible_pos = eligible_pos[:-1]
            if len(eligible_pos) == 0:
                sentence = None

        # queries are only parent, child #n, and sibling #n
        # we uniformly sample the item we want to ask the query about, and then sample the query type
        # TODO: wtf should we do about duplicate terminals?
        # TODO: diff query distribution?
        # TODO: order
        # TODO: this code is so sus, how do we handle path length > 1 (we will have to eventually)

        query_item = np.random.choice(eligible_pos)
        query_item_pos = 1 + query_item

        # generate the possible targets for each query type
        possible_queries_and_targets = {
            QueryType.PARENT: [i for i in range(len(sentence)) if sentence[i].id == sentence[query_item].head_id],
            QueryType.CHILD: [i for i in range(len(sentence)) if sentence[i].head_id == sentence[query_item].id] if not self.no_child_queries else [],
            QueryType.SIBLING: [i for i in range(len(sentence)) if (sentence[i].head_id == sentence[query_item].head_id and i != query_item)] if not self.no_sibling_queries else []
        }
        tokens = [self.BOS] + [int(x.label[1:]) + self.TERMINAL_START for x in sentence]
        # print(self.prettify(tokens))
        # print("    ", sentence[query_item])
        # for k, v in possible_queries_and_targets.items():
        #     print("    ", k.name, [sentence[i] for i in v])
        # input()

        query_item = int(sentence[query_item].label[1:]) + self.TERMINAL_START

        # sample a query type
        acceptable_query_types = [q for q in possible_queries_and_targets if len(possible_queries_and_targets[q]) > 0]
        query_type = np.random.choice(acceptable_query_types)

        # get the target distribution
        target_distribution = np.zeros(self.vocab_size)
        all_target_pos = []
        for item in possible_queries_and_targets[query_type]:
            tok = int(sentence[item].label[1:]) + self.TERMINAL_START
            target_distribution[tok] += 1
            all_target_pos.append(1 + item)
        target_distribution = target_distribution / np.sum(target_distribution)

        # sample a target item
        target_item = np.random.choice(possible_queries_and_targets[query_type])
        target_item_pos = 1 + target_item
        target_item = int(sentence[target_item].label[1:]) + self.TERMINAL_START

        # generate the overall sequence to train on
        only_parent_queries = self.no_child_queries and self.no_sibling_queries
        if only_parent_queries:
            query = [query_item, target_item]
        else:
            query = [query_item, query_type + self.QUERY_START, target_item]
        tokens = tokens + [self.EOS] + query + [self.EOS]

        # generate probing schema
        probing_schema = {
            "type": QueryType(query_type).name,
            "keys": {
                "query_item": [len(tokens) - 1 - len(query)],
                "query_type": [len(tokens) - 1 - len(query) + 1] if not only_parent_queries else [0],
                "target_item": [len(tokens) - 1 - len(query) + (2 if not only_parent_queries else 1)],
                "_target_item_orig": [target_item_pos],
                "query_item_orig": [query_item_pos],
                "divider": [len(tokens) - 1 - len(query) - 1],
                "_all_target_items_orig": all_target_pos,
                "children": [1 + i for i in possible_queries_and_targets[QueryType.CHILD]],
                "siblings": [1 + i for i in possible_queries_and_targets[QueryType.SIBLING]],
                "parent": [1 + i for i in possible_queries_and_targets[QueryType.PARENT]],
                "bos": [0],
            },
            "queries": {
                "query_item": {
                    "pos": len(tokens) - 1 - len(query),
                    "target_distribution": None if only_parent_queries else torch.tensor(target_distribution),
                },
                "query_type": {
                    "pos": len(tokens) - 1 - len(query) + 1,
                    "target_distribution": torch.tensor(target_distribution),
                },
                "target_item": {
                    "pos": len(tokens) - 1 - len(query) + (2 if not only_parent_queries else 1),
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

        if only_parent_queries:
            del probing_schema["queries"]["query_type"]
        
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
