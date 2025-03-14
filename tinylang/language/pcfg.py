from .language import Language
from collections import namedtuple, defaultdict
import numpy as np
from enum import IntEnum
import torch

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

        # make the terminals and nonterminals
        self.terminals = [f"t{i}" for i in range(num_terminals)]
        self.nonterminals = [f"nt{i}" for i in range(num_nonterminals)]

        # each nonterminal has a max depth
        # its immediate children must have a depth less than this
        self.max_depths = {nt: np.random.randint(1, max_depth) for nt in self.nonterminals}

        # make the rules
        # each nonterminal has a list of production rules
        self.rules = defaultdict(list)
        for nt in self.nonterminals:
            num_rules = np.random.randint(1, max_rules_per_nt)
            # select acceptable nonterminals
            nonterminals_subset = [x for x in self.nonterminals if self.max_depths[x] > self.max_depths[nt]]
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
        self.start_probs = np.random.dirichlet(np.ones(num_nonterminals))
    

    def prepare_sets(self, train_batch_size: int, eval_batch_size: int, num_train_steps: int, num_eval_steps: int):
        """Prepare the train and eval sets."""
        # we ignore train steps since we are generating on the fly
        self.eval_set = []
        for _ in range(num_eval_steps):
            self.eval_set.append(self.get_train_step(step=0, batch_size=eval_batch_size))
    

    def _sample(self):
        """Generate a random sentence from the PCFG."""
        # the start symbol is a random nonterminal
        # (nt, id, head_id)
        generated = [Node(str(np.random.choice(self.nonterminals, p=self.start_probs)), 0, None)]
        next_id = 1
        done = False
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
                    else:
                        raise NotImplementedError("Only right head position is implemented")
                    
                    # then sample the rhs
                    rhs = np.random.choice(len(self.rules[nt]), p=self.rule_probs[nt])
                    for child in self.rules[nt][rhs]:
                        generated_new.append(Node(str(child), next_id, this_head_id))
                        next_id += 1

            generated = generated_new
            done = done_new
            
        return generated
    
    def sample(self):
        """Generate a document from the PCFG, i.e. a sentence, a query, and a response."""
        sentence = self._sample()
        while len(sentence) <= 1:
            sentence = self._sample()

        # queries are only parent, child #n, and sibling #n
        # we uniformly sample the item we want to ask the query about, and then sample the query type
        # TODO: wtf should we do about duplicate terminals?
        # TODO: diff query distribution?
        # TODO: order
        # TODO: this code is so sus, how do we handle path length > 1 (we will have to eventually)

        # pick a random item to query
        query_item = np.random.randint(0, len(sentence))
        query_item_pos = 1 + query_item

        # generate the possible targets for each query type
        possible_queries_and_targets = {
            QueryType.PARENT: [i for i in range(len(sentence)) if sentence[i].id == sentence[query_item].head_id],
            QueryType.CHILD: [i for i in range(len(sentence)) if sentence[i].head_id == sentence[query_item].id],
            QueryType.SIBLING: [i for i in range(len(sentence)) if sentence[i].head_id == sentence[query_item].head_id and sentence[i].id != sentence[query_item].id]
        }
        query_item = int(sentence[query_item].label[1:]) + self.TERMINAL_START

        # sample a query type
        acceptable_query_types = [q for q in possible_queries_and_targets if len(possible_queries_and_targets[q]) > 0]
        query_type = np.random.choice(acceptable_query_types)

        # get the target distribution
        target_distribution = np.zeros(self.vocab_size)
        for item in possible_queries_and_targets[query_type]:
            tok = int(sentence[item].label[1:]) + self.TERMINAL_START
            target_distribution[tok] += 1
        target_distribution = target_distribution / np.sum(target_distribution)

        # sample a target item
        target_item = np.random.choice(possible_queries_and_targets[query_type])
        target_item_pos = 1 + target_item
        target_item = int(sentence[target_item].label[1:]) + self.TERMINAL_START

        # generate the overall sequence to train on
        query = [query_item, query_type + self.QUERY_START, target_item]
        tokens = [self.BOS] + [int(x.label[1:]) + self.TERMINAL_START for x in sentence] + [self.EOS] + query + [self.EOS]

        # generate probing schema
        probing_schema = {
            "type": QueryType(query_type).name,
            "tokens": {
                "query_item": len(tokens) - 1 - len(query),
                "query_type": len(tokens) - 1 - len(query) + 1,
                "target_item": len(tokens) - 1 - len(query) + 2,
                "query_item_orig": query_item_pos,
                "target_item_orig": target_item_pos,
            },
            "target_distribution": torch.tensor(target_distribution),
            "target_pos": len(tokens) - 1 - len(query) + 1,
        }
        
        return tokens, probing_schema
    
    def get_train_step(self, step: int, batch_size: int) -> dict:
        tokens, strs, probing_schemas = [], [], []
        for _ in range(batch_size):
            toks, probing_schema = self.sample()
            tokens.append(torch.tensor(toks))
            strs.append(" ".join([self.id_to_token[int(tok)] for tok in toks]))
            probing_schemas.append(probing_schema)

        # pad with self.PAD to max sequence length and stack with numpy
        # this is probably inefficient af
        # labels replace PAD with -100
        max_len = max(len(t) for t in tokens)
        tokens_padded = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=self.PAD)
        labels = tokens_padded.clone()
        labels[labels == self.PAD] = -100

        if self.mask_nonquery:
            for i in range(len(tokens)):
                length = len(tokens[i])
                query_token = length - 3
                # mask all except query token
                labels[i, :query_token] = -100
                labels[i, query_token + 1:] = -100

        return {
            "input_ids": tokens_padded,
            "labels": labels,
            "strs": strs,
            "probing_schemas": probing_schemas,
        }


    def get_eval_step(self, step: int, batch_size: int) -> dict:
        """Get an eval step."""
        return self.eval_set[step]