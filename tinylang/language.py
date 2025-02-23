"""
A simple language generator.
"""

import numpy as np
from enum import IntEnum
from collections import namedtuple, defaultdict


# just a wrapper for the PCFG and VerbArguments classes
class Language:
    def __init__(self):
        return

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
        self.TERMINAL_START = 2
        self.QUERY_START = self.TERMINAL_START + num_terminals
        self.vocab_size = self.QUERY_START + max(QueryType).value + 1

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

        # queries are only parent, child #n, and sibling #n
        # we uniformly sample the item we want to ask the query about, and then sample the query type
        # TODO: wtf should we do about duplicate terminals?
        # TODO: diff query distribution?
        # TODO: order
        # TODO: this code is so sus, how do we handle path length > 1 (we will have to eventually)

        query_item = np.random.randint(0, len(sentence))
        parents = [i for i in range(len(sentence)) if sentence[i].id == sentence[query_item].head_id]
        children = [i for i in range(len(sentence)) if sentence[i].head_id == sentence[query_item].id]
        siblings = [i for i in range(len(sentence)) if sentence[i].head_id == sentence[query_item].head_id and sentence[i].id != sentence[query_item].id]
        query_item = int(sentence[query_item].label[1:]) + self.TERMINAL_START

        acceptable_query_types = []
        if len(parents) > 0:
            acceptable_query_types.append(QueryType.PARENT)
        if len(children) > 0:
            acceptable_query_types.append(QueryType.CHILD)
        if len(siblings) > 0:
            acceptable_query_types.append(QueryType.SIBLING)
        
        query_type = np.random.choice(acceptable_query_types)
        if query_type == QueryType.PARENT:
            target_item = np.random.choice(parents)
        elif query_type == QueryType.CHILD:
            target_item = np.random.choice(children)
        elif query_type == QueryType.SIBLING:
            target_item = np.random.choice(siblings)
        target_item = int(sentence[target_item].label[1:]) + self.TERMINAL_START

        query = [query_item, query_type + self.QUERY_START, target_item]
        tokens = [self.BOS] + [int(x.label[1:]) + self.TERMINAL_START for x in sentence] + [self.EOS] + query + [self.EOS]
        return tokens
    
    def sample_n(self, n: int):
        tokens = []
        for _ in range(n):
            tokens.append(self.sample())

        # pad with self.PAD to max sequence length and stack with numpy
        # this is probably inefficient af
        # labels replace PAD with -100
        max_len = max(len(t) for t in tokens)
        tokens_padded = np.stack([t + [self.PAD] * (max_len - len(t)) for t in tokens], axis=0)
        labels = tokens_padded.copy()
        labels[labels == self.PAD] = -100

        if self.mask_nonquery:
            for i in range(len(tokens)):
                length = len(tokens[i])
                query_token = length - 3
                # mask all except query token
                labels[i, :query_token] = -100
                labels[i, query_token + 1:] = -100

        return tokens_padded, labels


class Order(IntEnum):
    OBJ_RECIP = 0
    RECIP_OBJ = 1

class ThetaRole(IntEnum):
    OBJ = 0
    RECIP = 1


# assume all verbs are ditransitive
class VerbArguments(Language):
    def __init__(
        self,
        num_verbs: int,
        num_nouns: int,
    ):
        super().__init__()
        self.num_verbs = num_verbs
        self.num_nouns = num_nouns
        self.vocab_size = num_verbs + num_nouns + 2
        self.BOS = 0
        self.EOS = 1
        self.VERB_START = 2
        self.NOUN_START = self.VERB_START + num_verbs

        # each noun has a preference over theta roles it takes (object or recipient)
        # maybe this should be sparser?
        self.p_noun_given_class = np.random.dirichlet(np.ones(num_nouns), size=2) # (2, num_nouns)
        # self.p_noun_given_class = np.zeros((2, num_nouns))
        # self.p_noun_given_class[0, :num_nouns // 2] = 1.0 / (num_nouns // 2)
        # self.p_noun_given_class[1, num_nouns // 2:] = 1.0 / (num_nouns // 2 + num_nouns % 2)

        self.p_class_given_noun = self.p_noun_given_class.T.copy() # (num_nouns, 2)
        self.p_class_given_noun /= self.p_class_given_noun.sum(axis=1, keepdims=True)

        # each verb has an order preference over theta roles (obj recip or recip obj)
        self.p_order_given_verb = np.random.dirichlet(np.ones(2), size=num_verbs)

        # each verb has a chance of appearing
        self.p_verb = np.random.dirichlet(np.ones(num_verbs), size=1)[0]
    
    def sample(self):
        # sample a verb
        verb = np.random.choice(self.num_verbs, p=self.p_verb)

        # sample the order preference for the verb
        order = np.random.choice(2, p=self.p_order_given_verb[verb])

        # sample the object and recipient for the verb
        obj = np.random.choice(self.num_nouns, p=self.p_noun_given_class[ThetaRole.OBJ])
        recip = np.random.choice(self.num_nouns, p=self.p_noun_given_class[ThetaRole.RECIP])

        # tokens
        verb_tok = verb + self.VERB_START
        obj_tok = obj + self.NOUN_START
        recip_tok = recip + self.NOUN_START
        tokens = [self.BOS, verb_tok, obj_tok, recip_tok, self.EOS] if order == Order.OBJ_RECIP else [self.BOS, verb_tok, recip_tok, obj_tok, self.EOS]
        noun_1 = obj if order == Order.OBJ_RECIP else recip
        noun_2 = recip if order == Order.OBJ_RECIP else obj

        # now we want to calculate the actual left-to-right probabilities of the tokens
        # this basically simplifies to bayes updates over the two possible orderings
        next_token_probs = np.zeros((len(tokens), self.vocab_size))
        next_token_probs[0, self.VERB_START:self.NOUN_START] = self.p_verb
        p_order_given_ctx = np.zeros((len(tokens), 2))
        p_order_given_ctx[0, :] = 0.5
        p_order_given_ctx[1, :] = self.p_order_given_verb[verb]
        for i in range(2, 4):
            token = noun_1 if i == 2 else noun_2
            for order in range(2):
                if i == 2:
                    role = ThetaRole.OBJ if order == Order.OBJ_RECIP else ThetaRole.RECIP
                elif i == 3:
                    role = ThetaRole.RECIP if order == Order.OBJ_RECIP else ThetaRole.OBJ
                p_order_given_ctx[i, order] = self.p_noun_given_class[role, token] * p_order_given_ctx[i - 1, order]
            p_order_given_ctx[i, :] /= p_order_given_ctx[i, :].sum()
        
        for i in range(1, 3):
            for order in range(2):
                next_token_probs[i, self.NOUN_START:] += p_order_given_ctx[i, order] * self.p_noun_given_class[role, :]
        next_token_probs[3, self.EOS] = 1.0
        
        return tokens, next_token_probs, p_order_given_ctx
    
    def sample_n(self, n: int):
        tokens = []
        next_token_probs = []
        p_order_given_ctxs = []
        for _ in range(n):
            token, next_token_prob, p_order_given_ctx = self.sample()
            tokens.append(token)
            next_token_probs.append(next_token_prob)
            p_order_given_ctxs.append(p_order_given_ctx)
        return np.stack(tokens, axis=0), np.stack(next_token_probs, axis=0), np.stack(p_order_given_ctxs, axis=0)