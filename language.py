"""
A simple language generator.
"""

import numpy as np
from enum import IntEnum


class Order(IntEnum):
    OBJ_RECIP = 0
    RECIP_OBJ = 1

class ThetaRole(IntEnum):
    OBJ = 0
    RECIP = 1


# assume all verbs are ditransitive
class Language:
    def __init__(
        self,
        num_verbs: int,
        num_nouns: int,
    ):
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