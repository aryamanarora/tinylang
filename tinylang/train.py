from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Config, AutoModel
from tinylang.language import PCFG, QueryType
import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict
device = "cuda" if torch.cuda.is_available() else "cpu"


def init_model(pcfg: PCFG):
    # initialize the model
    config = GPT2Config(
        vocab_size=pcfg.vocab_size,
        n_positions=1024,
        n_embd=128,
        n_layer=3,
        n_head=2,
        n_inner=4 * 128,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
    )
    model = GPT2LMHeadModel(config).to(device)
    return model


def eval_metrics(model: AutoModel, eval_inputs: torch.Tensor, eval_labels: torch.Tensor, pcfg: PCFG):
    with torch.no_grad():
        output = model(input_ids=eval_inputs, labels=eval_labels)
        logits = output.logits # (batch_size, seq_len, vocab_size)
        probs = torch.softmax(logits, dim=-1)
        loss = output.loss # always mean loss

        # get query answer token
        unpadded_len = (eval_inputs != pcfg.PAD).sum(dim=1)
        query_token = unpadded_len - 3
        query_answers = eval_inputs[torch.arange(len(eval_inputs)), query_token + 1]
        query_type = eval_inputs[torch.arange(len(eval_inputs)), query_token]
        query_token_prob = probs[torch.arange(len(probs)), query_token, query_answers]

        # take mean of query token prob by query type
        results = {"loss": loss.item(), "prob": query_token_prob.mean().item()}
        # ct = 0
        for t in set(query_type.tolist()):
            # use QueryType enum to get the label
            label = QueryType(t - pcfg.QUERY_START).name
            results[f"prob_{label}"] = query_token_prob[query_type == t].mean().item()
            # ct += query_token_prob[query_type == t].shape[0]
        # assert ct == len(query_token_prob)
        
        return results, {}


def train(
    model: AutoModel,
    pcfg: PCFG,
    batch_size: int = 16,
    num_steps: int = 10000,
):
    model.train()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6}M")

    optimizer = AdamW(model.parameters(), lr=1e-4)

    # load eval dataset
    eval_inputs, eval_labels = pcfg.sample_n(64)
    eval_inputs = torch.tensor(eval_inputs, device=model.device, dtype=torch.long)
    eval_labels = torch.tensor(eval_labels, device=model.device, dtype=torch.long)

    iterator = tqdm(range(num_steps), desc="Training")
    metrics = defaultdict(list)
    for step in iterator:
        input_ids, labels = pcfg.sample_n(batch_size)
        input_ids = torch.tensor(input_ids, device=model.device, dtype=torch.long)
        labels = torch.tensor(labels, device=model.device, dtype=torch.long)

        optimizer.zero_grad()
        output = model(input_ids=input_ids, labels=labels)
        logits = output.logits # (batch_size, seq_len, vocab_size)
        probs = torch.softmax(logits, dim=-1)
        loss = output.loss # always mean loss
        metrics["train_loss"].append(loss.item())
        postfix = {"loss": loss.item()}
        loss.backward()
        optimizer.step()

        # do eval
        eval_stats, more_stats = eval_metrics(model, eval_inputs, eval_labels, pcfg)
        for k, v in eval_stats.items():
            metrics[f"eval_{k}"].append(v)
            postfix[f"eval_{k}"] = v
        for k, v in more_stats.items():
            metrics[f"eval_{k}"].append(v)

        iterator.set_postfix(**postfix)
    
    return metrics


if __name__ == "__main__":
    pcfg = PCFG(num_terminals=20, num_nonterminals=10, max_rhs_len=10, max_rules_per_nt=5, max_depth=10, head_position="left")
    model = init_model(pcfg)
    train(model, pcfg)
