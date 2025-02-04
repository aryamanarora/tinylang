from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Config, AutoModel
from language import Language
import torch
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"


def init_model(language: Language):
    # initialize the model
    config = GPT2Config(
        vocab_size=language.vocab_size,
        n_positions=5,
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


def train(
    model: AutoModel,
    language: Language,
    batch_size: int = 16,
    num_steps: int = 10000,
):
    model.train()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6}M")

    optimizer = AdamW(model.parameters(), lr=1e-4)

    iterator = tqdm(range(num_steps), desc="Training")
    losses = []
    for step in iterator:
        tokens, next_token_probs, p_order_given_ctx = language.sample_n(batch_size)
        tokens = torch.tensor(tokens, device=model.device, dtype=torch.long)
        next_token_probs = torch.tensor(next_token_probs, device=model.device, dtype=torch.float32)
        p_order_given_ctx = torch.tensor(p_order_given_ctx, device=model.device, dtype=torch.float32)

        optimizer.zero_grad()
        output = model(input_ids=tokens, labels=tokens)
        logits = output.logits # (batch_size, seq_len, vocab_size)
        probs = torch.softmax(logits, dim=-1)
        loss = output.loss # always mean loss
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        # kl divergence between probs and next_token_probs
        # kl_div = torch.nn.functional.kl_div(probs[:, :-1, :], next_token_probs[:, :-1, :], reduction="batchmean")
        # print(probs[0, :-1, :], next_token_probs[0, :-1, :], p_order_given_ctx[0, :, :])
        # input()

        iterator.set_postfix(loss=loss.item())
    
    return losses


if __name__ == "__main__":
    language = Language(num_verbs=10, num_nouns=10)
    model = init_model(language)
    train(model, language)
