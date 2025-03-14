from transformers import GPT2Config, GPT2LMHeadModel
from .model import Model
import torch

class GPT2(Model):
    def __init__(
        self,
        vocab_size: int,
        n_positions: int,
        n_embd: int,
        n_layer: int,
        n_head: int,
        n_inner: int | None = None,
        device: torch.device | None = None,
    ):
        # initialize the model
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner if n_inner is not None else 4 * n_embd,
            activation_function="gelu_new",
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            summary_type="cls_index",
        )
        self.model = GPT2LMHeadModel(self.config).to(device)

        # print model size
        print(f"Model size: {sum(p.numel() for p in self.model.parameters())}")
    
    def step(self, input_ids: torch.Tensor, labels: torch.Tensor):
        """Run a single step.
        
        Args:
            input_ids: The input tokens
            labels: The labels

        Returns:
            A tuple containing the logits and the loss
        """
        output = self.model(input_ids, labels=labels)
        return output.logits, output.loss


    def save(self, path: str):
        self.model.save_pretrained(path)