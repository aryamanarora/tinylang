from transformers import LlamaConfig, LlamaForCausalLM
from .model import Model
import torch

class Llama(Model):
    def __init__(
        self,
        vocab_size: int,
        n_positions: int,
        n_embd: int,
        n_layer: int,
        n_head: int,
        n_inner: int | None = None,
        device: torch.device | None = None,
        rope_theta: float = 10000.0,
    ):
        # initialize the model
        self.config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=n_embd,
            intermediate_size=(4 * n_embd) if n_inner is None else n_inner,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            num_key_value_heads=n_head,
            hidden_act="silu",
            max_position_embeddings=n_positions,
            output_hidden_states=True,
            output_attentions=True,
            rope_theta=rope_theta,
        )
        self.model = LlamaForCausalLM(self.config).to(device)
        self.model.init_weights()
    
    def step(self, input_ids: torch.Tensor, labels: torch.Tensor):
        """Run a single step.
        
        Args:
            input_ids: The input tokens
            labels: The labels

        Returns:
            A tuple containing the logits and the loss
        """
        output = self.model(input_ids, labels=labels)
        return {
            "logits": output.logits.cpu(),
            "loss": output.loss, # keep on gpu for backprop
            "hidden_states": [h.cpu() for h in output.hidden_states],
            "attentions": [a.cpu() for a in output.attentions],
        }


    def save(self, path: str):
        self.model.save_pretrained(path)