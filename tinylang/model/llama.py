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
        self.n_layer = n_layer
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

        self.model.config = self.config
        self.model.to(device)
        self.components = ["attention_input", "attention_output", "block_input", "block_output"]
    
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

    @classmethod
    def from_pretrained(cls, path: str, device: torch.device | None = None):
        
        model = LlamaForCausalLM.from_pretrained(path)
        
        inst = cls(
            vocab_size=model.config.vocab_size,
            n_positions=model.config.max_position_embeddings,
            n_embd=model.config.hidden_size,
            n_layer=model.config.num_hidden_layers,
            n_head=model.config.num_attention_heads,
            n_inner=model.config.intermediate_size,
            rope_theta=model.config.rope_theta,
            device=device,
        )
        
        inst.model = model.to(device)
        
        return inst