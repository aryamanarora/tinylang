"""
This file re-implements the Zoology model such that each block actually just inputs and outputs a residual.
This is done to make it easier to enable interchange interventions on blocks.

TODO: more SSM architectures
"""

import torch.nn as nn
from zoology.model import ModelConfig, StochasticDepth, TokenEmbeddings, _init_weights
from functools import partial

from zoology.mixers.mamba_ssm.triton.layernorm import RMSNorm
from mamba_ssm.modules.mamba_simple import Mamba
import torch
from typing import Optional


class MambaBlock(nn.Module):
    def __init__(
        self, config, fused_add_norm=False, residual_in_fp32=True, norm_epsilon=1e-5, **factory_kwargs
    ):
        super().__init__()
        d_model = config.d_model
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.sequence_mixer = Mamba(d_model, **factory_kwargs, **config.sequence_mixer.kwargs)
        self.norm = RMSNorm(d_model, eps=norm_epsilon)


    def forward(
        self, residual: Optional[torch.Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.sequence_mixer(hidden_states, inference_params=inference_params)
        residual = (residual + hidden_states) if residual is not None else hidden_states
        return residual


class Mamba2Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        raise NotImplementedError("Mamba2Block is not implemented")
    
class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()

        self.sequence_mixer = config.sequence_mixer.instantiate(
            d_model=config.d_model,
            layer_idx=layer_idx,
        )
        self.state_mixer = config.state_mixer.instantiate(
            d_model=config.d_model,
            layer_idx=layer_idx,
        )
        self.dropout1 = nn.Dropout(config.embed_dropout if layer_idx == 0 else config.resid_dropout)
        self.drop_path1 = StochasticDepth(config.drop_path, mode="row")
        self.norm1 = nn.LayerNorm(config.d_model)
        self.dropout2 = nn.Dropout(config.resid_dropout)
        self.drop_path2 = StochasticDepth(config.drop_path, mode="row")
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(self, residual):
        # first pre-norm op (attention or equivalent)
        hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
        hidden_states = self.sequence_mixer(hidden_states)
        dropped = self.drop_path1(self.dropout1(hidden_states))
        residual = (dropped + residual) if residual is not None else dropped

        # second pre-norm op (mlp or equivalent)
        hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
        hidden_states = self.state_mixer(hidden_states)
        dropped = self.drop_path2(self.dropout2(hidden_states))
        residual = (dropped + residual) if residual is not None else dropped

        return residual


class LMBackbone(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embeddings = TokenEmbeddings(
            config.d_model, 
            config.vocab_size, 
            config.max_position_embeddings,
            learnable=config.learnable_word_embeddings
        )

        if config.block_type == 'TransformerBlock':
            block_cls = TransformerBlock
        elif config.block_type == 'MambaBlock':
            block_cls = MambaBlock
        elif config.block_type == "Mamba2Block": 
            block_cls = Mamba2Block

        self.layers = nn.Sequential(
            *[
                block_cls(config=config, layer_idx=i)
                for i in range(config.n_layers)
            ]
        )
        self.drop_path_i = StochasticDepth(config.drop_path, mode="row")
        self.drop_i = nn.Dropout(config.resid_dropout) # actually dropout for input
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.apply(partial(_init_weights, n_layers=config.n_layers, block_type=config.block_type))

    def forward(self, input_ids, position_ids=None):
        hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        hidden_states = self.drop_path_i(self.drop_i(hidden_states))
        residual = self.layers(hidden_states)
        hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
        return hidden_states


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )

        self.backbone = LMBackbone(config=config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, n_layers=config.n_layers, block_type=config.block_type))

        # tie weights
        self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight

    def forward(
        self, input_ids, position_ids=None, state=None
    ): 
        hidden_states = self.backbone(input_ids, position_ids=position_ids)
        return self.lm_head(hidden_states)
    
    def state_size(self, sequence_length: int):
        state_size = 0
        for layer in self.backbone.layers:
            if MambaBlock and isinstance(layer, MambaBlock):
                mixer = layer.mixer
            if Mamba2Block and isinstance(layer, Mamba2Block):
                mixer = layer.mixer
            elif isinstance(layer, TransformerBlock):
                mixer = layer.sequence_mixer
            else: 
                return None
            if hasattr(mixer, "state_size"):
                state_size += mixer.state_size(sequence_length=sequence_length)
            else:
                print(f"Layer {type(mixer).__name__} does not have state size")
                return None
            
        return state_size * 4
        