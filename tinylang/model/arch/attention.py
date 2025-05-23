import torch 
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math


class SelfAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super().__init__()
        self.dropout_p = attention_dropout

    def forward(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The query tensor. (B, S, H, D)
            k: The key tensor. (B, S, H, D)
            v: The value tensor. (B, S, H, D)
        """
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout_p, is_causal=True
        )
        # seqlen = qkv.shape[1]
        # q = q.contiguous()
        # k = k.contiguous()
        # v = v.contiguous()
        # softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        # scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        # causal_mask = torch.triu(
        #     torch.full((seqlen, seqlen), float('-inf'), device=scores.device), 1
        # )
        # scores = scores + causal_mask.to(dtype=scores.dtype)
        # attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        # attention_drop = F.dropout(attention, self.dropout_p if self.training else 0.0)
        # output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        # return output


class MHA(nn.Module):
    """Multi-head self-attention
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int=1,
        bias: bool=True,
        dropout: float=0.0,
        layer_idx: int=None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        assert (
            self.d_model % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        # self.Wqkv = nn.Linear(
        #     d_model, 3 * d_model, bias=bias
        # )
        self.inner_attn = SelfAttention(attention_dropout=dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        """"""
        q = rearrange(self.query(x), "... (h d) -> ... h d", h=self.num_heads)
        k = rearrange(self.key(x), "... (h d) -> ... h d", h=self.num_heads)
        v = rearrange(self.value(x), "... (h d) -> ... h d", h=self.num_heads)
        context = self.inner_attn(q, k, v)
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out
    
    def state_size(self, batch_size: int=1, sequence_length: int=2048):
        return 2 * self.d_model * sequence_length