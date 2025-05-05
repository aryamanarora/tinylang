# from zoology.model import LanguageModel
# from zoology.config import ModelConfig
from .arch.zoology import ModelConfig, LanguageModel
from transformers.loss.loss_utils import ForCausalLMLoss
from .model import Model
import torch
from einops import rearrange


class Zoology(Model):
    def __init__(
        self,
        vocab_size: int,
        n_positions: int,
        n_embd: int,
        n_layer: int,
        n_head: int = 1,
        n_inner: int | None = None, # discarded, defaults to 4 * n_embd
        mixer_type: str = "attention",
        state_mixer_type: str | None = None,
        device: torch.device | None = None,
        bias: bool = False,
    ):
        n_inner = 4 * n_embd if n_inner is None else n_inner
        self.n_layer = n_layer

        # configs taken from https://github.com/HazyResearch/zoology/blob/c42ae3370b9b13a04a23c5f9f4d967469ecb8958/zoology/experiments/iclr24_zoology_figure2/configs.py
        input_seq_len = n_positions
        MIXERS = {
            "attention": dict(
                # name="tinylang.model.MHA",
                name="zoology.mixers.attention.MHA",
                kwargs={
                    "dropout": 0.0,
                    "num_heads": n_head,
                    "bias": bias,
                },
            ),
            "hyena": dict(
                name="zoology.mixers.hyena.Hyena",
                kwargs={
                    "l_max": input_seq_len,
                    "bias": bias,
                },
            ),
            "rwkv": dict(
                name="zoology.mixers.rwkv.RWKVTimeMixer",
                kwargs={
                    "l_max": input_seq_len,
                    "bias": bias,
                },
            ),
            "base_conv": dict(
                name="zoology.mixers.base_conv.BaseConv",
                kwargs={
                    "l_max": input_seq_len,
                    # pass a list of kernel sizes for each of four layers
                    "kernel_size": [3, -1, 3, -1],
                    "bias": bias,
                }
            ),
            "h3": dict(
                name="zoology.mixers.h3.H3",
                kwargs={
                    "l_max": input_seq_len,
                    "d_state": input_seq_len,  # makes it mathematically equivalent to Hyena
                    "head_dim": 2,
                    "bias": bias,
                }
            ),
            "based": dict(
                name="zoology.mixers.hybrid.Hybrid",
                kwargs={
                    "configs": [
                        dict(
                            name="zoology.mixers.base_conv.BaseConv",
                            kwargs={
                                "l_max": input_seq_len,
                                # pass a list of kernel sizes for each of four layers
                                "kernel_size": 3,
                                "implicit_long_conv": True,
                                "bias": bias,
                            }
                        ),
                        dict(
                            name="zoology.mixers.based.Based",
                            kwargs={
                                "l_max": input_seq_len,
                                "feature_dim": 8,
                                "num_key_value_heads": n_head,
                                "num_heads": n_head,
                                "feature_name": "taylor_exp",
                                "train_view": "quadratic",
                                "bias": bias,
                            }
                        )
                    ]
                }
            ),
            "mamba": dict(
                name="mamba_ssm.modules.mamba_simple.Mamba",
                kwargs={
                    "bias": bias,
                }
            ),
            "mamba2": dict(
                name="mamba_ssm.modules.mamba2.Mamba2",
                kwargs={
                    "bias": bias,
                }
            ),
        }

        # add MLP or GLU to model
        state_mixer = dict(name="torch.nn.Identity", kwargs={})
        assert n_inner % n_embd == 0, "n_inner must be divisible by n_embd"
        if state_mixer_type in ["mlp", "glu"] and mixer_type != "mamba":
            name = state_mixer_type.upper()
            state_mixer = dict(name=f"zoology.mixers.mlp.{name}", kwargs={"hidden_mult": n_inner // n_embd})

        # set up config
        self.config = ModelConfig(
            sequence_mixer=MIXERS[mixer_type],
            state_mixer=state_mixer,
            d_model=n_embd,
            n_layers=n_layer,
            max_position_embeddings=n_positions,
            learnable_word_embeddings=True,
            vocab_size=vocab_size,
            resid_dropout=0.0,
            embed_dropout=0.0,
            drop_path=0.0,
            layer_norm_epsilon=1e-5,
            pad_vocab_size_multiple=1,
            block_type="TransformerBlock" if mixer_type != "mamba" else "MambaBlock",
            name="default",
        )
        self.model = LanguageModel(self.config)

        # to satisfy pyvene
        self.model.config = self.config
        self.model.device = device
        self.model.to(device)
        self.components = ["attention_input", "attention_output", "block_input", "block_output"]
        if mixer_type != "mamba":
            self.components.extend(["mlp_input", "mlp_output"])

    
    def step(self, input_ids: torch.Tensor, labels: torch.Tensor):
        """Run a single step.
        
        Args:
            input_ids: The input tokens
            labels: The labels

        Returns:
            A tuple containing the logits and the loss
        """
        # forward
        logits = self.model(input_ids)

        # aux loss
        auxiliary_loss = []
        def get_auxiliary_loss(module):
            if hasattr(module, "get_auxiliary_loss"):
                auxiliary_loss.append(module.get_auxiliary_loss())
        self.model.apply(get_auxiliary_loss)
        auxiliary_loss = sum(auxiliary_loss)

        # need to flatten batch and sequence dimensions
        main_loss = ForCausalLMLoss(logits, labels, vocab_size=self.config.vocab_size)
        # main_loss = self.loss_fn(
        #     rearrange(logits, "... c -> (...) c"), labels.flatten()
        # )
        loss = main_loss + auxiliary_loss

        return {
            "logits": logits.cpu(),
            "loss": loss, # keep on gpu for backprop
            "auxiliary_loss": auxiliary_loss,
            "hidden_states": [], # TODO: figure out how to get these
            "attentions": [], # TODO: figure out how to get these
        }


    def save(self, path: str):
        # save as torch model
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        # load as torch model
        self.model.load_state_dict(torch.load(path))
