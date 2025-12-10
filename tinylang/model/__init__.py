from .model import Model
from .gpt2 import GPT2
from .llama import Llama
from .lstm import LSTM
from .arch.attention import MHA

try:
    from .zoology import Zoology, LanguageModel
except ImportError:
    Zoology = None
    LanguageModel = None

__all__ = ['Model', 'GPT2', 'Llama', 'LSTM', 'MHA']
if Zoology is not None:
    __all__.append('Zoology')
if LanguageModel is not None:
    __all__.append('LanguageModel')
