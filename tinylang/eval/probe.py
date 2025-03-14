from .eval import Evaluator
from tinylang.model import Model


class ProbeEvaluator(Evaluator):
    def __str__(self):
        return "ProbeEvaluator"

    def __init__(self, run_every_n_steps: int):
        super().__init__(run_every_n_steps)
        self.probe = {}

    def eval(self, model: Model, inputs: dict, outputs: dict):
        hidden_states = outputs["hidden_states"]
        print(len(hidden_states))
        print(hidden_states[0].shape)
        input()