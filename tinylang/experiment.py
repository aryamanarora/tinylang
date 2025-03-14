from .language import Language
from .model import Model
from .eval import Evaluator
import torch
from tqdm import tqdm
from collections import defaultdict
import os
import pandas as pd
import plotnine as p9
import numpy as np
import random


# fix all seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class TrainingConfig:
    def __init__(
        self,
        train_batch_size: int,
        eval_batch_size: int,
        num_train_steps: int,
        num_eval_steps: int,
        lr: float | int | str,
        log_dir: str,
        save_every_n_steps: int | None = None,
    ):
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_train_steps = num_train_steps
        self.num_eval_steps = num_eval_steps
        self.lr = float(lr)
        self.save_every_n_steps = save_every_n_steps
        self.log_dir = log_dir

class Experiment:
    def __init__(
        self,
        model: Model,
        language: Language,
        evaluators: list[Evaluator],
        training_config: dict | TrainingConfig,
    ):
        self.model = model
        self.language = language
        self.evaluators = evaluators
        self.training_config = TrainingConfig(**training_config) if isinstance(training_config, dict) else training_config

        # set up optimizer and lr scheduler
        self.optimizer = torch.optim.AdamW(self.model.model.parameters(), lr=self.training_config.lr)

        # set up log dir
        os.makedirs(self.training_config.log_dir, exist_ok=True)


    @classmethod
    def from_config(cls, config: dict):
        """Load an experiment from a config file."""

        # set up language
        language = Language.from_config(config["language"])
        config["model"]["config"]["vocab_size"] = language.vocab_size

        # set up device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config["model"]["config"]["device"] = device

        # load configs
        model = Model.from_config(config["model"])
        evaluators = [Evaluator.from_config(evaluator) for evaluator in config["evaluators"]]
        training_config = TrainingConfig(**config["training"])

        # prepare train/eval sets
        language.prepare_sets(
            train_batch_size=training_config.train_batch_size,
            eval_batch_size=training_config.eval_batch_size,
            num_train_steps=training_config.num_train_steps,
            num_eval_steps=training_config.num_eval_steps,
        )

        # return experiment
        return cls(model, language, evaluators, training_config)
    

    def train(self):
        """Main training loop."""
        iterator = tqdm(range(self.training_config.num_train_steps), desc="Training")
        all_eval_stats = defaultdict(list)
        for step in iterator:
            # one train step
            train_loss = self.train_step(step)
            iterator.set_postfix(loss=train_loss)

            # one eval step
            eval_stats = self.eval_step(step)
            if eval_stats != {}:
                all_eval_stats[step] = eval_stats
                print(eval_stats)

            # optional save
            if self.training_config.save_every_n_steps is not None and step % self.training_config.save_every_n_steps == 0:
                self.save_checkpoint(step)

        # final eval
        # self.final_eval()
        self.make_plots(all_eval_stats)

        # save model and language to log dir for reproducibility
        self.model.save(os.path.join(self.training_config.log_dir, "model.pt"))
        self.language.save(os.path.join(self.training_config.log_dir, "language.pkl"))
    

    def train_step(self, step: int):
        """Run a single train step."""
        # get input and labels
        inputs = self.language.get_train_step(step=step, batch_size=self.training_config.train_batch_size)

        # train step
        logits, loss = self.model.step(inputs["input_ids"], inputs["labels"])

        # update optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # return loss
        return loss.item()


    @torch.no_grad()
    def eval_step(self, step: int):
        """Run all evaluators."""
        all_eval_stats = {}
        for evaluator in self.evaluators:
            if step % evaluator.run_every_n_steps == 0:
                all_eval_stats[str(evaluator)] = []
                for eval_step in range(self.training_config.num_eval_steps):
                    inputs = self.language.get_eval_step(step=eval_step, batch_size=self.training_config.eval_batch_size)
                    eval_stats = evaluator.eval(self.model, inputs)
                    all_eval_stats[str(evaluator)].append(eval_stats)
                all_eval_stats[str(evaluator)] = evaluator.aggregate(all_eval_stats[str(evaluator)])
        return all_eval_stats


    def make_plots(self, all_eval_stats: dict[int, dict]):
        """Make plots of the evaluation stats."""
        
        rows = []
        for step, eval_stats in all_eval_stats.items():
            eval_stats["step"] = step
            rows.append(eval_stats)
        df = pd.json_normalize(rows)

        # plot each column in df
        for col in df.columns:
            if col == "step": continue
            # make sure type is numeric
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            plot = p9.ggplot(df, p9.aes(x="step", y=col)) + p9.geom_line()
            plot.save(os.path.join(self.training_config.log_dir, f"{col}.png"))
        
        # save the df as parquet
        df.to_parquet(os.path.join(self.training_config.log_dir, "eval_stats.parquet"))
