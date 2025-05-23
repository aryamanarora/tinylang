from .language import Language
from .model import Model
from .eval import Evaluator
import torch
from tqdm import tqdm
from collections import defaultdict
import os
import numpy as np
import random
import wandb
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

# fix all seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# wandb config
WANDB_PROJECT = "tinylang"
WANDB_ENTITY = "aryamanarora"


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
        verbose: bool = False,
        wandb: bool = False,
        weight_decay: float = 0.0,
        warmup_percentage: float = 0.0,
        num_train_epochs: int = 1,
        cosine: bool = False,
        language_from_file: bool = False,
    ):
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_train_steps = num_train_steps
        self.num_eval_steps = num_eval_steps
        self.lr = float(lr)
        self.save_every_n_steps = save_every_n_steps
        self.log_dir = log_dir
        self.verbose = verbose
        self.wandb = wandb
        self.weight_decay = weight_decay
        self.warmup_percentage = warmup_percentage
        self.num_train_epochs = num_train_epochs
        self.cosine = cosine
        self.language_from_file = language_from_file


class Experiment:
    def __init__(
        self,
        model: Model,
        language: Language,
        evaluators: dict[str, list[Evaluator]],
        training_config: dict | TrainingConfig,
    ):
        self.model = model
        self.language = language
        self.evaluators = evaluators
        self.training_config = TrainingConfig(**training_config) if isinstance(training_config, dict) else training_config
        self.wandb = self.training_config.wandb
        self.verbose = self.training_config.verbose

        # set up optimizer and lr scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=self.training_config.lr,
            weight_decay=self.training_config.weight_decay,
        )
        scheduler_function = get_linear_schedule_with_warmup if (not self.training_config.cosine) else get_cosine_schedule_with_warmup
        self.scheduler = scheduler_function(
            self.optimizer,
            num_warmup_steps=int(self.training_config.num_train_steps * self.training_config.num_train_epochs * self.training_config.warmup_percentage),
            num_training_steps=self.training_config.num_train_steps * self.training_config.num_train_epochs,
        )

        # set up log dir
        os.makedirs(self.training_config.log_dir, exist_ok=True)

        # print model size
        size = sum(p.numel() for p in self.model.model.parameters())
        print(f"Model size: {size}")

        # log to wandb
        name = self.training_config.log_dir.split("logs/")[-1].replace("//", "/").replace("//", "/")
        name = name.replace(".", "_")
        training_dict = self.training_config.__dict__
        if self.wandb:
            self.wandb_run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=name,
                config={
                    "training": training_dict,
                    "language": self.language.config_dict,
                    "model": self.model.config_dict,
                    "model_size": size,
                }
            )


    @classmethod
    def from_config(cls, config: dict):
        """Load an experiment from a config file."""

        # set up language
        language_from_file = config["language"].get("file", None) is not None
        if language_from_file:
            language = Language.load(config["language"]["file"])
        else:
            language = Language.from_config(config["language"])
        config["model"]["config"]["vocab_size"] = language.vocab_size

        # set up device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config["model"]["config"]["device"] = device

        # load configs
        model = Model.from_config(config["model"])
        training_config = TrainingConfig(**config["training"])
        training_config.language_from_file = language_from_file

        # prepare train/eval sets
        if config["language"].get("file", None) is None:
            language.prepare_sets(
                train_set_size=training_config.train_batch_size * training_config.num_train_steps,
                eval_set_size=training_config.eval_batch_size * training_config.num_eval_steps,
            )

        # evals
        evaluators = {}
        for split in language.evalsets.keys():
            evaluators[split] = [Evaluator.from_config(evaluator) for evaluator in config["evaluators"]]

        # return experiment
        return cls(model, language, evaluators, training_config)
    

    def train(self):
        """Main training loop."""
        iterator = tqdm(range(self.training_config.num_train_epochs * self.training_config.num_train_steps + 1), desc="Training") if self.verbose else range(self.training_config.num_train_epochs * self.training_config.num_train_steps + 1)
        for step in iterator:
            # print current cuda memory usage
            # print(torch.cuda.memory_stats(device=self.model.model.device)["allocated_bytes.all.peak"])

            # one eval step (we want to eval at init for baselines, and after training)
            eval_stats = {}
            for split, evaluators in self.evaluators.items():
                eval_stats.update(self.eval_step(step, split=split, evaluators=evaluators))
            eval_stats["step"] = step

            # one train step
            if step != self.training_config.num_train_steps * self.training_config.num_train_epochs:
                train_loss = self.train_step(step % self.training_config.num_train_steps)
                eval_stats["train/loss"] = train_loss
                eval_stats["train/lr"] = self.scheduler.get_last_lr()[0]
                if self.verbose:
                    iterator.set_postfix(loss=train_loss)

            # optional save
            if self.training_config.save_every_n_steps is not None and step % self.training_config.save_every_n_steps == 0:
                self.save_checkpoint(step)
            
            # optional wandb log
            if self.wandb:
                self.wandb_run.log(eval_stats)

        # final eval
        self.make_plots()

        # save model and language to log dir for reproducibility
        self.model.save(os.path.join(self.training_config.log_dir, "model.pt"))
        if not self.training_config.language_from_file: # if language was not loaded from file, save it
            self.language.save(os.path.join(self.training_config.log_dir, "language.pkl"))
    

    def train_step(self, step: int):
        """Run a single train step."""
        # get input and labels
        inputs = self.language.get_train_step(step=step, batch_size=self.training_config.train_batch_size)

        # train step
        outputs = self.model.step(inputs["input_ids"], inputs["labels"])
        loss = outputs["loss"]

        # update optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # clear memory
        del inputs
        del outputs
        torch.cuda.empty_cache()

        # return loss
        return loss.item()


    def eval_step(self, step: int, split: str="test", evaluators: list[Evaluator]=None) -> dict:
        """Run all evaluators."""
        # skip if step is not divisible by run_every_n_steps
        if all(step % evaluator.run_every_n_steps != 0 for evaluator in evaluators):
            return {}

        # set model to eval
        self.model.model.eval()

        # batched evals
        eval_batch_size = self.training_config.eval_batch_size
        eval_steps = self.training_config.num_eval_steps
        all_outputs = []
        for eval_step in (tqdm(range(eval_steps), desc="Evals") if self.verbose else range(eval_steps)):
            inputs = self.language.get_eval_step(step=eval_step, batch_size=eval_batch_size, split=split)
            outputs = self.model.step(inputs["input_ids"], inputs["labels"])
            for k in outputs:
                if isinstance(outputs[k], torch.Tensor):
                    outputs[k] = outputs[k].cpu()
            all_outputs.append(outputs)
            for evaluator in evaluators:
                if step % evaluator.run_every_n_steps == 0:
                    evaluator.eval(self.model, self.language, inputs, outputs, step=step)

        # run all evaluators
        results = {}
        for evaluator in tqdm(evaluators, desc="Post-evals") if self.verbose else evaluators:
            if step % evaluator.run_every_n_steps == 0:
                evaluator.post_eval(step=step)
                if self.wandb:
                    for k, v in evaluator.wandb_log(step=step).items():
                        results[f"{split if split != 'test' else 'eval'}/{k}"] = v
                # print(torch.cuda.memory_stats(device=self.model.model.device)["allocated_bytes.all.peak"])
                torch.cuda.empty_cache()
        
        # clear memory
        del all_outputs
        torch.cuda.empty_cache()

        # set model to train
        for param in self.model.model.parameters():
            param.requires_grad = True
        self.model.model.train()
        return results


    def make_plots(self):
        """Make plots of the evaluation stats."""

        for split in self.evaluators:
            for evaluator in self.evaluators[split]:
                evaluator.prepare_plot()
                evaluator.plot(log_dir=self.training_config.log_dir)

                # save the df as csv
                os.makedirs(os.path.join(self.training_config.log_dir, f"{split}"), exist_ok=True)
                evaluator.df.to_csv(os.path.join(self.training_config.log_dir, f"{split}/{str(evaluator)}.csv"), index=False)
