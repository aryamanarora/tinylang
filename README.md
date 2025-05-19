<div align="center">
  <h1 align="center">MechEvals <sub>by <a href="https://github.com/stanfordnlp/pyvene">pyvene</a></sub></h1>
  <!-- <a href="..."><strong>Read our paper »</strong></a> -->
</div>

**MechEvals** is a framework for training and analysing different language model architectures on synthetic tasks. This repo includes all code for the synthetic tasks, intervention experiments (for interpretability), and some of the model architecture definitions (the remainder, for various SSMs are imported from the excellent [`zoology`](https://github.com/HazyResearch/zoology) library.)

---

## Structure

The `Experiment` class in `tinylang/experiment.py` wraps a whole experiment.

- `tinylang/model/` includes model architectures and training setups.
- `tinylang/language/` includes generators for formal languages we want to train on.
- `tinylang/eval/` includes evaluators which run during training to assess the learning progress of the model. These are both our proposed mechanistic evaluators, as well as traditional behavioural evaluators (e.g. task accuracy).

We generate language train/test sets in `languages/` and store config files in `yaml` format in `experiments/configs`. When you run an experiment, it will log to a subfolder of `experiments/logs`.

## Instructions

Make sure you have `uv`. Clone/pull the repo and run the following to install the package:

```
uv sync
```

You may have to mess with the `zoology`, `causal-conv1d`, and `mamba-ssm` package installs, since the various SSM dependencies may have to be built locally. E.g. to reinstall `zoology`:

```
uv remove zoology
uv add "zoology @ https://github.com/HazyResearch/zoology.git" --no-build-isolation
```

Then, to run an experiment, make a config file in the same subfolder as `experiments/configs/pcfg.yaml` using that as an example. Make sure `training.log_dir` is set to point to a path in `experiments/logs/...`. Then, run

```
uv run tinylang experiments/configs/[YOUR CONFIG FILE].yaml
```

This will run a full training experiment, and store plots and logging data along with the model checkpoint in the log dir.
