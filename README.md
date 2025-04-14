# tinylang

## Structure

The `Experiment` class in `tinylang/experiment.py` wraps a whole experiment.

- `tinylang/model/` includes model architectures and training setups. Currently just GPT2.
- `tinylang/language/` includes generators for formal languages we want to train on. Currently just the headed PCFG.
- `tinylang/eval/` includes evaluators which run during training to assess the learning progress of the model. Currently includes some behavioural evals for the PCFG, will soon include code for probes.

## Instructions

Make sure you have `uv`. Clone/pull the repo and run the following to install the package:

```
uv sync
```

You may have to mess with the `zoology` package install if non-transformer architectures don't work. Try running:

```
uv remove zoology
uv add "zoology @ https://github.com/aryamanarora/zoology.git" --no-build-isolation
```

Then, to run an experiment, make a config file in the same subfolder as `experiments/configs/pcfg.yaml` using that as an example. Make sure `training.log_dir` is set to point to a path in `experiments/logs/...`. Then, run

```
uv run tinylang experiments/configs/[YOUR CONFIG FILE].yaml
```

This will run a full training experiment, and store plots and logging data along with the model checkpoint in the log dir.