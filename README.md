<div align="center">
  <a align="center"><img src="https://github.com/user-attachments/assets/47a9d660-abb2-4d74-9a4d-da4bcfadd671" width="100" height="100" /></a>
  <h1 align="center">Tinylang <sub>by <a href="https://github.com/stanfordnlp/pyvene">pyvene</a></sub></h1>
  <!-- <a href="..."><strong>Read our paper »</strong></a> -->
</div>

**Tinylang** is a framework for performing mechanistic evaluations of language model architectures on synthetic tasks. This repo includes all code for the synthetic tasks, training and logging for models, interventions for the mechanistic evaluations, and model architecture definitions.

We would like to thank the creaters of the [`zoology`](https://github.com/HazyResearch/zoology) library for providing easy-to-use implementations of many SSMs used in this repo!

---

## Structure

The `Experiment` class in `tinylang/experiment.py` wraps a whole experiment.

- `tinylang/model/` includes model architectures and training setups.
- `tinylang/language/` includes generators for formal languages we want to train on.
- `tinylang/eval/` includes evaluators which run during training to assess the learning progress of the model. These are both our proposed mechanistic evaluators, as well as traditional behavioural evaluators (e.g. task accuracy).

We store language config files in `yaml` format in `languages/` and store experiment configs in `experiments/configs`. When you run an experiment, it will log to a subfolder of `experiments/logs`.

## Instructions

Make sure you have `uv`. Clone/pull the repo and run the following to install the package (requires GPU to build the `zoology` dependencies):

```
uv sync && uv sync --extra zoology
```

If you are running experiments that don't depend on `zoology` architectures (i.e. all SSMs in this repo, which require `mamba-ssm` and `causal-conv1d` as dependencies), you can simply run `uv sync`. This is untested though!

To run an experiment, make a config file in the same subfolder as `experiments/configs/`. Make sure `training.log_dir` is set to point to a path in `experiments/logs/...`. Then, run

```
uv run tinylang experiments/configs/[YOUR CONFIG FILE].yaml
```

This will run a full training experiment, and store plots and logging data along with the model checkpoint in the log dir.

To generate a language (or set of languages), make a config folder in `languages/`. Then, run

```
uv run make_languages.py -c [YOUR CONFIG FOLDER]
```

This will create a language and store it as a `pkl` file in the same subdirectory as the config file.