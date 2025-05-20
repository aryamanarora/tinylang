# Experiment Configs

Each subfolder of `configs` includes a `template.yaml` and a `make_sweep.py`. The python script sets the parameters of the ablation (i.e. which architectures to evaluate and which hyperparams to sweep). The YAML includes which language to train / evaluate on, as well as which hyperparameters to _not_ ablate.

In the paper, we report results from the following:
* Section 5.1: `ar_32`
* Section 5.2: `pcfg_easy` and `pcfg_medium`
* Section 5.3: `new_gen`
* Section 5.4: `ar_32_mamba_ablation`