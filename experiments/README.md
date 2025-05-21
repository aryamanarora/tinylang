# Experiment Configs

Each subfolder of `configs` includes a `template.yaml` and a `make_sweep.py`. The python script sets the parameters of the ablation (i.e. which architectures to evaluate and which hyperparams to sweep). The YAML includes which language to train / evaluate on, as well as which hyperparameters to _not_ ablate.

In the paper, we report aggregation of results from the following configs:
* Section 5.1:
    * `ar_32`, `ar_32_nope`, `ar_32_more_lr` (NoPE for non-attention only)
* Section 5.2:
    * `pcfg_easy`, `pcfg_easy_nope` (NoPE for non-attention only)
    * `pcfg_easy_8192`
    * `pcfg_medium`, `pcfg_medium_nope` (NoPE for non-attention only)
* Section 5.3:
    * `pcfg_gen`, `pcfg_gen_1L`, `pcfg_gen_3L`
* Section 5.4:
    * `ar_32_mamba_ablation`,
    * `ar_32_based_ablation`, `ar_32_based_ablation2`

And in the appendices:
* Appendix D.1:
    * `ar_32`, `ar_32_nope`, `ar_32_more_lr`
    * `pcfg_easy`, `pcfg_easy_nope`
    * `pcfg_easy_8192`
    * `pcfg_medium`, `pcfg_medium_nope`
* Appendix D.2:
    * `ar_32_1L`
    * `pcfg_easy_1L`
* Appendix D.3:
    * `ar_32_3L`
* Appendix D.4:
    * `sibling`, `sibling_1L`, `sibling_3L`