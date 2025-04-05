from .eval import Evaluator
from tinylang.model import Model
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.utils._testing import ignore_warnings
from tinylang.language import Language
import plotnine as p9
from collections import defaultdict
import torch.nn as nn
from tqdm import tqdm


class MLPProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ProbeEvaluator(Evaluator):
    def __str__(self):
        return "ProbeEvaluator"

    def __init__(self, run_every_n_steps: int):
        super().__init__(run_every_n_steps)
        self.activations = defaultdict(lambda: defaultdict(list))
        self.labels = defaultdict(lambda: defaultdict(list))

    def eval(self, model: Model, language: Language, inputs: dict, outputs: dict, step: int):
        probing_schemas = inputs["probing_schemas"]
        hidden_states = outputs["hidden_states"]
        types = set([x["type"] for x in probing_schemas])
        
        for type in types:
            for label_type in probing_schemas[0]["target_distributions"]:
                for query in probing_schemas[0]["queries"]:
                    for layer in range(len(hidden_states)):
                        activations, labels = [], []
                        for batch_idx in range(len(hidden_states[layer])):
                            if type != probing_schemas[batch_idx]["type"]:
                                continue
                            query_pos = probing_schemas[batch_idx]["queries"][query]["pos"]
                            activations.append(hidden_states[layer][batch_idx, query_pos].cpu())
                            labels.append(probing_schemas[batch_idx]["target_distributions"][label_type])
                        self.activations[step][f"{layer}.{type}.{label_type}.{query}"].extend(activations)
                        self.labels[step][f"{layer}.{type}.{label_type}.{query}"].extend(labels)
    

    @ignore_warnings(category=Warning)
    def post_eval(self, step: int):
        probe_weights = defaultdict(list)
        for subset in self.activations[step]:
            layer, type, label_type, query = subset.split(".")
            activations = torch.stack(self.activations[step][subset]).cpu().detach() # shape: (n, d)
            labels = torch.tensor(self.labels[step][subset]).cpu().detach() # shape: (n,)

            # first half is train set
            train_len = len(activations) // 2
            lr = LogisticRegression(random_state=0, max_iter=1000, l1_ratio=0.5,
                    fit_intercept=True, C=1.0,
                    penalty=None, solver="saga").fit(activations[:train_len], labels[:train_len])
            probe_weights[(layer, type, query)].append({
                "coef": torch.tensor(lr.coef_),
                "intercept": torch.tensor(lr.intercept_),
                "label_type": label_type,
            })
                            
            # get eval set accuracy
            preds = lr.predict(activations[train_len:])
            acc = ((preds == labels[train_len:]).sum() / len(preds)).item()
            print(f"{subset:>40}: {acc:.4%}")
            self.all_eval_stats[step][f"{subset}.acc"].append(acc)

            # # now do MLP probe
            # num_labels = max(labels) + 1
            # mlp = MLPProbe(activations.shape[1], activations.shape[1] * 2, num_labels)
            # mlp.train()
            # optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-2)
            # # iterator = tqdm(range(4000))
            # for _ in range(4000):
            #     optimizer.zero_grad()
            #     preds = mlp(activations[:train_len])
            #     loss = nn.functional.cross_entropy(preds, labels[:train_len])
            #     loss.backward()
            #     # iterator.set_postfix({"loss": loss.item()})
            #     optimizer.step()
            
            # # evaluate MLP probe
            # mlp.eval()
            # with torch.no_grad():
            #     preds = mlp(activations[train_len:])
            #     acc = ((preds.argmax(dim=1) == labels[train_len:]).sum() / len(preds)).item()
            #     print(f"{subset:>40} (MLP): {acc:.4%}")
            #     self.all_eval_stats[step][f"{subset}.mlp_acc"].append(acc)

        # compare similarity of probe weights
        for key in probe_weights:
            coefs = [x["coef"] for x in probe_weights[key]]
            for i in range(len(coefs)):
                label_i = probe_weights[key][i]["label_type"]   
                for j in range(i + 1, len(coefs)):
                    label_j = probe_weights[key][j]["label_type"]
                    diff = (coefs[i] - coefs[j]).norm().item()
                    per_row_cosine_sim = torch.nn.functional.cosine_similarity(coefs[i], coefs[j], dim=1)
                    self.all_eval_stats[step][f"{'.'.join(key)}.{label_i}.{label_j}.diff"].append(diff)
                    self.all_eval_stats[step][f"{'.'.join(key)}.{label_i}.{label_j}.cosine_sim"].append(per_row_cosine_sim.mean().item())

    def plot(self, log_dir: str):
        df = self.df
        df_all = df.groupby(["step", "variable"]).mean().reset_index()

        df = df_all[df_all["variable"].str.endswith(".acc")]
        df["layer"] = df["variable"].str.split(".").str[0]
        df["type"] = df["variable"].str.split(".").str[1]
        df["label_type"] = df["variable"].str.split(".").str[2]
        df["query"] = df["variable"].str.split(".").str[3]

        # make plot
        for type in df["type"].unique():
            df_subset = df[df["type"] == type]
            plot = (
                p9.ggplot(df_subset, p9.aes(x="step", y="value", color="query"))
                + p9.geom_line()
                + p9.facet_grid("label_type~layer")
            )
            plot.save(f"{log_dir}/{str(self)}.{type}.acc.png")

        df = df_all[df_all["variable"].str.endswith(".diff") | df_all["variable"].str.endswith(".cosine_sim")]
        df["layer"] = df["variable"].str.split(".").str[0]
        df["type"] = df["variable"].str.split(".").str[1]
        df["query"] = df["variable"].str.split(".").str[2]
        df["label_i"] = df["variable"].str.split(".").str[3]
        df["label_j"] = df["variable"].str.split(".").str[4]
        df["var"] = df["variable"].str.split(".").str[5]

        # make plot
        for var in df["var"].unique():
            for type in df["type"].unique():
                df_subset = df[df["var"] == var]
                df_subset = df_subset[df_subset["type"] == type]
                plot = (
                    p9.ggplot(df_subset, p9.aes(x="step", y="value", color="label_i + label_j"))
                    + p9.geom_line()
                    + p9.facet_grid("layer~query")
                    + (p9.scale_y_log10() if var == "diff" else p9.scale_y_continuous())
                )
                plot.save(f"{log_dir}/{str(self)}.{type}.{var}.png")
            