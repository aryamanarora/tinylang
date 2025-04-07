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
import umap
import os
import imageio
import pandas as pd


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
                self.all_eval_stats[step][f"{'.'.join(key)}.{label_i}.coef"].append(coefs[i])
                for j in range(i + 1, len(coefs)):
                    label_j = probe_weights[key][j]["label_type"]
                    diff = (coefs[i] - coefs[j]).norm().item()
                    per_row_cosine_sim = torch.nn.functional.cosine_similarity(coefs[i], coefs[j], dim=1)
                    self.all_eval_stats[step][f"{'.'.join(key)}.{label_i}.{label_j}.diff"].append(diff)
                    self.all_eval_stats[step][f"{'.'.join(key)}.{label_i}.{label_j}.cosine_sim"].append(per_row_cosine_sim.mean().item())

    def plot(self, log_dir: str):
        df = self.df
        df_all = df[~df["variable"].str.endswith(".coef")]
        df_coef = df[df["variable"].str.endswith(".coef")]
        df_all = df_all.groupby(["step", "variable"]).mean().reset_index()

        # probe accuracy
        df_acc = df_all[df_all["variable"].str.endswith(".acc")]
        df_acc["layer"] = df_acc["variable"].str.split(".").str[0]
        df_acc["type"] = df_acc["variable"].str.split(".").str[1]
        df_acc["label_type"] = df_acc["variable"].str.split(".").str[2]
        df_acc["query"] = df_acc["variable"].str.split(".").str[3]

        # probe difference/similarity comparisons
        df_sim = df_all[df_all["variable"].str.endswith(".diff") | df_all["variable"].str.endswith(".cosine_sim")]
        df_sim["layer"] = df_sim["variable"].str.split(".").str[0]
        df_sim["type"] = df_sim["variable"].str.split(".").str[1]
        df_sim["query"] = df_sim["variable"].str.split(".").str[2]
        df_sim["label_i"] = df_sim["variable"].str.split(".").str[3]
        df_sim["label_j"] = df_sim["variable"].str.split(".").str[4]
        df_sim["var"] = df_sim["variable"].str.split(".").str[5]

        # make gif of umap of coefs
        df_coef["layer"] = df_coef["variable"].str.split(".").str[0]
        df_coef["type"] = df_coef["variable"].str.split(".").str[1]
        df_coef["query"] = df_coef["variable"].str.split(".").str[2]
        df_coef["label_i"] = df_coef["variable"].str.split(".").str[3]
        df_coef["var"] = df_coef["variable"].str.split(".").str[4]
        vectors = torch.cat(df_coef["value"].tolist(), dim=0).cpu().detach()
        umap_model = umap.UMAP(n_components=2, random_state=42).fit(vectors)
        transformed_vectors = umap_model.transform(vectors)
        min_x, max_x = transformed_vectors[:, 0].min(), transformed_vectors[:, 0].max()
        min_y, max_y = transformed_vectors[:, 1].min(), transformed_vectors[:, 1].max()

        # make gif of umap of coefs
        frames_dir = f"{log_dir}/frames"
        gifs_dir = f"{log_dir}/gifs"
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(gifs_dir, exist_ok=True)
        for layer in df_coef["layer"].unique():
            for type in df_coef["type"].unique():
                for query in df_coef["query"].unique():
                    df_subset = df_coef[(df_coef["layer"] == layer) & (df_coef["type"] == type) & (df_coef["query"] == query)]
                    for step in df_coef["step"].unique():
                        df_subset_step = df_subset[df_subset["step"] == step]
                        vectors_step = torch.cat(df_subset_step["value"].tolist(), dim=0).cpu().detach()
                        vectors_labels = []
                        for label_i in df_subset_step["label_i"].unique():
                            vectors_labels.extend([label_i] * len(torch.cat(df_subset_step[df_subset_step["label_i"] == label_i]["value"].tolist(), dim=0)))
                        vectors_step = umap_model.transform(vectors_step)
                        vectors_step = pd.DataFrame(vectors_step, columns=["x", "y"])
                        vectors_step["label_i"] = vectors_labels
                        # make umap of coefs
                        probe_acc = df_acc[(df_acc["layer"] == layer) & (df_acc["type"] == type) & (df_acc["query"] == query) & (df_acc["step"] == step)]["value"].mean()
                        frame = p9.ggplot(vectors_step, p9.aes(x="x", y="y", color="label_i")) + p9.geom_point() + p9.labs(title=f"{layer}.{type}.{query}.{step} (avg acc: {probe_acc:.4%})") + p9.xlim((min_x, max_x)) + p9.ylim((min_y, max_y))
                        frame.save(f"{frames_dir}/{str(self)}.{layer}.{type}.{query}.{step}.png")
                    
                    # make gif
                    frames = [imageio.imread(f"{frames_dir}/{str(self)}.{layer}.{type}.{query}.{step}.png") 
                            for step in df_coef["step"].unique()]
                    imageio.mimsave(f"{gifs_dir}/{str(self)}.{layer}.{type}.{query}.gif", frames, duration=0.1)

        # make plot
        for type in df_acc["type"].unique():
            df_subset = df_acc[df_acc["type"] == type]
            plot = (
                p9.ggplot(df_subset, p9.aes(x="step", y="value", color="query"))
                + p9.geom_line()
                + p9.facet_grid("label_type~layer")
            )
            plot.save(f"{log_dir}/{str(self)}.{type}.acc.png")

        # make plot
        for var in df_sim["var"].unique():
            for type in df_sim["type"].unique():
                df_subset = df_sim[df_sim["var"] == var]
                df_subset = df_subset[df_subset["type"] == type]
                legend = len(df_subset["label_i"].unique()) + len(df_subset["label_j"].unique()) > 2
                plot = (
                    (p9.ggplot(df_subset, p9.aes(x="step", y="value", color="label_i + label_j")) if legend else p9.ggplot(df_subset, p9.aes(x="step", y="value")))
                    + p9.geom_line()
                    + p9.facet_wrap("layer + '.' + query", scales="free_y")
                    + (p9.scale_y_log10() if var == "diff" else p9.scale_y_continuous())
                )
                plot.save(f"{log_dir}/{str(self)}.{type}.{var}.png")
            