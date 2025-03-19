from .eval import Evaluator
from tinylang.model import Model
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.utils._testing import ignore_warnings
import plotnine as p9
from collections import defaultdict
from tqdm import tqdm

class ProbeEvaluator(Evaluator):
    def __str__(self):
        return "ProbeEvaluator"

    def __init__(self, run_every_n_steps: int):
        super().__init__(run_every_n_steps)
        self.activations = defaultdict(lambda: defaultdict(list))
        self.labels = defaultdict(lambda: defaultdict(list))

    def eval(self, model: Model, inputs: dict, outputs: dict, step: int):
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
                            activations.append(hidden_states[layer][batch_idx, query_pos])
                            labels.append(probing_schemas[batch_idx]["target_distributions"][label_type])
                        self.activations[step][f"{layer}.{type}.{label_type}.{query}"].extend(activations)
                        self.labels[step][f"{layer}.{type}.{label_type}.{query}"].extend(labels)
    

    @ignore_warnings(category=Warning)
    def post_eval(self, step: int):
        for subset in tqdm(self.activations[step]):
            activations = torch.stack(self.activations[step][subset])
            labels = torch.tensor(self.labels[step][subset])

            # first half is train set
            train_len = len(activations) // 2
            lr = LogisticRegression(random_state=0, max_iter=1000, l1_ratio=0.5,
                    fit_intercept=True, C=1.0,
                    penalty=None, solver="saga").fit(activations[:train_len], labels[:train_len])
                            
            # get eval set accuracy
            preds = lr.predict(activations[train_len:])
            acc = (preds == labels[train_len:]).sum() / len(preds)
            self.all_eval_stats[step][f"{subset}.acc"].append(acc)


    
    def plot(self, log_dir: str):
        df = self.df
        df = df.groupby(["step", "variable"]).mean().reset_index()
        df["layer"] = df["variable"].str.split(".").str[0]
        df["type"] = df["variable"].str.split(".").str[1]
        df["label_type"] = df["variable"].str.split(".").str[2]
        df["query"] = df["variable"].str.split(".").str[3]
        df = df.drop(columns=["variable"])

        # make plot
        for type in df["type"].unique():
            df_subset = df[df["type"] == type]
            plot = (
                p9.ggplot(df_subset, p9.aes(x="step", y="value", color="query"))
                + p9.geom_line()
                + p9.facet_grid("label_type~layer")
            )
            plot.save(f"{log_dir}/{type}.png")
            