from collections import defaultdict
from tinylang.language.pcfg import PCFG
from tqdm import tqdm

# max n-gram size to analyze
N = 24

def analyze_parent_child_distribution(language_path, num_samples=100000):
    language = PCFG.load(language_path)
    ngram_counts = [defaultdict(lambda: defaultdict(int)) for _ in range(N)]
    total_queries = 0
    
    for _ in tqdm(range(num_samples), desc="Analyzing parent-child relationships", unit="sample"):
        _, _, stats = language.sample(return_stats=True, return_sentence=True)
        sentence = stats['sentence']
        
        id_to_label = {}
        rightmost_terminals = {}
        for i, node in enumerate(sentence):
            rightmost_terminals[node.label] = i
            id_to_label[node.id] = node.label
        
        # compute ans for each rightmost terminal occ
        for child_label, i in rightmost_terminals.items():
            node = sentence[i]
            if node.head_id is not None:
                parent_label = id_to_label[node.head_id]
                context_tokens = [sentence[i-j].label if i > j else -1 for j in range(N-1, -1, -1)] # context for n gram is query tok + n-1 toks before it
                for n in range(N):
                    context = tuple(context_tokens[:(n+1)])
                    ngram_counts[n][context][parent_label] += 1
                
                total_queries += 1

    def calculate_accuracy(counts):
        total_accuracy = 0
        total_contexts = 0
        for key in counts:
            total = sum(counts[key].values())
            if total > 0:
                max_prob = max(counts[key].values()) / total
                total_accuracy += max_prob
                total_contexts += 1
        return total_accuracy / total_contexts if total_contexts > 0 else 0
    
    accuracies = [calculate_accuracy(ngram_counts[n]) for n in range(N)]
    return total_queries, accuracies

# Example usage:
# -------------
# language_path = "pcfg_medium.pkl"
# total_queries, accuracies = analyze_parent_child_distribution(language_path)
# 
# # print results
# print(f"\nTotal parent-child relationships analyzed: {total_queries}")
# 
# print("\nAccuracy Estimates (based on highest parent probability):")
# print("=====================================================")
# for n in range(N):
#     print(f"{n+1}-gram accuracy: {accuracies[n]:.4f}")
