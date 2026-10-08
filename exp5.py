import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

# 1: Load and Preprocess the Dataset
# Load a subset of 20 Newsgroups (e.g., sci.space) with train/test split
train_data = fetch_20newsgroups(
    subset='train',
    categories=['sci.space'],
    remove=('headers', 'footers', 'quotes')
)

test_data = fetch_20newsgroups(
    subset='test',
    categories=['sci.space'],
    remove=('headers', 'footers', 'quotes')
)

# Convert text documents to a Bag-of-Words count matrix
vectorizer = CountVectorizer(
    max_features=2000,       # Limit vocabulary size to 2000 frequent terms
    stop_words='english',    # Strip common stopwords
    min_df=2                 # Exclude words that appear only once
)

X_train = vectorizer.fit_transform(train_data.data)
X_test = vectorizer.transform(test_data.data)

vocab = np.array(vectorizer.get_feature_names_out())
V = len(vocab)

# Aggregate word counts across all training documents
train_counts = np.asarray(X_train.sum(axis=0)).ravel()
test_counts = np.asarray(X_test.sum(axis=0)).ravel()
N = train_counts.sum()

print(f"Vocabulary size (V): {V}")
print(f"Total training word tokens (N): {N}")
print(f"Total test word tokens: {test_counts.sum()}")


# 2: Implement MLE Parameter Estimation
def estimate_mle(counts):
    """
    Computes MLE: theta_v = N_v / sum(N)
    """
    total = counts.sum()
    return counts / total

theta_mle = estimate_mle(train_counts)


# 3: Implement MAP Parameter Estimation with Dirichlet Priors

def estimate_map(counts, alpha):
    """
    Computes MAP estimate under symmetric Dirichlet prior Dir(alpha, ..., alpha).
    Formula: theta_v = (N_v + alpha - 1) / (N + V * (alpha - 1))
    Requires alpha > 1.0 for the mode to exist.
    """
    if alpha < 1.0:
        raise ValueError("alpha must be >= 1.0 to define a mode (MAP estimate).")
    
    pseudo_counts = alpha - 1.0
    numerator = counts + pseudo_counts
    denominator = counts.sum() + (len(counts) * pseudo_counts)
    return numerator / denominator


# Explore different Dirichlet hyperparameter values
# alpha=1.0001 (near-uniform/weak prior), alpha=1.1, alpha=2.0 (Laplace), alpha=10.0 (strong prior)
prior_alphas = [1.0001, 1.1, 2.0, 10.0]
map_estimates = {alpha: estimate_map(train_counts, alpha) for alpha in prior_alphas}


# 4: Compare Results and Evaluate Effect of Different Priors

def compute_test_log_likelihood(theta, test_counts):
    """
    Computes the log-likelihood of test counts under the model parameters:
    LL = sum_{v} (test_counts_v * log(theta_v))
    Returns -inf if theta_v == 0 for any observed test word.
    """
    # Identify indices where test count > 0
    observed_idx = test_counts > 0
    
    if np.any(theta[observed_idx] == 0):
        return -np.inf
    
    return np.sum(test_counts[observed_idx] * np.log(theta[observed_idx]))

def compute_entropy(theta):
    """Computes Shannon entropy: H(theta) = -sum(theta * log(theta))"""
    non_zeros = theta[theta > 0]
    return -np.sum(non_zeros * np.log(non_zeros))

# Compile comparison metrics into a table
summary_data = []

# Evaluate MLE
mle_ll = compute_test_log_likelihood(theta_mle, test_counts)
mle_entropy = compute_entropy(theta_mle)
summary_data.append({
    "Estimator": "MLE",
    "Prior Parameter (alpha)": "None",
    "Test Log-Likelihood": mle_ll,
    "Entropy (nats)": round(mle_entropy, 4),
    "Zero Probabilities": int(np.sum(theta_mle == 0))
})

# Evaluate MAP across different priors
for alpha in prior_alphas:
    theta_curr = map_estimates[alpha]
    ll_curr = compute_test_log_likelihood(theta_curr, test_counts)
    entropy_curr = compute_entropy(theta_curr)
    summary_data.append({
        "Estimator": "MAP",
        "Prior Parameter (alpha)": alpha,
        "Test Log-Likelihood": round(ll_curr, 2),
        "Entropy (nats)": round(entropy_curr, 4),
        "Zero Probabilities": int(np.sum(theta_curr == 0))
    })

results_df = pd.DataFrame(summary_data)
print("\n--- Model Comparison Summary ---")
print(results_df.to_string(index=False))

# Compare probability shrinkage for top-5 most frequent words
top_indices = np.argsort(train_counts)[-5:][::-1]
comparison_dict = {"Word": vocab[top_indices], "Count": train_counts[top_indices], "MLE": theta_mle[top_indices]}

for alpha in prior_alphas:
    comparison_dict[f"MAP (alpha={alpha})"] = map_estimates[alpha][top_indices]

top_words_df = pd.DataFrame(comparison_dict)
print("\n--- Probability Shrinkage for Top 5 Words ---")
print(top_words_df.to_string(index=False))
