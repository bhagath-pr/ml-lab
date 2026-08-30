import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load dataset
print("Loading dataset (downloading if running for the first time)...")
train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
print("Dataset loaded successfully!")

X_train_raw, y_train = train_data.data, train_data.target
X_test_raw, y_test = test_data.data, test_data.target

# 2. Vectorize for Multinomial NB (TF-IDF)
print("Vectorizing text data...")
tfidf_vec = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_mnb = tfidf_vec.fit_transform(X_train_raw)
X_test_mnb = tfidf_vec.transform(X_test_raw)

# 3. Vectorize for Bernoulli NB (binary counts)
binary_vec = CountVectorizer(max_features=10000, stop_words='english', binary=True)
X_train_bnb = binary_vec.fit_transform(X_train_raw)
X_test_bnb = binary_vec.transform(X_test_raw)

# 4. Fit Multinomial Naïve Bayes
print("Training Naïve Bayes classifiers...")
mnb = MultinomialNB()
mnb.fit(X_train_mnb, y_train)
y_pred_mnb = mnb.predict(X_test_mnb)

# 5. Fit Bernoulli Naïve Bayes
bnb = BernoulliNB()
bnb.fit(X_train_bnb, y_train)
y_pred_bnb = bnb.predict(X_test_bnb)

# 6. Evaluation metrics helper function
def evaluate_model(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

metrics_mnb = evaluate_model(y_test, y_pred_mnb)
metrics_bnb = evaluate_model(y_test, y_pred_bnb)

# 7. Display results comparison
results_df = pd.DataFrame({
    'Metric': list(metrics_mnb.keys()),
    'Multinomial NB': list(metrics_mnb.values()),
    'Bernoulli NB': list(metrics_bnb.values()),
})

print("\nExecution Complete!")
print(results_df.to_string(index=False))
