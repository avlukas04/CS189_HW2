import glob
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

## LDA implementation
## helper function trains an LDA classifier 
# inputs: 
    # X : ndarray, shape (n_samples, n_features)
    #     The training feature matrix (bag-of-words, etc.).
    # y : ndarray, shape (n_samples,)
    #     Binary class labels (e.g., 0 for ham, 1 for spam).
# outputs: 
    # means : dict of {class_label -> mean vector}
    # pooled_cov : ndarray, shape (n_features, n_features)
    #     The regularized, shared covariance matrix for all classes.
    # priors : dict of {class_label -> prior probability}
def lda_train(X, y):
    classes = np.unique(y)
    n_features = X.shape[1]
    means = {}
    covs = {}
    priors = {}
    # computing the mean, covariance, and prior for each class 
    for c in classes:
        Xc = X[y == c]
        means[c] = np.mean(Xc, axis=0)
        covs[c] = np.cov(Xc, rowvar=False)  # unbiased estimate
        priors[c] = Xc.shape[0] / float(X.shape[0])

    # computing pooled covariance: weighted average of class covariances
    pooled_cov = np.zeros((n_features, n_features))
    for c in classes:
        n_c = (y == c).sum()
        pooled_cov += (n_c - 1) * covs[c]
    pooled_cov /= (X.shape[0] - len(classes))

    # -- adding regularize to the covariance to ensure it's positive definite ---
    alpha = 1e-5  
    pooled_cov += alpha * np.eye(n_features)
    return means, pooled_cov, priors
## helper function to Predicts class labels for X using the LDA parameters and sses a Cholesky decomposition for numerical stableness
# inputs: 
    # X : ndarray, shape (n_samples, n_features)
    # means : dict of class_label -> mean vector
    # pooled_cov : ndarray, shape (n_features, n_features)
    #     Shared, regularized covariance matrix for all classes.
    # priors : dict of class_label -> prior probability

# outputs: 
    # predictions : ndarray, shape (n_samples,)
    # Predicted class labels (e.g., 0 or 1).
def lda_predict(X, means, pooled_cov, priors):
    # trying out the Cholesky factorization
    L = np.linalg.cholesky(pooled_cov)
    log_det_cov = 2 * np.sum(np.log(np.diag(L)))  # log(det(cov)) = 2 * sum(log(diag(L)))
    predictions = []
    classes = list(means.keys())

    # def helper function to help lda predictions 
    def log_density(x, mean):
        # solving L*z = (x - mean) without inverting L explicitly
        z = np.linalg.solve(L, x - mean)
        mahal = z @ z  # squared the Mahalanobis distance
        return -0.5 * (mahal + log_det_cov)

    # evaluating the log-posterior for each sample
    for x in X:
        class_log_post = {}
        for c in classes:
            ld = log_density(x, means[c])
            # adding log prior
            class_log_post[c] = ld + np.log(priors[c])
        # picking the class with highest log-posterior
        predictions.append(max(class_log_post, key=class_log_post.get))
    return np.array(predictions)

# Helper function: Row Normalization
# purpose: normalizes each row vector in X to have L2 norm = 1 and helps reduce magnitude differences between documents
def normalize_rows(X):
    normed = []
    for row in X:
        norm = np.linalg.norm(row)
        normed.append(row / norm if norm > 0 else row)
    return np.array(normed)

# Main execution of the script 
if __name__ == "__main__":
    # reading raw text from spam/ and ham/
    spam_folder = 'data/spam/'
    ham_folder  = 'data/ham/'
    spam_files = glob.glob(os.path.join(spam_folder, '*.txt'))
    ham_files  = glob.glob(os.path.join(ham_folder, '*.txt'))

    spam_texts = []
    for fname in spam_files:
        with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
            spam_texts.append(f.read())
    ham_texts = []
    for fname in ham_files:
        with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
            ham_texts.append(f.read())

    y_spam = np.ones(len(spam_texts), dtype=int)  # spam = 1
    y_ham  = np.zeros(len(ham_texts), dtype=int)  # ham = 0
    X_train_text = spam_texts + ham_texts
    y_train = np.concatenate([y_spam, y_ham])

    # then converting to Bag-of-Words features using CountVectorizer()
    vectorizer = CountVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=2000
    )
    X_train_counts = vectorizer.fit_transform(X_train_text)
    X_train_features = X_train_counts.toarray()

    # normalizing the rows
    X_train_features = normalize_rows(X_train_features)

    # training LDA
    means, pooled_cov, priors = lda_train(X_train_features, y_train)

    # reading the test data from data/test/0.txt, 1.txtt,....
    num_test_examples = 1000
    test_texts = []
    for i in range(num_test_examples):
        fname = os.path.join('data/test', f"{i}.txt")
        with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
            test_texts.append(f.read())

    # transforming the test data with the same vectorizer
    X_test_counts = vectorizer.transform(test_texts)
    X_test_features = X_test_counts.toarray()
    X_test_features = normalize_rows(X_test_features)

    # predicting the test labels
    y_test_pred = lda_predict(X_test_features, means, pooled_cov, priors)
    # place to save the predictions to csv for kaggle
    df = pd.DataFrame({
        'Id': np.arange(1, len(y_test_pred) + 1),
        'Prediction': y_test_pred
    })
    df.to_csv('spam_predictions.csv', index=False)
    print("Done! Predictions saved to spam_predictions.csv")
