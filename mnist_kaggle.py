# MNIST KAGGLE 
import numpy as np

# LOADING + FLATTEN 
data = np.load("data/mnist-data-hw3.npz")
print(data.keys())
X_train = data["training_data"]
y_train = data["training_labels"]
X_test = data["test_data"]
# flatten everything except the first dimension
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

#LDA HELPER FUNCTIONS
# Help compute the mean vector mu_c for each class c, returning a dictionary s.t class_means[c] = the mean vector of a class c 
def help_class_means(X_train, y_train) :
    types_classes = np.unique(y_train)
    mean_classes = {}
    for cl in types_classes: 
        X_cl = X_train[y_train == cl]
        mean_classes[cl] = np.mean(X_cl, axis=0)
    return mean_classes

# Help compute the pooled within class covar matrix, s.t covar matrix = [sum_{c} sum_{x in class c} (x-mu_c)(x-mu_c)^T] / (N - #classes)
def helper_pooled_covar_comp(X_train, y_train, class_means): 
    types_classes = np.unique(y_train)
    d = X_train.shape[1]
    COVAR = np.zeros((d, d))
    total_samples = len(X_train)
    for cl in types_classes:
        X_cl = X_train[y_train == cl]
        diff = X_cl - class_means[cl] # goal is to accumulate sum of (x-mu_c)(x-mu_c)^T 
        COVAR_cl = diff.T @ diff
        COVAR += COVAR_cl
    #now to dividie by the num of total_samples - num of classes 
    COVAR /= (total_samples - len(types_classes))
    return COVAR

# Help to fit the LDA parameters, class means mu_c, pooled covar matrix, class pripors pi_c, then returns class means, covar matrix(sigm), class_priors
def fitting_lda(X_train, y_train):
    class_means = help_class_means(X_train, y_train)
    COVAR = helper_pooled_covar_comp(X_train, y_train, class_means)
    # Intro little regularization will help preent singular matrix
    COVAR += 1e-6 * np.eye(COVAR.shape[0])
    #class prior computation pi_c = # of pooint in class / total num points 
    types_classes = np.unique(y_train)
    total_sampl = len(X_train)
    prior_classes = {}
    for c in types_classes: 
        prior_classes[c] = np.sum(y_train == c ) / total_sampl
    return class_means, COVAR, prior_classes

# Helper to predict class labels by employing LDA discrimnant:
    # dis_c(x) = x^T COVAR^{-1} mu_c - 1/2 mu_c^T COVAR^{-1} mu_c + log pi_c
    # Avoiding explicit inverse by precomputing alpha_c = solve(COVAR, mu_c)
    # That means: 
        #x^T COVAR{-1} mu_c = x dot alpha_c
        #mu_c^T COVAR{-1} mu_c = mu_c dot alpha_c
def lda_prediction(X_test, class_means, COVAR, prior_classes):
    #sorting to make sure consistent ordering
    type_classes = sorted(class_means.keys())
    #precomputing the alpha_c and beta_c 
    dict_alpha = {}
    dict_beta = {}
    # need to factor out COVAR^{-1} * mu_c and using linear sys to solve 
    for cl in type_classes: 
        mu_c = class_means[cl]
        # alpha_c = COVAR^{-1} * mu_c
        alpha_c = np.linalg.solve(COVAR, mu_c)
        beta_c = -0.5 * mu_c.dot(alpha_c) + np.log(prior_classes[cl])
        dict_alpha[cl] = alpha_c
        dict_beta[cl] = beta_c
    #computigng the predictions here 
    y_predict = []
    for x in X_test: 
        #calc x dot alpha_c + beta_c for each class cl 
        bestest_class = None
        bestest_score = -np.inf
        for cl in type_classes: 
            cl_score = x.dot(dict_alpha[cl]) + dict_beta[cl]
            if cl_score > bestest_score: 
                bestest_score = cl_score
                bestest_class = cl
        y_predict.append(bestest_class)
    return np.array(y_predict)

# Helper: for each sample (row), subtract its mean pixel value & divide by its std
#   doing this helps LDA handle varying contrast
#   X shape: (N, d)
def center_and_scale(X):
    means = np.mean(X, axis=1, keepdims=True)           # shape (N, 1)
    stds  = np.std(X, axis=1, keepdims=True) + 1e-6     # add eps to avoid /0
    X_norm = (X - means) / stds
    return X_norm
# Helper: projects X onto its top-k principal components
    # Outputs: (X_reduced, mean_vec, Vt_top)
    #   X_reduced: shape (N, k)
    #   mean_vec: shape (d,)  for centering test data
    #   Vt_top: shape (k, d)  top principal components
def pca_transform(X, k=100):
    # 1) Compute mean of each feature (pixel)
    mean_vec = np.mean(X, axis=0)
    # 2) Center X
    X_centered = X - mean_vec
    # 3) SVD => X_centered = U * S * V^T
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Vt: shape (d, d), rows are principal components
    # 4) Keep top k components
    Vt_top = Vt[:k]  # shape (k, d)
    # 5) Project X onto top-k principal components
    X_reduced = X_centered @ Vt_top.T  # shape (N, k)
    return X_reduced, mean_vec, Vt_top

# normalize the training and test sets
X_train = center_and_scale(X_train)
X_test  = center_and_scale(X_test)

# apply PCA
X_train_pca, mean_vec, Vt_top = pca_transform(X_train, k=100)  # or 50, 150, etc.

# for our test set we must apply the same mean_vec & Vt_top
X_test_centered = X_test - mean_vec
X_test_pca = X_test_centered @ Vt_top.T

## TRAIN ON FULL DATA 
class_means, Sig, priors = fitting_lda(X_train_pca, y_train)

## PREDITCTING TEST LABELS 
test_predict = lda_prediction(X_test_pca, class_means, Sig, priors)

## save submission for kaggle 
import pandas as pd 
submission_df = pd.DataFrame({"Id": np.arange(1, len(test_predict)+1), "Label": test_predict})
submission_df.to_csv("lda_kaggle_submission.csv", index=False)
print("Saved 'lda_kaggle_submission.csv'. Upload it to Kaggle!")
