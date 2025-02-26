## 8.4 
import numpy as np 
import matplotlib.pyplot as plt 

## First, Loading and Splitting the the data 
data = np.load("data/mnist-data-hw3.npz")
print(data.keys())
X = data["training_data"]
y = data["training_labels"]
# Flatten everything except the first dimension
X = X.reshape(X.shape[0], -1)
print("X.shape after flatten:", X.shape)  # should be (60000, 784)
#Shuffle + create a validation set 
num_row = X.shape[0]
indexs = np.arange(num_row)
np.random.shuffle(indexs)
valid_size = 10000
valid_index = indexs[:valid_size]
train_index = indexs[valid_size:]
X_valid = X[valid_index]
y_valid = y[valid_index]
X_main = X[train_index]
y_main = y[train_index]
sizes_train = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000 ]
digits = np.unique(y_valid)

### LDA HELPERS
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

# Help to predict class labels by employing LDA discrimnant:
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

### QDA HELPERS 
def fitting_qda(X_train, y_train, reg=1e-6):
    classes = np.unique(y_train)
    class_covs = {}
    class_means = {}
    class_priors = {}
    #computing means first 
    for cl in classes: 
        X_cl = X_train[y_train == cl]
        class_means[cl] = np.mean(X_cl, axis = 0) 
    #computing covar (separately for each class)
    s = X_train.shape[1] # num cols 
    for cl in classes: 
        X_cl = X_train[y_train == cl]
        diff = X_cl - class_means[cl]
        cov_cl = diff.T @ diff / (len(X_cl) - 1) #unbiased estimate
        # if near singular, add small reg
        cov_cl += reg * np.eye(s)
        class_covs[cl] = cov_cl
    #computing class priors 
    total_samples = len(X_train)
    for cl in classes: 
        class_priors[cl] = np.sum(y_train == cl)/total_samples
    return class_means, class_covs, class_priors

## Predicting qda for each sample x, compute: 
#   predict class lables using the qda discriminant: 
#       delta_c(x) = -1/2 log|Sigma_c| 
#                    -1/2 (x - mu_c)^T Sigma_c^{-1} (x - mu_c)
#                   + log pi_c
def predicting_qda(X_test, class_means, class_covs, class_priors):
    type_classes = sorted(class_means.keys())
    #precompute the inverses and determinants in order to avoid repeated calls'
    covs_inverse = {}
    det_log = {}
    for cl in type_classes: 
        # goal to inverse signma_c
        # better to use if only need the product, but we do need (x - mu_c)^T * Sigma_c^-1 * (x - mu_c))
        covs_inverse[cl] = np.linalg.inv(class_covs[cl])
        #computing the log determinanat 
        sign, logdet = np.linalg.slogdet(class_covs[cl])
        # the sign should be +1 if covar is positive definite 
        det_log[cl] = logdet
    y_predict = []
    for x in X_test:
        bestest_class = None
        bestest_score = -np.inf 
        for cl in type_classes: 
            diff = x - class_means[cl]
            #quadratic form here 
            quad_term = diff @ covs_inverse[cl] @ diff
            #discriminant here
            cl_score =-0.5 * det_log[cl] - 0.5 * quad_term + np.log(class_priors[cl])
            if cl_score > bestest_score: 
                bestest_score = cl_score 
                bestest_class = cl 
        y_predict.append(bestest_class)
    return np.array(y_predict)

### 8.4 Part d code
#Goal: storing the per-digit error rate in a dict of lists 
# e.g lda_digit_error[c] is list of lengths len(train_sizes), where each entry 
# is the error rate for digit c at that training size 

lda_dig_err = {c: [] for c in digits}
qda_dig_err = {c: [] for c in digits}

for num in sizes_train: 
    num = min(num, len(X_main))
    subset_index = np.random.choice(len(X_main), size=num, replace=False)
    X_subset = X_main[subset_index]
    y_subset = y_main[subset_index]

    #LDA 
    lda_means, lda_Sig, lda_priors = fitting_lda(X_subset, y_subset)
    lda_valid_predict = lda_prediction(X_valid, lda_means, lda_Sig, lda_priors)
    #computing the per-digit error 
    for cl in digits: 
        c_msk  = (y_valid == cl)
        if np.sum(c_msk) == 0: 
            #if no ex of digit cl in valid set then skip it 
            lda_dig_err[cl].append(0.0)
        else: 
            misclass = np.sum(lda_valid_predict[c_msk]!= cl)
            err_c = misclass/ np.sum(c_msk)
            lda_dig_err[cl].append(err_c)

    #QDA 
    qda_means, qda_covs, qda_priors = fitting_qda(X_subset, y_subset, reg= 1e-6)
    qda_valid_predict = predicting_qda(X_valid, qda_means, qda_covs, qda_priors)
    for cl in digits: 
        c_msk = (y_valid == cl)
        if np.sum(c_msk) == 0: 
            qda_dig_err[cl].append(0.0)
        else: 
            misclass = np.sum(qda_valid_predict[c_msk]!= cl)
            err_c = misclass/ np.sum(c_msk)
            qda_dig_err[cl].append(err_c)

# ## Plotting 10 curves for LDA 
# plt.figure()
# for cl in digits:
#     plt.plot(sizes_train, lda_dig_err[cl], marker='o', label=f"Digit {cl}")
# plt.xlabel("Number of Training Points")
# plt.ylabel("Rate of Error")
# plt.title("LDA Validation Error Rate vs. # Training Points (Per Digit)")
# plt.legend()
# plt.grid(True)
# plt.show()

## Plotting 10 curves for QDA 
plt.figure()
for cl in digits:
    plt.plot(sizes_train, qda_dig_err[cl], marker='o', label=f"Digit {cl}")
plt.xlabel("Number of Training Points")
plt.ylabel("Rate of Error")
plt.title("QDA Validation Error Rate vs. # Training Points (Per Digit)")
plt.legend()
plt.grid(True)
plt.show()
