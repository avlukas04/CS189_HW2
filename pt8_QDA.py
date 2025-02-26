## 8.3 (b)
import numpy as np 
import matplotlib.pyplot as plt 

## First, Loading and Splitting the the data 
data = np.load("data/mnist-data-hw3.npz")
X = data["training_data"]
y = data["training_labels"]
# Flatten everything except the first dimension
X = X.reshape(X.shape[0], -1)
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

## Implement QDA;
# First fit QDA
# parameters for qda: 
#   class_covs[cl] = covar matrix of class c (w/small reg)
#   classes_means[cl] = mean vector of class cl
#   class_priors[cl] = # samples in class cl / total sampl
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

### Creating loop over all the training sample sizes
sizes_train = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000 ]
rates_of_error = []

for num in sizes_train: 
    #if num exceeds the actual size of X_main we clip it
    num = min(num, len(X_main))
    #choose randomly n points from our main trainin set 
    subset_indexs = np.random.choice(len(X_main), size=num, replace=False)
    X_subset = X_main[subset_indexs]
    y_subset = y_main[subset_indexs]
    
    #now to fit QDA on the subset
    class_means, class_covs, class_priors = fitting_qda(X_subset, y_subset, reg=1e-4)
    
    #prediction on validation set 
    y_predictions = predicting_qda(X_valid, class_means, class_covs, class_priors) 
    #validation error computation
    accuracy = np.mean(y_predictions == y_valid)
    error = 1.0 - accuracy
    rates_of_error.append(error)
    print(f"Training size = {num}, Validation Error Rate = {error:.4f}")

## Plotting error rate vs num of training 
plt.figure()
plt.plot(sizes_train, rates_of_error, marker= 'o', label='QDA')
plt.xlabel("Number of Training Samples")
plt.ylabel("Validation Rate of Error")
plt.title("QDA Validation Rate of Error vs. Number of Training Samples")
plt.grid(True)
plt.legend()
plt.show()

