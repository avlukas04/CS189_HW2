## 8.3 (a)
import numpy as np 
import matplotlib.pyplot as plt 

## First, Loading and Splitting the our data 
data = np.load("data/mnist-data-hw3.npz")
print(data.keys())
X = data["training_data"]
y = data["training_labels"]
# flatten everything except the first dimension
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

## Second creating helper functions to do LDA 
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

# help to predict class labels by employing LDA discrimnant:
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

## Creating the loop over all the training sizes 
sizes_train = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000 ]
rates_of_error = []

for num in sizes_train: 
    #if num exceeds the actual size of X_main we clip it
    num = min(num, len(X_main))
    #choose randomly n points from our main trainin set 
    subset_indexs = np.random.choice(len(X_main), size=num, replace=False)
    X_subset = X_main[subset_indexs]
    y_subset = y_main[subset_indexs]
    #now to fit the lda on the subset
    class_means, COVAR, prior_class = fitting_lda(X_subset, y_subset)
    #prediction on validation set 
    y_predictions = lda_prediction(X_valid, class_means, COVAR, prior_class) 
    accuracy = np.mean(y_predictions == y_valid)
    error = 1.0 - accuracy
    rates_of_error.append(error)
    print(f"Training size = {num}, Validation Error Rate = {error:.4f}")

## Plotting error rate vs num of training 
plt.figure()
plt.plot(sizes_train, rates_of_error, marker= 'o')
plt.xlabel("Number of Training Samples")
plt.ylabel("Validation Rate of Error")
plt.title("LDA Validation Rate of Error vs. Number of Training Samples")
plt.grid(True)
plt.show()



