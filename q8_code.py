import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## 8.1 

data = np.load("data/mnist-data-hw3.npz")
X_train = data["training_data"]  
y_train = data["training_labels"]

#normalize images 
def normalize(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)
#computing mean and covar matric for each class
unique_class = np.unique(y_train)
mean_class = {}
covar_class = {}
for cl in unique_class:
    X_cl = X_train[y_train == cl]
    X_cl = X_cl.reshape(X_cl.shape[0], -1) 
    mean_cl = np.mean(X_cl, axis=0)
    cov_cl = np.cov(X_cl, rowvar=False)
    mean_class[cl] = mean_cl
    covar_class[cl] = cov_cl
#print values 
for cl in unique_class: 
    print(f"Class {cl}:")
    print(f"Mean shape: {mean_class[cl].shape}")
    print(f"Covariance shape: {covar_class[cl].shape}\n")


## 8.2 

X_train = X_train.reshape(X_train.shape[0], -1)
#choosing a digit to visualize
chose_digit = 9 
digit_samples = X_train[y_train == chose_digit]
#computing the covar matrix
cov_matrix = np.cov(digit_samples, rowvar=False)
#plotting covar matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cov_matrix, cmap="coolwarm", cbar=True, xticklabels=False, yticklabels=False)
plt.title(f"Covariance Matrix for digit {chose_digit}")
plt.show()