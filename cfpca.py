import numpy as np
from utils import get_data


def cfpca(target, background, alpha=1):

# 1. Setup Data (Rows=Patients, Cols=Time)
# X_train: Your Private Data
# X_bg:    Your Public/Holdout Data
# X_syn:   Your Synthetic Data

# 2. Compute Covariance Matrices
# rowvar=False means columns are variables (Time points)
cov_target = np.cov(X_train, rowvar=False)
cov_bg     = np.cov(X_bg, rowvar=False)

# 3. Compute Contrastive Covariance (alpha=1)
cov_diff = cov_target - cov_bg

# 4. Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_diff)

# Sort them (eigh returns them in ascending order, we want descending)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# 5. Extract the Top Component (The "Primary Leak")
leak_shape = eigenvectors[:, 0]  # This is Psi_1

# 6. Compute Scores (Projecting Synthetic Data)
# CRITICAL: Subtract the TARGET mean, not the synthetic mean
mean_target = np.mean(X_train, axis=0)

# The Calculation
scores_syn = np.dot(X_syn - mean_target, leak_shape)
scores_train = np.dot(X_train - mean_target, leak_shape)

# 7. Privacy Check
# Plot Histogram(scores_train) vs Histogram(scores_syn)
# If they overlap perfectly -> Privacy Leak.