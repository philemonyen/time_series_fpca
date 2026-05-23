"""
Leveraging the concept of DOMIAS and compute local density ratio. 
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

def compute_bandwidth(X_ref, rule="silverman"):
    """
    Calculates analytical bandwidth h for multivariate KDE.
    X_ref: Reference matrix (N x d) used to calibrate the scale
    """
    N, d = X_ref.shape
    # Average standard deviation across the d embedding dimensions
    sigma_bar = np.sqrt(np.mean(np.var(X_ref, axis=0)))
    
    if rule == "scott":
        return (N ** (-1 / (d + 4))) * sigma_bar
    elif rule == "silverman":
        factor = (4 / (d + 2)) ** (1 / (d + 4))
        return factor * (N ** (-1 / (d + 4))) * sigma_bar
    else:
        raise ValueError("Unknown rule. Choose 'scott' or 'silverman'.")


def domias_vectorized_kde(X_synthetic, X_real, X_holdout, h=None):
    """
    Non-Parametric Matrix-Vectorized Algebraic KDE
    Evaluates density ratios of X_synthetic points against X_real vs X_holdout distributions.
    Safe against high-dimensions (d) via log-space scaling calculations.
    """
    N_synthetic, d = X_synthetic.shape
    N_real = X_real.shape[0]
    N_holdout = X_holdout.shape[0]
    
    if h is None:
        h = compute_bandwidth(X_real, rule="silverman")
        
    # Pairwise Squared Euclidean Distances
    dist_real = (np.sum(X_synthetic**2, axis=1, keepdims=True) + 
                 np.sum(X_real**2, axis=1) - 
                 2 * np.dot(X_synthetic, X_real.T))
    
    dist_holdout = (np.sum(X_synthetic**2, axis=1, keepdims=True) + 
                    np.sum(X_holdout**2, axis=1) - 
                    2 * np.dot(X_synthetic, X_holdout.T))
    
    # FIX: Compute Normalization in Log-Space to prevent float explosion/collapse
    log_norm_real = np.log(N_real) + (d / 2.0) * np.log(2.0 * np.pi * (h**2))
    log_norm_holdout = np.log(N_holdout) + (d / 2.0) * np.log(2.0 * np.pi * (h**2))
    
    # Sum over the kernel transformations
    sum_kernel_real = np.sum(np.exp(-dist_real / (2.0 * h**2)), axis=1)
    sum_kernel_holdout = np.sum(np.exp(-dist_holdout / (2.0 * h**2)), axis=1)
    
    # Compute final densities securely (adding epsilon to log inputs to prevent log(0))
    p_real = np.exp(np.log(sum_kernel_real + 1e-300) - log_norm_real)
    p_holdout = np.exp(np.log(sum_kernel_holdout + 1e-300) - log_norm_holdout)
    
    # Density Ratio Calculation 
    density_ratio = p_real / (p_holdout + 1e-10)
    return density_ratio


def domias_subspace_mahalanobis(X_synthetic, X_real, X_holdout):
    """
    Parametric Subspace Mahalanobis Density Ratio
    Exploits orthogonal/uncorrelated coordinate properties of FPC matrices.
    Used for FPC score matrix density ratio computation
    """
    # Compute parametric properties of the training subspace
    mu_real = np.mean(X_real, axis=0)
    var_real = np.var(X_real, axis=0) + 1e-8 # stability buffer
    
    # Compute parametric properties of the holdout subspace
    mu_holdout = np.mean(X_holdout, axis=0)
    var_holdout = np.var(X_holdout, axis=0) + 1e-8
    
    # Part 1: Log determinant ratio term 0.5 * sum(log(var_holdout / var_train))
    log_det_term = 0.5 * np.sum(np.log(var_holdout / var_real))
    
    # Part 2: Distance differences for each target coordinate vector
    # Broadcast subtraction across rows of X_eval
    sq_mahalanobis_holdout = np.sum(((X_synthetic - mu_holdout) ** 2) / var_holdout, axis=1)
    sq_mahalanobis_real = np.sum(((X_synthetic - mu_real) ** 2) / var_real, axis=1)
    
    # Combine terms to get final log ratio profile
    log_density_ratio = log_det_term + 0.5 * (sq_mahalanobis_holdout - sq_mahalanobis_real)
    
    # Exponentiate to return to regular ratio scale R(x)
    return np.exp(log_density_ratio)

def compute_bce_loss(y_true, y_pred, eps=1e-15):
    """
    Computes Binary Cross-Entropy Loss using NumPy.
    
    y_true: NumPy array of ground truth labels (0 or 1)
    y_pred: NumPy array of predicted probabilities (continuous between 0 and 1)
    eps: Small stability constant to prevent log(0)
    """
    # Convert inputs to numpy arrays just in case
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Clip probabilities to prevent log(0) and log(1) errors
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Calculate the cross entropy per sample
    loss_per_sample = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    
    # Take the negative average across all samples
    bce_loss = -np.mean(loss_per_sample)
    
    return bce_loss


def full_knowledge_mia(real_knowledge, synthetic_knowledge):
    # Downsample datasets to obtain same dataset size
    share_size  = min(real_knowledge.shape[0], synthetic_knowledge.shape[0])
    real_knowledge = real_knowledge[:share_size]
    synthetic_knowledge = synthetic_knowledge[:share_size]

    # Create training data and labels
    non_member = np.zeros(real_knowledge.shape[0])
    member = np.ones(synthetic_knowledge.shape[0])
    data = np.concatenate([real_knowledge, synthetic_knowledge], axis=0)
    y = np.concatenate([non_member, member], axis=0)
    X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Train the Classifier
    attack_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    attack_classifier.fit(X_train, y_train)

    # 3. Test the Classifier
    y_pred_proba = attack_classifier.predict_proba(X_test)[:, 1]
    print("BCE Test Loss: ", compute_bce_loss(y_test, y_pred_proba))


    print("BCE MIA Test Loss: ", compute_bce_loss(y_test, y_pred_proba))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    mia_auc_roc = roc_auc_score(y_test, y_pred_proba)

    return fpr, tpr, thresholds, mia_auc_roc