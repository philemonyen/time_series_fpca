"""
Leveraging the concept of DOMIAS and compute local density ratio. 
"""

import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics.pairwise import euclidean_distances

def domias(reference, real, synthetic):
    std_devs = np.std(real, axis=0)
    h_upper_bound = np.max(std_devs)
    pairwise_dists = euclidean_distances(real)
    non_zero_dists = pairwise_dists[pairwise_dists > 0]
    h_lower_bound = np.percentile(non_zero_dists, 1)

    bandwidth_grid = np.logspace(
        np.log10(h_lower_bound), 
        np.log10(h_upper_bound), 
        num=10
    )

    domias_scores = {}

    for bandwidth in bandwidth_grid:
        kde_ref = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde_ref.fit(reference)
        kde_real = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde_real.fit(real)
        log_density_ref = kde_ref.score_samples(synthetic)
        log_density_real = kde_real.score_samples(synthetic)
        domias_score = log_density_real - log_density_ref
        domias_scores[bandwidth] = domias_score

    return domias_scores

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