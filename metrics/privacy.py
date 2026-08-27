import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.spatial.distance import cdist

### DCR Score & DCR-based MIA ###
def dcr(real, synthetic):
    dists_train = cdist(real, synthetic, metric='euclidean')
    dcr_train = np.min(dists_train, axis=1)
    return dcr_train

def dcr_attack_percentile(canary, reference, synthetic):
    canary_dcr = dcr(canary, synthetic)
    reference_dcr = dcr(reference, synthetic)
    percentile_score = np.mean(reference_dcr > canary_dcr) * 100
    return canary_dcr, percentile_score

def dcr_mia(reference, real, synthetic):
    """
    Performs DCR-based Membership Inference Attack.
    """
    dcr_train = dcr(real, synthetic)
    dcr_holdout = dcr(reference, synthetic)
    
    scores = np.concatenate([dcr_train, dcr_holdout])
    labels = np.concatenate([np.ones(len(dcr_train)), np.zeros(len(dcr_holdout))])
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    mia_auc_roc = roc_auc_score(labels, scores)
    return fpr, tpr, thresholds, mia_auc_roc

## NNDR Score & NNDR-based MIA ###
def nndr_scores(real, synthetic, epsilon=1e-8):
    """
    Calculates NNDR (d1 / d2) for a target dataset against the synthetic dataset.
    Use np.partition for O(N) extraction of the two smallest distances instead of O(N log N) sorting.
    """
    dists = cdist(real, synthetic, metric='euclidean')
    dists_partitioned = np.partition(dists, 1, axis=1)
    d1 = dists_partitioned[:, 0]
    d2 = dists_partitioned[:, 1]
    
    nndr = d1 / (d2 + epsilon)
    return nndr

def nndr_attack_percentile(canary, reference, synthetic):
    canary_nndr = nndr_scores(canary, synthetic)
    reference_nndrs = nndr_scores(reference, synthetic)
    percentile_score = np.mean(reference_nndrs > canary_nndr) * 100
    
    return canary_nndr, percentile_score

def nndr_mia(reference, real, synthetic):
    """
    Performs NNDR-based Membership Inference Attack.
    """
    nndr_train = nndr_scores(real, synthetic)
    nndr_holdout = nndr_scores(reference, synthetic)

    scores = np.concatenate([nndr_train, nndr_holdout])
    labels = np.concatenate([np.ones(len(nndr_train)), np.zeros(len(nndr_holdout))])
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    mia_auc_roc = roc_auc_score(labels, scores)
    return fpr, tpr, thresholds, mia_auc_roc

### DOMIAS Ratio & Ratio Score-based MIA ###
def domias_attack_percentile(canary, reference, synthetic):
    """
    Computes the sample-wise DOMIAS attack percentile score.
    
    Args:
        canary_features: Array of shape (1, K) - The features of your single target.
        reference_features: Array of shape (M, K) - The features of your holdout set.
        synthetic_features: Array of shape (N, K) - The features of your synthetic set.
        
    Returns:
        canary_domias_score: Float, the density ratio score of the canary.
        percentile_score: Float (0 to 100), the attack confidence.
    """
    # Use your bandwidth logic (or a robust estimator like Scott's rule)
    bandwidth = np.max(np.std(reference, axis=0))
    if bandwidth == 0:
        bandwidth = 1e-3 # Prevent division by zero errors
        
    # 1. Fit KDEs on the datasets the attacker possesses
    kde_syn = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(synthetic)
    kde_ref = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(reference)
    
    # 2. Calculate the DOMIAS score for the Canary
    canary_log_syn = kde_syn.score_samples(canary)
    canary_log_ref = kde_ref.score_samples(canary)
    # Higher score means the synthetic data clusters heavily around this point
    canary_domias_score = canary_log_syn[0] - canary_log_ref[0] 
    
    # 3. Calculate the DOMIAS scores for all Reference records
    ref_log_syn = kde_syn.score_samples(reference)
    ref_log_ref = kde_ref.score_samples(reference)
    reference_domias_scores = ref_log_syn - ref_log_ref
    
    # 4. Calculate Attack Percentile
    # We want to know what % of reference records have a LOWER density ratio than the canary.
    percentile_score = np.mean(reference_domias_scores < canary_domias_score) * 100
    
    return canary_domias_score, percentile_score

def domias(reference, real, synthetic):
    std_devs = np.std(real, axis=0)
    bandwidth = np.max(std_devs)

    kde_ref = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde_ref.fit(reference)
    kde_real = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde_real.fit(real)
    log_density_ref = kde_ref.score_samples(synthetic)
    log_density_real = kde_real.score_samples(synthetic)
    return log_density_real - log_density_ref

def domias_mia(reference, real, synthetic):
    """
    Performs DOMIAS (Density ratio) Membership Inference Attack on embeddings.
    """
    std_devs = np.std(real, axis=0)
    bandwidth = np.max(std_devs)
    # 1. Fit KDE to the full real data (Train + Holdout) to get the prior/reference density
    X_real_all = np.vstack([real, reference])
    kde_real = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X_real_all)
    
    # 2. Fit KDE to the synthetic data to get the generated density
    kde_synth = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(synthetic)
    
    # Target points to evaluate (we want to guess if they are Members or Non-Members)
    targets = np.vstack([real, reference])
    labels = np.concatenate([np.ones(len(real)), np.zeros(len(reference))])
    
    # 3. Calculate log densities using KDE
    # (score_samples returns log-density)
    log_p_real = kde_real.score_samples(targets)
    log_p_synth = kde_synth.score_samples(targets)
    
    # 4. Compute the DOMIAS score: log( P_synth / P_real ) = log(P_synth) - log(P_real)
    # Higher score = synthetic model dumped too much probability mass here = likely memorized
    domias_scores = log_p_synth - log_p_real
    
    # 5. Evaluate
    fpr, tpr, thresholds = roc_curve(labels, domias_scores)
    mia_auc_roc = roc_auc_score(labels, domias_scores)
    return fpr, tpr, thresholds, mia_auc_roc