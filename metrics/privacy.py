import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics.pairwise import cdist

### DCR Score & DCR-based MIA ###
def dcr(reference, real, synthetic):
    dists_train = cdist(real, synthetic, metric='euclidean')
    dcr_train = np.min(dists_train, axis=1)
    dists_holdout = cdist(reference, synthetic, metric='euclidean')
    dcr_holdout = np.min(dists_holdout, axis=1)
    return dcr_train - dcr_holdout

def dcr_mia(reference, real, synthetic):
    """
    Performs DCR-based Membership Inference Attack.
    """
    # 1. Compute DCR for Members (Train)
    # cdist computes pairwise distances; min(axis=1) finds the closest synthetic point
    dists_train = cdist(real, synthetic, metric='euclidean')
    dcr_train = np.min(dists_train, axis=1)
    
    # 2. Compute DCR for Non-Members (Holdout)
    dists_holdout = cdist(reference, synthetic, metric='euclidean')
    dcr_holdout = np.min(dists_holdout, axis=1)
    
    # 3. Combine scores and labels
    scores = np.concatenate([dcr_train, dcr_holdout])
    labels = np.concatenate([np.ones(len(dcr_train)), np.zeros(len(dcr_holdout))])
    
    # 4. Invert scores for ROC AUC
    # Since smaller DCR means HIGHER likelihood of membership, we negate the distances
    # so that higher values = predicted member.
    inverted_scores = -scores
    
    # 5. Evaluate
    fpr, tpr, thresholds = roc_curve(labels, inverted_scores)
    mia_auc_roc = roc_auc_score(labels, inverted_scores)
    return fpr, tpr, thresholds, mia_auc_roc

### DOMIAS Ratio & Ratio Score-based MIA ###
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

### Classifier based MIA
def classifier_mia(real, synthetic):
    # Downsample datasets to obtain same dataset size
    share_size  = min(real.shape[0], synthetic.shape[0])
    real = real[:share_size]
    synthetic = synthetic[:share_size]

    # Create training data and labels
    non_member = np.zeros(real.shape[0])
    member = np.ones(synthetic.shape[0])
    data = np.concatenate([real, synthetic], axis=0)
    y = np.concatenate([non_member, member], axis=0)
    X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Train the Classifier
    attack_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    attack_classifier.fit(X_train, y_train)

    # 3. Test the Classifier
    y_pred_proba = attack_classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    mia_auc_roc = roc_auc_score(y_test, y_pred_proba)

    return fpr, tpr, thresholds, mia_auc_roc