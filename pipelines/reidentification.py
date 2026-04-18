import numpy as np
import json
from fpca import get_hyperparameters, fpca_pipeline, fpca_transform_pipeline
from utils import get_data, trim_ecg, load_synthetic_dataset

def distance_to_closest(real_scores, synth_scores):
    min_dist = float('inf')
    for real_score in real_scores:
        for synth_score in synth_scores:
            dist = np.linalg.norm(real_score - synth_score)
            if dist < min_dist:
                min_dist = dist
    return min_dist

def nn_adversarial_accuracy(real_scores, synth_scores):
    """
    Computes nearest neighbor adversarial (1-NN) accuracy: for each point in the joint set,
    is its closest (non-self) neighbor from the same group or the other group?
    Returns: adversarial accuracy (fraction of points whose nearest neighbor is from the *other* group).
    """
    X = np.concatenate([real_scores, synth_scores], axis=0)
    n_real = real_scores.shape[0]
    n_synth = synth_scores.shape[0]

    # Compute all pairwise distances
    dists = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=-1)

    # For each point, ignore self in min search by setting inf on diagonal
    np.fill_diagonal(dists, np.inf)

    # Identify real/synthetic labels for all
    labels = np.array([0]*n_real + [1]*n_synth)  # 0: real, 1: synth

    # For each sample, find index of nearest neighbor
    nn_indices = np.argmin(dists, axis=1)
    nn_labels = labels[nn_indices]

    # For each point: is the nearest neighbor from the *other* group?
    is_adv = labels != nn_labels
    accuracy = np.mean(is_adv[n_real:])
    return accuracy

if __name__ == "__main__":
    diagnostic = ["NORM"]
    lead = 1
    n_data = 1000
    n_beats, n_basis, n_components, domain_range = get_hyperparameters()

    real_all = get_data(diagnostic=diagnostic, lead=lead, holdout=False)
    synth_all = load_synthetic_dataset(diagnostic, lead)
    real = trim_ecg(real_all[n_data:2*n_data], n_beats)
    synth = trim_ecg(synth_all[:n_data], n_beats)

    # Create a pool of  real + synthetic
    pool_size = 200
    rng = np.random.default_rng()  # Use numpy's random generator for reproducibility
    idx_real = rng.choice(real.shape[0], size=pool_size, replace=False)
    idx_synth = rng.choice(synth.shape[0], size=pool_size, replace=False)
    pool_real = real[idx_real]
    pool_synth = synth[idx_synth]
    pool = np.concatenate([pool_real, pool_synth], axis=0)

    # Run FPCA on the pool
    pool_fpca = fpca_pipeline(pool, None)

    # Project the real and synthetic data onto the pool's FPCA space
    real_scores = fpca_transform_pipeline(pool_fpca.fpca_, real)
    synth_scores = fpca_transform_pipeline(pool_fpca.fpca_, synth)

    # Reidentification Evaluation
    dcr = distance_to_closest(real_scores, synth_scores)
    nnaa = nn_adversarial_accuracy(real_scores, synth_scores)
    result = {}
    result['dcr'] = dcr.tolist()
    result['nnaa'] = nnaa
    with open(f"results/reidentification.json", "w") as f:
        json.dump(result, f)
