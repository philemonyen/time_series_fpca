import pickle
import json
from pathlib import Path
from transformation.fda.mfpca import (
    as_multivariate_functional,
    match_beat_length, 
    tune_psplines,
    tune_n_components,
    mfpca_with_param,
)
# from preprocess.ecg_clinical_preprocess import (
#     get_pqrst,  
#     get_clinical_features_from_pqrst,
# )
from metrics.fidelity import *

if __name__ == "__main__":
    results_path = Path("images/sssd_ecg_test")
    results_path.mkdir(parents=True, exist_ok=True)
    result = {}
    sr = 100

    ### Load SSSD-ECG Synthetic Data
    synthetic_ecg = np.load("data/synthetic_final.npy")
    synthetic_label = np.load("data/synthetic_final_labels.npy")
    norm_indices = np.where(synthetic_label[:, 0] == 1)[0]
    synthetic_norm = synthetic_ecg[norm_indices][:, 1, :].squeeze()

    # synthetic_clinical_landmarks = get_pqrst(synthetic_norm, sr=100)
    # synthetic_clinical_features = get_clinical_features_from_pqrst(synthetic_norm, synthetic_clinical_landmarks, sr=sr)

    # with open(f"synthesis_test/synthetic_clinical_features.pkl", "wb") as f:
    #     pickle.dump(synthetic_clinical_features, f)
    # with open(f"synthesis_test/synthetic_clinical_landmarks.pkl", "wb") as f:
    #     pickle.dump(synthetic_clinical_landmarks, f)

    with open(f"synthesis_test/synthetic_clinical_features.pkl", "rb") as f:
        synthetic_clinical_features = pickle.load(f)
    with open(f"synthesis_test/synthetic_clinical_landmarks.pkl", "rb") as f:
        synthetic_clinical_landmarks = pickle.load(f)

    with open("data/validation/real_data.pkl", "rb") as f:
        real_data = pickle.load(f)
    with open("data/validation/clinical/real_clinical_features.pkl", "rb") as f:
        real_clinical_features = pickle.load(f)

    ### Global Clinical Features:Get Real Clinical Features & Run MFPCA
    smoothed_data = [tune_psplines(feature, degree=3) for feature in real_clinical_features]

    # MFPCA
    mfpca_data = as_multivariate_functional(smoothed_data)
    n_components = tune_n_components(mfpca_data)
    print(f"Tuned {n_components} components")
    real_mfpca = mfpca_with_param(mfpca_data, n_components)
    real_scores = real_mfpca.transform(mfpca_data)

    synthetic_clinical_features = match_beat_length(
        synthetic_clinical_features,
        real_clinical_features.shape[-1],
    )
    
    # MFPCA
    smoothed_synthetic_data = [tune_psplines(feature, degree=3) for feature in synthetic_clinical_features]
    mfpca_synthetic_data = as_multivariate_functional(smoothed_synthetic_data)

    synthetic_scores = real_mfpca.transform(mfpca_synthetic_data)
    # flaw_dmap_embedding = real_dmap.transform(flaw_scores)
    # flaw_umap_embedding = real_umap.transform(flaw_scores)

    #### ------------ Evaluation ------------ ####
    ## Clinical Feature Similarity: Wasserstein, Mahalanobis Distance
    # for i, feature in enumerate(features):
    #     clinical_feature_wasserstein_score = wasserstein(real_clinical_features[i], flaw_clinical_features[i])
    #     clinical_feature_mahalanobis_score = sample_wise_mahalanobis(real_clinical_features[i], flaw_clinical_features[i])
    #     result_tracking[scenario][key][feature] = {
    #         "wasserstein_score": clinical_feature_wasserstein_score,
    #         "mahalanobis_score": clinical_feature_mahalanobis_score,
    #     }

    ## FPC Score: Wasserstein Distance, Mahalanobis Distance
    fpca_wasserstein_score = wasserstein(real_scores, synthetic_scores)
    fpca_mahalanobis_score = sample_wise_mahalanobis(real_scores, synthetic_scores)
    fpca_mmd = mmd(real_scores, synthetic_scores)

    # Diffusion Map: JS Divergence, MMD, Spectral Distance
    # dmap_js_divergence = grid_js_divergence(real_dmap_embedding, flaw_dmap_embedding)
    # dmap_mahalanobis_score = sample_wise_mahalanobis(real_dmap_embedding, flaw_dmap_embedding)

    # UMAP: JS Divergence, MMD, Spectral Distance
    # umap_js_divergence = grid_js_divergence(real_umap_embedding, flaw_umap_embedding)
    # umap_mahalanobis_score = sample_wise_mahalanobis(real_umap_embedding, flaw_umap_embedding)

    # Precision, Recall
    precision, recall = precision_recall(real_scores, synthetic_scores)
    # precision_dmap, recall_dmap = precision_recall(real_dmap_embedding, flaw_dmap_embedding)
    # precision_umap, recall_umap = precision_recall(real_umap_embedding, flaw_umap_embedding)

    #### ------------ Save Results ------------ ####
    result = {
        "fpca_wasserstein_score": fpca_wasserstein_score,
        "fpca_mahalanobis_score": fpca_mahalanobis_score,
        "fpca_mmd": fpca_mmd,
        # "dmap_js_divergence": dmap_js_divergence,
        # "dmap_mahalanobis_score": dmap_mahalanobis_score,
        # "umap_js_divergence": umap_js_divergence,
        # "umap_mahalanobis_score": umap_mahalanobis_score,
        "precision": precision,
        "recall": recall,
        # "precision_dmap": precision_dmap,
        # "recall_dmap": recall_dmap,
        # "precision_umap": precision_umap,
        # "recall_umap": recall_umap,
    }

    # Save Results
    out_path = results_path / "sssd_ecg_clinical_fidelity_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results written to {out_path}")