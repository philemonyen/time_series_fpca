import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skfda import FDataGrid
from preprocess.ecg_clinical_preprocess import get_clinical_features_from_pqrst
from transformation.fda.fpca import fpca_with_param, basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, to_fd
from transformation.fda.mfpca import as_multivariate_functional, match_beat_length, tune_psplines, tune_n_components
from transformation.nonlinear.diffusion_map import DenseDiffusionMap
from transformation.nonlinear.umap import tune_umap
from scenario_engineering.dataset_creation import get_morphology_scenarios, get_temporal_scenarios,get_distributional_scenarios, get_flaw_scales
from metrics.fidelity import *

if __name__ == "__main__":
    diagnostic = "NORM"
    lead = 1
    sr = 100
    with open("data/validation/real_data.pkl", "rb") as f:
        real_data = pickle.load(f)
    with open("data/validation/clinical/real_clinical_landmarks.pkl", "rb") as f:
        real_clinical_landmarks = pickle.load(f)

    ### Global Clinical Features:Get Real Clinical Features & Run MFPCA
    real_clinical_features = get_clinical_features_from_pqrst(real_data, real_clinical_landmarks, sr=sr)
    smoothed_data = [tune_psplines(feature, degree=3) for feature in real_clinical_features]

    # MFPCA
    mfpca_data = as_multivariate_functional(smoothed_data)
    real_scores, real_mfpca = tune_n_components(mfpca_data)

    # Diffusion Map
    real_dmap = DenseDiffusionMap(n_evecs=30, k=20, metric='cosine').fit(real_scores)
    real_dmap_embedding = real_dmap.transform(real_scores)

    # UMAP
    real_umap = tune_umap(real_scores)
    real_umap_embedding = real_umap.transform(real_scores)

    # ### Run Clinical Fidelity Validation on All Scenarios
    scenarios = get_morphology_scenarios() + get_temporal_scenarios() + get_distributional_scenarios()
    result_tracking = {}
    for scenario in scenarios:
        # Result save path
        save_path = f"images/fidelity_val/clinical/{scenario}/"
        path=Path(save_path)
        path.mkdir(parents=True, exist_ok=True)

        result_tracking[scenario] = {}

        with open(f"data/validation/clinical/{scenario}_dataset.pkl", "rb") as f:
            datasets = pickle.load(f)
        
        for key, (flaw_data, flaw_clinical_landmarks) in datasets.items():
            flaw_clinical_features = match_beat_length(
                get_clinical_features_from_pqrst(flaw_data, flaw_clinical_landmarks, sr=sr),
                real_clinical_features.shape[-1],
            )
            
            # MFPCA
            smoothed_flaw_data = [tune_psplines(feature, degree=3) for feature in flaw_clinical_features]
            mfpca_flaw_data = as_multivariate_functional(smoothed_flaw_data)

            flaw_scores = real_mfpca.transform(mfpca_flaw_data)
            flaw_dmap_embedding = real_dmap.transform(flaw_scores)
            flaw_umap_embedding = real_umap.transform(flaw_scores)

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
            fpca_wasserstein_score = wasserstein(real_scores, flaw_scores)
            fpca_mahalanobis_score = sample_wise_mahalanobis(real_scores, flaw_scores)

            # Diffusion Map: JS Divergence, MMD, Spectral Distance
            dmap_js_divergence = grid_js_divergence(real_dmap_embedding, flaw_dmap_embedding)
            dmap_mahalanobis_score = sample_wise_mahalanobis(real_dmap_embedding, flaw_dmap_embedding)

            # UMAP: JS Divergence, MMD, Spectral Distance
            umap_js_divergence = grid_js_divergence(real_umap_embedding, flaw_umap_embedding)
            umap_mahalanobis_score = sample_wise_mahalanobis(real_umap_embedding, flaw_umap_embedding)

            # Precision, Recall
            precision, recall = precision_recall(real_scores, flaw_scores)
            precision_dmap, recall_dmap = precision_recall(real_dmap_embedding, flaw_dmap_embedding)
            precision_umap, recall_umap = precision_recall(real_umap_embedding, flaw_umap_embedding)

            #### ------------ Save Results ------------ ####
            result_tracking[scenario][key] = {
                "fpca_wasserstein_score": fpca_wasserstein_score,
                "fpca_mahalanobis_score": fpca_mahalanobis_score,
                "dmap_js_divergence": dmap_js_divergence,
                "dmap_mahalanobis_score": dmap_mahalanobis_score,
                "umap_js_divergence": umap_js_divergence,
                "umap_mahalanobis_score": umap_mahalanobis_score,
                "precision": precision,
                "recall": recall,
                "precision_dmap": precision_dmap,
                "recall_dmap": recall_dmap,
                "precision_umap": precision_umap,
                "recall_umap": recall_umap,
            }

            plt.scatter(real_umap_embedding[:, 0], real_umap_embedding[:, 1], label="Real")
            plt.scatter(flaw_umap_embedding[:, 0], flaw_umap_embedding[:, 1], label="Flaw")
            plt.title(f"UMAP Embedding: {scenario}, Flaw Scale: {key}")
            plt.legend()
            plt.savefig(save_path + f"UMAP_Embedding_{scenario}_{key}.png")
            plt.close()

    # Save Results
    with open(f"images/fidelity_val/fidelity_val_clinical_result.json", "w") as f:
        json.dump(result_tracking, f)

    # Channel-wise Clinical Features:Get Real Clinical Features & Run FPCA
    features = ["PR", "RR", "QRS", "QT", "R_amp", "QRS_area"]
    fpcas, dmaps, umaps, grids = {}, {}, {}, {}
    real_feature_scores, dmap_embeddings, umap_embeddings = {}, {}, {}
    for i, feature in enumerate(features):
        real_feature = real_clinical_features[i]        
        real_fd = to_fd(real_feature, 0, 1, "time", feature)
        n_timepoints, n_channel, _ = real_fd.data_matrix.shape
        n_basis = int(n_timepoints / 2)
        domain_range = (0, 1)
        n_components = 10
        lambda_ = basis_smoothing_hyperparameter_tuning(real_fd, n_basis, domain_range)
        real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(real_fd, lambda_, n_basis, domain_range)
        real_mean, real_components, real_scores, real_var_ratio, real_fpca_ = fpca_with_param(real_fd_smooth, n_components)
        real_fd_grid = real_fpca_.components_.grid_points[0]
        grids[feature] = real_fd_grid
        real_dmap = DenseDiffusionMap(n_evecs=30, k=20, metric='cosine').fit(real_scores)
        real_umap = tune_umap(real_scores)
        real_feature_scores[feature] = real_scores
        dmap_embeddings[feature] = real_dmap.transform(real_scores)
        umap_embeddings[feature] = real_umap.transform(real_scores)
        fpcas[feature] = real_fpca_
        dmaps[feature] = real_dmap
        umaps[feature] = real_umap
    
    result_tracking_pr, result_tracking_rr, result_tracking_qrs, result_tracking_qt, result_tracking_r_amp, result_tracking_qrs_area = {}, {}, {}, {}, {}, {}
    for scenario in scenarios:
        with open(f"data/validation/clinical/{scenario}_clinical_dataset.pkl", "rb") as f:
            datasets = pickle.load(f)

        save_path = f"images/fidelity_val/clinical/channe_wise/{scenario}/"
        path=Path(save_path)
        path.mkdir(parents=True, exist_ok=True)

        result_tracking_pr[scenario] = {}
        result_tracking_rr[scenario] = {}
        result_tracking_qrs[scenario] = {}
        result_tracking_qt[scenario] = {}
        result_tracking_r_amp[scenario] = {}
        result_tracking_qrs_area[scenario] = {}

        for key, (flaw_data, flaw_clinical_landmarks) in datasets.items():
            flaw_feature = get_clinical_features_from_pqrst(flaw_data, flaw_clinical_landmarks, sr=sr)
            for i, feature in enumerate(features):
                flaw_channel = flaw_feature[i]
                flaw_channel_fd = to_fd(flaw_channel, 0, 1, "time", feature)
                flaw_channel_data = flaw_channel_fd(grids[feature])
                flaw_channel_fd = FDataGrid(data_matrix=flaw_channel_data, grid_points=grids[feature])
                n_timepoints, n_channel, _ = flaw_channel_fd.data_matrix.shape
                n_basis = int(n_timepoints / 2)
                domain_range = (0, 1)
                n_components = 10
                lambda_ = basis_smoothing_hyperparameter_tuning(flaw_channel_fd, n_basis, domain_range)
                flaw_channel_fd_smooth, _, _, _ = basis_smoothing_with_lambda(flaw_channel_fd, lambda_, n_basis, domain_range)

                flaw_channel_scores = fpcas[feature].transform(flaw_channel_fd_smooth)
                flaw_channel_dmap = dmaps[feature].transform(flaw_channel_scores)
                flaw_channel_umap = umaps[feature].transform(flaw_channel_scores)

                #### ------------ Evaluation ------------ ####
                ## FPC Score: Wasserstein Distance, Mahalanobis Distance
                fpca_wasserstein_score = wasserstein(real_feature_scores[feature], flaw_channel_scores)
                fpca_mahalanobis_score = sample_wise_mahalanobis(real_feature_scores[feature], flaw_channel_scores)

                # Diffusion Map: JS Divergence, MMD, Spectral Distance
                dmap_js_divergence = grid_js_divergence(dmap_embeddings[feature], flaw_channel_dmap)
                dmap_mahalanobis_score = sample_wise_mahalanobis(dmap_embeddings[feature], flaw_channel_dmap)

                # UMAP: JS Divergence, MMD, Spectral Distance
                umap_js_divergence = grid_js_divergence(umap_embeddings[feature], flaw_channel_umap)
                umap_mahalanobis_score = sample_wise_mahalanobis(umap_embeddings[feature], flaw_channel_umap)

                # Precision, Recall
                precision, recall = precision_recall(real_feature_scores[feature], flaw_channel_scores)
                precision_dmap, recall_dmap = precision_recall(dmap_embeddings[feature], flaw_channel_dmap)
                precision_umap, recall_umap = precision_recall(umap_embeddings[feature], flaw_channel_umap)

                #### ------------ Save Results ------------ ####
                if feature == "PR":
                    result_tracking_pr[scenario][key] = {
                        "fpca_wasserstein_score": fpca_wasserstein_score,
                        "fpca_mahalanobis_score": fpca_mahalanobis_score,
                        "dmap_js_divergence": dmap_js_divergence,
                        "dmap_mahalanobis_score": dmap_mahalanobis_score,
                        "umap_js_divergence": umap_js_divergence,
                        "umap_mahalanobis_score": umap_mahalanobis_score,
                    }
                elif feature == "RR":
                    result_tracking_rr[scenario][key] = {
                        "fpca_wasserstein_score": fpca_wasserstein_score,
                        "fpca_mahalanobis_score": fpca_mahalanobis_score,
                        "dmap_js_divergence": dmap_js_divergence,
                        "dmap_mahalanobis_score": dmap_mahalanobis_score,
                        "umap_js_divergence": umap_js_divergence,
                        "umap_mahalanobis_score": umap_mahalanobis_score,
                    }
                elif feature == "QRS":
                    result_tracking_qrs[scenario][key] = {
                        "fpca_wasserstein_score": fpca_wasserstein_score,
                        "fpca_mahalanobis_score": fpca_mahalanobis_score,
                        "dmap_js_divergence": dmap_js_divergence,
                        "dmap_mahalanobis_score": dmap_mahalanobis_score,
                        "umap_js_divergence": umap_js_divergence,
                        "umap_mahalanobis_score": umap_mahalanobis_score,
                    }
                elif feature == "QT":
                    result_tracking_qt[scenario][key] = {
                        "fpca_wasserstein_score": fpca_wasserstein_score,
                        "fpca_mahalanobis_score": fpca_mahalanobis_score,
                        "dmap_js_divergence": dmap_js_divergence,
                        "dmap_mahalanobis_score": dmap_mahalanobis_score,
                        "umap_js_divergence": umap_js_divergence,
                        "umap_mahalanobis_score": umap_mahalanobis_score,
                    }
                elif feature == "R_amp":
                    result_tracking_r_amp[scenario][key] = {
                        "fpca_wasserstein_score": fpca_wasserstein_score,
                        "fpca_mahalanobis_score": fpca_mahalanobis_score,
                        "dmap_js_divergence": dmap_js_divergence,
                        "dmap_mahalanobis_score": dmap_mahalanobis_score,
                        "umap_js_divergence": umap_js_divergence,
                        "umap_mahalanobis_score": umap_mahalanobis_score,
                    }
                elif feature == "QRS_area":
                    result_tracking_qrs_area[scenario][key] = {
                        "fpca_wasserstein_score": fpca_wasserstein_score,
                        "fpca_mahalanobis_score": fpca_mahalanobis_score,
                        "dmap_js_divergence": dmap_js_divergence,
                        "dmap_mahalanobis_score": dmap_mahalanobis_score,
                        "umap_js_divergence": umap_js_divergence,
                        "umap_mahalanobis_score": umap_mahalanobis_score,
                    }
    with open(f"images/fidelity_val/clinical/channel_wise/result_tracking_pr.json", "w") as f:
        json.dump(result_tracking_pr, f)
    with open(f"images/fidelity_val/clinical/channel_wise/result_tracking_rr.json", "w") as f:
        json.dump(result_tracking_rr, f)
    with open(f"images/fidelity_val/clinical/channel_wise/result_tracking_qrs.json", "w") as f:
        json.dump(result_tracking_qrs, f)
    with open(f"images/fidelity_val/clinical/channel_wise/result_tracking_qt.json", "w") as f:
        json.dump(result_tracking_qt, f)
    with open(f"images/fidelity_val/clinical/channel_wise/result_tracking_r_amp.json", "w") as f:
        json.dump(result_tracking_r_amp, f)
    with open(f"images/fidelity_val/clinical/channel_wise/result_tracking_qrs_area.json", "w") as f:
        json.dump(result_tracking_qrs_area, f)