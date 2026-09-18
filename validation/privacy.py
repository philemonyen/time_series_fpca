import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from metrics.privacy import *

def plot_roc_curve(fpr, tpr, auc_score, title):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'MIA Classifier (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig(f"images/validation/privacy/roc_curve_{title.replace(' ', '_').replace(':', '')}.png")
    plt.close()

if __name__ == "__main__":
    with open("data/validation/privacy/real_data.pkl", "rb") as f:
        real_data = pickle.load(f)
    with open("data/validation/privacy/real_components.pkl", "rb") as f:
        real_components = pickle.load(f)
    with open("data/validation/privacy/holdout_train.pkl", "rb") as f:
        holdout_train = pickle.load(f)
    with open("data/validation/privacy/holdout_scores.pkl", "rb") as f:
        holdout_scores = pickle.load(f)
    with open("data/validation/privacy/query_data.pkl", "rb") as f:
        query_data = pickle.load(f)
    with open("data/validation/privacy/query_label.pkl", "rb") as f:
        query_label = pickle.load(f)
    with open("data/validation/privacy/query_scores.pkl", "rb") as f:
        query_scores = pickle.load(f)
    with open("data/validation/privacy/privacy_leaking_dataset.pkl", "rb") as f:
        privacy_leaking_dataset = pickle.load(f)
    
    # 1. Flatten the indices lists to prevent NumPy slicing dimension errors
    seq_len = real_data.shape[1]
    q1, q2, q3 = seq_len // 4, seq_len // 2, seq_len * 3 // 4
    imputation_missing_indices = list(range(q1, q3))
    imputation_known_indices = list(range(0, q1)) + list(range(q3, seq_len))
    forecasting_missing_indices = list(range(q2, seq_len))
    forecasting_known_indices = list(range(0, q2))

    ### Baseline: Adversary only has access to holdout set
    # MIA uses One-Class SVM on raw data
    baseline_result_track = {}
    auc_score, fpr, tpr = mia_baseline_ocsvm(holdout_train, query_data, query_label)
    plot_roc_curve(fpr, tpr, auc_score, title="Baseline MIA")
    
    # Imputation: Compare population mean to true target missing segments using MSE
    baseline_imp_mean = imputation_baseline(holdout_train, imputation_missing_indices)
    baseline_imp_predictions = np.tile(baseline_imp_mean, (query_data.shape[0], 1))
    baseline_imputation_mse = mean_squared_error(query_data[:, imputation_missing_indices], baseline_imp_predictions)

    # Forecasting: Compare population mean to true target missing segments using MSE
    baseline_fcst_mean = forecasting_baseline(holdout_train, forecasting_missing_indices)
    baseline_fcst_predictions = np.tile(baseline_fcst_mean, (query_data.shape[0], 1))
    baseline_forecasting_mse = mean_squared_error(query_data[:, forecasting_missing_indices], baseline_fcst_predictions)
    baseline_result_track = {
        "mia_auc": auc_score,
        "imputation_mse": baseline_imputation_mse,
        "forecasting_mse": baseline_forecasting_mse
    }

    black_box_result_track = {}
    grey_box_result_track = {}

    for key, value in privacy_leaking_dataset.items():
        portion_ratio, leaking_ratio = key
        synthetic_data, synthetic_scores = value
        synthetic_mean = np.mean(synthetic_data, axis=0)

        ### Black Box I: Adversary trains models on the raw synthetic dataset + holdout set
        train_data_raw = np.vstack((holdout_train, synthetic_data))
        train_label = np.concatenate((np.zeros(holdout_train.shape[0]), np.ones(synthetic_data.shape[0])))
        
        # MIA
        auc_score_rf, fpr_rf, tpr_rf = mia_rf(train_data_raw, train_label, query_data, query_label)
        plot_roc_curve(fpr_rf, tpr_rf, auc_score_rf, title=f"Black Box RF MIA {portion_ratio}_{leaking_ratio}")
        
        auc_score_xgb, fpr_xgb, tpr_xgb = mia_xgb(train_data_raw, train_label, query_data, query_label)
        plot_roc_curve(fpr_xgb, tpr_xgb, auc_score_xgb, title=f"Black Box XGB MIA {portion_ratio}_{leaking_ratio}")

        # Imputation (Regression Attack)
        bb_imp_predictions = regression_attack(
            X_train_known=train_data_raw[:, imputation_known_indices],
            X_train_missing=train_data_raw[:, imputation_missing_indices],
            X_test_known=query_data[:, imputation_known_indices]
        )
        black_box_imputation_mse = mean_squared_error(query_data[:, imputation_missing_indices], bb_imp_predictions)

        # Forecasting (Regression Attack)
        bb_fcst_predictions = regression_attack(
            X_train_known=train_data_raw[:, forecasting_known_indices],
            X_train_missing=train_data_raw[:, forecasting_missing_indices],
            X_test_known=query_data[:, forecasting_known_indices]
        )
        black_box_forecasting_mse = mean_squared_error(query_data[:, forecasting_missing_indices], bb_fcst_predictions)

        black_box_result_track[key] = {
            "mia_rf_auc": auc_score_rf,
            "mia_xgb_auc": auc_score_xgb,
            "imputation_mse": black_box_imputation_mse,
            "forecasting_mse": black_box_forecasting_mse
        }

        ### Grey Box: Adversary trains exclusively on the FPC score structural fingerprints
        # Prevent feature dilution: Stack ONLY the scores, not the raw arrays
        train_features_scores = np.vstack((holdout_scores, synthetic_scores))

        # MIA
        gb_auc_score_rf, gb_fpr_rf, gb_tpr_rf = mia_rf(train_features_scores, train_label, query_scores, query_label)
        plot_roc_curve(gb_fpr_rf, gb_tpr_rf, gb_auc_score_rf, title=f"Grey Box RF MIA {portion_ratio}_{leaking_ratio}")
        
        gb_auc_score_xgb, gb_fpr_xgb, gb_tpr_xgb = mia_xgb(train_features_scores, train_label, query_scores, query_label)
        plot_roc_curve(gb_fpr_xgb, gb_tpr_xgb, gb_auc_score_xgb, title=f"Grey Box XGB MIA {portion_ratio}_{leaking_ratio}")

        # Imputation (Iterate over targets to avoid lstsq dimensionality crash)
        gb_imp_predictions = np.zeros((query_data.shape[0], len(imputation_missing_indices)))
        for i in range(query_data.shape[0]):
            gb_imp_predictions[i] = imputation_fpc(
                target_known=query_data[i, imputation_known_indices], 
                known_indices=imputation_known_indices, 
                missing_indices=imputation_missing_indices, 
                synth_fpcs=real_components, 
                synth_mean=synthetic_mean
            )
        grey_box_imputation_mse = mean_squared_error(query_data[:, imputation_missing_indices], gb_imp_predictions)

        # Forecasting (Iterate over targets)
        gb_fcst_predictions = np.zeros((query_data.shape[0], len(forecasting_missing_indices)))
        for i in range(query_data.shape[0]):
            gb_fcst_predictions[i] = forecasting_fpc(
                target_known=query_data[i, forecasting_known_indices], 
                known_indices=forecasting_known_indices, 
                forecast_indices=forecasting_missing_indices, 
                synth_fpcs=real_components, 
                synth_mean=synthetic_mean
            )
        grey_box_forecasting_mse = mean_squared_error(query_data[:, forecasting_missing_indices], gb_fcst_predictions)

        grey_box_result_track[key] = {
            "mia_rf_auc": gb_auc_score_rf,
            "mia_xgb_auc": gb_auc_score_xgb,
            "imputation_mse": grey_box_imputation_mse,
            "forecasting_mse": grey_box_forecasting_mse
        }

    path = Path("images/validation/privacy")
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "baseline_result.json", "w") as f:
        json.dump(baseline_result_track, f, indent=4)
    with open(path / "black_box_result.json", "w") as f:
        json.dump(black_box_result_track, f, indent=4)
    with open(path / "grey_box_result.json", "w") as f:
        json.dump(grey_box_result_track, f, indent=4)