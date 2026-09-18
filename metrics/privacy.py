import numpy as np
import xgboost as xgb
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, roc_curve

def mia_baseline_ocsvm(X_train, X_test, y_test, nu=0.5, kernel='rbf'):
    clf = OneClassSVM(nu=nu, kernel=kernel)
    clf.fit(X_train)
    y_pred_scores = -clf.decision_function(X_test)
    auc_score = roc_auc_score(y_test, y_pred_scores)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_scores)
    return auc_score, fpr, tpr

def mia_rf(X_train, y_train, X_test, y_test, n_estimators=100, random_state=42):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    return auc_score, fpr, tpr

def mia_xgb(X_train, y_train, X_test, y_test, random_state=42):
    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    return auc_score, fpr, tpr

def imputation_baseline(holdout_train, missing_indices):
    population_mean = np.mean(holdout_train, axis=0)
    return population_mean[missing_indices]

def forecasting_baseline(holdout_train, forecast_indices):
    population_mean = np.mean(holdout_train, axis=0)
    return population_mean[forecast_indices]

def regression_attack(X_train_known, X_train_missing, X_test_known, n_estimators=50, random_state=42):
    """
    Black Box Trajectory Attack using a Random Forest Regressor.
    Learns to predict the missing segment from the known segment based entirely on the training set.
    """
    regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    regressor.fit(X_train_known, X_train_missing)
    return regressor.predict(X_test_known)

def imputation_fpc(target_known, known_indices, missing_indices, synth_fpcs, synth_mean):
    target_centered = target_known - synth_mean[known_indices]
    psi_known = synth_fpcs[:, known_indices]
    xi, _, _, _ = np.linalg.lstsq(psi_known.T, target_centered, rcond=None)
    psi_missing = synth_fpcs[:, missing_indices]
    return synth_mean[missing_indices] + np.dot(xi, psi_missing)

def forecasting_fpc(target_known, known_indices, forecast_indices, synth_fpcs, synth_mean):
    target_centered = target_known - synth_mean[known_indices]
    psi_known = synth_fpcs[:, known_indices]
    xi, _, _, _ = np.linalg.lstsq(psi_known.T, target_centered, rcond=None)
    psi_forecast = synth_fpcs[:, forecast_indices]
    return synth_mean[forecast_indices] + np.dot(xi, psi_forecast)