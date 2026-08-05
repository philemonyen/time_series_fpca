import numpy as np
from typing import Union, Dict
from scipy.linalg import sqrtm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import pdist, squareform, jensenshannon
from scipy.linalg import eigh
from statsmodels.tsa.stattools import acf
from pydiffmap import diffusion_map as dm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import similaritymeasures


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


###### ------ Raw Time Sequence Metrics ------ ######
### Raw time sequence discriminator 
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (hn, cn) = self.lstm(x)
        # Use the last time-step output for classification
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return self.sigmoid(out)

def raw_data_discriminative_score(real_data, synthetic_data, epochs=20, batch_size=64, lr=1e-3):
    """
    real_data: np.ndarray of shape (N_real, seq_len, num_features)
    synthetic_data: np.ndarray of shape (N_synth, seq_len, num_features)
    
    Discriminative Score is defined as |Accuracy - 0.5|. Lower is better.
    """
    # Create Labels: 1 for Real, 0 for Synthetic
    y_real = np.ones(len(real_data), dtype=np.float32)
    y_synth = np.zeros(len(synthetic_data), dtype=np.float32)

    # Train/Test Split for Real and Synthetic independently
    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(real_data, y_real, test_size=0.3, random_state=42)
    X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(synthetic_data, y_synth, test_size=0.3, random_state=42)

    # Combine Training and Testing subsets
    X_train = np.concatenate([X_r_train, X_s_train], axis=0)
    y_train = np.concatenate([y_r_train, y_s_train], axis=0)
    X_test = np.concatenate([X_r_test, X_s_test], axis=0)
    y_test = np.concatenate([y_r_test, y_s_test], axis=0)

    # Convert to PyTorch Tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len, input_dim = real_data.shape[1], real_data.shape[2]
    
    model = Discriminator(input_dim=input_dim, hidden_dim=64, num_layers=2).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            preds = model(bx)
            all_preds.extend((preds.cpu().numpy() > 0.5).astype(int))
            all_targets.extend(by.numpy())

    acc = accuracy_score(all_targets, all_preds)
    disc_score = abs(acc - 0.5)

    return disc_score

### Autocorrelation score
def autocorrelation_score(real_data, synthetic_data, max_lag=24):
    """
    Computes the Autocorrelation Score (ACS) between real and synthetic time series.
    
    Parameters:
    - real_data: np.ndarray of shape (num_samples, seq_len, num_features)
    - synthetic_data: np.ndarray of shape (num_samples, seq_len, num_features)
    - max_lag: int, the maximum time lag to compute the autocorrelation for.
    
    Returns:
    - acs: float, the Mean Absolute Error between the real and synthetic ACF profiles.
           A lower score indicates better preservation of temporal dynamics.
    """
    # Ensure inputs have the same shape
    assert real_data.shape == synthetic_data.shape, "Data shapes must match."
    
    N, T, F = real_data.shape
    
    # Cap max_lag if the sequence length is shorter than the requested lag
    max_lag = min(max_lag, T - 1)
    
    real_acf_profiles = np.zeros((F, max_lag + 1))
    synth_acf_profiles = np.zeros((F, max_lag + 1))
    
    # Iterate through each feature (channel)
    for f in range(F):
        
        # 1. Compute ACF for every sample in the real dataset and average them
        real_acfs = [
            acf(real_data[i, :, f], nlags=max_lag, fft=True) 
            for i in range(N)
        ]
        real_acf_profiles[f] = np.mean(real_acfs, axis=0)
        
        # 2. Compute ACF for every sample in the synthetic dataset and average them
        synth_acfs = [
            acf(synthetic_data[i, :, f], nlags=max_lag, fft=True) 
            for i in range(N)
        ]
        synth_acf_profiles[f] = np.mean(synth_acfs, axis=0)
        
    # 3. Compute the Mean Absolute Error (MAE) across all features and lags
    # We exclude lag 0 because autocorrelation at lag 0 is always exactly 1.0
    acs_score = np.mean(np.abs(real_acf_profiles[:, 1:] - synth_acf_profiles[:, 1:]))
    
    return acs_score

def dtw_score(real_data, synthetic_data, num_samples=100):
    """
    Computes the Expected DTW Distance between real and synthetic time series.
    
    Parameters:
    - real_data: np.ndarray of shape (N, T, F)
    - synthetic_data: np.ndarray of shape (N, T, F)
    - num_samples: int, number of random pairs to evaluate (to avoid O(N^2) bottleneck)
    
    Returns:
    - avg_dtw: float, average DTW distance. Lower is better.
    """
    N_real = len(real_data)
    N_synth = len(synthetic_data)
    
    # Randomly sample indices to create pairs
    idx_real = np.random.choice(N_real, size=num_samples, replace=False)
    idx_synth = np.random.choice(N_synth, size=num_samples, replace=False)
    
    total_dtw = 0.0
    
    for r_idx, s_idx in zip(idx_real, idx_synth):
        # Extract the sequence: shape (T, F)
        seq_real = real_data[r_idx]
        seq_synth = synthetic_data[s_idx]
        
        # fastdtw supports multidimensional sequences automatically
        distance, path = fastdtw(seq_real, seq_synth, dist=euclidean)
        total_dtw += distance
        
    avg_dtw = total_dtw / num_samples
    return avg_dtw

def frechet_score(real_data, synthetic_data, num_samples=100):
    """
    Computes the Expected Fréchet Distance between real and synthetic time series.
    
    Parameters:
    - real_data: np.ndarray of shape (N, T, F)
    - synthetic_data: np.ndarray of shape (N, T, F)
    - num_samples: int, number of random pairs to evaluate
    
    Returns:
    - avg_frechet: float, average Fréchet distance. Lower is better.
    """
    N_real = len(real_data)
    N_synth = len(synthetic_data)
    
    # Randomly sample indices to create pairs
    idx_real = np.random.choice(N_real, size=num_samples, replace=False)
    idx_synth = np.random.choice(N_synth, size=num_samples, replace=False)
    
    total_frechet = 0.0
    
    for r_idx, s_idx in zip(idx_real, idx_synth):
        # Extract the sequence: shape (T, F)
        seq_real = real_data[r_idx]
        seq_synth = synthetic_data[s_idx]
        
        # similaritymeasures.frechet_dist handles 2D arrays natively
        distance = similaritymeasures.frechet_dist(seq_real, seq_synth)
        total_frechet += distance
        
    avg_frechet = total_frechet / num_samples
    return avg_frechet


###### ------ Feature Metrics ------ ######
### Feature discriminator 
def feature_discriminative_score(real, synthetic):
    """
    Train a random forest classifier to distinguish real and synthetic data.
    Args:
        real: (N, dim) numpy array of real features.
        synthetic: (M, dim) numpy array of synthetic features.
    Returns:
        Discriminative Score is defined as |Accuracy - 0.5|. Lower is better.
    """
    y_real = np.ones(len(real))
    y_synth = np.zeros(len(synthetic))

    # Train/test split
    X_r_tr, X_r_te, y_r_tr, y_r_te = train_test_split(real, y_real, test_size=0.3, random_state=42)
    X_s_tr, X_s_te, y_s_tr, y_s_te = train_test_split(synthetic, y_synth, test_size=0.3, random_state=42)

    X_train = np.vstack([X_r_tr, X_s_tr])
    y_train = np.hstack([y_r_tr, y_s_tr])
    X_test = np.vstack([X_r_te, X_s_te])
    y_test = np.hstack([y_r_te, y_s_te])

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return abs(acc - 0.5)

def mmd(X, Y, gamma=None):
    """
    Computes the True Maximum Mean Discrepancy using an RBF kernel.
    Captures differences in means, variances, and non-linear shapes.
    """
    # If gamma is None, use the median heuristic or default to 1 / n_features
    if gamma is None:
        gamma = 1.0 / X.shape[1]
        
    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)
    
    # MMD^2 formula
    mmd_squared = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    
    # Relu to prevent tiny negative numbers due to floating point precision
    return np.sqrt(np.max([mmd_squared, 0.0]))

def wasserstein(X, Y):
    """
    Computes the Multidimensional Fréchet Distance (2-Wasserstein distance 
    assuming Gaussian distributions) between two matrices.
    """
    mu_X, mu_Y = np.mean(X, axis=0), np.mean(Y, axis=0)
    sigma_X, sigma_Y = np.cov(X, rowvar=False), np.cov(Y, rowvar=False)
    
    # Difference between means
    diff = mu_X - mu_Y
    mean_term = diff.dot(diff)
    
    # Product of covariances
    covmean, _ = sqrtm(sigma_X.dot(sigma_Y), disp=False)
    
    # Handle imaginary numbers from numerical instability
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    covariance_term = np.trace(sigma_X + sigma_Y - 2 * covmean)
    
    return np.sqrt(mean_term + covariance_term)

### Shared UMAP & Diffusion Map
def grid_js_divergence(real_coords: np.ndarray, 
                                 synthetic_coords: np.ndarray, 
                                 bins: int = 50, 
                                 epsilon: float = 1e-10,
                                 max_dims: int = 2) -> float:
    """
    Computes the Jensen-Shannon Divergence between two sets of coordinates in a shared embedding space
    by constructing an n-dimensional probability density grid.
    
    Parameters:
    -----------
    real_coords : np.ndarray
        Shape (N_real, n_dimensions). E.g., shared 2D UMAP coordinates for real data.
    synthetic_coords : np.ndarray
        Shape (N_synth, n_dimensions). E.g., shared 2D UMAP coordinates for synthetic data.
    bins : int
        Number of grid bins per dimension.
    epsilon : float
        Small smoothing factor to prevent division by zero or log(0).
    max_dims : int
        Number of leading coordinates to histogram. Grid size grows as bins**max_dims, so
        high-dimensional embeddings (e.g. diffusion maps with n_evecs=30) must be truncated.
        
    Returns:
    --------
    float : The JS Divergence bounded between 0.0 (identical) and 1.0 (completely disjoint in log base 2).
    """

    if real_coords.shape[1] > max_dims:
        real_coords = real_coords[:, :max_dims]
        synthetic_coords = synthetic_coords[:, :max_dims]

    # 1. Determine global bounding box across both datasets so grid edges align perfectly
    combined = np.vstack([real_coords, synthetic_coords])
    min_edges = np.min(combined, axis=0)
    max_edges = np.max(combined, axis=0)
    
    # Create bin edges for each dimension
    grid_edges = [np.linspace(min_edges[i], max_edges[i], bins + 1) for i in range(combined.shape[1])]
    
    # 2. Compute N-dimensional histograms
    real_hist, _ = np.histogramdd(real_coords, bins=grid_edges)
    synth_hist, _ = np.histogramdd(synthetic_coords, bins=grid_edges)
    
    # 3. Flatten and apply Laplace/Epsilon smoothing
    real_pdf = real_hist.flatten() + epsilon
    synth_pdf = synth_hist.flatten() + epsilon
    
    # 4. Normalize to valid probability distributions (summing to 1)
    real_pdf /= np.sum(real_pdf)
    synth_pdf /= np.sum(synth_pdf)
    
    # 5. Compute JS Divergence
    # Note: scipy's jensenshannon computes the JS Distance (square root of divergence).
    # We square it and use base=2 so the final divergence is strictly bounded in [0, 1].
    js_distance = jensenshannon(real_pdf, synth_pdf, base=2.0)
    js_divergence = float(js_distance ** 2)
    
    return js_divergence

def local_mixing_ratio(real_iso, synth_iso):
    """
    Computes the local mixing ratio of real data in synthetic neighborhoods.
    Args:
        real_iso: (N, dim) numpy array of real data embeddings/scores.
        synth_iso: (M, dim) numpy array of synthetic data embeddings/scores.
    Returns:
        The average ratio of real data in synthetic neighborhoods.
    """
    X_combined = np.vstack((real_iso, synth_iso))
    y_combined = np.concatenate([np.zeros(len(real_iso)), np.ones(len(synth_iso))])

    baseline = len(real_iso) / (len(real_iso) + len(synth_iso))

    # Fit kNN on the joint space
    ratios = [] 
    ks = []
    for k in [5, 10, 30, 50, 100]:
        if k > np.sqrt(len(X_combined)):
            break
        ks.append(k)

        nbrs = NearestNeighbors(n_neighbors=k).fit(X_combined)
        distances, indices = nbrs.kneighbors(X_combined)

        # Isolate synthetic points to check their neighborhoods
        synth_indices = np.where(y_combined == 1)[0]
        neighbor_labels = y_combined[indices[synth_indices, 1:]] # Skip the first index (itself)

        # Calculate ratio of real data in synthetic neighborhoods
        real_ratios = np.mean(neighbor_labels == 0, axis=1)
        ratios.append(np.mean(real_ratios))
    return ratios, baseline, ks

def get_diffusion_eigenvalues(data, k=10, sigma=None):
    """
    Constructs the diffusion operator and returns its top k eigenvalues.
    """
    # 1. Compute pairwise squared Euclidean distances
    sq_dists = squareform(pdist(data, metric='sqeuclidean'))
    
    # 2. Estimate kernel bandwidth (sigma) using the median distance heuristic if not provided
    if sigma is None:
        sigma = np.median(sq_dists)
        if sigma == 0.0:
            sigma = 1e-5
            
    # 3. Compute the Gaussian (Heat) affinity matrix W
    W = np.exp(-sq_dists / (2 * sigma))
    
    # 4. Compute the symmetric normalized diffusion matrix
    # D^(-1/2) * W * D^(-1/2) shares the same eigenvalues as the random walk matrix D^(-1) * W
    d = np.sum(W, axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    
    # Matrix multiplication for D^(-1/2) * W * D^(-1/2)
    M_sym = W * np.outer(d_inv_sqrt, d_inv_sqrt)
    
    # 5. Extract the top k eigenvalues
    # eigh is optimized for symmetric matrices. It returns eigenvalues in ascending order.
    # We take the last k eigenvalues (which are the largest) and reverse them.
    eigenvalues, _ = eigh(M_sym, subset_by_index=[len(data)-k, len(data)-1])
    top_k_evals = eigenvalues[::-1]
    
    return top_k_evals

def compute_spectral_distance(real_data, synth_data, k=10):
    """
    Computes the Spectral Distance between the diffusion operators of real and synthetic data.
    
    Parameters:
    - real_data: np.ndarray (e.g., FPC score matrix or pre-embedding features)
    - synth_data: np.ndarray 
    - k: int, number of top eigenvalues to compare.
    
    Returns:
    - spectral_dist: float, Mean Squared Error between the top k eigenvalues.
    """
    # Ensure k does not exceed the number of samples
    k = min(k, len(real_data), len(synth_data))
    
    # Calculate eigenvalues
    real_evals = get_diffusion_eigenvalues(real_data, k=k)
    synth_evals = get_diffusion_eigenvalues(synth_data, k=k)
    
    # Compute the distance (MSE) between the eigenvalue spectra
    spectral_dist = np.mean((real_evals - synth_evals) ** 2)
    
    return spectral_dist