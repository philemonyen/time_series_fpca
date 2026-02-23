import numpy as np
from sklearn.cross_decomposition import CCA
from scipy.linalg import subspace_angles
from utils import get_sr

def euclidean(fd1, fd2):
    data1 = fd1.data_matrix.squeeze()
    data2 = fd2.data_matrix.squeeze()
    return np.sqrt(np.sum((data1 - data2)**2, axis=0))

def cosine_similarity(fd1, fd2):
    data1 = fd1.data_matrix.squeeze()
    data2 = fd2.data_matrix.squeeze()
    return np.sum(data1 * data2, axis=0) / (np.linalg.norm(data1, axis=0) * np.linalg.norm(data2, axis=0))

def sobolev(fd1, fd2, lambda_=0.1):
    sr = get_sr()
    data1 = fd1.data_matrix.squeeze()
    data2 = fd2.data_matrix.squeeze()
    derivative1 = fd1.derivative(order=1).data_matrix.squeeze()
    derivative2 = fd2.derivative(order=1).data_matrix.squeeze()
    val_diff = np.trapz((data1 - data2)**2, dx=1/sr)
    derivatie_diff = np.trapz((derivative1 - derivative2)**2, dx=1/sr)
    return np.sqrt(val_diff + lambda_*derivatie_diff)

def abs_cosine_similarity(fd1, fd2):
    f1 = fd1.data_matrix.squeeze()
    f2 = fd2.data_matrix.squeeze()
    return np.sum(np.abs(f1 * f2), axis=0) / (np.linalg.norm(f1, axis=0) * np.linalg.norm(f2, axis=0))

def cca(f1, f2, num_points, n_components):

    grid = np.linspace(0, 1, num_points)
    U = f1(grid).squeeze().T
    V = f2(grid).squeeze().T

    cca = CCA(n_components=n_components)
    U_c, V_c = cca.fit_transform(U, V)
    cos_thetas = np.array([np.corrcoef(U_c[:, i], V_c[:, i])[0, 1] for i in range(U_c.shape[1])])

    mean_corr = np.mean(cos_thetas)
    jordan = np.min(cos_thetas)
    procrustes = np.sqrt(np.sum(1 - cos_thetas**2))
    return mean_corr, jordan, procrustes

def fisher_rao(w1, w2):
    derivative1 = np.sqrt(w1.derivative(order=1).data_matrix.squueze())
    derivative2 = np.sqrt(w2.derivative(order=1).data_matrix.squeeze())
    return np.arccos(np.sum(derivative1 * derivative2, axis=0))