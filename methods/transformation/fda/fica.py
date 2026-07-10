import numpy as np
from sklearn.decomposition import FastICA
from skfda.representation.grid import FDataGrid

def compute_fica(fpca_scores, fpca_components: FDataGrid, random_state=42):
    """
    Computes Functional ICA by applying FastICA on FPCA scores.

    Parameters:
    -----------
    fpca_scores : array-like, shape (n_samples, n_components)
        The score matrix obtained from FPCA.
    fpca_components : FDataGrid
        The functional principal components from the FPCA model.
        
    Returns:
    --------
    fica_scores : numpy.ndarray, shape (n_samples, n_components)
        The independent FICA scores.
    fics : FDataGrid
        The Functional Independent Components (FICs).
    """
    n_components = fpca_scores.shape[1]
    
    # 1. Fit FastICA on the FPCA scores
    ica = FastICA(n_components=n_components, random_state=random_state)
    fica_scores = ica.fit_transform(fpca_scores)
    
    # The mixing matrix A has shape (n_fpca_components, n_ica_components)
    # ica.mixing_.T gives us the weights to combine FPCs into FICs
    mixing_matrix_T = ica.mixing_.T 
    
    # 2. Extract underlying data from the FPCs
    # Shape: (n_components, n_eval_points, codomain_dim)
    fpc_data = fpca_components.data_matrix 
    
    # 3. Compute FICs: Tensor dot product of (mixing_matrix_T) and (FPC data)
    # We contract the last axis of mixing_matrix_T (n_fpca_comps) 
    # with the first axis of fpc_data (n_fpca_comps)
    fic_data = np.tensordot(mixing_matrix_T, fpc_data, axes=(1, 0))
    
    # 4. Package back into an skfda FDataGrid
    fics = FDataGrid(
        data_matrix=fic_data,
        grid_points=fpca_components.grid_points,
        dataset_name="Functional Independent Components"
    )
    
    return fica_scores, fics