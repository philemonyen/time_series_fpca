import numpy as np
from skfda.representation.grid import FDataGrid

def compute_principal_curve(score_matrix, first_pc: FDataGrid, mean_curve: FDataGrid = None) -> FDataGrid:
    """
    Constructs the principal curve representation using the first principal component.

    Parameters:
    -----------
    score_matrix : array-like, shape (n_samples, n_components) or (n_samples,)
        The score matrix from FPCA. The function will extract the first column 
        (the scores corresponding to the 1st PC).
    first_pc : FDataGrid
        An FDataGrid containing exactly one sample representing the first principal component.
    mean_curve : FDataGrid, optional
        An FDataGrid representing the mean of the functional data. 
        If None, the function assumes a zero-mean sequence.

    Returns:
    --------
    FDataGrid
        An FDataGrid containing the reconstructed curves along the first principal component.
    """
    # 1. Isolate the scores for the first principal component
    scores = np.asarray(score_matrix)
    if scores.ndim > 1:
        scores_1st_pc = scores[:, 0]
    else:
        scores_1st_pc = scores

    # 2. Extract the underlying data arrays
    # first_pc.data_matrix shape is (1, n_eval_points, codomain_dimension)
    pc_data = first_pc.data_matrix[0] 
    
    # 3. Compute the functional linear combination: Score * PC1
    # Broadcasting: (n_samples, 1, 1) * (n_eval_points, codomain_dim) -> (n_samples, n_eval_points, codomain_dim)
    curve_data = scores_1st_pc[:, np.newaxis, np.newaxis] * pc_data[np.newaxis, ...]

    # 4. Add the mean curve to shift the principal curve back to the original data space
    if mean_curve is not None:
        mean_data = mean_curve.data_matrix[0]
        curve_data += mean_data[np.newaxis, ...]

    # 5. Package back into an skfda FDataGrid
    return FDataGrid(
        data_matrix=curve_data,
        grid_points=first_pc.grid_points,
        dataset_name="1st PC Principal Curve"
    )