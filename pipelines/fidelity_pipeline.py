import numpy as np
import json
import numpy as np
from fpca import fpca_pipeline, get_ecg_info
from utils import get_data, trim_ecg, load_synthetic_dataset, get_diagnostics
from evaluation import euclidean, abs_cosine_similarity, krzanowski_similarity


def fidelity_evaluation_pipeline(target_fpca, reference_fpca, name):
    l2_target_reference = euclidean(target_fpca.mean, reference_fpca.mean)
    cos_target_reference = abs_cosine_similarity(target_fpca.components, reference_fpca.components)
    krzanowski_target_reference = krzanowski_similarity(target_fpca.components, reference_fpca.components)
    result = {}
    result['variance_ratios'] = target_fpca.var_ratio.tolist()
    result['variance_sum'] = np.sum(target_fpca.var_ratio)
    result['l2_target_reference'] = l2_target_reference
    result['cos_target_reference'] = cos_target_reference.tolist()
    result['krzanowski_target_reference'] = krzanowski_target_reference.tolist()
    result['Score'] = l2_target_reference + (1-krzanowski_target_reference)
    
    with open(f"results/{name}.json", "w") as f:
        json.dump(result, f)

# if __name__ == "__main__":
#     diagnostic = ["NORM"]
#     lead = 1
#     n_data = 1000
#     n_beats, domain_range = get_ecg_info()

#     # Get Data
#     real_all = get_data(diagnostic=diagnostic, lead=lead, holdout=False)
#     synth_all = load_synthetic_dataset(diagnostic, lead)
#     holdout = trim_ecg(real_all[:n_data], n_beats)
#     real = trim_ecg(real_all[n_data:2*n_data], n_beats)
#     synth = trim_ecg(synth_all[:n_data], n_beats)

#     #### Hyperparameter Tuning Visualization ####
#     basis_mult_range = [6, 5, 4, 3]
#     for basis_mult in basis_mult_range:
#         n_basis = int(n_data / basis_mult)
#         print(f"Number of basis functions: {n_basis}")

#         output = fpca_pipeline(real_all, n_basis, None)

#         print(f"    Smoothing parameter: {output.lambda_}")
#         print(f"    Number of eigenfunctions: {output.n_components}")
#         print(f"    Variance ratio: {output.var_ratio}")
#         print(f"    Variance sum: {np.sum(output.var_ratio)}")

#     #### Fidelity Evaluation - Shared Preprocessing ####
    


#     #### Fidelity Evaluation - Independent Preprocessing ####
