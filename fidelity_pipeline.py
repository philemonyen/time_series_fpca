import numpy as np
import json
import numpy as np
from fpca import get_hyperparameters, fpca_pipeline
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

if __name__ == "__main__":
    diagnostic = ["NORM"]
    lead = 1
    n_data = 1000
    n_beats, n_basis, n_components, domain_range = get_hyperparameters()

    #### Holdout-Real-Synthetic Experiment
    # Get Data
    real_all = get_data(diagnostic=diagnostic, lead=lead, holdout=False)
    synth_all = load_synthetic_dataset(diagnostic, lead)
    holdout = trim_ecg(real_all[:n_data], n_beats)
    real = trim_ecg(real_all[n_data:2*n_data], n_beats)
    synth = trim_ecg(synth_all[:n_data], n_beats)

    # Run FPCA
    holdout_fpca = fpca_pipeline(holdout, None)
    real_fpca = fpca_pipeline(real, holdout_fpca.template)
    synth_fpca = fpca_pipeline(synth, holdout_fpca.template)

    # Evaluation
    fidelity_evaluation_pipeline(synth_fpca, real_fpca, "Synthetic-Real")

    # Plotting
    holdout_fpca.plot("Holdout", "holdout")
    real_fpca.plot("Real", "real")
    synth_fpca.plot("Synthetic", "synthetic")

    #### Holdout-Multi-Class Experiment
    for diag in get_diagnostics():
        diag_all= get_data(diagnostic=[diag], lead=lead, holdout=False)

        # Run FPCA
        diag_partial = trim_ecg(diag_all[:n_data], n_beats)
        diag_fpca = fpca_pipeline(diag_partial, holdout_fpca.template)

        # Evaluation
        fidelity_evaluation_pipeline(diag_fpca, real_fpca, f"{diag}-Real")

        # Plotting 
        diag_fpca.plot(diag, diag.lower())


    # # Absolute cosine similarity between corresponding eigenfunctions
    # abs_cos = abs_cosine_similarity(holdout_fpca.components, real_fpca.components)
    # for i in range(len(abs_cos)):
    #     print(f"Absolute Cosine Similarity (NORM2 vs NORM) - Eigenfunction {i+1}: ", abs_cos[i])
    # print("\n")
    # abs_cos = abs_cosine_similarity(holdout_fpca.components, MI_output.components)
    # for i in range(len(abs_cos)):
    #     print(f"Absolute Cosine Similarity (MI vs NORM) - Eigenfunction {i+1}: ", abs_cos[i])