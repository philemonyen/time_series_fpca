import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from fpca import to_fd, basis_smoothing, elastic_registration, fpca
from utils import get_data, trim_ecg, load_synthetic_dataset, get_sr, get_diagnostics
from evaluation import euclidean, abs_cosine_similarity, krzanowski_similarity,fisher_rao

# Hyperparameter setting
n_beats = 8
n_basis = 500
n_components = 4
domain_range = (0, n_beats)

class FPCAOutput:
    def __init__(self, fd, 
            smoothed,
            aligned,
            warping,
            template,
            mean,
            components,
            scores,
            var_ratio):
        self.fd = fd
        self.smoothed = smoothed
        self.aligned = aligned
        self.warping = warping
        self.template = template
        self.mean = mean
        self.components = components
        self.scores = scores
        self.var_ratio = var_ratio

    def plot(self, name, directory):
        save_path = f"images/{directory}"
        path=Path(save_path)
        path.mkdir(parents=True, exist_ok=True)

        self.fd.plot()
        plt.title(f"{name}: Raw ({n_beats} beats)")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.savefig(save_path + '/raw.png')
        plt.close()
        self.smoothed.plot()
        plt.title(f"{name}: Smoothed ({n_beats} beats)")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.savefig(save_path + "/smoothed.png")
        plt.close()
        self.aligned.plot()
        plt.title(f"{name}: Aligned ({n_beats} beats)")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.savefig(save_path + "/aligned.png")
        plt.close()
        self.mean.plot()
        plt.title(f"{name}: FPCA Mean Curve ({n_beats} beats)")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.savefig(save_path + "/mean.png")
        plt.close()
        component_matrix = self.components.data_matrix
        fig, axes = plt.subplots(n_components, 1, figsize=(8, 12))
        xvals = np.linspace(0, n_beats, n_beats*get_sr())
        for i in range(n_components):
            axes[i].plot(xvals, component_matrix[i])
            axes[i].set_title(f"{name}: Eigenfunction {i+1} ({n_beats} beats)")
            axes[i].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(save_path + "/components.png")
        plt.close()


def fpca_pipeline(data, template_):
    fd = to_fd(data, 0, n_beats, "Time (s)", "Voltage (ms)")
    smoothed = basis_smoothing(fd, n_basis, domain_range)
    if template_:
        aligned, warping = elastic_registration(smoothed, template=template_)
        template = None
    else:
        aligned, warping, template = elastic_registration(smoothed)
    mean, components, scores, var_ratio = fpca(aligned, n_components)
    return FPCAOutput(
        fd, 
        smoothed,
        aligned,
        warping,
        template,
        mean,
        components,
        scores,
        var_ratio
    )

def evaluation_pipeline(target_fpca, reference_fpca, name):
    l2_target_reference = euclidean(target_fpca.mean, reference_fpca.mean)
    cos_target_reference = abs_cosine_similarity(target_fpca.components, reference_fpca.components)
    krzanowski_target_reference = krzanowski_similarity(target_fpca.components, reference_fpca.components)
    result = {}
    result['variance_ratios'] = target_fpca.var_ratio.tolist()
    result['variance_sum'] = np.sum(target_fpca.var_ratio)
    result['l2_target_reference'] = l2_target_reference
    result['cos_target_reference'] = cos_target_reference.tolist()
    result['krzanowski_target_reference'] = krzanowski_target_reference.tolist()
    
    with open(f"results/{name}.json", "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    diagnostic = ["NORM"]
    lead = 1

    #### Holdout-Real-Synthetic Experiment
    # Get Data
    real_all = get_data(diagnostic=diagnostic, lead=lead, holdout=False)
    synth_all = load_synthetic_dataset(diagnostic, lead)
    holdout = trim_ecg(real_all[10:20], n_beats)
    real = trim_ecg(real_all[20:30], n_beats)
    synth = trim_ecg(synth_all[10:20], n_beats)

    # Run FPCA
    holdout_fpca = fpca_pipeline(holdout, None)
    real_fpca = fpca_pipeline(real, holdout_fpca.template)
    synth_fpca = fpca_pipeline(synth, holdout_fpca.template)

    # Evaluation
    evaluation_pipeline(synth_fpca, real_fpca, "Synthetic-Real")

    # Plotting
    holdout_fpca.plot("Holdout", "holdout")
    real_fpca.plot("Real", "real")
    synth_fpca.plot("Synthetic", "synthetic")

    #### Holdout-Multi-Class Experiment
    for diag in get_diagnostics():
        diag_all= get_data(diagnostic=[diag], lead=lead, holdout=False)

        # Run FPCA
        diag_partial = trim_ecg(diag_all[10:20], n_beats)
        diag_fpca = fpca_pipeline(diag_partial, holdout_fpca.template)

        # Evaluation
        evaluation_pipeline(diag_fpca, real_fpca, f"{diag}-Real")

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