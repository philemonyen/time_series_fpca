import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from fpca import to_fd, basis_smoothing, elastic_registration, fpca
from utils import get_data, trim_ecg
from evaluation import euclidean, cosine_similarity, sobolev, abs_cosine_similarity, cca, fisher_rao

# Hyperparameter setting
n_beats = 5
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
        for i in range(n_components):
            axes[i].plot(component_matrix[i])
            axes[i].set_title(f"{name}: Component {i+1} ({n_beats} beats)")
            axes[i].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(save_path + "/components.png")
        plt.close()


def pipeline(data, template_):
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

if __name__ == "__main__":
    diagnostic = ["NORM"]
    lead = 1
    real, holdout = get_data(diagnostic=diagnostic, lead=lead, holdout=True)

    # Run FPCA on Holdout
    norm = trim_ecg(holdout[:10], n_beats)
    norm_output = pipeline(norm, None)

    diagnostic2 = ["MI"]
    real2, holdout2 = get_data(diagnostic=diagnostic2, lead=lead, holdout=True)

    # Run FPCA on Holdout
    holdout2 = trim_ecg(holdout2[10:20], n_beats)
    MI_output = pipeline(holdout2, None)


    #### Evaluation ####
    # Mean function
    print(f"Variance Ratios (Diagnostics {diagnostic}, Lead {lead}): {norm_output.var_ratio}. Variance Sum: {np.sum(norm_output.var_ratio)}")
    print(f"Variance Ratios (Diagnostics {diagnostic2}, Lead {lead}): {MI_output.var_ratio}. Variance Sum: {np.sum(MI_output.var_ratio)}")
    l2 = euclidean(MI_output.mean, norm_output.mean)
    cos = cosine_similarity(MI_output.mean, norm_output.mean)
    sd = sobolev(MI_output.mean, norm_output.mean)
    print("Euclidean: ", l2)
    print("Cosine Similarity: ", cos)
    print("Sobolev: ", sd)
    print("\n")


    # # Absolute cosine similarity between corresponding eigenfunctions
    # abs_cos_sim = abs_cosine_similarity(train_components, holdout_components)
    # for i in range(len(abs_cos_sim)):
    #     print(f"Eigenfunction Absolute Cosine Similarity - Eigenfunction {i+1}: ", abs_cos_sim[i])


    #### Plotting ####
    norm_output.plot("NORM", "norm")
    MI_output.plot("MI", "mi")