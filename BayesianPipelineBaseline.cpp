//
// Created by plakk on 18/12/2025.
//


#include "BayesianPipelineBaseline.h"
#include <algorithm>
#include <iostream>

void BayesianPipelineBaseline::populate_from_pipeline(
    const std::vector<std::vector<int>>& clusterings,
    const std::vector<std::map<int, ComponentParameter>>& parameters,
    const std::vector<ComponentParameter>& phi,
    const std::vector<int>& meta_labels,
    SamplerType sampler,
    double alpha_dp_, double lambda_appm_,
    int burn_in_, int samples_,
    double alpha_meta_, double delta_in_, double lambda_in_,
    double delta_out_, double theta_out_,
    int best_meta_iteration_,
    bool has_conjugate_prior_,
    double mu0_, double kappa0_,
    double alpha0_, double beta0_
) {
    all_clusterings = clusterings;
    all_parameters = parameters;
    phi_data = phi;
    best_meta_labels = meta_labels;
    sampler_type = sampler;

    alpha_dp = alpha_dp_;
    lambda_appm = lambda_appm_;
    burn_in = burn_in_;
    samples = samples_;

    alpha_meta = alpha_meta_;
    delta_in = delta_in_;
    lambda_in = lambda_in_;
    delta_out = delta_out_;
    theta_out = theta_out_;

    best_meta_iteration = best_meta_iteration_;

    has_conjugate_prior = has_conjugate_prior_;
    if (has_conjugate_prior) {
        mu0 = mu0_;
        kappa0 = kappa0_;
        alpha0 = alpha0_;
        beta0 = beta0_;
    }

    compute_global_posteriors_from_meta();
}

void BayesianPipelineBaseline::compute_global_posteriors_from_meta() {
    global_posteriors.clear();
    for (size_t i = 0; i < phi_data.size(); ++i) {
        int meta = best_meta_labels[i];
        global_posteriors[meta] = phi_data[i]; // lasciato invariato
    }
}

void BayesianPipelineBaseline::print_summary() const {
    std::cout << "\n================== BAYESIAN PIPELINE SUMMARY ==================\n";

    std::cout << "Sampler type: "
              << (sampler_type == SamplerType::Neal2
                      ? "Neal2 (conjugate)"
                      : "Neal8 (non-conjugate / Beta)")
              << "\n";

    if (has_conjugate_prior) {
        std::cout << "Prior (conjugate NIG): mu0=" << mu0
                  << " kappa0=" << kappa0
                  << " alpha0=" << alpha0
                  << " beta0=" << beta0 << "\n";
    } else {
        std::cout << "Prior: non-conjugate / not available\n";
    }

    std::cout << "DP hyperparameters: alpha=" << alpha_dp
              << " lambda=" << lambda_appm << "\n";

    std::cout << "MCMC: burn-in=" << burn_in
              << " samples=" << samples << "\n";

    std::cout << "Meta-clustering: alpha=" << alpha_meta
              << " delta_in=" << delta_in
              << " lambda_in=" << lambda_in
              << " delta_out=" << delta_out
              << " theta_out=" << theta_out << "\n";

    std::cout << "Best meta-clustering iteration (Binder loss): "
              << best_meta_iteration << "\n";

    std::cout << "Number of chains: " << all_parameters.size() << "\n";
    std::cout << "Number of components: " << phi_data.size() << "\n";
    std::cout << "Best meta-labels size: " << best_meta_labels.size() << "\n";

    std::cout << "\nGlobal posteriors:\n";
    for (const auto& kv : global_posteriors) {
        std::cout << "Cluster " << kv.first
                  << " | param[0]=" << kv.second.parameters[0]
                  << " param[1]=" << kv.second.parameters[1] << "\n";
    }

    std::cout << "================================================================\n";
}
