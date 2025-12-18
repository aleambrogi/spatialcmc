//
// Created by plakk on 18/12/2025.
//

#ifndef BAYESIANPIPELINEBASELINE_H
#define BAYESIANPIPELINEBASELINE_H

#include <vector>
#include <map>
#include <iostream>
#include "MCMC/IModelPrior.h"

// Tipo di sampler
enum class SamplerType {
    Neal2,
    Neal8
};

// Interfaccia generica per visualizzare pipeline già runnata
class BayesianPipelineBaseline {
public:
    // OUTPUT dal pipeline
    std::vector<std::vector<int>> all_clusterings;
    std::vector<std::map<int, ComponentParameter>> all_parameters;
    std::vector<ComponentParameter> phi_data;
    std::vector<int> best_meta_labels;

    // DP + MCMC + Meta hyperparameters
    double alpha_dp, lambda_appm;
    int burn_in, samples;
    double alpha_meta, delta_in, lambda_in, delta_out, theta_out;

    // Optional prior info
    double mu0=0, kappa0=0, alpha0=0, beta0=0;

    // Sampler usato
    SamplerType sampler_type;

    // Global posteriors
    std::map<int, ComponentParameter> global_posteriors;

public:
    BayesianPipelineBaseline() = default;

    void populate_from_pipeline(
        const std::vector<std::vector<int>>& clusterings,
        const std::vector<std::map<int, ComponentParameter>>& parameters,
        const std::vector<ComponentParameter>& phi,
        const std::vector<int>& meta_labels,
        SamplerType sampler,
        double alpha_dp_, double lambda_appm_,
        int burn_in_, int samples_,
        double alpha_meta_, double delta_in_, double lambda_in_,
        double delta_out_, double theta_out_
    );

    void compute_global_posteriors_from_meta();

    void print_summary() const;
};


#endif //BAYESIANPIPELINEBASELINE_H

