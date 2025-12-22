//
// Created by albor on 17/12/2025.
//

#ifndef BETA_NONCONJUGATE_MODEL_H
#define BETA_NONCONJUGATE_MODEL_H

#include "IModelPrior.h"

class BetaNonConjugateModel : public IModelPrior {
private:
    // Independent Gamma priors on alpha and beta:
    // alpha ~ Gamma(shape_a, rate_a), beta ~ Gamma(shape_b, rate_b)
    double shape_a, rate_a;
    double shape_b, rate_b;

    // MH proposal std dev in log-space
    double prop_sd_log_a;
    double prop_sd_log_b;

    // MH steps per call to sample_from_posterior()
    int mh_steps;

    // Numerics: clamp data to (eps, 1-eps)
    double eps;

    // Log posterior for (a,b) given data (excluding constants)
    double log_posterior(double a, double b, const std::vector<double>& data) const;

    // Log Beta(a,b) pdf for y
    double log_beta_pdf(double y, double a, double b) const;

    // Log Gamma(shape, rate) pdf (excluding constants is fine for MH ratios)
    static double log_gamma_prior(double x, double shape, double rate);

public:
    BetaNonConjugateModel(double shape_a_, double rate_a_,
                          double shape_b_, double rate_b_,
                          double prop_sd_log_a_ = 0.2,
                          double prop_sd_log_b_ = 0.2,
                          int mh_steps_ = 2,
                          double eps_ = 1e-12);

    double likelihood(double y_i, const ComponentParameter& phi) override;

    ComponentParameter sample_from_G0() override;
    ComponentParameter sample_from_H_i(double y_i) override;

    // Neal-8 uses this as a transition kernel (MH)
    ComponentParameter sample_from_posterior(const std::vector<double>& cluster_data) override;

    // Not needed for Neal-8; you can throw to avoid accidental use
    double calculate_prior_predictive(double y_i) override;

    IModelPrior* clone() const override;
};

#endif
