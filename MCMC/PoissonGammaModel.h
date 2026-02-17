#ifndef POISSON_GAMMA_MODEL_H
#define POISSON_GAMMA_MODEL_H

#include "IModelPrior.h"

// ------------------------------------------------------------
// PoissonGammaModel: Conjugate Poisson Likelihood with Gamma Prior
// ------------------------------------------------------------
class PoissonGammaModel : public IModelPrior {
private:
    // Gamma prior hyperparameters (shape, rate)
    double alpha_0; // shape
    double beta_0;  // rate (inverse scale)

    // Helper to calculate Poisson log-density
    double log_poisson_pdf(double y, double lambda) const;

    // Helper to sample from Gamma
    double gamma_rng(double alpha, double beta);

public:
    PoissonGammaModel(double alpha0, double beta0);

    // ------------------------------------------------------------
    // IModelPrior Methods
    // ------------------------------------------------------------

    // Computes P(y | lambda)
    double likelihood(double y_i, const ComponentParameter& phi) override;

    // Samples lambda from prior G0
    ComponentParameter sample_from_G0() override;

    // Samples lambda from posterior given a single observation (H_i)
    ComponentParameter sample_from_H_i(double y_i) override;

    // Samples lambda from posterior given a data cluster (Algorithm 2 step)
    ComponentParameter sample_from_posterior(const std::vector<double>& cluster_data) override;

    // Computes integral P(y_new | G0) -> Negative Binomial
    double calculate_prior_predictive(double y_i) override;

    // Clones the object for parallelism
    IModelPrior* clone() const override;
};

#endif //POISSON_GAMMA_MODEL_H