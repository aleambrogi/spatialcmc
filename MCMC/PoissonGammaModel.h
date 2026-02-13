//
// Created by ricca on 18/12/2025.
//

#ifndef POISSON_GAMMA_MODEL_H
#define POISSON_GAMMA_MODEL_H

#include "IModelPrior.h"

class PoissonGammaModel : public IModelPrior {
private:
    // Iperparametri della prior Gamma(shape, rate)
    double alpha_0; // shape
    double beta_0;  // rate (inverse scale)

    // Helper per calcolare la log-density della Poisson
    double log_poisson_pdf(double y, double lambda) const;

    // Helper per campionare dalla Gamma
    double gamma_rng(double alpha, double beta);

public:
    PoissonGammaModel(double alpha0, double beta0);

    // --- Metodi IModelPrior ---

    // Calcola P(y | lambda)
    double likelihood(double y_i, const ComponentParameter& phi) override;

    // Campiona lambda dalla prior G0
    ComponentParameter sample_from_G0() override;

    // Campiona lambda dalla posteriore data una singola osservazione (H_i)
    ComponentParameter sample_from_H_i(double y_i) override;

    // Campiona lambda dalla posteriore dato un cluster di dati (Algorithm 2 step)
    ComponentParameter sample_from_posterior(const std::vector<double>& cluster_data) override;

    // Calcola l'integrale P(y_new | G0) -> Negative Binomial
    double calculate_prior_predictive(double y_i) override;

    // Clona l'oggetto per il parallelismo
    IModelPrior* clone() const override;
};

#endif //POISSON_GAMMA_MODEL_H