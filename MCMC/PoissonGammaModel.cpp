//
// Created by ricca on 18/12/2025.
//

#include "PoissonGammaModel.h"
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

PoissonGammaModel::PoissonGammaModel(double alpha0, double beta0)
    : alpha_0(alpha0), beta_0(beta0) {
    // Inizializza il generatore casuale se non fatto nella classe base
    std::random_device rd;
    generator.seed(rd());
}

// Helper: Log Poisson PDF
// log( lambda^y * exp(-lambda) / y! ) = y*log(lambda) - lambda - lgamma(y+1)
double PoissonGammaModel::log_poisson_pdf(double y, double lambda) const {
    if (lambda <= 0) return -std::numeric_limits<double>::infinity();
    if (y < 0) return -std::numeric_limits<double>::infinity(); // Poisson definita su y >= 0
    return y * std::log(lambda) - lambda - std::lgamma(y + 1.0);
}

// Helper: Gamma RNG
// C++ std::gamma_distribution usa (shape, scale), dove scale = 1/rate
double PoissonGammaModel::gamma_rng(double alpha, double beta) {
    if (alpha <= 0 || beta <= 0) return 0.1; // fallback di sicurezza
    std::gamma_distribution<double> dist(alpha, 1.0 / beta);
    return dist(generator);
}

// Implementazione Likelihood
double PoissonGammaModel::likelihood(double y_i, const ComponentParameter& phi) {
    if (phi.parameters.empty()) throw std::runtime_error("PoissonGammaModel: parametri vuoti.");
    double lambda = phi.parameters[0];
    // Ritorniamo la densità, non il log (l'algoritmo chiamante la usa direttamente o ne fa il log dopo)
    return std::exp(log_poisson_pdf(y_i, lambda));
}

ComponentParameter PoissonGammaModel::sample_from_G0() {
    ComponentParameter phi;
    double lambda = gamma_rng(alpha_0, beta_0);
    phi.parameters.push_back(lambda);
    return phi;
}

ComponentParameter PoissonGammaModel::sample_from_H_i(double y_i) {
    // Posteriore con un solo dato: Gamma(alpha0 + y, beta0 + 1)
    ComponentParameter phi;
    double lambda = gamma_rng(alpha_0 + y_i, beta_0 + 1.0);
    phi.parameters.push_back(lambda);
    return phi;
}

ComponentParameter PoissonGammaModel::sample_from_posterior(const std::vector<double>& cluster_data) {
    double sum_y = 0.0;
    double n = static_cast<double>(cluster_data.size());

    for (double val : cluster_data) {
        sum_y += val;
    }

    // Posteriore Gamma(alpha_n, beta_n)
    double alpha_n = alpha_0 + sum_y;
    double beta_n = beta_0 + n;

    ComponentParameter phi;
    double lambda = gamma_rng(alpha_n, beta_n);
    phi.parameters.push_back(lambda);
    return phi;
}

// Prior Predictive: Negative Binomial
// P(y) = Gamma(y + alpha) / (Gamma(y+1)Gamma(alpha)) * p^alpha * (1-p)^y
// dove p = beta / (beta + 1)
double PoissonGammaModel::calculate_prior_predictive(double y_i) {
    // Usiamo i logaritmi per stabilità numerica
    double log_gamma_term = std::lgamma(y_i + alpha_0) - std::lgamma(y_i + 1.0) - std::lgamma(alpha_0);

    // Termini di probabilità: p = beta0 / (beta0 + 1)
    // Logica: beta0^alpha0 / (beta0 + 1)^(alpha0 + y)
    double log_prob_term = alpha_0 * std::log(beta_0) - (alpha_0 + y_i) * std::log(beta_0 + 1.0);

    return std::exp(log_gamma_term + log_prob_term);
}

IModelPrior* PoissonGammaModel::clone() const {
    return new PoissonGammaModel(*this);
}