#include "PoissonGammaModel.h"
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------
PoissonGammaModel::PoissonGammaModel(double alpha0, double beta0)
    : alpha_0(alpha0), beta_0(beta0) {
    // Initialize random generator if not done in base class
    std::random_device rd;
    generator.seed(rd());
}

// ------------------------------------------------------------
// Mathematical Helpers
// ------------------------------------------------------------

// Helper: Log Poisson PDF
// log( lambda^y * exp(-lambda) / y! ) = y*log(lambda) - lambda - lgamma(y+1)
double PoissonGammaModel::log_poisson_pdf(double y, double lambda) const {
    if (lambda <= 0) return -std::numeric_limits<double>::infinity();
    if (y < 0) return -std::numeric_limits<double>::infinity(); // Poisson defined for y >= 0
    return y * std::log(lambda) - lambda - std::lgamma(y + 1.0);
}

// Helper: Gamma RNG
// C++ std::gamma_distribution uses (shape, scale), where scale = 1/rate
double PoissonGammaModel::gamma_rng(double alpha, double beta) {
    if (alpha <= 0 || beta <= 0) return 0.1; // Safety fallback
    std::gamma_distribution<double> dist(alpha, 1.0 / beta);
    return dist(generator);
}

// ------------------------------------------------------------
// Likelihood Implementation
// ------------------------------------------------------------
double PoissonGammaModel::likelihood(double y_i, const ComponentParameter& phi) {
    if (phi.parameters.empty()) throw std::runtime_error("PoissonGammaModel: empty parameters.");
    double lambda = phi.parameters[0];
    // Return density, not log (the calling algorithm uses it directly or takes log later)
    return std::exp(log_poisson_pdf(y_i, lambda));
}

// ------------------------------------------------------------
// Base Prior and Posterior Sampling
// ------------------------------------------------------------

// Sample lambda from G0 prior
ComponentParameter PoissonGammaModel::sample_from_G0() {
    ComponentParameter phi;
    double lambda = gamma_rng(alpha_0, beta_0);
    phi.parameters.push_back(lambda);
    return phi;
}

// Posterior with a single observation: Gamma(alpha0 + y, beta0 + 1)
ComponentParameter PoissonGammaModel::sample_from_H_i(double y_i) {
    ComponentParameter phi;
    double lambda = gamma_rng(alpha_0 + y_i, beta_0 + 1.0);
    phi.parameters.push_back(lambda);
    return phi;
}

// Sample lambda from posterior given a cluster of data
ComponentParameter PoissonGammaModel::sample_from_posterior(const std::vector<double>& cluster_data) {
    double sum_y = 0.0;
    double n = static_cast<double>(cluster_data.size());

    for (double val : cluster_data) {
        sum_y += val;
    }

    // Posterior Gamma(alpha_n, beta_n)
    double alpha_n = alpha_0 + sum_y;
    double beta_n = beta_0 + n;

    ComponentParameter phi;
    double lambda = gamma_rng(alpha_n, beta_n);
    phi.parameters.push_back(lambda);
    return phi;
}

// ------------------------------------------------------------
// Prior Predictive: Negative Binomial
// ------------------------------------------------------------
// P(y) = Gamma(y + alpha) / (Gamma(y+1)Gamma(alpha)) * p^alpha * (1-p)^y
// where p = beta / (beta + 1)
double PoissonGammaModel::calculate_prior_predictive(double y_i) {
    // Use logarithms for numerical stability
    double log_gamma_term = std::lgamma(y_i + alpha_0) - std::lgamma(y_i + 1.0) - std::lgamma(alpha_0);

    // Probability terms: p = beta0 / (beta0 + 1)
    // Logic: beta0^alpha0 / (beta0 + 1)^(alpha0 + y)
    double log_prob_term = alpha_0 * std::log(beta_0) - (alpha_0 + y_i) * std::log(beta_0 + 1.0);

    return std::exp(log_gamma_term + log_prob_term);
}

IModelPrior* PoissonGammaModel::clone() const {
    return new PoissonGammaModel(*this);
}