//
// Created by albor on 14/11/2025.
//

#include "IModelPrior.h"

/**
 * @brief Utility function to perform the NIG posterior calculation and sampling.
 * @param data Data for the cluster (or {y_i} for H_i, or empty for G_0).
 */

NormalConjugateModel::NormalConjugateModel(double mu0, double kappa0, double alpha0, double beta0)
: mu_0(mu0), kappa_0(kappa0), alpha_0(alpha0), beta_0(beta0) {
    std::random_device rd;
    generator.seed(rd());
}

ComponentParameter NormalConjugateModel::sample_nig_posterior(const std::vector<double>& data, double mu0, double kappa0, double alpha0, double beta0) {
    size_t k = data.size();

    // 1. Calculate Sufficient Statistics
    double sum_y = std::accumulate(data.begin(), data.end(), 0.0);
    double y_bar = (k > 0) ? (sum_y / k) : 0.0;
    double sum_sq_dev = 0.0; // S_c
    for (double y_i : data) {
        sum_sq_dev += std::pow(y_i - y_bar, 2);
    }

    // 2. Compute Posterior Hyperparameters
    double kappa_k = kappa0 + k;
    double mu_k = (kappa0 * mu0 + sum_y) / kappa_k;
    double alpha_k = alpha0 + (double)k / 2.0;

    double beta_k = beta0 + sum_sq_dev / 2.0;
    if (k > 0) {
        beta_k += (kappa0 * k * std::pow(y_bar - mu0, 2)) / (2.0 * kappa_k);
    }

    // 3. Sample Sigma^2
    double sigma_sq_new = inv_gamma_rng(alpha_k, beta_k);

    // 4. Sample Mu

    ComponentParameter phi_mu;
    phi_mu.parameters.push_back(mu_k);
    phi_mu.parameters.push_back(sigma_sq_new / kappa_k);
    double mu_new = normal_rng(phi_mu);
    // double mu_new = normal_rng(mu_k, sigma_sq_new / kappa_k);

    // 5. Package into the generic struct (MAPPING)
    ComponentParameter phi;
    phi.parameters.push_back(mu_new);     // values[0] = mu
    phi.parameters.push_back(sigma_sq_new); // values[1] = sigma_sq

    return phi;
}

// G0 (Base Prior)
// G0 (Base Prior)
ComponentParameter NormalConjugateModel::sample_from_G0() {
    // Call the utility function with empty data (k=0) to return G0
    ComponentParameter phi;
    phi = sample_nig_posterior({}, mu_0, kappa_0, alpha_0, beta_0);

    // Ensure the vector is correctly populated by the utility function
    if (phi.parameters.size() != 2) throw std::runtime_error("NIG posterior returned wrong parameter count.");
    return phi;
}

// H_i (Single-Observation Posterior)
// H_i (Single-Observation Posterior)
ComponentParameter NormalConjugateModel::sample_from_H_i(double y_i) {
    // Call the utility function with data = {y_i} (k=1)
    ComponentParameter phi;
    phi = sample_nig_posterior({y_i}, mu_0, kappa_0, alpha_0, beta_0);

    // Assuming sample_nig_posterior handles population of phi.values
    return phi;
}

// Full Cluster Posterior
// Full Cluster Posterior
ComponentParameter NormalConjugateModel::sample_from_posterior(const std::vector<double>& cluster_data) {
    // Call the utility function with the full cluster data
    ComponentParameter phi;
    phi = sample_nig_posterior(cluster_data, mu_0, kappa_0, alpha_0, beta_0);
    return phi;
}

double NormalConjugateModel::likelihood(double y_i, const ComponentParameter& phi) {
    // F is Normal(mu, sigma_sq)
    if (phi.parameters.size() < 2) throw std::runtime_error("ComponentParameter missing parameters for NIG.");
    return normal_pdf(y_i, phi);
}

double NormalConjugateModel::calculate_prior_predictive(double y_i) {
    double df = 2.0 * alpha_0;
    double scale_sq = (beta_0 / alpha_0) * ((kappa_0 + 1.0) / kappa_0); // This is the variance factor for the scaled t-dist.

    // The integral is the PDF of the predictive Student's t-distribution
    return student_t_pdf(y_i, mu_0, df, scale_sq);
}

double NormalConjugateModel::normal_pdf(double x, const ComponentParameter& phi) const {

    if (phi.parameters.size() < 2) throw std::runtime_error("ComponentParameter missing parameters for NIG.");
    const double mu = phi.parameters[0];
    const double sigma_sq = phi.parameters[1];

    if (sigma_sq <= 0.0) {
        // Handle invalid variance gracefully
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double PI = 3.14159265358979323846;

    // Constant term: 1 / sqrt(2 * PI * sigma_sq)
    double constant = 1.0 / std::sqrt(2.0 * PI * sigma_sq);

    // Exponent term: -(x - mu)^2 / (2 * sigma_sq)
    double exponent = -0.5 * std::pow(x - mu, 2) / sigma_sq;

    return constant * std::exp(exponent);
}

double NormalConjugateModel::normal_rng(const ComponentParameter& phi) {

    if (phi.parameters.size() < 2) throw std::runtime_error("ComponentParameter missing parameters for NIG.");
    const double mu = phi.parameters[0];
    const double sigma_sq = phi.parameters[1];

    if (sigma_sq <= 0.0) {
        return mu; // Return mean or throw error for invalid variance
    }

    // std::normal_distribution uses standard deviation (sqrt(sigma_sq))
    std::normal_distribution<double> distribution(mu, std::sqrt(sigma_sq));

    // 'generator' must be defined as a std::mt19937 member in the class scope
    return distribution(generator);
}

double NormalConjugateModel::student_t_pdf(double x, double mu, double df, double sigma_sq) {
    if (df <= 0.0 || sigma_sq <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double PI = 3.14159265358979323846;

    // A = sigma_sq (Scale factor in the NIG predictive formula)

    // 1. Constant term C: Gamma((df+1)/2) / [Gamma(df/2) * sqrt(PI * df * sigma_sq)]
    // We use the log-Gamma function (lgamma) for numerical stability.
    double log_constant_term =
    std::lgamma((df + 1.0) / 2.0) -
    std::lgamma(df / 2.0) -
    0.5 * std::log(PI * df * sigma_sq);

    // 2. Power term: [1 + (x - mu)^2 / (df * sigma_sq)] ^ -(df+1)/2
    double term_in_brackets = 1.0 + std::pow(x - mu, 2) / (df * sigma_sq);
    double power = -(df + 1.0) / 2.0;

    // Log-PDF = log(C) + power * log(term)
    double log_pdf = log_constant_term + power * std::log(term_in_brackets);

    return std::exp(log_pdf);
}

double NormalConjugateModel::inv_gamma_rng(double alpha, double beta) {
    if (alpha <= 0.0 || beta <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // The standard Gamma distribution in C++ uses shape (alpha) and scale (1/beta).
    double scale = 1.0 / beta;

    // 1. Define Gamma distribution
    std::gamma_distribution<double> gamma_dist(alpha, scale);

    // 2. Sample X ~ Gamma(alpha, 1/beta)
    // 'generator' must be defined in scope (e.g., static global or class member)
    double x = gamma_dist(generator);

    if (x == 0.0) {
        return std::numeric_limits<double>::infinity();
    }

    // 3. Return 1/X for the Inverse Gamma variate
    return 1.0 / x;
}

IModelPrior* NormalConjugateModel::clone() const {
    return new NormalConjugateModel(*this);
}


