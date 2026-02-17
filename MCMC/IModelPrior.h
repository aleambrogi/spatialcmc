#ifndef IMODELPRIOR_H
#define IMODELPRIOR_H

#include <vector>     // For std::vector
#include <map>        // For std::map
#include <cmath>      // For standard math functions (e.g., log, exp)
#include <numeric>    // For std::accumulate (for summing weights)
#include <algorithm>  // For std::max
#include <random>     // For C++ random number generation (e.g., std::mt19937, std::uniform_real_distribution)
#include <stdexcept>  // For exceptions (e.g., std::runtime_error)
#include <iostream>   // For basic I/O (debugging/logging)
#include <ctime>
#include <limits> // For std::numeric_limits
#include <boost/math/special_functions/gamma.hpp>

struct ComponentParameter {
    // This could hold parameters like (mu, sigma) for a Normal distribution
    std::vector<double> parameters;
};

// ------------------------------------------------------------
// IModelPrior: Interface for Model and Prior Operations
// Abstraction for distribution-specific logic (Likelihood, Sampling)
// ------------------------------------------------------------
class IModelPrior {
protected:
    ComponentParameter current_phi;
public:

    std::mt19937 generator;

    void set_seed(unsigned int s) { generator.seed(s); }

    // Computes the likelihood F(y_i, phi_c)
    virtual double likelihood(double y_i, const ComponentParameter& phi) = 0;

    // Samples a new phi from the base prior G0 (used when a new component is created)
    virtual ComponentParameter sample_from_G0() = 0;

    // Samples a new phi from the posterior H_i (posterior given prior G0 and a single observation y_i)
    virtual ComponentParameter sample_from_H_i(double y_i) = 0;

    // Samples a new phi from the posterior given all data points associated with cluster c
    // (Algorithm 2, last step)
    virtual ComponentParameter sample_from_posterior(const std::vector<double>& cluster_data) = 0;

    // Computes the prior predictive (integral term: ∫ F(y_i, phi) dG_0(phi))
    virtual double calculate_prior_predictive(double y_i) = 0;

    // Allows cloning the prior for parallel implementation
    virtual IModelPrior* clone() const = 0;

    void set_current_phi(const ComponentParameter& phi) { current_phi = phi; }

    virtual ~IModelPrior() = default;
};

// ------------------------------------------------------------
// NormalConjugateModel: Concrete Implementation
// F = Normal(mu, sigma^2), G0 = Normal-Inverse-Gamma(NIG)
// ------------------------------------------------------------
class NormalConjugateModel : public IModelPrior {
private:
    // Fixed NIG hyperparameters for the base prior G0 = NIG(mu_0, kappa_0, alpha_0, beta_0)
    const double mu_0;
    const double kappa_0;
    const double alpha_0;
    const double beta_0;

    // Helper functions (implementation not shown, but necessary)
    [[nodiscard]] double normal_pdf(double x,  const ComponentParameter& phi) const;
    double normal_rng(const ComponentParameter& phi);
    double inv_gamma_rng(double alpha, double beta);
    double student_t_pdf(double x, double mu, double df, double sigma_sq); // Student's t-PDF

    // Core utility function for Bayesian updating of a Normal likelihood with a Normal-Inverse-Gamma (NIG) prior
    ComponentParameter sample_nig_posterior(const std::vector<double>& data, double mu0, double kappa0, double alpha0, double beta0);

public:
    NormalConjugateModel(double mu0, double kappa0, double alpha0, double beta0);

    // ------------------------------------------------------------
    // IModelPrior Interface Implementation
    // ------------------------------------------------------------

    // Computes the likelihood F(y_i, phi_c)
    double likelihood(double y_i, const ComponentParameter& phi) override;

    // Samples a new phi from the base prior G0 (used when a new component is created)
    ComponentParameter sample_from_G0() override;

    // Samples a new phi from the posterior H_i (posterior given prior G0 and a single observation y_i)
    ComponentParameter sample_from_H_i(double y_i) override;

    // Samples a new phi from the posterior given all data points associated with cluster c
    // (Algorithm 2, last step)
    ComponentParameter sample_from_posterior(const std::vector<double>& cluster_data) override;

    [[nodiscard]] IModelPrior* clone() const override;

    // Computes the prior predictive (integral term: ∫ F(y_i, phi) dG_0(phi))
    double calculate_prior_predictive(double y_i) override;
};

#endif //IMODELPRIOR_H
