#include "BetaNonConjugateModel.h"
#include <random>
#include <cmath>
#include <stdexcept>
#include <limits>

// ------------------------------------------------------------
// Constructor: Validates hyperparameters and MCMC settings
// ------------------------------------------------------------
BetaNonConjugateModel::BetaNonConjugateModel(double shape_a_, double rate_a_,
                                             double shape_b_, double rate_b_,
                                             double prop_sd_log_a_,
                                             double prop_sd_log_b_,
                                             int mh_steps_,
                                             double eps_)
: shape_a(shape_a_), rate_a(rate_a_),
  shape_b(shape_b_), rate_b(rate_b_),
  prop_sd_log_a(prop_sd_log_a_), prop_sd_log_b(prop_sd_log_b_),
  mh_steps(mh_steps_), eps(eps_)
{
    if (shape_a <= 0 || rate_a <= 0 || shape_b <= 0 || rate_b <= 0)
        throw std::runtime_error("BetaNonConjugateModel: gamma prior hyperparameters must be > 0.");
    if (prop_sd_log_a <= 0 || prop_sd_log_b <= 0)
        throw std::runtime_error("BetaNonConjugateModel: proposal std dev must be > 0.");
    if (mh_steps <= 0)
        throw std::runtime_error("BetaNonConjugateModel: mh_steps must be > 0.");
    if (eps <= 0 || eps >= 0.5)
        throw std::runtime_error("BetaNonConjugateModel: eps must be in (0, 0.5).");
}

// ------------------------------------------------------------
// Mathematical Helpers: Log-PDFs
// ------------------------------------------------------------
double BetaNonConjugateModel::log_gamma_prior(double x, double shape, double rate)
{
    // log p(x) = (shape-1)log x - rate*x + const
    if (x <= 0.0) return -std::numeric_limits<double>::infinity();
    return (shape - 1.0) * std::log(x) - rate * x;
}

double BetaNonConjugateModel::log_beta_pdf(double y, double a, double b) const
{
    if (a <= 0.0 || b <= 0.0) return -std::numeric_limits<double>::infinity();

    // Clamp y away from 0/1 for stability
    if (y <= 0.0) y = eps;
    else if (y >= 1.0) y = 1.0 - eps;

    // log Beta pdf:
    // log f(y|a,b) = (a-1)log y + (b-1)log(1-y) - log B(a,b)
    // log B(a,b) = lgamma(a)+lgamma(b)-lgamma(a+b)
    double logB = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
    return (a - 1.0) * std::log(y) + (b - 1.0) * std::log(1.0 - y) - logB;
}

double BetaNonConjugateModel::log_posterior(double a, double b,
                                            const std::vector<double>& data) const
{
    double lp = 0.0;

    // priors
    lp += log_gamma_prior(a, shape_a, rate_a);
    lp += log_gamma_prior(b, shape_b, rate_b);

    if (!std::isfinite(lp)) return -std::numeric_limits<double>::infinity();

    // likelihood
    for (double y : data) {
        double ll = log_beta_pdf(y, a, b);
        lp += ll;
        if (!std::isfinite(lp)) return -std::numeric_limits<double>::infinity();
    }
    return lp;
}

double BetaNonConjugateModel::likelihood(double y_i, const ComponentParameter& phi)
{
    if (phi.parameters.size() < 2)
        throw std::runtime_error("BetaNonConjugateModel::likelihood: phi must have [alpha, beta].");

    const double a = phi.parameters[0];
    const double b = phi.parameters[1];

    double logf = log_beta_pdf(y_i, a, b);
    if (!std::isfinite(logf)) return 0.0;
    return std::exp(logf);
}

// ------------------------------------------------------------
// Base Measure Sampling (G0)
// ------------------------------------------------------------
ComponentParameter BetaNonConjugateModel::sample_from_G0()
{
    // alpha ~ Gamma(shape_a, scale=1/rate_a), beta ~ Gamma(shape_b, scale=1/rate_b)
    std::gamma_distribution<double> ga(shape_a, 1.0 / rate_a);
    std::gamma_distribution<double> gb(shape_b, 1.0 / rate_b);

    ComponentParameter phi;
    phi.parameters.resize(2);
    phi.parameters[0] = ga(generator);
    phi.parameters[1] = gb(generator);

    // Ensure positivity (gamma should already)
    if (phi.parameters[0] <= 0.0) phi.parameters[0] = eps;
    if (phi.parameters[1] <= 0.0) phi.parameters[1] = eps;

    return phi;
}

ComponentParameter BetaNonConjugateModel::sample_from_H_i(double y_i)
{
    // Not used by Neal-8 in your current design; keep simple.
    // A reasonable choice is to start from G0 and do a couple MH steps with singleton data.
    ComponentParameter phi = sample_from_G0();
    std::vector<double> singleton{y_i};

    // Use current_phi trick locally
    current_phi = phi;
    return sample_from_posterior(singleton);
}

// ------------------------------------------------------------
// Posterior Sampling via Metropolis-Hastings (Log-Space)
// ------------------------------------------------------------
ComponentParameter BetaNonConjugateModel::sample_from_posterior(const std::vector<double>& cluster_data)
{
    // MH in log(alpha), log(beta) with symmetric Normal proposals.
    if (current_phi.parameters.size() < 2) {
        // If caller didn't set current_phi, fall back to G0 initialization.
        current_phi = sample_from_G0();
    }

    double a = current_phi.parameters[0];
    double b = current_phi.parameters[1];
    if (a <= 0.0) a = eps;
    if (b <= 0.0) b = eps;

    std::normal_distribution<double> N_a(0.0, prop_sd_log_a);
    std::normal_distribution<double> N_b(0.0, prop_sd_log_b);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    double loga = std::log(a);
    double logb = std::log(b);

    for (int t = 0; t < mh_steps; ++t) {

        double loga_prop = loga + N_a(generator);
        double logb_prop = logb + N_b(generator);

        double a_prop = std::exp(loga_prop);
        double b_prop = std::exp(logb_prop);

        // Target is in (a,b) space; since we propose in log-space with symmetric proposal,
        // we must include Jacobian terms log|da/dloga| = loga, log|db/dlogb| = logb.
        double log_target_cur  = log_posterior(std::exp(loga),      std::exp(logb),      cluster_data) + loga + logb;
        double log_target_prop = log_posterior(a_prop,              b_prop,              cluster_data) + loga_prop + logb_prop;

        double log_acc = log_target_prop - log_target_cur;
        double u = std::log(U(generator));

        if (u < log_acc) {
            loga = loga_prop;
            logb = logb_prop;
        }
    }

    ComponentParameter out;
    out.parameters.resize(2);
    out.parameters[0] = std::exp(loga);
    out.parameters[1] = std::exp(logb);

    return out;
}

double BetaNonConjugateModel::calculate_prior_predictive(double)
{
    // Neal-8 does not require this; throwing prevents accidental Neal-2 usage.
    throw std::runtime_error("BetaNonConjugateModel: calculate_prior_predictive not available (non-conjugate). Use Neal-8.");
}

IModelPrior* BetaNonConjugateModel::clone() const
{
    return new BetaNonConjugateModel(*this);
}