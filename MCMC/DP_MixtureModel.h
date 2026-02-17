#ifndef DP_MIXTUREMODEL_H
#define DP_MIXTUREMODEL_H

#include <chrono>
#include <vector>     // For std::vector
#include <map>        // For std::map
#include <cmath>      // For standard math functions (e.g., log, exp)
#include <numeric>    // For std::accumulate (for summing weights)
#include <algorithm>  // For std::max
#include <random>     // For C++ random number generation (e.g., std::mt19937, std::uniform_real_distribution)
#include <stdexcept>  // For exceptions (e.g., std::runtime_error)
#include <iostream>   // For basic I/O (debugging/logging)
#include <queue>
#include <unordered_set>
#include "IModelPrior.h"

int binder_loss(const std::vector<int>& a,
                const std::vector<int>& b);

// ------------------------------------------------------------
// DPMixtureModel: Base class for Dirichlet Process Mixtures
// Implements Neal's Algorithm 2 (Gibbs Sampling)
// ------------------------------------------------------------
class DPMixtureModel {
protected:
    // Core Model Parameters (from DP(G0, alpha))
    double alpha; // Concentration parameter alpha
    int n;        // Number of observations
    std::vector<double> y; // The observed data y_1, ..., y_n

    // State Variables for Algorithm 2
    std::vector<int> c; // Component indicators: c_1, ..., c_n

    // Map of unique component labels (c) to their parameters (phi_c).
    // The component parameters are kept in the state for Algorithm 2.
    std::map<int, ComponentParameter> phi_c;

    // Pointer to the specific implementation of the component distribution and prior
    IModelPrior* model_prior;

    // ------------------------------------------------------------
    // Weight calculations for Gibbs Sampling
    // ------------------------------------------------------------
    virtual double cluster_weight(int i,
                                  int cluster_label,
                                  const ComponentParameter& phi);

    virtual double new_cluster_weight(int i);

    // Helper: Computes the integral term for the probability of a new component
    // ∫ F(y_i, phi) dG_0(phi) -> Feasible because G0 is conjugate for F
    double compute_prior_predictive(double y_i);

    // Helper: Calculates n_{-i, c} (count of observations in cluster c excluding observation i)
    int get_cluster_count_minus_i(int i, int c_label);

    int next_component_id;

    // Step 1 and Step 2 must be overridable for Neal-8 / non-conjugate
    virtual void update_component_indicator(int i);
    virtual void update_component_parameters();

public:
    std::vector<std::vector<int>> samples_c;
    std::vector<std::map<int, ComponentParameter>> samples_phi_c;

    // ------------------------------------------------------------
    // Constructor & Main MCMC Loop
    // ------------------------------------------------------------
    DPMixtureModel(double alpha, const std::vector<double>& data, IModelPrior* prior);
    void run_mcmc(int burn_in, int sampling);
    void gibbs_scan(); // Executes one full scan (step 1 and 2 of Algorithm 2)
    virtual ~DPMixtureModel() = default;

    // Returns the index of the sample that minimizes Binder Loss
    [[nodiscard]] int opt_param_index_fast() const;


private:
    std::vector<std::vector<double>> compute_co_clustering() const;

    double binder_expected_loss(
            const std::vector<int>& est,
            const std::vector<std::vector<double>>& P,
            double a = 1.0,
            double b = 1.0) const;

    // compute the binder_loss between two vectors a and b
    static int binder_loss(const std::vector<int>& a,
                    const std::vector<int>& b);
};

#endif //DP_MIXTUREMODEL_H