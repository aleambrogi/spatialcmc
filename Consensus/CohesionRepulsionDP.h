//
// Created by albor on 01/12/2025.
//

#ifndef COHESION_REPULSION_DP_H
#define COHESION_REPULSION_DP_H

#include <vector>
#include <random>
#include "IModelPrior.h"

class CohesionRepulsionDP {
public:
    CohesionRepulsionDP(double alpha,
                        const std::vector<std::vector<double>>& D,
                        double delta1, double lambda_in,
                        double delta2, double theta_out,
                        unsigned int seed = std::random_device{}());

    // Initialize with a single cluster or random labels
    void initialize_single_cluster();
    void initialize_random(int max_init_clusters);

    [[nodiscard]] int best_sample_binder() const;

    // One full Gibbs sweep over all items
    void gibbs_scan();

    // Run MCMC: burn-in + sampling (with optional thinning)
    void run_mcmc(int burn_in, int n_samples);

    const std::vector<int>& get_current_labels() const { return c; }
    const std::vector<std::vector<int>>& get_samples_labels() const { return samples_c; }

    void run_mcmc_to_csv(int burn_in, int n_samples, const std::string& filename);
    void save_meta_cluster_assignments_csv(const std::vector<ComponentParameter>& phi_data,
                                           const std::string& filename) const;

private:
    double alpha;                      // DP concentration
    const std::vector<std::vector<double>>& D; // distance matrix (NxN)
    int n;                             // number of objects

    // Gamma parameters for cohesion and repulsion
    double delta1;     // shape for within
    double lambda_in;  // rate for within
    double delta2;     // shape for between
    double theta_out;  // rate for between

    // precomputed log-constants for Gamma pdfs
    double logC_in;
    double logC_out;

    // Partition state
    std::vector<int> c;                // cluster assignment for each i
    std::vector<int> cluster_counts;   // size of each cluster (index = cluster id)

    std::mt19937 rng;

    // MCMC samples of labels
    std::vector<std::vector<int>> samples_c;

    // Internal helpers
    int sample_from_log_weights(const std::vector<double>& logw);
    void remove_item_from_cluster(int i);
    void check_empty_and_relabel(int emptied_cluster);
    double log_f_in(double d) const;
    double log_g_out(double d) const;
    void update_assignment_for_index(int i);
};

#endif // COHESION_REPULSION_DP_H
