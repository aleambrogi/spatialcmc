#ifndef PARALLEL_DP_MIXTURE_H
#define PARALLEL_DP_MIXTURE_H

#include <vector>
#include <future>
#include <iostream>
#include <Sharding.h>
#include <unordered_map>
#include "aPPM_MixtureModel.h"
#include "IModelPrior.h"

class ParallelDPMixture {
public:
    ParallelDPMixture(double alpha_,
                      const std::vector<Step1_Sharding::Sharding::Shard>& shards,
                      IModelPrior* prior_,
                      const std::vector<std::vector<std::vector<int>>>& adjacency_per_shard,
                      double lambda_appm_);

    void run_parallel_mcmc(int burn_in, int sampling);

    const std::vector<std::vector<std::vector<int>>>& get_all_samples_c() const;
    const std::vector<std::vector<std::map<int, ComponentParameter>>>& get_all_samples_phi() const;

    const std::vector<int>& get_best_k_per_chain() const;
    const std::vector<std::vector<int>>& get_best_clustering_per_chain() const;
    const std::vector<std::map<int, ComponentParameter>>& get_best_parameters_per_chain() const;

    const std::vector<std::vector<double>>& get_partitioned_data() const {
        return partitions_data;
    }

private:
    double alpha;
    int num_partitions;
    double lambda_appm;
    IModelPrior* prior;

    // RESULTS PER CHAIN
    std::vector<int> best_k_per_chain;                                         // best index per chain
    std::vector<std::vector<int>> best_clustering_per_chain;                   // best c_i per chain
    std::vector<std::map<int, ComponentParameter>> best_parameters_per_chain;  // best φ per chain

    std::vector<std::vector<double>> partitions_data;
    std::vector<std::vector<std::vector<int>>> partitions_adj;

    std::vector<std::vector<std::vector<int>>> all_samples_c;
    std::vector<std::vector<std::map<int, ComponentParameter>>> all_samples_phi;

    IModelPrior* clone_prior() const;

};

#endif
