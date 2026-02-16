
#ifndef BAYESIANPIPELINEBASELINE_H
#define BAYESIANPIPELINEBASELINE_H

#include <vector>
#include <map>
#include <iostream>
#include "../MCMC/IModelPrior.h"
#include <string>
#include "../Step0/DataRow.h"

// Tipo di sampler
enum class SamplerType {
    Neal2,
    Neal8
};

class BayesianPipelineBaseline {
public:
    // OUTPUT dal pipeline
    std::vector<std::vector<int>> all_clusterings;
    std::vector<std::map<int, ComponentParameter>> all_parameters;
    std::vector<ComponentParameter> phi_data;
    std::vector<int> best_meta_labels;
    std::vector<std::vector<int>> global_assignments;

    // DP + MCMC + Meta hyperparameters
    double alpha_dp = 0.0, lambda_appm = 0.0;
    int burn_in = 0, samples = 0;
    double alpha_meta = 0.0, delta_in = 0.0, lambda_in = 0.0;
    double delta_out = 0.0, theta_out = 0.0;

    // Prior (solo Neal2)
    bool has_conjugate_prior = false;
    double mu0 = 0.0, kappa0 = 0.0, alpha0 = 0.0, beta0 = 0.0;

    // Sampler
    SamplerType sampler_type;

    // Meta info
    int best_meta_iteration = -1;

    // Global posteriors (lasciati invariati)
    std::map<int, ComponentParameter> global_posteriors;

public:
    BayesianPipelineBaseline() = default;

    void populate_from_pipeline(
        std::vector<std::vector<int>> clusterings,           // <--- Pass by Value
        std::vector<std::map<int, ComponentParameter>> parameters,
        std::vector<ComponentParameter> phi,
        std::vector<int> meta_labels,
        SamplerType sampler,
        double alpha_dp_, double lambda_appm_,
        int burn_in_, int samples_,
        double alpha_meta_, double delta_in_, double lambda_in_,
        double delta_out_, double theta_out_,
        int best_meta_iteration_,
        bool has_conjugate_prior_ = false,
        double mu0_ = 0.0, double kappa0_ = 0.0,
        double alpha0_ = 0.0, double beta0_ = 0.0
    );

    void save_global_clustering_gpkg(
        const std::string& filename,
        const std::vector<DataRow>& sorted_data,
        const std::string& layer_name = "clusters"
    ) const;


    void save_meta_mcmc_point_trace(
        const std::string& filename,
        // The history from CohesionRepulsionDP: [Iteration][MetaItemIndex] -> GlobalID
        const std::vector<std::vector<int>>& meta_samples,
        // Optional: for column headers
        const std::vector<DataRow>& original_data_sorted = {}
    );

    void compute_global_posteriors_from_meta();
    void print_summary() const;
    void save_global_clustering_csv(const std::string& filename, const std::vector<DataRow>& original_data );
    std::map<int, int> build_global_index_to_cluster_map(const std::vector<std::vector<DataRow>>& shards) const;
    void save_dataset_with_global_clusters(const std::string& filename, const std::vector<DataRow>& original_data, const std::vector<std::vector<DataRow>>& shards) const;
private:
    void map_local_to_global();

};

#endif // BAYESIANPIPELINEBASELINE_H
