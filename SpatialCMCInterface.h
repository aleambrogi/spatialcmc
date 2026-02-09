#ifndef SPATIAL_CMC_INTERFACE_H
#define SPATIAL_CMC_INTERFACE_H

#include <string>
#include <vector>
#include <memory>

#include "Step0/DataLoader.h"
#include "Step0/GeometryAdjacency.h"
#include "Step1_Sharding/Sharding.h"
#include "MCMC/ParallelDPMixture.h"
#include "MCMC/IModelPrior.h"
#include "Consensus/CohesionRepulsionDP.h"
#include "BayesianPipelineBaseline.h"

struct SpatialCMCConfig {
    std::string data_path;
    DataFormat format = DataFormat::CSV;

    // For GIS: optional layer name (GPKG)
    std::string layer_name = "";

    // Sampler and prior
    SamplerType sampler = SamplerType::Neal2; // Neal2 / Neal8 (your enum)
    std::string prior_name;                   // "normal", "beta", "poisson"
    std::vector<double> hyperparams;

    // Sharding
    std::string shard_key = "PROVINCIA";

    // DP/MCMC
    double alpha_dp = 5.0;
    double lambda_appm = 0.01;
    int burn_in = 100;
    int samples = 500;
    int m_aux = 5; // Neal8 auxiliary components

    // Adjacency
    int knn_k = 8; // for point geometry

    // Meta-clustering
    double alpha_meta = 5.0;
    double delta_in = 0.5;
    double lambda_in = 10.0;
    double delta_out = 2.0;
    double theta_out = 10.0;
    int meta_seed = 999;
    int meta_init_K = 6;
    int meta_burnin = 1;
    int meta_samples = 100;

    // Outputs
    bool write_outputs = true;
    std::string meta_assignments_csv = "meta_cluster_assignments.csv";
    std::string global_clustering_csv = "FINAL_GLOBAL_CLUSTERING.csv";
};

struct SpatialCMCResult {
    BayesianPipelineBaseline pipeline;
    int n_rows = 0;
    int n_shards = 0;
    int Q_phi = 0;
    int K_meta = 0;
};

class SpatialCMCInterface {
public:
    explicit SpatialCMCInterface(SpatialCMCConfig cfg);
    SpatialCMCResult run();

private:
    SpatialCMCConfig cfg_;

    std::vector<DataRow> load_data() const;
    void preprocess_for_prior(std::vector<DataRow>& rows) const;

    std::unique_ptr<IModelPrior> make_prior() const;

    std::vector<std::vector<std::vector<int>>>
    build_adjacency_per_shard(const std::vector<Step1_Sharding::Sharding::Shard>& shards) const;

    double phi_distance(const ComponentParameter& a, const ComponentParameter& b) const;
};

#endif
