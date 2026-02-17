#include "SpatialCMCInterface.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>

// Priors
#include "MCMC/IModelPrior.h"
#include "MCMC/PoissonGammaModel.h"
#include "MCMC/BetaNonConjugateModel.h"
#include "Step0/GeometryAdjacency.h"

// -----------------------------------------------------------------------------
// Helper function to calculate Log-Beta: ln(B(x, y))
// B(x,y) = Gamma(x)Gamma(y) / Gamma(x+y)
// Uses lgamma to avoid numerical overflow with factorials
// -----------------------------------------------------------------------------
double log_beta(double x, double y) {
    return std::lgamma(x) + std::lgamma(y) - std::lgamma(x + y);
}

SpatialCMCInterface::SpatialCMCInterface(SpatialCMCConfig cfg)
: cfg_(std::move(cfg))
{}

std::vector<DataRow> SpatialCMCInterface::load_data() const
{
    LoadSpec spec;
    spec.path = cfg_.data_path;
    spec.layer_name = cfg_.layer_name;
    spec.format = cfg_.format;

    // If your vector layers use different field names, expose them via cfg_ and set here.
    return load_dataset(spec);
}

void SpatialCMCInterface::preprocess_for_prior(std::vector<DataRow>& rows) const
{
    if (cfg_.prior_name != "beta") return;

    // Beta requires y in (0,1): rescale median_val
    double mn = std::numeric_limits<double>::infinity();
    double mx = -std::numeric_limits<double>::infinity();

    for (const auto& r : rows) {
        mn = std::min(mn, r.median_val);
        mx = std::max(mx, r.median_val);
    }

    if (!(mx > mn))
        throw std::runtime_error("Beta preprocess: median_val has zero range.");

    const double eps = 1e-6;
    for (auto& r : rows) {
        double x = (r.median_val - mn) / (mx - mn);
        r.median_val = std::min(1.0 - eps, std::max(eps, x));
    }
}

std::unique_ptr<IModelPrior> SpatialCMCInterface::make_prior() const
{
    const auto& h = cfg_.hyperparams;

    if (cfg_.prior_name == "normal") {
        // mu0, kappa0, alpha0, beta0
        if (h.size() != 4)
            throw std::runtime_error("normal prior requires 4 hyperparams: mu0,kappa0,alpha0,beta0");
        return std::unique_ptr<IModelPrior>(new NormalConjugateModel(h[0], h[1], h[2], h[3]));
    }

    if (cfg_.prior_name == "beta") {
        // shape_a, rate_a, shape_b, rate_b, propA, propB, mh_steps
        if (h.size() != 7)
            throw std::runtime_error("beta prior requires 7 hyperparams: shape_a,rate_a,shape_b,rate_b,propA,propB,mh_steps");
        return std::unique_ptr<IModelPrior>(
            new BetaNonConjugateModel(h[0], h[1], h[2], h[3], h[4], h[5], (int)h[6])
        );
    }

    if (cfg_.prior_name == "poisson") {
        if (h.size() != 2)
            throw std::runtime_error("poisson prior requires 2 hyperparams: alpha,beta");
        return std::unique_ptr<IModelPrior>(
            new PoissonGammaModel(h[0], h[1])
        );
    }

    throw std::runtime_error("Unknown prior: " + cfg_.prior_name);
}

std::vector<std::vector<std::vector<int>>>
SpatialCMCInterface::build_adjacency_per_shard(const std::vector<Step1_Sharding::Sharding::Shard>& shards) const
{
    std::vector<std::vector<std::vector<int>>> out;
    out.reserve(shards.size());

    for (const auto& shard : shards) {
        if (cfg_.format == DataFormat::CSV) {
            // CSV: no geometry => empty adjacency (or load external JSON adjacency elsewhere)
            out.emplace_back(shard.size());
            continue;
        }

        // SHP/GPKG: infer adjacency mode from geometry
        auto mode = Step0::infer_adjacency_mode(shard);
        out.push_back(Step0::build_local_adjacency_from_rows(shard, mode, cfg_.knn_k));
    }
    return out;
}

double SpatialCMCInterface::phi_distance(const ComponentParameter& a,
                                         const ComponentParameter& b) const
{
    if (a.parameters.empty() || b.parameters.empty())
        return 0.0;

    // --- BETA DISTRIBUTION ---
    // Uses Hellinger Distance.
    // Expected parameters: [0] = alpha, [1] = beta
    if (cfg_.prior_name == "beta") {
        if (a.parameters.size() < 2 || b.parameters.size() < 2) {
            throw std::runtime_error("phi_distance: Beta model requires 2 parameters (alpha, beta).");
        }

        double alpha1 = a.parameters[0];
        double beta1  = a.parameters[1];
        double alpha2 = b.parameters[0];
        double beta2  = b.parameters[1];

        // Hellinger Formula: sqrt( 1 - BC(P,Q) )
        // Where BC (Bhattacharyya Coefficient) = B((a1+a2)/2, (b1+b2)/2) / sqrt( B(a1,b1) * B(a2,b2) )

        // Log-space calculation for numerical stability
        double log_num = log_beta((alpha1 + alpha2) / 2.0, (beta1 + beta2) / 2.0);
        double log_den = 0.5 * (log_beta(alpha1, beta1) + log_beta(alpha2, beta2));

        double affinity = std::exp(log_num - log_den);

        // Clamp for floating point errors (affinity must not exceed 1.0)
        if (affinity > 1.0) affinity = 1.0;
        if (affinity < 0.0) affinity = 0.0; // safety

        return std::sqrt(1.0 - affinity);
    }

    // --- NORMAL DISTRIBUTION ---
    // Uses 2-Wasserstein distance.
    // Expected parameters: [0] = mean, [1] = variance (sigma^2)
    if (cfg_.prior_name == "normal") {
        double mu1 = a.parameters[0];
        double mu2 = b.parameters[0];

        // Case: fixed variance (only mean)
        if (a.parameters.size() == 1) {
            return std::abs(mu1 - mu2);
        }

        // Case: mean and variance
        double var1 = a.parameters[1];
        double var2 = b.parameters[1];

        // Wasserstein requires standard deviations
        double sigma1 = std::sqrt(var1);
        double sigma2 = std::sqrt(var2);

        // W_2 = sqrt( (mu1-mu2)^2 + (sigma1-sigma2)^2 )
        double mean_diff_sq = (mu1 - mu2) * (mu1 - mu2);
        double std_diff_sq  = (sigma1 - sigma2) * (sigma1 - sigma2);

        return std::sqrt(mean_diff_sq + std_diff_sq);
    }

    // --- POISSON DISTRIBUTION ---
    // Uses 1-Wasserstein distance (Earth Mover's Distance)
    // Coincides with the absolute difference of means (lambda)
    if (cfg_.prior_name == "poisson") {
        return std::abs(a.parameters[0] - b.parameters[0]);
    }

    // --- UNKNOWN MODEL ---
    throw std::invalid_argument("phi_distance: Unknown prior_name '" + cfg_.prior_name + "'");
}

SpatialCMCResult SpatialCMCInterface::run()
{
    SpatialCMCResult out;

    // ---------------------------------------------------------
    // 1-3) Load, Preprocess, Shard
    // ---------------------------------------------------------
    auto rows = load_data();
    if (rows.empty()) throw std::runtime_error("Dataset is empty.");
    out.n_rows = (int)rows.size();

    preprocess_for_prior(rows);
    for (size_t i = 0; i < rows.size(); ++i) rows[i].global_index = (int)i;

    Step1_Sharding::Sharding sharder(rows);
    auto shards = sharder.generateShards(cfg_.shard_key);
    out.n_shards = (int)shards.size();

    // MEMORY OPTIMIZATION: Clear raw input now that we have shards
    rows.clear();
    rows.shrink_to_fit();

    // ---------------------------------------------------------
    // 4-6) Parallel DP (Batched)
    // ---------------------------------------------------------
    auto prior = make_prior();
    auto adjacency_per_shard = build_adjacency_per_shard(shards);

    // Note: Assuming ParallelDPMixture is correctly implemented to take these arguments
    ParallelDPMixture pdp(cfg_.alpha_dp, shards, prior.get(), adjacency_per_shard,
                          cfg_.lambda_appm, cfg_.sampler, cfg_.m_aux);
    pdp.run_parallel_mcmc(cfg_.burn_in, cfg_.samples);

    // Helper methods to extract results from the parallel engine
    // (Ensure these methods exist in your ParallelDPMixture class)
    auto best_c   = pdp.get_best_clustering_per_chain(); // Renamed to match main.cpp usage
    auto best_phi = pdp.get_best_parameters_per_chain(); // Renamed to match main.cpp usage

    // ---------------------------------------------------------
    // 7-8) Meta Clustering
    // ---------------------------------------------------------
    std::vector<ComponentParameter> phi_data;
    for (const auto& chain_phi : best_phi)
        for (const auto& kv : chain_phi)
            phi_data.push_back(kv.second);

    out.Q_phi = (int)phi_data.size();

    // Safety Check for Matrix Size
    long long Q = out.Q_phi;
    if ((double)Q*Q*8.0 / 1e9 > 24.0)
        throw std::runtime_error("Meta-clustering matrix too big! Increase alpha_dp.");

    // Build Distance Matrix
    std::vector<std::vector<double>> D_phi(Q, std::vector<double>(Q, 0.0));

    // OpenMP parallelization if enabled in compiler settings
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < Q; ++i) {
        for (int j = i + 1; j < Q; ++j) {
            D_phi[i][j] = D_phi[j][i] = phi_distance(phi_data[i], phi_data[j]);
        }
    }

    CohesionRepulsionDP phiDP(cfg_.alpha_meta, D_phi, cfg_.delta_in, cfg_.lambda_in,
                              cfg_.delta_out, cfg_.theta_out, cfg_.meta_seed);
    phiDP.initialize_random(cfg_.meta_init_K);
    phiDP.run_mcmc(cfg_.meta_burnin, cfg_.meta_samples);

    // Free Matrix Memory Immediately
    D_phi.clear();
    std::vector<std::vector<double>>().swap(D_phi);

    // Extract History
    auto meta_history = phiDP.get_samples_labels();

    // Find Best Iteration
    int best_phi_idx = phiDP.best_sample_binder();

    // We copy this because we need to move it into pipeline later
    std::vector<int> final_phi_labels;
    if (best_phi_idx >= 0 && best_phi_idx < meta_history.size()) {
        final_phi_labels = meta_history[best_phi_idx];
        if (!final_phi_labels.empty()) {
            out.K_meta = *std::max_element(final_phi_labels.begin(), final_phi_labels.end()) + 1;
        }
    }

    // ---------------------------------------------------------
    // 9) Populate Pipeline
    // ---------------------------------------------------------
    bool has_conjugate = (cfg_.prior_name == "normal" || cfg_.prior_name == "poisson");
    double mu0=0, kappa0=0, alpha0=0, beta0=0; // Fill if using normal...

    // Populate the results pipeline
    out.pipeline.populate_from_pipeline(
        best_c,
        best_phi,
        phi_data,
        final_phi_labels,
        cfg_.sampler, cfg_.alpha_dp, cfg_.lambda_appm, cfg_.burn_in, cfg_.samples,
        cfg_.alpha_meta, cfg_.delta_in, cfg_.lambda_in, cfg_.delta_out, cfg_.theta_out,
        best_phi_idx, has_conjugate, mu0, kappa0, alpha0, beta0
    );

    out.pipeline.print_summary();

    // ---------------------------------------------------------
    // 10) Export Results (Optimized)
    // ---------------------------------------------------------
    if (cfg_.write_outputs) {
        std::cout << "--- Exporting Results ---\n";

        // Reconstruct data ONCE in shard order
        std::vector<DataRow> shard_order;
        // Estimate size from out.n_rows
        shard_order.reserve(out.n_rows > 0 ? out.n_rows : 0);

        for (const auto& shard : shards)
            for (const auto& r : shard)
                shard_order.push_back(r);

        // Export 1: Meta Trace (if your pipeline class supports it)
        // Check if save_meta_mcmc_point_trace is available in BayesianPipelineBaseline
        // out.pipeline.save_meta_mcmc_point_trace(cfg_.meta_assignments_csv, meta_history, shard_order);
        // Assuming standard save for now based on main.cpp usage:
        out.pipeline.save_global_clustering_csv(cfg_.global_clustering_csv, shard_order);

        // Export 2: GPKG (if supported)
        std::string gpkg_out = cfg_.global_clustering_csv;
        size_t lastindex = gpkg_out.find_last_of(".");
        if (lastindex != std::string::npos) gpkg_out = gpkg_out.substr(0, lastindex);
        gpkg_out += ".gpkg";

        // out.pipeline.save_global_clustering_gpkg(gpkg_out, shard_order);
        out.pipeline.save_dataset_with_global_clusters(gpkg_out, rows, shards); // Using available method from main.cpp context
    }

    return out;
}