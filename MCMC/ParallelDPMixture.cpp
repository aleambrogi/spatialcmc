#include "ParallelDPMixture.h"
#include "Sharding.h"
#include <thread> // Include this at top
#include "ParallelDPMixture.h"
#include <thread>
#include <future>
#include <algorithm>

// ... constructor ...

void ParallelDPMixture::run_parallel_mcmc(int burn_in, int sampling)
{
    // Resize outputs
    all_samples_c.resize(num_partitions);
    all_samples_phi.resize(num_partitions);
    best_k_per_chain.resize(num_partitions);
    best_clustering_per_chain.resize(num_partitions);
    best_parameters_per_chain.resize(num_partitions);

    // Batching Strategy
    unsigned int threads = std::thread::hardware_concurrency();
    if (threads == 0) threads = 4;

    int total_tasks = num_partitions;
    int batches = (total_tasks + threads - 1) / threads;

    for (int b = 0; b < batches; ++b) {
        int start = b * threads;
        int end = std::min(start + (int)threads, total_tasks);

        std::vector<std::future<void>> current_jobs;

        for (int k = start; k < end; ++k) {
            current_jobs.push_back(std::async(std::launch::async, [this, k, burn_in, sampling]() {

                IModelPrior* prior_copy = clone_prior();

                if (sampler == SamplerType::Neal2) {
                    aPPM_MixtureModel model(alpha, partitions_data[k], prior_copy, partitions_adj[k], lambda_appm);
                    model.run_mcmc(burn_in, sampling);

                    // --- FIX START ---
                    // 1. Calculate best index WHILE data is still in 'model'
                    int best_idx = model.opt_param_index_fast();

                    // 2. Save best results based on that index
                    best_k_per_chain[k] = best_idx;
                    best_clustering_per_chain[k] = model.samples_c[best_idx]; // Copy (safe/small)
                    best_parameters_per_chain[k] = model.samples_phi_c[best_idx]; // Copy (safe/small)

                    // 3. NOW move the massive history vectors out
                    all_samples_c[k] = std::move(model.samples_c);
                    all_samples_phi[k] = std::move(model.samples_phi_c);
                    // --- FIX END ---

                } else {
                    aPPM_MixtureModel_Neal8 model(alpha, partitions_data[k], prior_copy, partitions_adj[k], lambda_appm, m_aux);
                    model.run_mcmc(burn_in, sampling);

                    // --- FIX START (Same for Neal8) ---
                    int best_idx = model.opt_param_index_fast();

                    best_k_per_chain[k] = best_idx;
                    best_clustering_per_chain[k] = model.samples_c[best_idx];
                    best_parameters_per_chain[k] = model.samples_phi_c[best_idx];

                    all_samples_c[k] = std::move(model.samples_c);
                    all_samples_phi[k] = std::move(model.samples_phi_c);
                    // --- FIX END ---
                }

                delete prior_copy;
            }));
        }

        for (auto& f : current_jobs) f.get();

        // Progress check
        if ((b+1) % 10 == 0 || b == batches - 1) {
             std::cout << "    Batch " << (b+1) << "/" << batches << " done.\r" << std::flush;
        }
    }
    std::cout << "\n";
}
// ---------------------------------------------------------
// Constructor
// ---------------------------------------------------------

ParallelDPMixture::ParallelDPMixture(double alpha_,
                  const std::vector<Step1_Sharding::Sharding::Shard>& shards,
                  IModelPrior* prior_,
                  const std::vector<std::vector<std::vector<int>>>& adjacency_per_shard,
                  double lambda_appm_,
                  SamplerType sampler_,
                  int m_aux_)
        : alpha(alpha_),
          lambda_appm(lambda_appm_),
          prior(prior_),
          sampler(sampler_),
          m_aux(m_aux_)
{
    num_partitions = shards.size();

    partitions_data.resize(num_partitions);
    partitions_adj  = adjacency_per_shard;   // <-- copia diretta, zero trasformazioni

    for (size_t k = 0; k < shards.size(); ++k)
    {
        size_t shard_size = shards[k].size();
        partitions_data[k].resize(shard_size);

        for (size_t i = 0; i < shard_size; ++i)
        {
            partitions_data[k][i] = shards[k][i].median_val;
        }
    }

    std::cout << "Created ParallelDPMixture with "
              << num_partitions << " shards\n";
}


// ---------------------------------------------------------
// Clone prior (thread-safe)
// ---------------------------------------------------------
IModelPrior* ParallelDPMixture::clone_prior() const
{
    auto* p = prior->clone();

    static std::atomic<int> counter{1};
    unsigned int seed =
        std::random_device{}() ^ (counter++ * 2654435761u);

    p->set_seed(seed);
    return p;
}

// ---------------------------------------------------------
// Run parallel MCMC
// ---------------------------------------------------------

// In ParallelDPMixture.cpp



// ---------------------------------------------------------
// Accessors
// ---------------------------------------------------------

const std::vector<int>&
ParallelDPMixture::get_best_k_per_chain() const {
    return best_k_per_chain;
}

const std::vector<std::vector<int>>&
ParallelDPMixture::get_best_clustering_per_chain() const {
    return best_clustering_per_chain;
}

const std::vector<std::map<int, ComponentParameter>>&
ParallelDPMixture::get_best_parameters_per_chain() const {
    return best_parameters_per_chain;
}

const std::vector<std::vector<std::vector<int>>>&
ParallelDPMixture::get_all_samples_c() const {
    return all_samples_c;
}

const std::vector<std::vector<std::map<int, ComponentParameter>>>&
ParallelDPMixture::get_all_samples_phi() const {
    return all_samples_phi;
}