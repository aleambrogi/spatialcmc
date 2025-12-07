#include "ParallelDPMixture.h"
#include "Sharding.h"

// ---------------------------------------------------------
// Constructor
// ---------------------------------------------------------

ParallelDPMixture::ParallelDPMixture(
        double alpha_,
        const std::vector<Step1_Sharding::Sharding::Shard>& shards,
        IModelPrior* prior_,
        const std::vector<std::vector<int>>& full_adjacency,
        double lambda_appm_)
        : alpha(alpha_),
          lambda_appm(lambda_appm_),
          prior(prior_)
{
    num_partitions = shards.size();
    partitions_data.resize(num_partitions);
    partitions_adj.resize(num_partitions);

    // Concatenazione globale per calcolare gli indici
    std::vector<double> all_data;
    std::vector<int> shard_start(num_partitions, 0);
    for (size_t k = 0; k < shards.size(); ++k) {
        shard_start[k] = all_data.size();
        for (const auto& row : shards[k])
            all_data.push_back(std::stod(row.fields.at("X")));
    }

    // Per ogni shard
    for (size_t k = 0; k < shards.size(); ++k) {
        size_t shard_size = shards[k].size();
        partitions_data[k].resize(shard_size);
        partitions_adj[k].resize(shard_size);

        int idx = shard_start[k];

        // Copia dati
        for (size_t i = 0; i < shard_size; ++i)
            partitions_data[k][i] = all_data[idx + i];

        // Map global -> local
        std::unordered_map<int,int> global2local;
        for (int g = idx; g < idx + (int)shard_size; ++g)
            global2local[g] = g - idx;

        // Costruzione grafo locale filtrando i vicini
        for (int g = idx; g < idx + (int)shard_size; ++g) {
            int local = g - idx;
            for (int nb : full_adjacency[g]) {
                if (global2local.count(nb))
                    partitions_adj[k][local].push_back(global2local[nb]);
            }
            // Self-loop se vuoto
            if (partitions_adj[k][local].empty())
                partitions_adj[k][local].push_back(local);
        }
    }

    std::cout << "Created ParallelDPMixture with "
              << num_partitions << " shards and local adjacency\n";
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
void ParallelDPMixture::run_parallel_mcmc(int burn_in, int sampling)
{
    all_samples_c.resize(num_partitions);
    all_samples_phi.resize(num_partitions);

    best_k_per_chain.resize(num_partitions);
    best_clustering_per_chain.resize(num_partitions);
    best_parameters_per_chain.resize(num_partitions);

    std::vector<std::future<void>> jobs;

    for (int k = 0; k < num_partitions; ++k)
    {
        jobs.push_back(std::async(std::launch::async,
                                  [this, k, burn_in, sampling]() {

            IModelPrior* prior_copy = clone_prior();

            aPPM_MixtureModel model(alpha,
                                    partitions_data[k],
                                    prior_copy,
                                    partitions_adj[k],
                                    lambda_appm);

            model.run_mcmc(burn_in, sampling);

            all_samples_c[k]   = model.samples_c;
            all_samples_phi[k] = model.samples_phi_c;

            int best_idx = model.opt_param_index();

            best_k_per_chain[k]          = best_idx;
            best_clustering_per_chain[k] = model.samples_c[best_idx];
            best_parameters_per_chain[k] = model.samples_phi_c[best_idx];

            delete prior_copy;
        }));
    }

    for (auto& f : jobs)
        f.get();
}


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
