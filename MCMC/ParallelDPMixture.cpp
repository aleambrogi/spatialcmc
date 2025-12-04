#include "ParallelDPMixture.h"

// ---------------------------------------------------------
// Constructor
// ---------------------------------------------------------
ParallelDPMixture::ParallelDPMixture(double alpha_,
                                     const std::vector<double>& full_data,
                                     IModelPrior* prior_,
                                     const std::vector<std::vector<int>>& full_adjacency,
                                     int num_partitions_,
                                     double lambda_appm_)
    : alpha(alpha_),
      num_partitions(num_partitions_),
      lambda_appm(lambda_appm_),
      prior(prior_)
{
    split_data_and_graph(full_data, full_adjacency);
}


// ---------------------------------------------------------
// Split data and adjacency into K subproblems
// ---------------------------------------------------------
void ParallelDPMixture::split_data_and_graph(
        const std::vector<double>& data,
        const std::vector<std::vector<int>>& full_adj)
{
    int N = data.size();
    int block = N / num_partitions;
    int remainder = N % num_partitions;

    partitions_data.resize(num_partitions);
    partitions_adj.resize(num_partitions);

    int idx = 0;

    for (int k = 0; k < num_partitions; ++k)
    {
        int size = block + (k < remainder ? 1 : 0);

        partitions_data[k].assign(data.begin() + idx,
                                  data.begin() + idx + size);

        partitions_adj[k].clear();
        partitions_adj[k].resize(size);

        // map global indices → local (for this shard)
        std::unordered_map<int,int> global2local;
        for (int g = idx; g < idx + size; ++g)
            global2local[g] = g - idx;

        for (int g = idx; g < idx + size; ++g)
        {
            int local = g - idx;

            for (int nb : full_adj[g])
            {
                if (global2local.count(nb))
                    partitions_adj[k][local].push_back(global2local[nb]);
            }

            // Safety: ensure at least self-loop to avoid empty clusters collapsing
            if (partitions_adj[k][local].empty())
                partitions_adj[k][local].push_back(local);
        }

        idx += size;
    }
}



// ---------------------------------------------------------
// Clone prior for thread safety
// ---------------------------------------------------------
IModelPrior* ParallelDPMixture::clone_prior() const
{
    auto* p = prior->clone();  // deep clone of the prior

    // Give each clone a unique seed
    static std::atomic<int> counter{1};
    unsigned int seed = std::random_device{}() ^ (counter++ * 2654435761u);

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
        jobs.push_back(std::async(std::launch::async, [&, k]() {

            IModelPrior* prior_copy = clone_prior();

            aPPM_MixtureModel model(alpha,
                                    partitions_data[k],
                                    prior_copy,
                                    partitions_adj[k],
                                    lambda_appm);

            // Run MCMC -----------------------------------

            model.run_mcmc(burn_in, sampling);

            std::cout << "Chain " << k << " finished MCMC \n";

            // Store all raw samples
            all_samples_c[k]   = model.samples_c;
            all_samples_phi[k] = model.samples_phi_c;

            // Compute optimal sample index
            int best_idx = model.opt_param_index();

            std::cout << "Chain " << k << " finished binder loss optimization \n";

            // Store per-chain results
            best_k_per_chain[k]            = best_idx;
            best_clustering_per_chain[k]   = model.samples_c[best_idx];
            best_parameters_per_chain[k]   = model.samples_phi_c[best_idx];

            delete prior_copy;

            std::cout << "Chain " << k << " finished. Best index = "
                      << best_idx << "\n";
        }));
    }

    for (auto& f : jobs)
        f.get();
}


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



// ---------------------------------------------------------
// Result accessors
// ---------------------------------------------------------
const std::vector<std::vector<std::vector<int>>>&
ParallelDPMixture::get_all_samples_c() const {
    return all_samples_c;
}

const std::vector<std::vector<std::map<int, ComponentParameter>>>&
ParallelDPMixture::get_all_samples_phi() const {
    return all_samples_phi;
}
