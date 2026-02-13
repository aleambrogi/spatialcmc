#include "ParallelDPMixture.h"
#include "Sharding.h"

// NEW: Includes for threading and time
#include <atomic>
#include <thread>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <mutex>

// ---------------------------------------------------------
// Helper: Progress Bar Display
// ---------------------------------------------------------
void display_progress_bar(std::atomic<long>& current, long total, std::chrono::steady_clock::time_point start_time) {
    const int bar_width = 50;

    // Calculate percentage
    double progress = (double)current / total;
    if (progress > 1.0) progress = 1.0;

    // Calculate ETA
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
    long eta = 0;
    if (progress > 0.001) {
        eta = (long)(elapsed / progress) - elapsed;
    }

    std::cout << "\r[";
    int pos = bar_width * progress;
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " % "
              << "| Elapsed: " << elapsed << "s "
              << "| ETA: " << eta << "s   " << std::flush;
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
    partitions_adj  = adjacency_per_shard;

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
    unsigned int seed = std::random_device{}() ^ (counter++ * 2654435761u);
    p->set_seed(seed);
    return p;
}

// ---------------------------------------------------------
// Run parallel MCMC (With Progress Bar)
// ---------------------------------------------------------
void ParallelDPMixture::run_parallel_mcmc(int burn_in, int sampling)
{
    all_samples_c.resize(num_partitions);
    all_samples_phi.resize(num_partitions);

    best_k_per_chain.resize(num_partitions);
    best_clustering_per_chain.resize(num_partitions);
    best_parameters_per_chain.resize(num_partitions);

    // 1. Setup Progress Tracking
    long total_iterations = (long)num_partitions * (burn_in + sampling);
    std::atomic<long> completed_iterations{0};
    std::atomic<bool> done{false};
    auto start_time = std::chrono::steady_clock::now();

    // 2. Launch Progress Monitor Thread
    std::thread monitor([&]() {
        while (!done) {
            display_progress_bar(completed_iterations, total_iterations, start_time);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        // Final draw to ensure 100%
        display_progress_bar(completed_iterations, total_iterations, start_time);
        std::cout << std::endl;
    });

    // 3. Define the Callback for inner models
    // This lambda will be passed to your models to increment progress
    auto progress_callback = [&completed_iterations]() {
        completed_iterations.fetch_add(1, std::memory_order_relaxed);
    };

    std::vector<std::future<void>> jobs;

    for (int k = 0; k < num_partitions; ++k)
    {
        jobs.push_back(std::async(std::launch::async,
                                  [this, k, burn_in, sampling, &progress_callback]() {

            IModelPrior* prior_copy = clone_prior();

             if (sampler == SamplerType::Neal2) {
                aPPM_MixtureModel model(alpha, partitions_data[k], prior_copy, partitions_adj[k], lambda_appm);

                // *** IMPORTANT: You must modify your aPPM_MixtureModel::run_mcmc to accept this callback ***
                // If you haven't modified the inner class yet, use the wrapper below (Option B in notes)
                model.run_mcmc(burn_in, sampling, progress_callback);

                 all_samples_c[k]   = model.samples_c;
                 all_samples_phi[k] = model.samples_phi_c;

                 int best_idx = model.opt_param_index_fast();
                 best_k_per_chain[k]          = best_idx;
                 best_clustering_per_chain[k] = model.samples_c[best_idx];
                 best_parameters_per_chain[k] = model.samples_phi_c[best_idx];

                 delete prior_copy;

             } else {
                aPPM_MixtureModel_Neal8 model(alpha, partitions_data[k], prior_copy, partitions_adj[k], lambda_appm, m_aux);

                // *** IMPORTANT: Modify run_mcmc here as well ***
                model.run_mcmc(burn_in, sampling, progress_callback);

                 all_samples_c[k]   = model.samples_c;
                 all_samples_phi[k] = model.samples_phi_c;

                 int best_idx = model.opt_param_index_fast();
                 best_k_per_chain[k]          = best_idx;
                 best_clustering_per_chain[k] = model.samples_c[best_idx];
                 best_parameters_per_chain[k] = model.samples_phi_c[best_idx];

                 delete prior_copy;
             }
        }));
    }

    // Wait for workers
    for (auto& f : jobs)
        f.get();

    // Stop monitor thread
    done = true;
    if(monitor.joinable()) monitor.join();
}

// Accessors remain the same...


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
