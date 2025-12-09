#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <CSVLoader.h>
#include <map>
#include "DP_MixtureModel.h"
#include "aPPM_MixtureModel.h"
#include "CohesionRepulsionDP.h"
#include "ParallelDPMixture.h"
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// ------------------------------------------------------------
// Generate synthetic Gaussian cluster parameters (mu, sigma^2)
// ------------------------------------------------------------
ComponentParameter generate_cluster_param(double true_mu,
                                         double true_sigma_sq,
                                         std::mt19937& gen)
{
    std::normal_distribution<double> mu_noise(true_mu, 0.3);
    std::gamma_distribution<double> ig(5.0, 1.0 / (5.0 * true_sigma_sq));

    double mu_k = mu_noise(gen);
    double sigma2_k = 1.0 / ig(gen);

    ComponentParameter cp;
    cp.parameters = {mu_k, sigma2_k};
    return cp;
}

// ------------------------------------------------------------
// 1D Wasserstein distance between Gaussians
// W₂² = (μ₁–μ₂)² + (σ₁ – σ₂)²
// ------------------------------------------------------------
double wasserstein_1d(const ComponentParameter& a,
                      const ComponentParameter& b)
{
    double mu1 = a.parameters[0];
    double s1  = std::sqrt(a.parameters[1]);

    double mu2 = b.parameters[0];
    double s2  = std::sqrt(b.parameters[1]);

    double dmu = mu1 - mu2;
    double ds  = s1 - s2;

    return std::sqrt(dmu*dmu + ds*ds);
}

// ------------------------------------------------------------
// TEST for CohesionRepulsionDP (distance-based clustering)
// ------------------------------------------------------------
void test_CohesionRepulsionDP()
{
    std::cout << "\n===== TEST: Cohesion - Repulsion DP Clustering =====\n";

    std::mt19937 gen(4233);

    // ------------------------------------------------------------
    // 1. Simulate 40 Gaussian cluster summaries
    //    First 20 around μ=-5, second 20 around μ=+5
    // ------------------------------------------------------------
    std::vector<ComponentParameter> phi_data;
    phi_data.reserve(40);

    for (int i = 0; i < 20; ++i)
        phi_data.push_back(generate_cluster_param(-5.0, 1.0, gen));

    for (int i = 0; i < 20; ++i)
        phi_data.push_back(generate_cluster_param(+5.0, 1.0, gen));

    std::cout << "Generated 40 Gaussian clusters.\n";
    std::cout << "Example cluster k (first 5):\n";
    for (int i = 0; i < 5; ++i)
        std::cout << "  mu=" << phi_data[i].parameters[0]
                  << "  sigma2=" << phi_data[i].parameters[1] << "\n";

    // ------------------------------------------------------------
    // 2. Compute pairwise Wasserstein-2 distances
    // ------------------------------------------------------------
    int n = phi_data.size();
    std::vector<std::vector<double>> D(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            D[i][j] = wasserstein_1d(phi_data[i], phi_data[j]);

    std::cout << "Computed Wasserstein distance matrix.\n";

    // ------------------------------------------------------------
    // 3. Set parameters for cohesion–repulsion DP
    // ------------------------------------------------------------
    double alpha = 1.0;    // DP concentration
    double delta1 = 0.5;   // within-cluster shape (<1 = cohesion)
    double lambda_in = 1.0;

    double delta2 = 3.0;   // between-cluster shape (>1 = repulsion)
    double theta_out = 1.0;

    // ------------------------------------------------------------
    // 4. Build model and run MCMC
    // ------------------------------------------------------------
    CohesionRepulsionDP model(
        alpha,
        D,
        delta1, lambda_in,
        delta2, theta_out,
        /*seed=*/123
    );

    // You can initialize random or single cluster
    model.initialize_random(/*max_init_clusters=*/4);

    std::cout << "Running MCMC...\n";
    model.run_mcmc(/*burn-in=*/500, /*samples=*/1000);

    const auto& samples = model.get_samples_labels();
    int best_idx = model.best_sample_binder();
    const auto& final_labels = samples[best_idx];

    // ------------------------------------------------------------
    // 5. Print final assignments
    // ------------------------------------------------------------
    std::cout << "\nBest-sample cluster assignments (Binder-minimizing):\n";
    for (int i = 0; i < n; ++i)
        std::cout << "cluster[" << i << "] mu=" << phi_data[i].parameters[0]
                  << "  sigma2=" << phi_data[i].parameters[1]
                  << "  -> cluster " << final_labels[i] << "\n";

    // ------------------------------------------------------------
    // 6. Check clustering accuracy
    // ------------------------------------------------------------
    int mismatch_left = 0;
    for (int i = 0; i < 20; ++i)
        if (final_labels[i] != final_labels[0])
            mismatch_left++;

    int mismatch_right = 0;
    for (int i = 20; i < 40; ++i)
        if (final_labels[i] != final_labels[20])
            mismatch_right++;

    std::cout << "\nLeft-group mismatches:  " << mismatch_left << "/20\n";
    std::cout << "Right-group mismatches: " << mismatch_right << "/20\n";

    std::cout << "\n===== END TEST =====\n";
}


void test_DP_MixtureModel() {
    std::vector<double> data;
    std::mt19937 gen(42);
    std::normal_distribution<double> dist1(-5.0, 1.0);
    std::normal_distribution<double> dist2(5.0, 1.0);

    for (int i = 0; i < 500; ++i) data.push_back(dist1(gen));
    for (int i = 0; i < 500; ++i) data.push_back(dist2(gen));

    NormalConjugateModel prior(0.0, 1.0, 2.0, 2.0);

    DPMixtureModel model(1.0, data, &prior);
    model.run_mcmc(1000, 1000);

    // Print last iteration cluster assignments
    const auto& index_star = model.opt_param_index_fast();
    const auto& best_c = model.samples_c[index_star];
    const auto& best_phi = model.samples_phi_c[index_star];

    std::cout << "Final cluster assignments:\n";
    for (size_t i = 0; i < best_c.size(); ++i)
        std::cout << "y[" << i << "]=" << data[i] << " in cluster " << best_c[i] << "\n";

    std::cout << "\nFinal cluster parameters:\n";
    for (auto& kv : best_phi)
        std::cout << "Cluster " << kv.first << ": mu=" << kv.second.parameters[0]
                  << ", sigma2=" << kv.second.parameters[1] << "\n";
}

void test_DP_MixtureModel_with_APPM()
{
    std::cout << "\n===== TEST: DP Mixture Model with APPM =====\n";

    // ------------------------------------------
    // 1. Generate synthetic data (2 true clusters)
    // ------------------------------------------

    int n1 = 50;
    int n2 = 50;
    int N  = n1 + n2;

    std::vector<double> data;
    data.reserve(N);

    std::mt19937 rng(1789);
    std::normal_distribution<double> dist1(-5.0, 1.0);   // cluster 1
    std::normal_distribution<double> dist2(5.0, 1.0);  // cluster 2

    // cluster 1
    for (int i = 0; i < n1; ++i)
        data.push_back(dist1(rng));

    // cluster 2
    for (int i = 0; i < n2; ++i)
        data.push_back(dist2(rng));

    std::cout << "Generated " << N << " points (two well-separated Gaussians).\n";

    // -----------------------------------------------------
    // 2. Build adjacency: two independent fully connected sets
    //    Members 0..49 are connected only among themselves.
    //    Members 50..99 are connected only among themselves.
    // -----------------------------------------------------

    std::vector<std::vector<int>> adjacency(N);

    // cluster 1 adjacency
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n1; ++j) {
            if (i != j)
                adjacency[i].push_back(j);
        }
    }

    // cluster 2 adjacency
    for (int i = n2; i < N; ++i) {
        for (int j = n2; j < N; ++j) {
            if (i != j)
                adjacency[i].push_back(j);
        }
    }

    std::cout << "Created adjacency with 2 spatially separated components.\n";

    // -----------------------------------------------------
    // 3. Define model prior (Normal-Inverse-Gamma for 1D)
    // -----------------------------------------------------
    // Replace with your actual class name
    IModelPrior* prior = new NormalConjugateModel(
        0.0,
        1.0,
        2.0,
        2.0
    );
    // -----------------------------------------------------
    // 4. Instantiate APPM DP Mixture
    // -----------------------------------------------------
    double alpha = 1.0;
    double lambda = 3.0;   // strong spatial cohesion

    aPPM_MixtureModel model(alpha, data, prior, adjacency, lambda);

    std::cout << "Running APPM DP Mixture Model...\n";

    // -----------------------------------------------------
    // 5. Run MCMC
    // -----------------------------------------------------
    int burn_in = 500;
    int sampling = 1000;

    model.run_mcmc(burn_in, sampling);

    // -----------------------------------------------------
    // 6. Recover optimal clustering
    // -----------------------------------------------------
    int idx = model.opt_param_index_fast();
    const auto& best_clust = model.samples_c[idx];

    std::cout << "\nBest partition index = " << idx << "\n";
    std::cout << "Cluster assignment:\n";

    for (int i = 0; i < N; ++i)
        std::cout << "y[" << i << "]=" << data[i] << " -> cluster " << best_clust[i] << "\n";

    std::cout << "\nFinal cluster parameters:\n";
    for (auto& kv : model.samples_phi_c[idx])
        std::cout << "Cluster " << kv.first << ": mu=" << kv.second.parameters[0]
                  << ", sigma2=" << kv.second.parameters[1] << "\n";

    std::cout << "===== END TEST =====\n";
}

std::vector<std::vector<int>> load_adjacency_json(const std::string& filename)
{
    std::ifstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("Cannot open adjacency JSON file.");

    json j;
    f >> j;

    std::vector<std::vector<int>> adj(j.size());
    for (int i = 0; i < j.size(); i++)
        for (auto v : j[i])
            adj[i].push_back(v);

    return adj;
}


std::vector<std::vector<int>> extract_local_adjacency(
        const std::vector<DataRow>& shard,
        const std::vector<std::vector<int>>& global_adj)
{
    int n = shard.size();

    // mappa: global_index → local_index
    std::unordered_map<int,int> map_global_to_local;
    for (int i = 0; i < n; i++)
        map_global_to_local[ shard[i].global_index ] = i;

    // adjacency locale
    std::vector<std::vector<int>> local_adj(n);

    for (int i = 0; i < n; i++)
    {
        int gi = shard[i].global_index;

        for (int g_nei : global_adj[gi])
        {
            if (map_global_to_local.count(g_nei))
            {
                int local_j = map_global_to_local[g_nei];
                local_adj[i].push_back(local_j);
            }
        }
    }

    return local_adj;
}


void test_Parallel_Mixture_with_MetaClustering()
{
    std::cout << "\n==================== FULL PIPELINE TEST ====================\n";

    // =====================================================================
    // 0. LOAD CSV
    // =====================================================================
    std::cout << "Loading CSV...\n";
    std::ifstream file("dataset_median.csv");
    std::string header;
    std::getline(file, header);
    std::cout << "Header CSV: " << header << std::endl;

    CSVReader reader("dataset_median.csv", ',');
    std::vector<DataRow> rows = reader.read();

    if (rows.empty()) {
        std::cerr << "CSV is empty. Abort.\n";
        return;
    }
    std::cout << "Loaded " << rows.size() << " rows.\n";

    for (size_t i = 0; i < std::min<size_t>(20, rows.size()); ++i) {
        std::cout << "row[" << i << "] PROVINCIA=\"" << rows[i].PROVINCIA
                  << "\" COMUNE=\"" << rows[i].COMUNE
                  << "\" median=" << rows[i].median_value << "\n";
    }

    // =====================================================================
    // 1. SHARDING
    // =====================================================================
    std::cout << "Running sharding...\n";

    // assegna GLOBAL INDEX ai dati
    for (size_t i = 0; i < rows.size(); i++)
        rows[i].global_index = i;

    Step1_Sharding::Sharding sharder(rows);

    std::string shard_column = "PROVINCIA";
    auto shards = sharder.generateShards(shard_column);

    int num_partitions = shards.size();
    std::cout << "Generated " << num_partitions << " shards.\n";

    // =====================================================================
    // 2. PRIOR
    // =====================================================================
    double mu0 = 0.0, kappa0 = 1.0, alpha0 = 2.0, beta0 = 2.0;
    IModelPrior* prior = new NormalConjugateModel(mu0, kappa0, alpha0, beta0);

    // =====================================================================
    // 3. LOAD GLOBAL ADJACENCY (JSON)
    // =====================================================================
    auto global_adjacency = load_adjacency_json("adjacency_list.json");
    std::cout << "Loaded adjacency list with " << global_adjacency.size() << " nodes.\n";

    // =====================================================================
    // 3a. COSTRUZIONE ADIACENZA PER SHARD
    // =====================================================================
    std::vector<std::vector<std::vector<int>>> adjacency_per_shard;

    for (const auto& shard : shards)
    {
        adjacency_per_shard.push_back(
                extract_local_adjacency(shard, global_adjacency)
        );
    }

    std::cout << "Built local adjacency matrices for "
              << adjacency_per_shard.size() << " shards.\n";

    // =====================================================================
    // 4. PARALLEL DP ON SHARDS
    // =====================================================================
    double alpha_dp    = 1.0;
    double lambda_appm = 0.2;
    int burn_in        = 200;
    int sampling       = 100;

    ParallelDPMixture pdp(alpha_dp,
                          shards,               // ora vettore di vettori DataRow
                          prior,
                          adjacency_per_shard,  // adjacency locale corretta
                          lambda_appm);

    pdp.run_parallel_mcmc(burn_in, sampling);

    // =====================================================================
    // 5. RISULTATI: BEST CLUSTER PER SHARD
    // =====================================================================
    const auto& all_samples_c   = pdp.get_best_clustering_per_chain();
    const auto& all_samples_phi = pdp.get_best_parameters_per_chain();
    int M = all_samples_c.size();

    std::cout << "\nCollected best output from " << M << " chains.\n";

    // =====================================================================
    // 6. META-CLUSTERING SU PHI
    // =====================================================================
    std::vector<ComponentParameter> phi_data;

    for (int ch = 0; ch < M; ++ch)
        for (const auto& kv : all_samples_phi[ch])
            phi_data.push_back(kv.second);

    int Q = phi_data.size();
    std::cout << "Collected " << Q << " parameter vectors.\n";

    std::vector<std::vector<double>> D_phi(Q, std::vector<double>(Q, 0.0));

    for (int i = 0; i < Q; ++i)
        for (int j = i + 1; j < Q; ++j)
            D_phi[i][j] = D_phi[j][i] =
                    wasserstein_1d(phi_data[i], phi_data[j]);

    CohesionRepulsionDP phiDP(1.0, D_phi, 0.5, 1.0, 3.0, 1.0, 999);
    phiDP.initialize_random(6);
    phiDP.run_mcmc(200, 400);

    int best_phi_idx = phiDP.best_sample_binder();
    const auto& final_phi_labels = phiDP.get_samples_labels()[best_phi_idx];

    int Kphi = *std::max_element(final_phi_labels.begin(), final_phi_labels.end()) + 1;
    std::cout << "Detected " << Kphi << " parameter clusters.\n";

    for (int i = 0; i < Q; ++i)
        std::cout << "cluster[" << i << "]: mu=" << phi_data[i].parameters[0]
                  << " sigma2=" << phi_data[i].parameters[1]
                  << " --> meta cluster " << final_phi_labels[i] << "\n";

    std::cout << "\n==================== END FULL PIPELINE TEST ====================\n";

    delete prior;
}


int main()
{
    //test_DP_MixtureModel();
    //test_DP_MixtureModel_with_APPM();
    //test_CohesionRepulsionDP();
    //test_Parallel_DP_MixtureModel_with_APPM();
    test_Parallel_Mixture_with_MetaClustering();
    return 0;
}