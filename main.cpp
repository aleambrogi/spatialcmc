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
#include "BetaNonConjugateModel.h"
#include "ParallelDPMixture.h"
#include "BayesianPipelineBaseline.h"
#include <nlohmann/json.hpp>
#include "MCMC/PoissonGammaModel.h"
#include "distances.h"
#include "adjacency.h"
#include "SpatialCMCInterface.h"
#include "Step0/DataLoader.h"


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
                  << "\" median=" << rows[i].median_val << "\n";
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
    double mu0 = 0.0, kappa0 = 0.1, alpha0 = 2.0, beta0 = 0.01;
    IModelPrior* prior = new NormalConjugateModel(mu0, kappa0, alpha0, beta0);

    // =====================================================================
    // 3. LOAD GLOBAL ADJACENCY (JSON)
    // =====================================================================
    auto global_adjacency = AdjacencyUtils::load_adjacency_json("adjacency_list.json");
    std::cout << "Loaded adjacency list with " << global_adjacency.size() << " nodes.\n";

    // =====================================================================
    // 3a. COSTRUZIONE ADIACENZA PER SHARD
    // =====================================================================
    std::vector<std::vector<std::vector<int>>> adjacency_per_shard;

    for (const auto& shard : shards)
    {
        adjacency_per_shard.push_back(
                AdjacencyUtils::extract_local_adjacency(shard, global_adjacency)
        );
    }

    std::cout << "Built local adjacency matrices for "
              << adjacency_per_shard.size() << " shards.\n";

    // =====================================================================
    // 4. PARALLEL DP ON SHARDS
    // =====================================================================
    double alpha_dp    = 5.0;
    double lambda_appm = 0.01;
    int burn_in        = 1;
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

    Distances dist(Distances::Type::Gaussian);
    for (int i = 0; i < Q; ++i)
        for (int j = i + 1; j < Q; ++j)
            D_phi[i][j] = D_phi[j][i] =
                    dist(phi_data[i], phi_data[j]);

    CohesionRepulsionDP phiDP(5.0, D_phi, 0.5, 10.0, 2.0, 10.0, 999);
    phiDP.initialize_random(6);
    phiDP.run_mcmc(1, 100);

    //phiDP.run_mcmc_to_csv(1, 100, "output_clusters.csv");
    phiDP.save_meta_cluster_assignments_csv(phi_data, "meta_cluster_assignments.csv");

    int best_phi_idx = phiDP.best_sample_binder();
    const auto& final_phi_labels = phiDP.get_samples_labels()[best_phi_idx];

    int Kphi = *std::max_element(final_phi_labels.begin(), final_phi_labels.end()) + 1;
    std::cout << "Detected " << Kphi << " parameter clusters.\n";

    for (int i = 0; i < Q; ++i)
        std::cout << "cluster[" << i << "]: mu=" << phi_data[i].parameters[0]
                  << " sigma2=" << phi_data[i].parameters[1]
                  << " --> meta cluster " << final_phi_labels[i] << "\n";
    // Creazione e stampa della pipeline

    BayesianPipelineBaseline pipeline;
    // Meta-clustering
    double alpha_meta  = 5.0;
    double delta_in    = 0.5;
    double lambda_in   = 10.0;
    double delta_out   = 2.0;
    double theta_out   = 10.0;


    pipeline.populate_from_pipeline(
    all_samples_c, all_samples_phi, phi_data, final_phi_labels,
    SamplerType::Neal2,
    alpha_dp, lambda_appm,
    burn_in, sampling,
    alpha_meta, delta_in, lambda_in, delta_out, theta_out,
    best_phi_idx,          // <-- BEST ITERATION (Binder)
    true,                  // <-- ha prior coniugata
    mu0, kappa0, alpha0, beta0
);

    pipeline.print_summary();

    // A. Ricostruiamo il vettore ordinato per l'export
    std::vector<DataRow> sorted_rows_for_export;
    for(const auto& shard : shards) {
        for(const auto& row : shard) {
            sorted_rows_for_export.push_back(row);
        }
    }

    // B. Salviamo
    pipeline.save_global_clustering_csv("FINAL_GLOBAL_CLUSTERING.csv", sorted_rows_for_export);
    //pipeline.save_global_clustering_csv("FINAL_GLOBAL_CLUSTERING.csv");
    pipeline.save_dataset_with_global_clusters(
        "dataset_with_clusters.csv",
        rows,
        shards
    );
    std::cout << "\n==================== END FULL PIPELINE TEST ====================\n";

    delete prior;
}

void test_Parallel_Mixture_with_MetaClustering_Neal8_beta()
{
    std::cout << "\n==================== FULL PIPELINE TEST (NEAL 8 + BETA) ====================\n";

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
    /*
        // =====================================================================
        // 0a. RESCALE MEDIANS TO (0,1) FOR BETA MODEL
        // =====================================================================
        double min_val = std::numeric_limits<double>::infinity();
        double max_val = -std::numeric_limits<double>::infinity();

        for (const auto& r : rows) {
            min_val = std::min(min_val, r.median_val);
            max_val = std::max(max_val, r.median_val);
        }

        const double eps = 1e-6;
        for (auto& r : rows) {
            double x = (r.median_val - min_val) / (max_val - min_val);
            r.median_val = std::min(1.0 - eps, std::max(eps, x));
        }

        std::cout << "Rescaled median_val to (0,1).\n";
    */
    // =====================================================================
    // 1. SHARDING
    // =====================================================================
    for (size_t i = 0; i < rows.size(); i++)
        rows[i].global_index = i;

    Step1_Sharding::Sharding sharder(rows);
    auto shards = sharder.generateShards("PROVINCIA");

    std::cout << "Generated " << shards.size() << " shards.\n";

    // =====================================================================
    // 2. NON-CONJUGATE BETA PRIOR
    // =====================================================================
    // Independent Gamma priors on alpha and beta
    double shape_a = 2.0, rate_a = 1.0;
    double shape_b = 2.0, rate_b = 1.0;

    IModelPrior* prior = new BetaNonConjugateModel(
        shape_a, rate_a,
        shape_b, rate_b,
        0.25,   // proposal sd log(alpha)
        0.25,   // proposal sd log(beta)
        2       // MH steps per update
    );

    // =====================================================================
    // 3. LOAD GLOBAL ADJACENCY
    // =====================================================================
    auto global_adjacency = AdjacencyUtils::load_adjacency_json("adjacency_list.json");

    std::vector<std::vector<std::vector<int>>> adjacency_per_shard;
    for (const auto& shard : shards) {
        adjacency_per_shard.push_back(
            AdjacencyUtils::extract_local_adjacency(shard, global_adjacency)
        );
    }

    // =====================================================================
    // 4. PARALLEL DP (NEAL 8)
    // =====================================================================
    double alpha_dp    = 2;
    double lambda_appm = 0.5;
    int burn_in        = 1000;
    int sampling       = 200;
    int m_aux          = 5;

    ParallelDPMixture pdp(
        alpha_dp,
        shards,
        prior,
        adjacency_per_shard,
        lambda_appm,
        SamplerType::Neal8,
        m_aux
    );

    pdp.run_parallel_mcmc(burn_in, sampling);

    // =====================================================================
    // 5. COLLECT BEST CLUSTERS PER SHARD
    // =====================================================================
    const auto& best_c   = pdp.get_best_clustering_per_chain();
    const auto& best_phi = pdp.get_best_parameters_per_chain();

    std::cout << "Collected best output from "
              << best_phi.size() << " chains.\n";

    // =====================================================================
    // 6. META-CLUSTERING ON BETA PARAMETERS
    // =====================================================================
    std::vector<ComponentParameter> phi_data;
    for (const auto& chain_phi : best_phi)
        for (const auto& kv : chain_phi)
            phi_data.push_back(kv.second);

    int Q = phi_data.size();
    std::cout << "Collected " << Q << " Beta parameter vectors.\n";

    Distances dist(Distances::Type::Beta);
    std::vector<std::vector<double>> D_phi(Q, std::vector<double>(Q, 0.0));
    for (int i = 0; i < Q; ++i)
        for (int j = i + 1; j < Q; ++j)
            D_phi[i][j] = D_phi[j][i] =
                dist(phi_data[i], phi_data[j]);

    CohesionRepulsionDP phiDP(
        2.0,   // alpha
        D_phi,
        0.5,   // delta_in
        5.0,  // lambda_in
        0.5,   // delta_out
        5.0,  // theta_out
        999
    );

    phiDP.initialize_random(6);
    phiDP.run_mcmc(1000, 200);
    phiDP.save_meta_cluster_assignments_csv(
        phi_data, "meta_cluster_assignments_neal8_beta.csv"
    );

    int best_phi_idx = phiDP.best_sample_binder();
    const auto& final_phi_labels =
        phiDP.get_samples_labels()[best_phi_idx];

    int Kphi = *std::max_element(
        final_phi_labels.begin(),
        final_phi_labels.end()
    ) + 1;

    std::cout << "Detected " << Kphi << " Beta meta-clusters.\n";

    for (int i = 0; i < Q; ++i) {
        std::cout << "component[" << i << "]: "
                  << "alpha=" << phi_data[i].parameters[0]
                  << " beta="  << phi_data[i].parameters[1]
                  << " --> meta cluster "
                  << final_phi_labels[i] << "\n";
    }

    // Creazione e stampa della pipeline

    BayesianPipelineBaseline pipeline;
    // Meta-clustering
    double alpha_meta  = 5.0;
    double delta_in    = 0.5;
    double lambda_in   = 10.0;
    double delta_out   = 2.0;
    double theta_out   = 10.0;

    pipeline.populate_from_pipeline(
    best_c, best_phi, phi_data, final_phi_labels,
    SamplerType::Neal8,        // <-- Neal8
    alpha_dp, lambda_appm,
    burn_in, sampling,
    alpha_meta, delta_in, lambda_in, delta_out, theta_out,
    best_phi_idx,              // <-- BEST ITERATION Binder
    false                      // <-- NON coniugata
    );
    pipeline.print_summary();

    // Creiamo un vettore piatto dei dati riordinati come negli shard
    std::vector<DataRow> sorted_rows_for_export;
    for(const auto& shard : shards) {
        for(const auto& row : shard) {
            sorted_rows_for_export.push_back(row);
        }
    }

    // Passiamo questo vettore alla funzione di salvataggio
    // Nota: devi aggiornare la firma in BayesianPipelineBaseline.h/cpp per accettare questo vettore
    // (Nel codice Step 2 ho già predisposto l'argomento opzionale const std::vector<DataRow>& original_data = {})

    pipeline.save_global_clustering_csv("FINAL_GLOBAL_CLUSTERING.csv", sorted_rows_for_export);
    pipeline.save_dataset_with_global_clusters(
    "dataset_with_clusters.csv",
    rows,
    shards
);
    std::cout << "\n==================== END FULL PIPELINE TEST (NEAL8 + BETA) ====================\n";

    delete prior;
}

void test_PoissonGamma_DP() {
    std::cout << "\n===== TEST: DP Mixture Model with Poisson-Gamma (Count Data) =====\n";

    // 1. Generazione dati sintetici (Conteggi)
    std::vector<double> data;
    std::mt19937 gen(1234);

    // Cluster 1: Lambda = 5 (conteggi bassi)
    std::poisson_distribution<int> pois1(5);
    for (int i = 0; i < 100; ++i) data.push_back(static_cast<double>(pois1(gen)));

    // Cluster 2: Lambda = 20 (conteggi alti)
    std::poisson_distribution<int> pois2(20);
    for (int i = 0; i < 100; ++i) data.push_back(static_cast<double>(pois2(gen)));

    std::cout << "Generated " << data.size() << " count observations.\n";
    std::cout << "Cluster 1 True Lambda: 5, Cluster 2 True Lambda: 20.\n";

    // 2. Definizione Prior
    // Prior Gamma(alpha=2, beta=0.5). Media a priori = alpha/beta = 4. Varianza alta.
    // beta basso => prior "piatta" (informativa debole)
    IModelPrior* prior = new PoissonGammaModel(2.0, 0.5);

    // 3. Modello DPMixture (Neal 2)
    // Nota: Passiamo 'data' che è std::vector<double>, ma contiene interi.
    double alpha_dp = 1.0;
    DPMixtureModel model(alpha_dp, data, prior);

    // 4. Esecuzione MCMC
    std::cout << "Running MCMC (Poisson Likelihood)...\n";
    model.run_mcmc(500, 1000); // 500 burn-in, 1000 samples

    // 5. Risultati
    const auto& index_star = model.opt_param_index_fast();
    const auto& best_phi = model.samples_phi_c[index_star];
    const auto& best_c = model.samples_c[index_star];

    std::cout << "\nBest iteration index: " << index_star << "\n";
    std::cout << "Estimated Clusters (Lambda):\n";
    for (auto& kv : best_phi) {
        // parameter[0] è il lambda stimato
        std::cout << "Cluster " << kv.first
                  << " -> Lambda estimated: " << kv.second.parameters[0]
                  << " (Size: " << std::count(best_c.begin(), best_c.end(), kv.first) << ")\n";
    }

    delete prior;
    std::cout << "===== END TEST =====\n";
}

void test_Parallel_Poisson_FullPipeline() {
    std::cout << "\n==================== FULL PIPELINE TEST (POISSON - COUNT DATA) ====================\n";

    // 1. GENERAZIONE DATI SINTETICI (Simuliamo lettura CSV)
    // Creiamo 200 dati: 100 con lambda=5 (Shard A), 100 con lambda=50 (Shard B)
    std::vector<DataRow> rows;
    std::mt19937 gen(999);
    std::poisson_distribution<int> p1(5);
    std::poisson_distribution<int> p2(50);

    // Shard A (Lambda 5)
    for(int i=0; i<100; ++i) {
        DataRow r;
        r.global_index = i;
        r.median_val = (double)p1(gen); // Il conteggio va in median_val
        r.PROVINCIA = "SHARD_A";          // Fittizio per lo sharding
        r.COMUNE = "Comune_" + std::to_string(i);
        rows.push_back(r);
    }
    // Shard B (Lambda 50)
    for(int i=0; i<100; ++i) {
        DataRow r;
        r.global_index = 100 + i;
        r.median_val = (double)p2(gen);
        r.PROVINCIA = "SHARD_B";
        r.COMUNE = "Comune_" + std::to_string(100+i);
        rows.push_back(r);
    }

    std::cout << "Generated " << rows.size() << " synthetic count data.\n";

    // 2. SHARDING
    Step1_Sharding::Sharding sharder(rows);
    auto shards = sharder.generateShards("PROVINCIA");
    std::cout << "Generated " << shards.size() << " shards based on synthetic PROVINCIA.\n";

    // 3. PRIOR POISSON-GAMMA
    // Prior poco informativa: Gamma(alpha=1, beta=0.1) -> Media 10, Varianza 100
    IModelPrior* prior = new PoissonGammaModel(1.0, 0.1);

    // 4. ADIACENZA (Fittizia: identità o vuota, dato che i dati sono sintetici e non spaziali)
    // Creiamo un'adiacenza vuota per ogni shard (nessun vicino spaziale)
    std::vector<std::vector<std::vector<int>>> adjacency_per_shard;
    for(const auto& shard : shards) {
        std::vector<std::vector<int>> adj(shard.size()); // vettori vuoti
        adjacency_per_shard.push_back(adj);
    }

    // 5. PARALLEL MCMC
    double alpha_dp = 1.0;
    double lambda_appm = 0.0; // Disattiviamo influenza spaziale (dato che l'adj è vuota)
    int burn_in = 2000;
    int sampling = 500;

    ParallelDPMixture pdp(alpha_dp, shards, prior, adjacency_per_shard, lambda_appm);
    pdp.run_parallel_mcmc(burn_in, sampling);

    // 6. RACCOLTA RISULTATI LOCALI
    const auto& best_c = pdp.get_best_clustering_per_chain();
    const auto& best_phi = pdp.get_best_parameters_per_chain();

    // 7. META-CLUSTERING (Consensus)
    // Raccogliamo tutti i parametri Lambda trovati
    std::vector<ComponentParameter> phi_data;
    for (const auto& chain_phi : best_phi)
        for (const auto& kv : chain_phi)
            phi_data.push_back(kv.second);

    int Q = phi_data.size();
    std::cout << "Collected " << Q << " local Poisson parameters (Lambdas).\n";

    // Calcolo matrice distanze D_phi
    // Per Poisson, il parametro è uno scalare (Lambda).
    // Usiamo semplicemente la differenza assoluta: |lambda1 - lambda2|
    Distances dist(Distances::Type::Poisson);

    std::vector<std::vector<double>> D_phi(Q, std::vector<double>(Q, 0.0));
    for (int i = 0; i < Q; ++i) {
        for (int j = i + 1; j < Q; ++j) {
            double lam1 = phi_data[i].parameters[0];
            double lam2 = phi_data[j].parameters[0];
            double dist = std::abs(lam1 - lam2); // Distanza semplice 1D
            D_phi[i][j] = D_phi[j][i] = dist;
        }
    }

    // Eseguiamo CohesionRepulsionDP sulla distanza dei Lambda
    CohesionRepulsionDP phiDP(
        1.0,   // alpha meta
        D_phi,
        0.5, 1.0,  // Cohesion (delta_in, lambda_in)
        5.0, 1.0,  // Repulsion (delta_out, theta_out) - Parametri forti per separare 5 da 50
        1234       // seed
    );
    phiDP.initialize_random(5);
    phiDP.run_mcmc(200, 500);

    int best_phi_idx = phiDP.best_sample_binder();
    const auto& final_phi_labels = phiDP.get_samples_labels()[best_phi_idx];

    // 8. PIPELINE & SALVATAGGIO CSV
    BayesianPipelineBaseline pipeline;

    pipeline.populate_from_pipeline(
        best_c, best_phi, phi_data, final_phi_labels,
        SamplerType::Neal2, // PoissonGamma è coniugato -> Neal2
        alpha_dp, lambda_appm,
        burn_in, sampling,
        1.0, 0.5, 1.0, 5.0, 1.0, // Meta params
        best_phi_idx,
        true, // Coniugato = true
        1.0, 0.1, 0.0, 0.0 // Prior params (alpha0, beta0 negli slot mu0, kappa0)
    );

    pipeline.print_summary();

    // --- SALVATAGGIO CSV CON DATI ORIGINALI ---

    // A. Ricostruiamo il vettore ordinato per l'export
    std::vector<DataRow> sorted_rows_for_export;
    for(const auto& shard : shards) {
        for(const auto& row : shard) {
            sorted_rows_for_export.push_back(row);
        }
    }

    // B. Salviamo
    pipeline.save_global_clustering_csv("FINAL_POISSON_CLUSTERING.csv", sorted_rows_for_export);
    pipeline.save_dataset_with_global_clusters(
        "dataset_with_clusters.csv",
        rows,
        shards
    );
    std::cout << "\n==================== END POISSON TEST ====================\n";
    delete prior;
}


void test_spatialcmc_interface()
{
    try {
        std::cout << "\n================ TEST: SpatialCMCInterface ================\n";
/*
        // ------------------------------------------------------------
        // 1) CSV TEST (no geometry, province sharding)
        // ------------------------------------------------------------

        {
            std::cout << "\n[TEST] CSV input\n";

            SpatialCMCConfig cfg;
            cfg.data_path = "dataset_median.csv";
            cfg.format = DataFormat::CSV;

            cfg.sampler = SamplerType::Neal8;
            cfg.prior_name = "beta";
            cfg.hyperparams = {2,1, 2,1, 0.25,0.25, 5}; // BetaNonConjugate

            cfg.shard_key = "PROVINCIA";

            cfg.alpha_dp = 5.0;
            cfg.lambda_appm = 0.01;
            cfg.burn_in = 20;
            cfg.samples = 50;
            cfg.m_aux = 5;

            cfg.write_outputs = true;
            cfg.global_clustering_csv = "TEST_csv_global.csv";
            cfg.meta_assignments_csv  = "TEST_csv_meta.csv";

            SpatialCMCInterface engine(cfg);
            auto res = engine.run();

            std::cout << "CSV OK | rows=" << res.n_rows
                      << " shards=" << res.n_shards
                      << " Q_phi=" << res.Q_phi
                      << " K_meta=" << res.K_meta << "\n";
        }

        // ------------------------------------------------------------
        // 2) GPKG TEST (auto geometry + adjacency)
        // ------------------------------------------------------------
        {
            std::cout << "\n[TEST] GPKG input\n";

            SpatialCMCConfig cfg;
            cfg.data_path = "full_dataset_pulito.gpkg";
            cfg.format = DataFormat::GPKG;
            cfg.layer_name = ""; // empty = first layer

            cfg.sampler = SamplerType::Neal2;
            cfg.prior_name = "poisson";
            cfg.hyperparams = {2.0, 1.0}; // example Poisson hyperparams

            cfg.shard_key = "PROVINCIA"; // must exist as attribute

            cfg.alpha_dp = 5.0;
            cfg.lambda_appm = 0.01;
            cfg.burn_in = 20;
            cfg.samples = 50;
            cfg.m_aux = 5;

            cfg.knn_k = 8; // used for point geometries

            cfg.write_outputs = true;
            cfg.global_clustering_csv = "TEST_gpkg_global.csv";
            cfg.meta_assignments_csv  = "TEST_gpkg_meta.csv";

            SpatialCMCInterface engine(cfg);
            auto res = engine.run();

            std::cout << "GPKG OK | rows=" << res.n_rows
                      << " shards=" << res.n_shards
                      << " Q_phi=" << res.Q_phi
                      << " K_meta=" << res.K_meta << "\n";
        }
*/
        // ------------------------------------------------------------
        // 3) SHP TEST (auto geometry + adjacency)
        // ------------------------------------------------------------
        {
            std::cout << "\n[TEST] SHP input\n";

            SpatialCMCConfig cfg;
            cfg.data_path = "output_shapefile/full_dataset_pulito.shp";
            cfg.format = DataFormat::SHP;

            cfg.sampler = SamplerType::Neal2;
            cfg.prior_name = "normal";
            cfg.hyperparams = {2,2, 1,1};

            cfg.shard_key = "PROVINCIA"; // must exist in DBF

            cfg.alpha_dp = 5.0;
            cfg.lambda_appm = 0.01;
            cfg.burn_in = 20;
            cfg.samples = 50;
            cfg.m_aux = 5;

            cfg.knn_k = 8;

            cfg.write_outputs = true;
            cfg.global_clustering_csv = "TEST_shp_global.csv";
            cfg.meta_assignments_csv  = "TEST_shp_meta.csv";

            SpatialCMCInterface engine(cfg);
            auto res = engine.run();

            std::cout << "SHP OK | rows=" << res.n_rows
                      << " shards=" << res.n_shards
                      << " Q_phi=" << res.Q_phi
                      << " K_meta=" << res.K_meta << "\n";
        }

        std::cout << "\n================ ALL TESTS COMPLETED =================\n";
    }
    catch (const std::exception& e) {
        std::cerr << "\n[TEST FAILED] " << e.what() << "\n";
        throw; // rethrow so debugger / CI catches it
    }
}

int main()
{
    //test_DP_MixtureModel();
    //test_DP_MixtureModel_with_APPM();
    //test_CohesionRepulsionDP();
    //test_Parallel_DP_MixtureModel_with_APPM();
    //test_Parallel_Mixture_with_MetaClustering();
    //test_Parallel_Mixture_with_MetaClustering_Neal8_beta();
    //test_PoissonGamma_DP();
    //test_Parallel_Poisson_FullPipeline();
    test_spatialcmc_interface();
    return 0;
}
