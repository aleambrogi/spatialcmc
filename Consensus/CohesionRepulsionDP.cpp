//
// Created by albor on 01/12/2025.
//

#include "CohesionRepulsionDP.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <iomanip> // Required for progress bar formatting
#include <fstream>

// ------------------------------------------------------------
// Helper: Static Progress Bar
// ------------------------------------------------------------
static void print_progress(int current, int total, const std::string& prefix) {
    const int bar_width = 50;
    float progress = (float)current / total;

    std::cout << "\r" << prefix << " [";
    int pos = bar_width * progress;
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %" << std::flush;
}

static long binder_loss(const std::vector<int>& z1,
                        const std::vector<int>& z2)
{
    int n = z1.size();
    long loss = 0;

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {

            bool same1 = (z1[i] == z1[j]);
            bool same2 = (z2[i] == z2[j]);

            if (same1 != same2)
                loss++;
        }
    }
    return loss;
}

// ------------------------------------------------------------
// Select the MCMC sample index with minimal Binder loss
// ------------------------------------------------------------
// ------------------------------------------------------------
// Select the MCMC sample index with minimal Binder loss
// ------------------------------------------------------------
// ------------------------------------------------------------
// Select the MCMC sample index with minimal Binder loss
// Optimized: Uses Posterior Similarity Matrix (O(T * N^2))
// ------------------------------------------------------------
int CohesionRepulsionDP::best_sample_binder() const
{
    int T = samples_c.size();
    if (T == 0)
        throw std::runtime_error("No samples stored. Run MCMC first.");

    // 1. Compute Posterior Similarity Matrix P
    // P[i][j] = probability that i and j are in the same cluster
    std::cout << "Computing Posterior Similarity Matrix..." << std::endl;
    std::vector<std::vector<double>> P(n, std::vector<double>(n, 0.0));

    for (const auto& sample : samples_c) {
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (sample[i] == sample[j]) {
                    P[i][j] += 1.0;
                }
            }
        }
    }

    // Normalize counts to probabilities
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            P[i][j] /= static_cast<double>(T);
        }
    }

    // 2. Find sample minimizing expected Binder loss against P
    int best_index = 0;
    double best_loss = std::numeric_limits<double>::max();

    // Weights for Binder Loss (standard: a=1, b=1)
    const double a = 1.0;
    const double b = 1.0;

    for (int t = 0; t < T; ++t) {
        double current_loss = 0.0;
        const auto& est = samples_c[t];

        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                bool same_cluster = (est[i] == est[j]);
                double pij = P[i][j];

                if (same_cluster) {
                    // Penalty if they are same in est, but prob says different
                    current_loss += a * (1.0 - pij);
                } else {
                    // Penalty if they are different in est, but prob says same
                    current_loss += b * pij;
                }
            }
        }

        if (current_loss < best_loss) {
            best_loss = current_loss;
            best_index = t;
        }

        print_progress(t + 1, T, "Minimizing Binder Loss");
    }
    std::cout << std::endl;

    std::cout << "Best iteration found: " << best_index
              << " (Min Expected Binder Loss: " << best_loss << ")\n";

    return best_index;
}


// ---------- constructor ----------
CohesionRepulsionDP::CohesionRepulsionDP(
        double alpha_,
        const std::vector<std::vector<double>>& D_,
        double delta1_, double lambda_in_,
        double delta2_, double theta_out_,
        unsigned int seed)
    : alpha(alpha_),
      D(D_),
      n(static_cast<int>(D_.size())),
      delta1(delta1_),
      lambda_in(lambda_in_),
      delta2(delta2_),
      theta_out(theta_out_),
      rng(seed)
{
    if (n <= 0) {
        throw std::runtime_error("Distance matrix must be non-empty.");
    }
    // Check that D is square
    for (int i = 0; i < n; ++i) {
        if (static_cast<int>(D[i].size()) != n) {
            throw std::runtime_error("Distance matrix must be square.");
        }
    }
    if (delta1 <= 0.0 || delta2 <= 0.0 || lambda_in <= 0.0 || theta_out <= 0.0) {
        throw std::runtime_error("Gamma parameters must be > 0.");
    }

    logC_in  = delta1 * std::log(lambda_in)  - std::lgamma(delta1);
    logC_out = delta2 * std::log(theta_out) - std::lgamma(delta2);

    initialize_single_cluster();
}

// ---------- initialization: all items in one cluster ----------
void CohesionRepulsionDP::initialize_single_cluster()
{
    c.assign(n, 0);
    cluster_counts.assign(1, n);
}

// ---------- initialization: random labels ----------
void CohesionRepulsionDP::initialize_random(int max_init_clusters)
{
    if (max_init_clusters <= 0) {
        throw std::runtime_error("max_init_clusters must be > 0.");
    }
    std::uniform_int_distribution<int> unif(0, max_init_clusters - 1);

    c.resize(n);
    for (int i = 0; i < n; ++i) {
        c[i] = unif(rng);
    }

    int max_label = *std::max_element(c.begin(), c.end());
    std::vector<int> tmp_counts(max_label + 1, 0);
    for (int i = 0; i < n; ++i) {
        tmp_counts[c[i]]++;
    }

    std::vector<int> label_map(max_label + 1, -1);
    int newK = 0;
    for (int old = 0; old <= max_label; ++old) {
        if (tmp_counts[old] > 0) {
            label_map[old] = newK++;
        }
    }

    for (int i = 0; i < n; ++i) {
        c[i] = label_map[c[i]];
    }

    cluster_counts.assign(newK, 0);
    for (int i = 0; i < n; ++i) {
        cluster_counts[c[i]]++;
    }
}

// ---------- log Gamma pdfs ----------
double CohesionRepulsionDP::log_f_in(double d) const
{
    if (d <= 0.0) return -1e30;
    return logC_in + (delta1 - 1.0) * std::log(d) - lambda_in * d;
}

double CohesionRepulsionDP::log_g_out(double d) const
{
    if (d <= 0.0) return -1e30;
    return logC_out + (delta2 - 1.0) * std::log(d) - theta_out * d;
}

// ---------- helper: sample from unnormalised log-weights ----------
int CohesionRepulsionDP::sample_from_log_weights(const std::vector<double>& logw)
{
    const int K = static_cast<int>(logw.size());
    if (K == 0) throw std::runtime_error("sample_from_log_weights: empty weights.");

    double max_logw = logw[0];
    for (int k = 1; k < K; ++k) {
        if (logw[k] > max_logw) max_logw = logw[k];
    }

    std::vector<double> w(K);
    double sum_w = 0.0;
    for (int k = 0; k < K; ++k) {
        double val = std::exp(logw[k] - max_logw);
        w[k] = val;
        sum_w += val;
    }

    if (sum_w <= 0.0 || !std::isfinite(sum_w)) {
        std::uniform_int_distribution<int> unif(0, K - 1);
        return unif(rng);
    }

    std::uniform_real_distribution<double> u_dist(0.0, sum_w);
    double u = u_dist(rng);
    double cum = 0.0;
    for (int k = 0; k < K; ++k) {
        cum += w[k];
        if (u <= cum) return k;
    }
    return K - 1;
}

// ---------- remove item i from its cluster ----------
void CohesionRepulsionDP::remove_item_from_cluster(int i)
{
    int old_k = c[i];
    if (old_k < 0 || old_k >= static_cast<int>(cluster_counts.size())) {
        throw std::runtime_error("Invalid cluster index in remove_item_from_cluster.");
    }

    cluster_counts[old_k]--;
    c[i] = -1;

    if (cluster_counts[old_k] == 0) {
        check_empty_and_relabel(old_k);
    }
}

// ---------- if a cluster is empty, drop it and relabel ----------
void CohesionRepulsionDP::check_empty_and_relabel(int emptied_cluster)
{
    cluster_counts.erase(cluster_counts.begin() + emptied_cluster);
    for (int i = 0; i < n; ++i) {
        if (c[i] > emptied_cluster) {
            c[i]--;
        }
    }
}

// ---------- update assignment for index i (Gibbs step) ----------
void CohesionRepulsionDP::update_assignment_for_index(int i)
{
    remove_item_from_cluster(i);
    const int K = static_cast<int>(cluster_counts.size());
    std::vector<double> logw(K + 1, 0.0);

    for (int k = 0; k < K; ++k) {
        double log_prior = std::log(static_cast<double>(cluster_counts[k]));
        double log_like = 0.0;

        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            int cj = c[j];
            if (cj == -1) continue;

            double d = D[i][j];
            if (cj == k) log_like += log_f_in(d);
            else         log_like += log_g_out(d);
        }
        logw[k] = log_prior + log_like;
    }

    {
        double log_prior_new = std::log(alpha);
        double log_like_new = 0.0;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            int cj = c[j];
            if (cj == -1) continue;
            double d = D[i][j];
            log_like_new += log_g_out(d);
        }
        logw[K] = log_prior_new + log_like_new;
    }

    int chosen = sample_from_log_weights(logw);

    if (chosen < K) {
        c[i] = chosen;
        cluster_counts[chosen]++;
    } else {
        int new_k = static_cast<int>(cluster_counts.size());
        cluster_counts.push_back(1);
        c[i] = new_k;
    }
}

// ---------- one full Gibbs sweep ----------
void CohesionRepulsionDP::gibbs_scan()
{
    for (int i = 0; i < n; ++i) {
        update_assignment_for_index(i);
    }
}

// ---------- run MCMC (with progress bars) ----------
void CohesionRepulsionDP::run_mcmc(int burn_in, int n_samples)
{
    if (burn_in < 0 || n_samples <= 0) {
        throw std::runtime_error("Invalid burn_in, n_samples in run_mcmc.");
    }

    std::cout << "[MCMC] Number of clusters found: " << n << std::endl;

    samples_c.clear();
    samples_c.reserve(n_samples);

    // 1. Burn-in Phase
    for (int it = 0; it < burn_in; ++it) {
        gibbs_scan();
        print_progress(it + 1, burn_in, "Burn-in ");
    }
    if (burn_in > 0) std::cout << std::endl;

    // 2. Sampling Phase
    for (int it = 0; it < n_samples; ++it) {
        gibbs_scan();
        samples_c.push_back(c);
        print_progress(it + 1, n_samples, "Sampling");
    }
    std::cout << std::endl;
}

// ---------- run MCMC to CSV (with progress bars) ----------
void CohesionRepulsionDP::run_mcmc_to_csv(int burn_in, int n_samples, const std::string& filename)
{
    if (burn_in < 0 || n_samples <= 0) {
        throw std::runtime_error("Invalid burn_in or n_samples in run_mcmc_to_csv.");
    }

    samples_c.clear();
    samples_c.reserve(n_samples);

    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    for (int i = 0; i < n; ++i) {
        outfile << "elem" << i;
        if (i != n-1) outfile << ",";
    }
    outfile << "\n";

    // 1. Burn-in Phase
    for (int it = 0; it < burn_in; ++it) {
        gibbs_scan();
        print_progress(it + 1, burn_in, "Burn-in ");
    }
    if (burn_in > 0) std::cout << std::endl;

    // 2. Sampling Phase
    for (int it = 0; it < n_samples; ++it) {
        gibbs_scan();
        samples_c.push_back(c);

        for (int i = 0; i < n; ++i) {
            outfile << c[i];
            if (i != n-1) outfile << ",";
        }
        outfile << "\n";

        print_progress(it + 1, n_samples, "Sampling");
    }
    std::cout << std::endl;

    outfile.close();
    std::cout << "Saved " << n_samples << " MCMC samples to " << filename << "\n";
}

void CohesionRepulsionDP::save_meta_cluster_assignments_csv(
    const std::vector<ComponentParameter>& phi_data,
    const std::string& filename) const
{
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    int Q = phi_data.size();
    auto samples_labels = get_samples_labels();
    int n_iter = samples_labels.size();

    out << "iteration";
    for (int i = 0; i < Q; ++i) {
        out << ",(" << phi_data[i].parameters[0] << "," << phi_data[i].parameters[1] << ")";
    }
    out << "\n";

    for (int it = 0; it < n_iter; ++it) {
        out << (it + 1);
        const auto& labels = samples_labels[it];
        for (int i = 0; i < Q; ++i) {
            out << "," << labels[i];
        }
        out << "\n";
    }

    out.close();
    std::cout << "Saved meta-cluster assignments CSV to: " << filename << "\n";
}