//
// Created by albor on 01/12/2025.
//

#include "CohesionRepulsionDP.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>

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
// (comparing each sample to all samples)
// ------------------------------------------------------------
int CohesionRepulsionDP::best_sample_binder() const
{
    int T = samples_c.size();
    if (T == 0)
        throw std::runtime_error("No samples stored. Run MCMC first.");

    int best_index = 0;
    long best_loss = std::numeric_limits<long>::max();

    for (int t = 0; t < T; ++t) {
        long total_loss = 0;

        for (int s = 0; s < T; ++s) {
            total_loss += binder_loss(samples_c[t], samples_c[s]);
        }

        if (total_loss < best_loss) {
            best_loss = total_loss;
            best_index = t;
        }
    }

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

    // Precompute log-constants for Gamma densities
    // Gamma(shape=delta, rate=lambda):
    // log f(d) = delta*log(lambda) - lgamma(delta) + (delta-1)*log(d) - lambda*d
    logC_in  = delta1 * std::log(lambda_in)  - std::lgamma(delta1);
    logC_out = delta2 * std::log(theta_out) - std::lgamma(delta2);

    // default init: single cluster
    initialize_single_cluster();
}

// ---------- initialization: all items in one cluster ----------
void CohesionRepulsionDP::initialize_single_cluster()
{
    c.assign(n, 0);
    cluster_counts.assign(1, n);
}

// ---------- initialization: random labels up to max_init_clusters ----------
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

    // compress labels so that only used labels remain
    // and cluster_counts are correct
    // (we'll rebuild cluster_counts)
    int max_label = *std::max_element(c.begin(), c.end());
    std::vector<int> tmp_counts(max_label + 1, 0);
    for (int i = 0; i < n; ++i) {
        tmp_counts[c[i]]++;
    }
    // build mapping old_label -> new_label (keeping only non-empty)
    std::vector<int> label_map(max_label + 1, -1);
    int newK = 0;
    for (int old = 0; old <= max_label; ++old) {
        if (tmp_counts[old] > 0) {
            label_map[old] = newK++;
        }
    }
    // relabel c[i]
    for (int i = 0; i < n; ++i) {
        c[i] = label_map[c[i]];
    }
    // build cluster_counts
    cluster_counts.assign(newK, 0);
    for (int i = 0; i < n; ++i) {
        cluster_counts[c[i]]++;
    }
}

// ---------- log Gamma pdfs for cohesion / repulsion ----------
double CohesionRepulsionDP::log_f_in(double d) const
{
    // within-cluster distance density
    if (d <= 0.0) {
        // Ideally should never happen; distances should be > 0
        return -1e30;
    }
    return logC_in + (delta1 - 1.0) * std::log(d) - lambda_in * d;
}

double CohesionRepulsionDP::log_g_out(double d) const
{
    // between-cluster distance density
    if (d <= 0.0) {
        // repulsive Gamma has 0 density at d=0 when delta2>1
        return -1e30;
    }
    return logC_out + (delta2 - 1.0) * std::log(d) - theta_out * d;
}

// ---------- helper: sample from unnormalised log-weights ----------
int CohesionRepulsionDP::sample_from_log_weights(const std::vector<double>& logw)
{
    const int K = static_cast<int>(logw.size());
    if (K == 0) {
        throw std::runtime_error("sample_from_log_weights: empty weights.");
    }

    // subtract max for numerical stability
    double max_logw = logw[0];
    for (int k = 1; k < K; ++k) {
        if (logw[k] > max_logw) {
            max_logw = logw[k];
        }
    }

    std::vector<double> w(K);
    double sum_w = 0.0;
    for (int k = 0; k < K; ++k) {
        double val = std::exp(logw[k] - max_logw);
        w[k] = val;
        sum_w += val;
    }

    if (sum_w <= 0.0 || !std::isfinite(sum_w)) {
        // fallback: uniform
        std::uniform_int_distribution<int> unif(0, K - 1);
        return unif(rng);
    }

    std::uniform_real_distribution<double> u_dist(0.0, sum_w);
    double u = u_dist(rng);
    double cum = 0.0;
    for (int k = 0; k < K; ++k) {
        cum += w[k];
        if (u <= cum) {
            return k;
        }
    }
    return K - 1; // safety
}

// ---------- remove item i from its cluster ----------
void CohesionRepulsionDP::remove_item_from_cluster(int i)
{
    int old_k = c[i];
    if (old_k < 0 || old_k >= static_cast<int>(cluster_counts.size())) {
        throw std::runtime_error("Invalid cluster index in remove_item_from_cluster.");
    }

    cluster_counts[old_k]--;
    c[i] = -1; // mark as unassigned for now

    if (cluster_counts[old_k] == 0) {
        check_empty_and_relabel(old_k);
    }
}

// ---------- if a cluster is empty, drop it and relabel higher indices ----------
void CohesionRepulsionDP::check_empty_and_relabel(int emptied_cluster)
{
    // remove that cluster from counts
    cluster_counts.erase(cluster_counts.begin() + emptied_cluster);

    // relabel assignments c[i] > emptied_cluster
    for (int i = 0; i < n; ++i) {
        if (c[i] > emptied_cluster) {
            c[i]--;
        }
    }
}

// ---------- update assignment for index i (Gibbs step) ----------
void CohesionRepulsionDP::update_assignment_for_index(int i)
{
    // 1. Remove i from its current cluster (and possibly drop that cluster)
    remove_item_from_cluster(i);

    const int K = static_cast<int>(cluster_counts.size());

    // 2. Build log weights for each candidate cluster (existing + "new")
    // candidates: 0..K-1 existing, K = new cluster
    std::vector<double> logw(K + 1, 0.0);

    // For each candidate cluster k, compute:
    // log posterior ∝ log prior + log likelihood contribution of i's distances
    // Prior (DP): existing cluster k: log(n_k), new cluster: log(alpha)
    // Likelihood: sum over j != i of log f_in(d_ij) or log g_out(d_ij)
    // Only pairs involving i change; others are constant wrt assignment of i.

    // Precompute which j's are in which cluster to avoid repeated scanning
    // (but for simplicity we scan c[j] directly, which is O(n*K) worst-case
    // and is fine for moderate n).

    // existing clusters
    for (int k = 0; k < K; ++k) {
        double log_prior = std::log(static_cast<double>(cluster_counts[k]));
        double log_like = 0.0;

        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            int cj = c[j];
            if (cj == -1) continue; // j might be temporarily unassigned only when j==i

            double d = D[i][j];
            if (cj == k) {
                // within-cluster pair (i,j)
                log_like += log_f_in(d);
            } else {
                // between-cluster pair
                log_like += log_g_out(d);
            }
        }
        logw[k] = log_prior + log_like;
    }

    // new cluster candidate (label = K)
    {
        double log_prior_new = std::log(alpha);
        double log_like_new = 0.0;

        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            int cj = c[j];
            if (cj == -1) continue;

            double d = D[i][j];
            // new cluster has no other members, so every pair (i,j) is "between"
            log_like_new += log_g_out(d);
        }

        logw[K] = log_prior_new + log_like_new;
    }

    // 3. Sample cluster index from these log-weights
    int chosen = sample_from_log_weights(logw);

    if (chosen < K) {
        // existing cluster
        c[i] = chosen;
        cluster_counts[chosen]++;
    } else {
        // new cluster
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

// ---------- run MCMC ----------
void CohesionRepulsionDP::run_mcmc(int burn_in, int n_samples)
{
    if (burn_in < 0 || n_samples <= 0) {
        throw std::runtime_error("Invalid burn_in, n_samples or thin in run_mcmc.");
    }

    samples_c.clear();
    samples_c.reserve(n_samples);



    for (int it = 0; it < burn_in; ++it) {
        gibbs_scan();
        //std::cout << "Burn-in, iteration: " << it << "\n";
    }

    for (int it = 0; it < n_samples; ++it) {
        gibbs_scan();
        //std::cout << "Sampling , iteration: " << it << "\n";
        samples_c.push_back(c);
    }
}
