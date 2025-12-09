//
// Created by albor on 14/11/2025.
//

#include "DP_MixtureModel.h"

DPMixtureModel::DPMixtureModel(double alpha_, const std::vector<double>& data, IModelPrior *prior):
alpha(alpha_), y(data), model_prior(prior)
    {
        n = y.size();
        c.resize(n, -1);  // initialize all indicators to -1 (unassigned)
        next_component_id = 0;  // will assign new clusters starting from 0
    }

void DPMixtureModel::run_mcmc(int burn_in, int sampling)
{
    using clock = std::chrono::high_resolution_clock;
    auto start_time = clock::now();

    if (burn_in < 0 || sampling <= 0) {
        throw std::runtime_error("run_mcmc: burn_in must be >= 0 and sampling > 0.");
    }

    samples_c.clear();
    samples_phi_c.clear();
    samples_c.reserve(sampling);
    samples_phi_c.reserve(sampling);

    // --------------------------
    // BURN-IN PHASE
    // --------------------------
    //std::cout << "Burn-in (" << burn_in << " iterations)...\n";

    for (int it = 0; it < burn_in; ++it) {
        gibbs_scan();

        //std::cout << "\rBurn-in iteration " << (it + 1)
               //   << " / " << burn_in << std::flush;
    }

    //if (burn_in > 0) std::cout << "\nBurn-in completed.\n";

    // --------------------------
    // SAMPLING PHASE
    // --------------------------
    //std::cout << "Sampling (" << sampling << " iterations)...\n";

    for (int it = 0; it < sampling; ++it) {
        gibbs_scan();

        // Store samples
        samples_c.push_back(c);
        samples_phi_c.push_back(phi_c);

        //std::cout << "\rSample iteration " << (it + 1)
                 // << " / " << sampling << std::flush;
    }

    auto end_time = clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    //std::cout << "\nMCMC completed in "
           //   << elapsed.count() << " seconds.\n";
    //std::cout << "Total saved samples: "
        //      << samples_c.size() << "\n";
}

void DPMixtureModel::gibbs_scan() {
    for (int i = 0; i < n; ++i) {
        update_component_indicator(i);
    }
    update_component_parameters();
}

std::vector<std::vector<double>> DPMixtureModel::compute_co_clustering() const
{
    int S = samples_c.size();
    if (S == 0) {
        throw std::runtime_error("No MCMC samples available.");
    }

    int N = samples_c[0].size();

    std::vector<std::vector<double>> P(N, std::vector<double>(N, 0.0));

    // Count co-occurrences
    for (int s = 0; s < S; ++s) {
        const auto& c = samples_c[s];
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                if (c[i] == c[j]) {
                    P[i][j] += 1.0;
                }
            }
        }
    }

    // Normalize
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            P[i][j] /= S;
        }
    }

    return P;
}


double DPMixtureModel::binder_expected_loss(
        const std::vector<int>& est,
        const std::vector<std::vector<double>>& P,
        double a,
        double b) const
{
    int N = est.size();
    double loss = 0.0;

    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {

            bool same_est = (est[i] == est[j]);
            double pij = P[i][j];

            if (same_est) {
                loss += a * (1.0 - pij);
            } else {
                loss += b * pij;
            }
        }
    }

    return loss;
}


int DPMixtureModel::opt_param_index_fast() const
{
    if (samples_c.empty()) {
        throw std::runtime_error("No MCMC samples available for opt_param_index_fast().");
    }

    int S = samples_c.size();

    // Build co-clustering matrix
    auto P = compute_co_clustering();

    double best_loss = std::numeric_limits<double>::infinity();
    int best_index = 0;

    // Evaluate expected Binder loss for each sample
    for (int s = 0; s < S; ++s) {
        double loss = binder_expected_loss(samples_c[s], P);

        if (loss < best_loss) {
            best_loss = loss;
            best_index = s;
        }
    }

    return best_index;
}


double DPMixtureModel::cluster_weight(int i,
                                      int cluster_label,
                                      const ComponentParameter& phi)
{
    int count = get_cluster_count_minus_i(i, cluster_label);
    double like = model_prior->likelihood(y[i], phi);
    return count * like;   // standard DP mixture formula
}

double DPMixtureModel::new_cluster_weight(int i)
{
    return alpha * compute_prior_predictive(y[i]);
}


void DPMixtureModel::update_component_indicator(int i) {
    //remove i from the current cluster
    int old_cluster = c[i];
    if (old_cluster != -1) {
        // Decrease cluster count
        int count = get_cluster_count_minus_i(i, old_cluster);
        if (count == 0) {
            // Remove empty cluster
            phi_c.erase(old_cluster);
        }
    }
    //compute probabilities
    std::vector<double> probs;
    std::vector<int> cluster_labels;

    // Existing clusters
    for (auto& kv : phi_c) {
        int label = kv.first;
        ComponentParameter& phi = kv.second;
        probs.push_back(cluster_weight(i, label, phi));
        cluster_labels.push_back(label);
    }

    // New cluster
    double new_prob = new_cluster_weight(i);
    probs.push_back(new_prob);
    cluster_labels.push_back(next_component_id);  // temporary label

    //sample cluster assignements
    // Normalize probabilities
    double sum = std::accumulate(probs.begin(), probs.end(), 0.0);
    for (auto& p : probs) p /= sum;

    // Draw a cluster
    double u = ((double) rand() / RAND_MAX);
    double cum = 0.0;
    int selected = cluster_labels.back();  // default: new cluster
    for (size_t k = 0; k < probs.size(); ++k) {
        cum += probs[k];
        if (u < cum) {
            selected = cluster_labels[k];
            break;
        }
    }
    c[i] = selected;

    // If selected is a new cluster, generate phi_c from G0
    if (selected == next_component_id) {
        phi_c[selected] = model_prior->sample_from_G0();
        ++next_component_id;
    }

}

void DPMixtureModel::update_component_parameters() {
    for (auto& kv : phi_c) {
        int label = kv.first;
        std::vector<double> cluster_data;
        for (int i = 0; i < n; ++i) {
            if (c[i] == label)
                cluster_data.push_back(y[i]);
        }
        kv.second = model_prior->sample_from_posterior(cluster_data);
    }
}


int DPMixtureModel::get_cluster_count_minus_i(int i, int c_label) {
    int count = 0;
    for (int j = 0; j < n; ++j) {
        if (j != i && c[j] == c_label) count++;
    }
    return count;
}


double DPMixtureModel::compute_prior_predictive(double y_i) {
    return model_prior->calculate_prior_predictive(y_i);
}


int DPMixtureModel::binder_loss(const std::vector<int>& a,
                                const std::vector<int>& b)
{
    if (a.size() != b.size()) {
        throw std::runtime_error("binder_loss: clusterings must have same length.");
    }

    int N = a.size();
    int loss = 0;

    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            bool same_a = (a[i] == a[j]);
            bool same_b = (b[i] == b[j]);
            if (same_a != same_b) {
                loss++;
            }
        }
    }

    return loss;
}
/*
int DPMixtureModel::opt_param_index() const
{
    if (samples_c.empty()) {
        throw std::runtime_error("No MCMC samples available for opt().");
    }

    int S = samples_c.size();     // number of MCMC draws

    // Evaluate loss for each sample against all others
    int best_index = 0;
    long best_total_loss = std::numeric_limits<long>::max();

    for (int s = 0; s < S; ++s) {
        long total_loss = 0;

        for (int t = 0; t < S; ++t) {
            if (s == t) continue;
            total_loss += binder_loss(samples_c[s], samples_c[t]);
        }

        if (total_loss < best_total_loss) {
            best_total_loss = total_loss;
            best_index = s;
        }
    }

    // Return the optimal cluster index (smallest binder loss)
    return best_index;
}
*/