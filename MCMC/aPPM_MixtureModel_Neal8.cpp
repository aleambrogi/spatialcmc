#include "aPPM_MixtureModel_Neal8.h"
#include <limits>
#include <cmath>
#include <random>

// ------------------------------------------------------------
// Constructor: initializes Neal 8 with m auxiliary components
// ------------------------------------------------------------
aPPM_MixtureModel_Neal8::aPPM_MixtureModel_Neal8(
        double alpha,
        const std::vector<double>& data,
        IModelPrior* prior,
        const std::vector<std::vector<int>>& adjacency,
        double lambda,
        int m_aux)
: aPPM_MixtureModel(alpha, data, prior, adjacency, lambda),
  m(m_aux)
{
    if (m <= 0) throw std::runtime_error("Neal8: m_aux must be > 0");
}

// ------------------------------------------------------------
// Core Step: Updates cluster assignment for observation i
// ------------------------------------------------------------
void aPPM_MixtureModel_Neal8::update_component_indicator(int i)
{
    // 1) Remove i from its old cluster; delete empty clusters
    int old_cluster = c[i];
    if (old_cluster != -1) {
        int count = get_cluster_count_minus_i(i, old_cluster);
        if (count == 0) {
            phi_c.erase(old_cluster);
        }
    }

    // 2) Build candidate set: existing clusters + m auxiliary components
    std::vector<int> existing_labels;
    existing_labels.reserve(phi_c.size());

    std::vector<double> logw; // log-weights aligned with candidates
    logw.reserve(phi_c.size() + m);

    // Existing clusters: use APPM cluster_weight (count * like * spatial)
    for (auto& kv : phi_c) {
        int label = kv.first;
        const ComponentParameter& phi = kv.second;

        double w = cluster_weight(i, label, phi); // inherited from aPPM_MixtureModel
        if (w <= 0.0) {
            logw.push_back(-std::numeric_limits<double>::infinity());
        } else {
            logw.push_back(std::log(w));
        }
        existing_labels.push_back(label);
    }

    // ------------------------------------------------------------
    // 2b) Generate m auxiliary candidates from Base Measure (G0)
    // ------------------------------------------------------------
    std::vector<ComponentParameter> phi_aux;
    phi_aux.reserve(m);

    const double log_alpha_over_m = std::log(alpha) - std::log((double)m);

    for (int j = 0; j < m; ++j) {
        ComponentParameter phij = model_prior->sample_from_G0();
        phi_aux.push_back(phij);

        double like = model_prior->likelihood(y[i], phij);
        if (like <= 0.0) {
            logw.push_back(-std::numeric_limits<double>::infinity());
        } else {
            // New-cluster weight: (alpha/m) * like
            // (If later you want cohesion(singleton), multiply here; for singleton it is typically 1 in your current APPM.)
            logw.push_back(log_alpha_over_m + std::log(like));
        }
    }

    // ------------------------------------------------------------
    // 3) Sample new label using Log-Sum-Exp for stability
    // ------------------------------------------------------------
    double maxlog = -std::numeric_limits<double>::infinity();
    for (double lw : logw) if (lw > maxlog) maxlog = lw;

    // If all are -inf, something is wrong (e.g., likelihood returns 0 everywhere)
    if (!std::isfinite(maxlog)) {
        throw std::runtime_error("Neal8: all candidate weights are zero/invalid for observation i.");
    }

    std::vector<double> wexp(logw.size());
    double sum = 0.0;
    for (size_t t = 0; t < logw.size(); ++t) {
        wexp[t] = std::exp(logw[t] - maxlog);
        sum += wexp[t];
    }

    std::uniform_real_distribution<double> U(0.0, 1.0);
    double u = U(model_prior->generator) * sum;

    size_t chosen = wexp.size() - 1;
    double cum = 0.0;
    for (size_t t = 0; t < wexp.size(); ++t) {
        cum += wexp[t];
        if (u <= cum) { chosen = t; break; }
    }

    // 4) Assign i
    const size_t num_existing = existing_labels.size();
    if (chosen < num_existing) {
        c[i] = existing_labels[chosen];
    } else {
        // chose auxiliary => create new active cluster
        int new_label = next_component_id;
        phi_c[new_label] = phi_aux[chosen - num_existing];
        ++next_component_id;
        c[i] = new_label;
    }
}
