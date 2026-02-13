//
// Created by albor on 30/11/2025.
//

#include "aPPM_MixtureModel.h"

// Constructor
aPPM_MixtureModel::aPPM_MixtureModel(
        double alpha,
        const std::vector<double>& data,
        IModelPrior* prior,
        const std::vector<std::vector<int>>& adjacency,
        double lambda)
    :
      DPMixtureModel(alpha, data, prior),
      neighbors(adjacency),
      lambda_appm(lambda)
{}

double aPPM_MixtureModel::cluster_weight(
        int i,
        int cluster_label,
        const ComponentParameter& phi)
{
    // 1) DP size term n_{-i,k}
    int count = get_cluster_count_minus_i(i, cluster_label);
    if (count <= 0) return 0.0;

    // 2) Likelihood term
    double like = model_prior->likelihood(y[i], phi);

    // 3) Spatial term: how many neighbors of i are already in cluster k?
    int neighbors_in_cluster = 0;
    for (int nb : neighbors[i]) {
        if (nb != i && c[nb] == cluster_label) {
            neighbors_in_cluster++;
        }
    }

    double spatial = std::exp(lambda_appm * neighbors_in_cluster);

    return count * like * spatial;
}



double aPPM_MixtureModel::new_cluster_weight(int i)
{
    double prior_pred = compute_prior_predictive(y[i]);

    // new cluster consists only of observation i
    std::vector<int> singleton = {i};

    double cohesion = std::exp(log_cohesion(singleton));

    return alpha * prior_pred * cohesion;
}


int aPPM_MixtureModel::count_internal_edges(
        const std::vector<int>& members) const
{
    std::unordered_set<int> S(members.begin(), members.end());
    int edges = 0;

    for (int v : members) {
        for (int u : neighbors[v]) {
            if (S.count(u))
                edges++;
        }
    }

    return edges / 2; // undirected graph
}

bool aPPM_MixtureModel::is_connected(
        const std::vector<int>& members) const
{
    if (members.empty()) return true;
    if (members.size() == 1) return true;

    std::unordered_set<int> S(members.begin(), members.end());
    std::unordered_set<int> visited;

    std::queue<int> q;
    q.push(members[0]);

    while (!q.empty()) {
        int v = q.front();
        q.pop();
        visited.insert(v);

        for (int u : neighbors[v])
            if (S.count(u) && !visited.count(u))
                q.push(u);
    }

    return visited.size() == S.size();
}

double aPPM_MixtureModel::log_cohesion(
        const std::vector<int>& members) const
{
    if (!is_connected(members))
        return -std::numeric_limits<double>::infinity();

    int E = count_internal_edges(members);
    return lambda_appm * E;
}
