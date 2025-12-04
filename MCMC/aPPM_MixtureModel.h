//
// Created by albor on 30/11/2025.
//

#ifndef NEAL_ALGORITHM_APPM_MIXTUREMODEL_H
#define NEAL_ALGORITHM_APPM_MIXTUREMODEL_H

#include "DP_MixtureModel.h"
#include <unordered_set>
#include <queue>

class aPPM_MixtureModel : public DPMixtureModel {
private:
    std::vector<std::vector<int>> neighbors;
    double lambda_appm;

public:
    aPPM_MixtureModel(double alpha,
                      const std::vector<double>& data,
                      IModelPrior* prior,
                      const std::vector<std::vector<int>>& adjacency,
                      double lambda);

private:
    // Override cluster weight in DP mixture
    double cluster_weight(int i,
                          int cluster_label,
                          const ComponentParameter& phi) override;

    // Override weight of a potential new cluster
    double new_cluster_weight(int i) override;
    // APPM helpers
    [[nodiscard]] double log_cohesion(const std::vector<int>& members) const;
    [[nodiscard]] int count_internal_edges(const std::vector<int>& members) const;
    [[nodiscard]] bool is_connected(const std::vector<int>& members) const;
};

#endif //NEAL_ALGORITHM_APPM_MIXTUREMODEL_H