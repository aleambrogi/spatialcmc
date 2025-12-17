//
// Created by albor on 17/12/2025.
//

#ifndef NEAL_ALGORITHM_APPM_MIXTUREMODEL_NEAL8_H
#define NEAL_ALGORITHM_APPM_MIXTUREMODEL_NEAL8_H

#include "aPPM_MixtureModel.h"

class aPPM_MixtureModel_Neal8 : public aPPM_MixtureModel {
public:
    aPPM_MixtureModel_Neal8(double alpha,
                            const std::vector<double>& data,
                            IModelPrior* prior,
                            const std::vector<std::vector<int>>& adjacency,
                            double lambda,
                            int m_aux);

protected:
    void update_component_indicator(int i) override;

private:
    int m; // number of auxiliary components
};

#endif
