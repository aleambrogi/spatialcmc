#ifndef NEAL_ALGORITHM_APPM_MIXTUREMODEL_NEAL8_H
#define NEAL_ALGORITHM_APPM_MIXTUREMODEL_NEAL8_H

#include "aPPM_MixtureModel.h"

// ------------------------------------------------------------
// aPPM_MixtureModel_Neal8: Implements Neal's Algorithm 8
// for non-conjugate models (e.g., Beta) with spatial APPM
// ------------------------------------------------------------
class aPPM_MixtureModel_Neal8 : public aPPM_MixtureModel {
public:
    // ------------------------------------------------------------
    // Constructor: requires 'm_aux' for auxiliary components
    // ------------------------------------------------------------
    aPPM_MixtureModel_Neal8(double alpha,
                            const std::vector<double>& data,
                            IModelPrior* prior,
                            const std::vector<std::vector<int>>& adjacency,
                            double lambda,
                            int m_aux);

protected:
    // ------------------------------------------------------------
    // Overrides standard Gibbs step with Algorithm 8 logic
    // ------------------------------------------------------------
    void update_component_indicator(int i) override;

private:
    int m; // number of auxiliary components
};

#endif
