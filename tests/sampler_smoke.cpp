#include "DP_MixtureModel.h"
#include "aPPM_MixtureModel.h"
#include "CohesionRepulsionDP.h"
#include <stdexcept>
#include <set>

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(const DPMixtureModel& model, int observations, int draws) {
    require(model.samples_c.size() == draws, "Incorrect number of saved draws");
    for (size_t j = 0; j < model.samples_c.size(); ++j) {
        require(model.samples_c[j].size() == observations, "Incorrect partition size");
        for (int label : model.samples_c[j]) {
            require(label >= 0, "Unassigned observation");
            const auto& phi = model.samples_phi_c[j].at(label).parameters;
            require(phi.size() == 2 && std::isfinite(phi[0]) &&
                    std::isfinite(phi[1]) && phi[1] > 0, "Invalid Gaussian parameters");
        }
    }
}
int main() {
    const std::vector<double> y{-4.8, -5.1, -4.9, 4.8, 5.1, 4.9};
    NormalConjugateModel prior(0, 1, 2, 2);
    prior.set_seed(42);
    DPMixtureModel dp(1, y, &prior);
    dp.run_mcmc(10, 20);
    check(dp, 6, 20);
    std::vector<std::vector<int>> graph{{1,2},{0,2},{0,1},{4,5},{3,5},{3,4}};
    prior.set_seed(42);
    aPPM_MixtureModel spatial(1, y, &prior, graph, 0.5);
    spatial.run_mcmc(10, 20);
    check(spatial, 6, 20);
    std::vector<std::vector<double>> distance(6, std::vector<double>(6));
    for (int i=0;i<6;++i) for(int j=0;j<6;++j) distance[i][j]=std::abs(y[i]-y[j]);
    CohesionRepulsionDP consensus(1, distance, .5, 1, 3, 1, 42);
    consensus.initialize_single_cluster();
    consensus.run_mcmc(10, 20);
    require(consensus.get_samples_labels().size()==20, "Missing consensus draws");
    for (const auto& draw:consensus.get_samples_labels()) {
        require(draw.size()==6, "Invalid consensus partition size");
        for(int label:draw) require(label>=0, "Invalid consensus label");
    }
    return 0;
}
