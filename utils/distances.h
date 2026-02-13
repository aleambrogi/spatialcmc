#ifndef DISTANCES_H
#define DISTANCES_H

#include <IModelPrior.h>

class Distances
{
public:
    enum class Type {
        Gaussian,
        Beta,
        Poisson
    };

    explicit Distances(Type type);

    double operator()(const ComponentParameter& a,
                      const ComponentParameter& b) const;

private:
    Type type_;

    double gaussian_distance(const ComponentParameter& a,
                             const ComponentParameter& b) const;

    double beta_distance(const ComponentParameter& a,
                         const ComponentParameter& b) const;

    double poisson_distance(const ComponentParameter& a,
                            const ComponentParameter& b) const;
};

#endif