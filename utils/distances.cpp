#include "distances.h"

#include <cmath>
#include <IModelPrior.h>
#include <stdexcept>

// --------------------------------------------------

Distances::Distances(Type type)
    : type_(type) {}

// --------------------------------------------------

double Distances::operator()(const ComponentParameter& a,
                             const ComponentParameter& b) const
{
    switch (type_)
    {
        case Type::Gaussian:
            return gaussian_distance(a, b);

        case Type::Beta:
            return beta_distance(a, b);

        case Type::Poisson:
            return poisson_distance(a, b);
    }

    throw std::runtime_error("Unknown distance type");
}

// --------------------------------------------------
// NORMALE: (mu, sqrt(var))
// --------------------------------------------------

double Distances::gaussian_distance(const ComponentParameter& a,
                                    const ComponentParameter& b) const
{
    double mu1 = a.parameters[0];
    double s1  = std::sqrt(a.parameters[1]);

    double mu2 = b.parameters[0];
    double s2  = std::sqrt(b.parameters[1]);

    return std::hypot(mu1 - mu2, s1 - s2);
}

// --------------------------------------------------
// BETA: (mean, log precision)
// --------------------------------------------------

double Distances::beta_distance(const ComponentParameter& a,
                                const ComponentParameter& b) const
{
    double a1 = a.parameters[0];
    double b1 = a.parameters[1];
    double a2 = b.parameters[0];
    double b2 = b.parameters[1];

    double mu1  = a1 / (a1 + b1);
    double mu2  = a2 / (a2 + b2);

    double phi1 = std::log(a1 + b1);
    double phi2 = std::log(a2 + b2);

    return std::hypot(mu1 - mu2, phi1 - phi2);
}

// --------------------------------------------------
// POISSON: sqrt(lambda)
// --------------------------------------------------

double Distances::poisson_distance(const ComponentParameter& a,
                                   const ComponentParameter& b) const
{
    double l1 = std::sqrt(a.parameters[0]);
    double l2 = std::sqrt(b.parameters[0]);

    return std::fabs(l1 - l2);
}