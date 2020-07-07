#include "mixture_model.h"
#include "multivariate_tangent_normal.h"
#include "multivariate_normal.h"

template class jmm::MixtureModel<
    3,
    8192,
    3,
    double,
    jmm::MultivariateTangentNormal,
    jmm::MultivariateNormal
>;

template class jmm::MixtureModel<
    4,
    8192,
    3,
    double,
    jmm::MultivariateTangentNormal,
    jmm::MultivariateNormal
>;