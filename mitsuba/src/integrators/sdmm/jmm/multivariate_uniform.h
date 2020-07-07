#ifndef __MULTIVARIATE_UNIFORM_H
#define __MULTIVARIATE_UNIFORM_H

#include <Eigen/Geometry>

#include "distribution.h"

namespace jmm {

template<int t_dims, typename Scalar>
class MultivariateUniform : public Distribution<t_dims, Scalar> {
public:
    using Vectord = Eigen::Matrix<Scalar, t_dims, 1>;

    MultivariateUniform(const Vectord& low_=Vectord::Zero(), const Vectord& high_=Vectord::Ones()) : low(low_), high(high_) {
        diff = high - low;
        volume = diff.prod();
    }

    Vectord sample(const std::function<Scalar()>& rng) const {
        Vectord uniform;
        for(int dim_i = 0; dim_i < t_dims; ++dim_i) {
            uniform(dim_i) = rng();
            if(uniform(dim_i) < 0 || uniform(dim_i) > 1) {
                throw std::runtime_error("RNG generating numbers outside of [0, 1].");
            }
        }
        return uniform.cwiseProduct(diff) + low;
    } 

    Scalar pdf(const Vectord& sample) const {
        Vectord normalized = (sample - low).cwiseProduct(diff.cwiseInverse());
        if(isInUnitHypercube(normalized)) {
            return 1.f / volume;
        }
        return 0.f;
    }
private:
    Vectord low;
    Vectord high;
    Vectord diff;
    Scalar volume;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}

#endif /* __MULTIVARIATE_UNIFORM_H */
