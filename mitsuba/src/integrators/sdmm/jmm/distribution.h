#ifndef __DISTRIBUTION_H
#define __DISTRIBUTION_H

// #define EIGEN_DONT_PARALLELIZE
// #define EIGEN_NO_MALLOC
#define EIGEN_INITIALIZE_MATRICES_BY_ZERO // TODO: THIS SHOULD NOT BE NECESSARY!
// #define EIGEN_STACK_ALLOCATION_LIMIT 0
// #define EIGEN_NO_DEBUG
// #define EIGEN_USE_MKL_ALL
// #define EIGEN_DONT_VECTORIZE
// #define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT

#include "eigen_boost_serialization.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>

namespace jmm {

template<int t_dims, typename Scalar>
inline static bool isInUnitHypercube(const Eigen::Matrix<Scalar, t_dims, 1>& v) {
    return !((v.array() < 0).any() || (v.array() > 1).any());
}

template<int t_dims, typename Scalar>
class Distribution {
public:
    using Vectord = Eigen::Matrix<Scalar, t_dims, 1>;
    using Matrixd = Eigen::Matrix<Scalar, t_dims, t_dims>;

    virtual ~Distribution() {}
    virtual Vectord sample(const std::function<Scalar()>& rng) const = 0;
    virtual Scalar pdf(const Vectord& sample) const = 0;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}

#endif /* __DISTRIBUTION_H */
