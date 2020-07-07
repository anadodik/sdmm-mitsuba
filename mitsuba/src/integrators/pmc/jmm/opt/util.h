#ifndef __MIXTURE_MODEL_OPT_UTIL_H
#define __MIXTURE_MODEL_OPT_UTIL_H

#include <vector>
#include <functional>
#include <numeric>
#include <cassert>
#include <atomic>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>

#include <boost/thread.hpp>
#include <boost/thread/barrier.hpp>

#include "../kdtree-eigen/kdtree_eigen.h"

#include "../distribution.h"
#include "../mixture_model.h"
#include "../samples.h"
#include "../sphere_volume.h"


namespace jmm {


template<int t_dims, typename Scalar>
bool isPositiveDefinite(const Eigen::Matrix<Scalar, t_dims, t_dims>& covMatrix) {
    using Matrixd = Eigen::Matrix<Scalar, t_dims, t_dims>;
    using Vectord = Eigen::Matrix<Scalar, t_dims, 1>;
    Matrixd copy = covMatrix;
    Eigen::SelfAdjointEigenSolver<Matrixd> solver;
    solver.compute(copy);
    if(solver.info() == Eigen::NumericalIssue) {
        std::cerr << "Cannot compute eigendecomposition of matrix\n";
    }
    Vectord eigenvalues = solver.eigenvalues();
    return (eigenvalues.array() > 0.f).all();
}

template<int t_dims, typename Scalar>
void makePositiveDefinite(
    Eigen::Matrix<Scalar, t_dims, t_dims>& covMatrix,
    Scalar epsilon,
    int component_i
) {
    using Matrixd = Eigen::Matrix<Scalar, t_dims, t_dims>;
    using Vectord = Eigen::Matrix<Scalar, t_dims, 1>;
    Matrixd copy = covMatrix;
    Eigen::SelfAdjointEigenSolver<Matrixd> solver;
    solver.compute(copy);
    if(solver.info() == Eigen::NumericalIssue) {
        std::cerr << "Cannot compute eigendecomposition of matrix\n";
    }
    Vectord eigenvalues = solver.eigenvalues();
    if((eigenvalues.array() <= 0.f).any()) {
        Matrixd eigenvectors = solver.eigenvectors();
        for(int dim_i = 0; dim_i < t_dims; ++dim_i) {
            if(eigenvalues(dim_i) <= 0.f) {
                covMatrix += (epsilon - eigenvalues(dim_i)) * eigenvectors.col(dim_i) * eigenvectors.col(dim_i).transpose();
                std::cerr << "component = " << component_i << ", eigenvalue= " << eigenvalues(dim_i)
                    << ", eigenvector = " << eigenvectors.col(dim_i).transpose() 
                    << ", epsilon = " << epsilon << '\n';
            }
        }
    }
}

}

#endif /* __MIXTURE_MODEL_OPT_UTIL_H */