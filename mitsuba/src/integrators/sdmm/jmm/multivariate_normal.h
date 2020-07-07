#ifndef __MULTIVARIATE_NORMAL_H
#define __MULTIVARIATE_NORMAL_H

#include "distribution.h"

#include <vector>
#include <iostream>

#include <boost/math/distributions/chi_squared.hpp>

#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#define USE_CANONICAL 0

namespace jmm {

template<int t_dims, int t_conditionalDims, typename Scalar>
class MultivariateNormal : public Distribution<t_dims, Scalar> {
public:
    constexpr static int t_conditionDims = t_dims - t_conditionalDims;

    using Vectord = Eigen::Matrix<Scalar, t_dims, 1>;
    using Matrixd = Eigen::Matrix<Scalar, t_dims, t_dims>;

    using ConditionalVectord = Eigen::Matrix<Scalar, t_conditionalDims, 1>;
    using ConditionalMatrixd = Eigen::Matrix<Scalar, t_conditionalDims, t_conditionalDims>;

    using ConditionVectord = Eigen::Matrix<Scalar, t_conditionDims, 1>;
    using ConditionMatrixd = Eigen::Matrix<Scalar, t_conditionDims, t_conditionDims>;

    using AABB = Eigen::AlignedBox<Scalar, t_dims>;
    
    double containedMass = 0.99;

    MultivariateNormal(const Vectord& mean, const Matrixd& cov) {
        set(mean, cov);
    }

    MultivariateNormal() {
        set(Vectord::Zero(), Matrixd::Identity());
    }
    
    ~MultivariateNormal() = default;
    MultivariateNormal(const MultivariateNormal& other) = default;
    MultivariateNormal(MultivariateNormal&& other) noexcept = default;
    MultivariateNormal& operator=(const MultivariateNormal& other) = default;
    MultivariateNormal& operator=(MultivariateNormal&& other) = default;

    const Vectord& mean() const { return m_mean; }
    const Matrixd& cov() const { return m_cov; }
    const Eigen::LLT<Matrixd>& cholesky() const { return m_cholesky; }

    template <int dimCheck = t_dims>
    typename std::enable_if<(dimCheck == 0), bool>::type
    set(const Vectord& mean, const Matrixd& cov) { return true; }

    template <int dimCheck = t_dims>
    typename std::enable_if<(dimCheck > 0), bool>::type
    set(const Vectord& mean, const Matrixd& cov) {
        if(t_dims == 0) {
            return true;
        }
        m_mean = mean;
        if((cov.array() != cov.transpose().array()).all()) { // !cov.isApprox(cov.transpose())) {
            m_cov = (cov + cov.transpose()) / 2.f;
            std::cerr << "COV MAT NOT SYMMETRIC " << cov << "\n";
        } else {
            m_cov = cov;
        }

#if USE_CANONICAL
        m_covCanonical = m_cov.inverse();
        if(!m_covCanonical.isApprox(m_covCanonical.transpose())) {
            m_covCanonical = (m_covCanonical + m_covCanonical.transpose()) / 2.f;
        }
        m_meanCanonical = m_covCanonical * m_mean;
#endif

        if(t_conditionalDims != 0) {
            precomputeConditioning();
        }

        m_cholesky = m_cov.llt();
        if(m_cholesky.info() == Eigen::NumericalIssue) {
            std::cerr << "MATRIX NOT PD!" << "\n";
        }
        if(m_cholesky.info() != Eigen::Success) {
            std::stringstream ss;
            ss << "ERROR CREATING DISTRIBUTION WITH\nMEAN=\n" << mean.transpose() << ", AND COV=\n" << cov << "\n\n";
            std::cerr << ss.str();
            return false;
        }
        m_detInv = 1.f / m_cholesky.matrixL().determinant();
		bool aabbSuccess;
		m_aabb = getAABB(aabbSuccess);
        return true;
    }

    Vectord toStandardNormal(const Vectord& sample) const {
        return m_cholesky.matrixL().solve(sample - m_mean);
    }

    Vectord sample(const std::function<Scalar()>& rng) const {
        // TODO: Ziggurat or vectorized box-mueller
        Vectord sample;
        for(int dim_i = 0; dim_i < t_dims; dim_i += 2) {
            Eigen::Matrix<Scalar, 2, 1> normal2d = boxMullerTransform(rng(), rng());
            sample(dim_i) = normal2d(0);
            if(dim_i + 1 < t_dims) {
                sample(dim_i + 1) = normal2d(1);
            }
        }
        return m_cholesky.matrixL() * sample + m_mean;
    }
    
    Scalar pdf(const Vectord& sample, bool& isInside) const {
        static const boost::math::chi_squared chiSquared(t_dims);
        static const Scalar maxRadiusSqr = (Scalar) boost::math::quantile(chiSquared, containedMass);
        constexpr static Scalar INV_SQRT_TWO_PI = 0.39894228040143267793994605993438186847585863116492;
        constexpr static Scalar NORMALIZATION = std::pow(INV_SQRT_TWO_PI, t_dims);
        Scalar standardizedNormSqr = toStandardNormal(sample).squaredNorm();
        isInside = standardizedNormSqr < maxRadiusSqr;
        
        Scalar pdf = NORMALIZATION * std::exp(-0.5 * standardizedNormSqr);
        return pdf * m_detInv;
    }

    Scalar pdf(const Vectord& sample) const {
        constexpr static Scalar INV_SQRT_TWO_PI = 0.39894228040143267793994605993438186847585863116492;
        constexpr static Scalar NORMALIZATION = std::pow(INV_SQRT_TWO_PI, t_dims);
        
        Scalar pdf = NORMALIZATION * std::exp(-0.5 * toStandardNormal(sample).squaredNorm());
        return pdf * m_detInv;
    }

    template <int dimCheck = t_conditionDims>
    typename std::enable_if<(dimCheck == 0), void>::type precomputeConditioning() {}

    template <int dimCheck = t_conditionDims>
    typename std::enable_if<(dimCheck > 0), void>::type precomputeConditioning() {
        auto muA = m_mean.template topRows<t_conditionDims>();
        auto muB = m_mean.template bottomRows<t_conditionalDims>();
        auto covAA = m_cov.template topLeftCorner<t_conditionDims, t_conditionDims>();
        auto covAB = m_cov.template topRightCorner<t_conditionDims, t_conditionalDims>();
        auto covBA = m_cov.template bottomLeftCorner<t_conditionalDims, t_conditionDims>();
        auto covBB = m_cov.template bottomRightCorner<t_conditionalDims, t_conditionalDims>();

        // if(!covBA.isApprox(covAB.transpose())) {
        //     std::cerr << "covBA: " << covBA << "\ncovAB.T: " << covAB.transpose() << "\n";
        // }
        // if(!m_cov.isApprox(m_cov.transpose())) {
        //     std::cerr << "cov: " << m_cov << "\ncov.T: " << m_cov.transpose() << "\n";
        // }
        // assert(covBA.isApprox(covAB.transpose()));
        auto covAAInv = covAA.inverse();

        m_muA = muA;
        m_muB = muB;
        m_conditionalCov = covBB - covBA * covAAInv * covAB;
        m_muPremult = covBA * covAAInv;
    }

    template <int dimCheck = t_conditionDims>
    typename std::enable_if<(dimCheck == 0), bool>::type conditional(
        const ConditionVectord& condition,
        MultivariateNormal<t_conditionalDims, t_conditionalDims, Scalar>& conditional
    ) const { return true; }

    // https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf
    template <int dimCheck = t_conditionDims>
    typename std::enable_if<(dimCheck > 0), bool>::type conditional(
        const ConditionVectord& condition,
        MultivariateNormal<t_conditionalDims, t_conditionalDims, Scalar>& conditional
    ) const {
#if USE_CANONICAL
        auto muB = m_meanCanonical.template bottomRows<t_conditionalDims>();
        auto covBA = m_covCanonical.template bottomLeftCorner<t_conditionalDims, t_conditionDims>();
        auto covBB = m_covCanonical.template bottomRightCorner<t_conditionalDims, t_conditionalDims>();

        auto conditionalCov = covBB.inverse();
        auto conditionalMean = conditionalCov * (muB - covBA * condition);
        return conditional.set(std::move(conditionalMean), std::move(conditionalCov));
#else
        ConditionalVectord conditionalMean = m_muB + m_muPremult * (condition - m_muA);
        return conditional.set(conditionalMean, m_conditionalCov);
#endif
    }

    template<int t_marginalDims=t_conditionDims>
    typename std::enable_if<(t_marginalDims == 0), void>::type
    marginal(MultivariateNormal<t_marginalDims, t_marginalDims, Scalar>& m) const {
    }

    template<int t_marginalDims=t_conditionDims>
    typename std::enable_if<(t_marginalDims > 0), void>::type
    marginal(MultivariateNormal<t_marginalDims, t_marginalDims, Scalar>& m) const {
        auto muA = m_mean.template topRows<t_marginalDims>();
        auto covAA = m_cov.template topLeftCorner<t_marginalDims, t_marginalDims>();
        m.set(muA, covAA);
    }

    bool isInside(const Vectord& sample, Scalar scale) const {
        static const boost::math::chi_squared chiSquared(t_dims);
        static const Scalar maxRadius = std::sqrt((Scalar) boost::math::quantile(chiSquared, containedMass));
        Vectord standardized = toStandardNormal(sample);
        return standardized.norm() < maxRadius;
    }

    bool isInsideAABB(const Vectord& sample) const {
		return m_aabb.contains(sample);
    }

    template <int dimCheck = t_dims>
    typename std::enable_if<(dimCheck == 0), 
        Eigen::AlignedBox<Scalar, t_dims>
    >::type getAABB(bool& success) {
        success = false;
        return {};
    }

    template <int dimCheck = t_dims>
    typename std::enable_if<(dimCheck > 0), 
        Eigen::AlignedBox<Scalar, t_dims>
    >::type getAABB(bool& success) {
        static const boost::math::chi_squared chiSquared(t_dims);
        static const Scalar maxRadius = std::sqrt((Scalar) boost::math::quantile(chiSquared, containedMass));
        
        Eigen::SelfAdjointEigenSolver<Matrixd> solver;
        solver.compute(m_cov);
        Eigen::AlignedBox<Scalar, t_dims> aabb(m_mean);
        if(solver.info() != Eigen::Success) {
            std::cerr << "Cannot compute eigendecomposition of matrix for AABB. \n";
            success = false;
            return aabb;
        }
        Vectord eigenvalues = solver.eigenvalues().array();
        Matrixd eigenvectors = solver.eigenvectors();
        Vectord lengths = maxRadius * eigenvalues.array().sqrt();


        Matrixd Lambda = Matrixd::Zero();
        for(int d = 0; d < t_dims; ++d) {
            Lambda += eigenvectors.col(d) * eigenvectors.col(d).transpose() / (lengths(d) * lengths(d));
        }
        Vectord lambdas_sqr = Lambda.inverse().diagonal();
        if(lambdas_sqr.array().isInf().any()) {
            std::cerr << "Lambas inf: " << lambdas_sqr.transpose() << "\n";
            success = false;
            return aabb;
        }
        for(int d = 0; d < t_dims; ++d) {
            Vectord point = Vectord::Zero();

            point(d) = std::sqrt(lambdas_sqr(d));
            aabb.extend(m_mean + point);
            point(d) = -std::sqrt(lambdas_sqr(d));
            aabb.extend(m_mean + point);
        }
        success = true;
        return aabb;
    }

    Scalar KLDivergence(const MultivariateNormal& other) const {
        Scalar detj = other.m_cov.determinant();
        Scalar deti = m_cov.determinant();
        Scalar logDetRatio = std::log(detj) - std::log(deti);
        Matrixd invj = m_cov.inverse();
        Scalar trace = (invj * m_cov).trace(); 
        Vectord muj_mui = other.m_mean - m_mean;
        Scalar distance = (muj_mui.transpose() * invj * muj_mui)[0];
        Scalar divergence2 = logDetRatio + trace + distance - t_conditionDims;
        return divergence2 / 2;
    }

private:
    friend class boost::serialization::access;
    template<class Archive>
    void save(Archive & ar, const unsigned int version) const
    {
        ar  & m_mean;
        ar  & m_cov;
    }
    template<class Archive>
    void load(Archive & ar, const unsigned int version)
    {
        ar & m_mean;
        ar & m_cov;
        set(m_mean, m_cov);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    Vectord m_mean;
    Matrixd m_cov;
    Matrixd m_covInv;
#if USE_CANONICAL
    Matrixd m_covCanonical;
    Vectord m_meanCanonical;
#endif
    ConditionalVectord m_muB;
    ConditionVectord m_muA;
    ConditionalMatrixd m_conditionalCov;
    Eigen::Matrix<Scalar, t_conditionalDims, t_conditionDims> m_muPremult;

    Eigen::LLT<Matrixd> m_cholesky;
    Scalar m_detInv;
	AABB m_aabb;

    Eigen::Matrix<Scalar, 2, 1> boxMullerTransform(Scalar u1, Scalar u2) const {
        Scalar radius = std::sqrt(-2 * std::log(1 - u1));
        Scalar theta = 2 * M_PI * u2;
        Eigen::Matrix<Scalar, 2, 1> result;
        double res0, res1;
        sincos((double) theta, &res0, &res1);
        result(0) = res0;
        result(1) = res1;
        return radius * result;
    }
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template<int t_dims, int t_conditionalDims, typename Scalar>
class MultivariateNormalConditioning {
};

}

#endif /* __MULTIVARIATE_NORMAL_H */
