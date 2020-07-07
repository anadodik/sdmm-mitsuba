#ifndef __MULTIVARIATE_TANGENT_NORMAL_H
#define __MULTIVARIATE_TANGENT_NORMAL_H

#include "distribution.h"
#include "multivariate_normal.h"
#include "utils.h"

#include <vector>
#include <type_traits>
#include <iostream>

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/special_functions/sinc.hpp>

#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#define USE_CANONICAL 0
#define ENABLE_CONDITIONING 0

namespace jmm {


template<int t_dims, int t_conditionalDims, typename Scalar>
class TangentSpace {
public:
    constexpr static int t_conditionDims = t_dims - t_conditionalDims;

    constexpr static int t_tangentDims = 2;
    constexpr static int t_embeddingDims = 3;
    static_assert(t_conditionalDims == t_embeddingDims, "Tangent Gaussians must have 3 conditional dimensions.");
    
    constexpr static int t_jointTangentDims = t_conditionDims + t_tangentDims;
    constexpr static int t_jointEmbeddingDims = t_conditionDims + t_embeddingDims;

    using Vectord = Eigen::Matrix<Scalar, t_jointEmbeddingDims, 1>;
    using JointTangentVectord = Eigen::Matrix<Scalar, t_jointTangentDims, 1>;
    using EmbeddingVectord = Eigen::Matrix<Scalar, t_embeddingDims, 1>;
    using TangentVectord = Eigen::Matrix<Scalar, t_tangentDims, 1>;
    using Matrixd = Eigen::Matrix<Scalar, t_jointTangentDims, t_jointTangentDims>;
    using TangentMatrixd = Eigen::Matrix<Scalar, t_tangentDims, t_tangentDims>;

    using ConditionVectord = Eigen::Matrix<Scalar, t_conditionDims, 1>;
    using ConditionMatrixd = Eigen::Matrix<Scalar, t_conditionDims, t_conditionDims>;
    
    TangentSpace() { }
    
    TangentSpace(const Vectord& mean)
    : m_mean(mean) {
        Coordinates<Scalar> coordinates(directionalComponent(mean));
		// Eigen::Quaternion<Scalar> rotation;
		// rotation.setFromTwoVectors(EmbeddingVectord::UnitZ(), directionalComponent(m_mean)).normalize(); 
        m_rotation = coordinates.to.transpose(); // rotation.toRotationMatrix();
        // m_invRotation = m_rotation.conjugate();
        // m_invRotation.w() = m_rotation.w();
        // m_invRotation.coeffs() = -m_rotation.coeffs();
        m_invRotation = coordinates.to; // Eigen::Quaternion<Scalar>(rotation.w(), -rotation.x(), -rotation.y(), -rotation.z()).toRotationMatrix();

        // std::cerr << m_rotation.vec().transpose() << ", w=" << m_rotation.w() << std::endl;
        // std::cerr << m_invRotation.vec().transpose() << ", w=" << m_invRotation.w() << std::endl;
        
        // std::cerr << "Rotation matrix:\n" << m_rotation.toRotationMatrix() << std::endl;
        // std::cerr << "Inverse rotation matrix:\n" << m_invRotation.toRotationMatrix() << std::endl;
    }

    auto directionalComponent(const Vectord& vector) const {
        return vector.template bottomRows<t_embeddingDims>();
    }
    auto directionalComponent(const JointTangentVectord& vector) const {
        return vector.template bottomRows<t_tangentDims>();
    }

    template<typename Derived>
    auto spatialComponent(const Eigen::MatrixBase<Derived>& vector) const {
        return vector.template topRows<t_conditionDims>();
    }

    Vectord concatComponents(const ConditionVectord& spatial, const EmbeddingVectord& embedding) const {
        Vectord joint;
        joint.template topRows<t_conditionDims>() << spatial;
        joint.template bottomRows<t_embeddingDims>() << embedding;
        return joint;
    }

    JointTangentVectord concatComponents(const ConditionVectord& spatial, const TangentVectord& tangent) const {
        JointTangentVectord joint;
        joint.template topRows<t_conditionDims>() << spatial;
        joint.template bottomRows<t_tangentDims>() << tangent;
        return joint;
    }

    bool exp(
        const JointTangentVectord& jointTangent,
        Vectord& jointEmbedding,
        Scalar& jacobian
    ) const {
        auto tangent = directionalComponent(jointTangent);
        Scalar length = tangent.norm();
        if(length >= M_PI) {
            jointEmbedding = Vectord::Zero();
            return false;
        }
        Scalar sinOverAngle = boost::math::sinc_pi(length);  

        EmbeddingVectord relToNorthPole;
        relToNorthPole <<
            tangent(0) * sinOverAngle,
            tangent(1) * sinOverAngle,
            std::cos(length);
        
		EmbeddingVectord rotated = m_rotation * relToNorthPole;
        jointEmbedding = concatComponents(
            spatialComponent(jointTangent), rotated
        );

        jacobian = sinOverAngle;

        return true;
    }

    bool conditionalExp(
        const TangentVectord& tangent,
        EmbeddingVectord& embedding,
        Scalar& jacobian
    ) const {
        Scalar length = tangent.norm();
        if(length >= M_PI) {
            embedding = EmbeddingVectord::Zero();
            return false;
        }
        Scalar sinOverAngle = boost::math::sinc_pi(length);

        EmbeddingVectord relToNorthPole;
        relToNorthPole <<
            tangent(0) * sinOverAngle,
            tangent(1) * sinOverAngle,
            std::cos(length);
        embedding = m_rotation * relToNorthPole;

        jacobian = sinOverAngle;

        return true;
    }

    bool log(
        const Vectord& embedding,
        JointTangentVectord& jointTangent,
        Scalar& jacobian
    ) const {
        auto direction = directionalComponent(embedding);
        if((direction.array() == 0).all()) {
            return false;
        }
        EmbeddingVectord relToNorthPole = m_invRotation * direction;
        Scalar cosAngle = relToNorthPole(2);
        if(cosAngle <= -1) {
            return false;
        }
        cosAngle = std::min(Scalar(1), cosAngle);

        Scalar angle = approx_acos(cosAngle);
		Scalar sinAngle = tsqrtf(1 - cosAngle * cosAngle);
        Scalar angleOverSin = (sinAngle < 1e-3) ? 1 : (angle / sinAngle); // boost::math::sinc_pi(angle);
        // Scalar angleOverSin = 1.f / boost::math::sinc_pi(angle);

        TangentVectord tangent;
        tangent <<
            relToNorthPole(0) * angleOverSin,
            relToNorthPole(1) * angleOverSin;
        
        jointTangent = concatComponents(spatialComponent(embedding), tangent);
        
        jacobian = angleOverSin;

        return true;
    }

    Eigen::Matrix<Scalar, 3, 2> expJacobian(const TangentVectord& tangent) const {
        Eigen::Matrix<Scalar, 3, 2> jacobian;
        Scalar length = tangent.norm();
        if(length == 0) {
            jacobian.setZero();
            jacobian(0, 0) = 1;
            jacobian(1, 1) = 1;
            return jacobian;
        }
        Scalar lengthSqr = length * length;
        Scalar cos = std::cos(length);
        Scalar sinc = boost::math::sinc_pi(length);

        Scalar cosMinusSincOverLengthSqr = 
            (cos - sinc) / lengthSqr;
        jacobian(0, 0) = sinc + tangent(0) * tangent(0) * cosMinusSincOverLengthSqr;
        jacobian(1, 1) = sinc + tangent(1) * tangent(1) * cosMinusSincOverLengthSqr;

        Scalar j_off_diag = tangent(0) * tangent(1) * cosMinusSincOverLengthSqr;
        jacobian(1, 0) = j_off_diag;
        jacobian(0, 1) = j_off_diag;

        jacobian(2, 0) = -tangent(0) * sinc;
        jacobian(2, 1) = -tangent(1) * sinc;

        // jacobian(0, 2) = 0;
        // jacobian(1, 2) = 0;
        // jacobian(2, 2) = 0;

        return jacobian; // m_rotation.toRotationMatrix() * jacobian;
    }

    Eigen::Matrix<Scalar, 2, 3> logJacobian(const EmbeddingVectord& embedding) const {
        Eigen::Matrix<Scalar, 2, 3> jacobian;
        EmbeddingVectord relToNorthPole = m_invRotation * embedding;
        // std::cerr << relToNorthPole << std::endl;
        Scalar cosAngle = relToNorthPole(2);
        cosAngle = std::min(Scalar(1), cosAngle);
        if(cosAngle <= -1) {
            jacobian.setZero();
            return jacobian;
        }
        if(cosAngle == 1 || (embedding.array() == m_mean.bottomRows(t_embeddingDims).array()).all()) {
            jacobian.setZero();
            jacobian(0, 0) = 1.f;
            jacobian(1, 1) = 1.f;
            // std::cerr << relToNorthPole << std::endl;
            return jacobian;
        }
        // std::cerr << "Calculating full log-jacobian.\n";
        Scalar angle = std::acos(cosAngle);
        Scalar angleOverSin = 1.f / boost::math::sinc_pi(angle);

        jacobian(0, 0) = angleOverSin;
        jacobian(1, 1) = angleOverSin;

        jacobian(1, 0) = 0;
        jacobian(0, 1) = 0;

        Scalar invSinAngleSqr = 1 / (1 - cosAngle * cosAngle);
        jacobian(0, 2) = 
            relToNorthPole(0) * cosAngle * angleOverSin * invSinAngleSqr -
            relToNorthPole(0) * invSinAngleSqr;
        jacobian(1, 2) = 
            relToNorthPole(1) * cosAngle * angleOverSin * invSinAngleSqr -
            relToNorthPole(1) * invSinAngleSqr;

        return jacobian; // * m_invRotation.toRotationMatrix();
    }

    Eigen::Matrix<Scalar, t_embeddingDims, t_embeddingDims> rotation() const { return m_rotation; }
    Eigen::Matrix<Scalar, t_embeddingDims, t_embeddingDims> invRotation() const { return m_invRotation; }

    Vectord mean() const { return m_mean; }

private:
    Vectord m_mean;
    Eigen::Matrix<Scalar, 3, 3> m_rotation, m_invRotation;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


template<int t_dims, int t_conditionalDims, typename Scalar>
class MultivariateTangentNormal : public Distribution<t_dims, Scalar> {
public:
    constexpr static int t_conditionDims = t_dims - t_conditionalDims;

    constexpr static int t_tangentDims = 2;
    constexpr static int t_embeddingDims = 3;
    static_assert(t_conditionalDims == t_embeddingDims, "Tangent Gaussians must have 3 conditional dimensions.");
    
    constexpr static int t_jointTangentDims = t_conditionDims + t_tangentDims;
    constexpr static int t_jointEmbeddingDims = t_conditionDims + t_embeddingDims;

    double containedMass = 0.999;

    using Vectord = Eigen::Matrix<Scalar, t_jointEmbeddingDims, 1>;
    using JointTangentVectord = Eigen::Matrix<Scalar, t_jointTangentDims, 1>;
    using EmbeddingVectord = Eigen::Matrix<Scalar, t_embeddingDims, 1>;
    using TangentVectord = Eigen::Matrix<Scalar, t_tangentDims, 1>;
    using Matrixd = Eigen::Matrix<Scalar, t_jointTangentDims, t_jointTangentDims>;
    using TangentMatrixd = Eigen::Matrix<Scalar, t_tangentDims, t_tangentDims>;

    using ConditionVectord = Eigen::Matrix<Scalar, t_conditionDims, 1>;
    using ConditionMatrixd = Eigen::Matrix<Scalar, t_conditionDims, t_conditionDims>;

    using AABB = Eigen::AlignedBox<Scalar, t_dims>;

    MultivariateTangentNormal(const Vectord& mean, const Matrixd& cov) {
        set(mean, cov);
    }

    MultivariateTangentNormal() {
        set(Vectord::Ones().normalized(), Matrixd::Identity());
    }
    
    ~MultivariateTangentNormal() = default;
    MultivariateTangentNormal(const MultivariateTangentNormal& other) = default;
    MultivariateTangentNormal(MultivariateTangentNormal&& other) noexcept = default;
    MultivariateTangentNormal& operator=(const MultivariateTangentNormal& other) = default;
    MultivariateTangentNormal& operator=(MultivariateTangentNormal&& other) = default;

    const Vectord& mean() const { return m_mean; }

    JointTangentVectord tangentMean() const {
        JointTangentVectord tangentMean;
        tangentMean.template bottomRows<t_tangentDims>().setZero();
        if(t_conditionDims > 0) {
            tangentMean.template topRows<t_conditionDims>() 
                << m_mean.template topRows<t_conditionDims>();
        }
        return tangentMean;
    }
    const Matrixd& cov() const { return m_cov; }
    const TangentSpace<t_dims, t_conditionalDims, Scalar>& tangentSpace() const { return m_tangentSpace; }
    
    bool set(const Vectord& mean, const Matrixd& cov);

    JointTangentVectord toStandardNormal(const JointTangentVectord& sample) const {
        return m_cholLInv * sample;
    }

    Vectord sample(const std::function<Scalar()>& rng) const {
        // TODO: Ziggurat or vectorized box-mueller
        JointTangentVectord tangent;
        for(int dim_i = 0; dim_i < t_jointTangentDims; dim_i += 2) {
            Eigen::Matrix<Scalar, 2, 1> normal2d = boxMullerTransform(rng(), rng());
            tangent(dim_i) = normal2d(0);
            if(dim_i + 1 < t_jointTangentDims) {
                tangent(dim_i + 1) = normal2d(1);
            }
        }
        JointTangentVectord covTransformed = m_cholL * tangent;
        Vectord embedding;
        Scalar jacobian;
        bool success = m_tangentSpace.exp(covTransformed, embedding, jacobian);
        if(!success) {
            return Vectord::Zero();
        }
        return embedding + m_zeroedOutMean;
    }

	static inline float fakegaussian_pdf(const float x)
	{
		const float norm_c1 = 2.0f * 0x1.1903a6p+0;
		const int i1 = 0x3f800000u, i2 = 0x40000000u;
		const int k0 = i1 - x*x * (i2 - i1);
		const int k = k0 > 0 ? k0 : 0;
		return (*(const float *)&k)/norm_c1;
	}

    Scalar pdfAndLog(const Vectord& sample, JointTangentVectord& tangent) const {
        constexpr static Scalar INV_SQRT_TWO_PI = 0.39894228040143267793994605993438186847585863116492;
        constexpr static Scalar NORMALIZATION = std::pow(INV_SQRT_TWO_PI, t_jointTangentDims);

        Scalar jacobian;
        if(!m_tangentSpace.log(sample - m_zeroedOutMean, tangent, jacobian)) {
            return 0;
        }
        JointTangentVectord standardized = toStandardNormal(tangent);
        Scalar pdf = NORMALIZATION * std::exp(-0.5 * standardized.squaredNorm());
		// Scalar pdf = fakegaussian_pdf(0.833 * standardized.norm());
        pdf *= m_detInv * jacobian;
        tangent.template topRows<t_conditionDims>() +=
            m_zeroedOutMean.template topRows<t_conditionDims>();
        return pdf;
    }

    Scalar pdf(const Vectord& sample) const {
        constexpr static Scalar INV_SQRT_TWO_PI = 0.39894228040143267793994605993438186847585863116492;
        constexpr static Scalar NORMALIZATION = std::pow(INV_SQRT_TWO_PI, t_jointTangentDims);

        JointTangentVectord tangent;
        Scalar jacobian;
        if(!m_tangentSpace.log(sample - m_zeroedOutMean, tangent, jacobian)) {
            return 0;
        }
        JointTangentVectord standardized = toStandardNormal(tangent);
        Scalar pdf = NORMALIZATION * std::exp(-0.5 * standardized.squaredNorm());
		// Scalar pdf = fakegaussian_pdf(0.833 * standardized.norm());
        pdf *= m_detInv * jacobian;
        return pdf;
    }

    template <int dimCheck = t_conditionDims>
    typename std::enable_if<(dimCheck == 0), void>::type precomputeConditioning() {}

    template <int dimCheck = t_conditionDims>
    typename std::enable_if<(dimCheck > 0), void>::type precomputeConditioning() {
        auto muA = m_mean.template topRows<t_conditionDims>();
        auto muB = m_mean.template bottomRows<t_embeddingDims>();
        auto covAA = m_cov.template topLeftCorner<t_conditionDims, t_conditionDims>();
        auto covAB = m_cov.template topRightCorner<t_conditionDims, t_tangentDims>();
        auto covBA = m_cov.template bottomLeftCorner<t_tangentDims, t_conditionDims>();
        auto covBB = m_cov.template bottomRightCorner<t_tangentDims, t_tangentDims>();

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
        MultivariateTangentNormal<t_embeddingDims, t_embeddingDims, Scalar>& conditional
    ) const { return true; }

    // https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf
    template <int dimCheck = t_conditionDims>
    typename std::enable_if<(dimCheck > 0), bool>::type conditional(
        const ConditionVectord& condition,
        MultivariateTangentNormal<t_embeddingDims, t_embeddingDims, Scalar>& conditional
    ) const {
#if USE_CANONICAL
        auto muB = m_meanCanonical.template bottomRows<t_conditionalDims>();
        auto covBA = m_covCanonical.template bottomLeftCorner<t_conditionalDims, t_conditionDims>();
        auto covBB = m_covCanonical.template bottomRightCorner<t_conditionalDims, t_conditionalDims>();

        auto conditionalCov = covBB.inverse();
        auto conditionalMean = conditionalCov * (muB - covBA * condition);
        return conditional.set(std::move(conditionalMean), std::move(conditionalCov));
#else
        TangentVectord conditionalMean = m_muPremult * (condition - m_muA);
        EmbeddingVectord embedded;
        Scalar jacobian;
        if(!m_tangentSpace.conditionalExp(conditionalMean, embedded, jacobian)) {
            return false;
        }
        return conditional.set(embedded, m_conditionalCov);
#endif
    }

    template<int t_marginalDims=t_conditionDims>
    typename std::enable_if<(t_marginalDims == 0), void>::type
    marginal(MultivariateNormal<t_marginalDims, t_marginalDims, Scalar>& m) const {
    }

    template<int t_marginalDims=t_conditionDims>
    typename std::enable_if<(t_marginalDims > 0), void>::type
    marginal(MultivariateNormal<t_marginalDims, t_marginalDims, Scalar>& m) const {
        Eigen::Matrix<Scalar, t_marginalDims, 1> muA =
            m_mean.template topRows<t_marginalDims>();
        Eigen::Matrix<Scalar, t_marginalDims, t_marginalDims> covAA =
            m_cov.template topLeftCorner<t_marginalDims, t_marginalDims>();
        m.set(muA, covAA);
    }

    bool isInside(const Vectord& sample, Scalar scale) const {
        static const boost::math::chi_squared chiSquared(t_dims);
        static const Scalar maxRadius = std::sqrt((Scalar) boost::math::quantile(chiSquared, containedMass));

        JointTangentVectord tangent;
        Scalar jacobian;
        if(!m_tangentSpace.log(sample - m_zeroedOutMean, tangent, jacobian)) {
            return 0;
        }
        JointTangentVectord standardized = toStandardNormal(tangent);

        return standardized.norm() < maxRadius;
    }

#if ENABLE_CONDITIONING == 1
    Eigen::AlignedBox<Scalar, t_dims> getAABB(bool& success) {
        static const boost::math::chi_squared chiSquared(t_dims);
        static const Scalar maxRadius = std::sqrt((Scalar) boost::math::quantile(chiSquared, containedMass));
        Eigen::SelfAdjointEigenSolver<Matrixd> solver;
        solver.compute(m_cov);
        Eigen::AlignedBox<Scalar, t_dims> aabb(m_mean);
        if(solver.info() != Eigen::Success) {
            // std::cerr << "Cannot compute eigendecomposition of matrix for AABB. \n";
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
            // std::cerr << "Lambas inf: " << lambdas_sqr.transpose() << "\n";
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
        Scalar distance = muj_mui.transpose() * invj * muj_mui;
        Scalar divergence2 = logDetRatio + trace + distance - t_conditionDims;
        return divergence2 / 2;
    }
#endif

    Scalar pdf(
        const TangentSpace<t_dims, t_conditionalDims, Scalar>& tangentSpace,
        const Vectord& zeroedOutMean,
        const Matrixd& cov,
        const Vectord& sample
    ) const {
        constexpr static Scalar INV_SQRT_TWO_PI = 0.39894228040143267793994605993438186847585863116492;
        constexpr static Scalar NORMALIZATION = std::pow(INV_SQRT_TWO_PI, t_jointTangentDims);

        Eigen::LLT<Matrixd> cholesky;
        cholesky = cov.llt();
        if(cholesky.info() == Eigen::NumericalIssue) {
            std::cerr << "MATRIX NOT PD!" << "\n";
        }
        if(cholesky.info() != Eigen::Success) {
            std::stringstream ss;
            ss << "ERROR SAMPLING FROM DISTRIBUTION WITH COV=\n" << cov << "\n\n";
            std::cerr << ss.str();
            return false;
        }
        Scalar invDet = 1.f / cholesky.matrixL().determinant();

        JointTangentVectord tangent;
        Scalar jacobian;
        if(!tangentSpace.log(sample - zeroedOutMean, tangent, jacobian)) {
            return 0;
        }
        JointTangentVectord standardized = cholesky.matrixL().solve(tangent);
        Scalar pdf = NORMALIZATION * std::exp(-0.5 * standardized.squaredNorm());
        pdf *= invDet * jacobian;

        return pdf;
    }

    Matrixd covIntoTangentSpace(const MultivariateTangentNormal& other) const {
        Matrixd jointJacobian = Matrixd::Identity();
        Eigen::Matrix<Scalar, 3, 2> expJacobian =
            other.tangentSpace().expJacobian(
                other.tangentMean().template bottomRows<2>()
            );
        Eigen::Matrix<Scalar, 2, 3> logJacobian = 
            m_tangentSpace.logJacobian(
                other.mean().template bottomRows<3>()
            );
        jointJacobian.template bottomRightCorner<2, 2>() =
            logJacobian * m_tangentSpace.invRotation() *
            other.tangentSpace().rotation() * expJacobian;
        return jointJacobian * other.cov() * jointJacobian.transpose();
    }

    void multiply(
        const MultivariateTangentNormal& other,
        Scalar& weight,
        MultivariateTangentNormal& result
    ) const {
        JointTangentVectord otherMean;
        Scalar jacobianDet;
        bool success = m_tangentSpace.log(other.mean(), otherMean, jacobianDet);
        if(!success) {
            std::cerr <<
                "Could not multiply two MVTN. Mean 1: " <<
                m_mean.transpose() <<
                ", mean 2: " <<
                other.mean().transpose();
        }
        assert(success);
        
        Matrixd otherCov = covIntoTangentSpace(other);
        
        Matrixd covSum = m_cov + otherCov;
        Matrixd invCovSum = covSum.inverse();
        JointTangentVectord meanNewTangent = 
            otherCov * invCovSum * tangentMean() +
            m_cov * invCovSum * otherMean;
        Matrixd covNewTangent = m_cov * invCovSum * otherCov;

        Vectord meanNewEmbedded;
        success = m_tangentSpace.exp(meanNewTangent, meanNewEmbedded, jacobianDet);
        assert(success);
        TangentSpace<t_dims, t_conditionalDims, Scalar> tangentSpaceNew(meanNewEmbedded);

        Matrixd jointJacobian = Matrixd::Identity();
        Eigen::Matrix<Scalar, 3, 2> expJacobian = m_tangentSpace.expJacobian(meanNewTangent.template bottomRows<2>());
        Eigen::Matrix<Scalar, 2, 3> logJacobian = tangentSpaceNew.logJacobian(meanNewEmbedded.template bottomRows<3>());
        jointJacobian.bottomRightCorner(2, 2) =
            logJacobian * tangentSpaceNew.invRotation() *
            m_tangentSpace.rotation() * expJacobian;
        Matrixd covNewEmbedded = jointJacobian * covNewTangent * jointJacobian.transpose();

        weight = pdf(
            m_tangentSpace,
            m_zeroedOutMean,
            covSum,
            other.mean()
        );
        result.set(meanNewEmbedded, covNewEmbedded);
    }

    MultivariateTangentNormal rotateTo(const EmbeddingVectord& otherMean) const {
        Vectord mean;
        if(t_conditionDims > 0) {
            mean.template topRows<t_conditionDims>() =
                m_mean.template topRows<t_conditionDims>();
        }
        mean.template bottomRows<t_embeddingDims>() = otherMean;
        return MultivariateTangentNormal{mean, m_cov};
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
    Vectord m_zeroedOutMean;
    Matrixd m_cov;
    Matrixd m_covInv;

#if USE_CANONICAL
    Matrixd m_covCanonical;
    Vectord m_meanCanonical;
#endif

    EmbeddingVectord m_muB;
    ConditionVectord m_muA;
    TangentMatrixd m_conditionalCov;
    Eigen::Matrix<Scalar, t_tangentDims, t_conditionDims> m_muPremult;

	Matrixd m_cholL;
	Matrixd m_cholLInv;
	
    TangentSpace<t_dims, t_conditionalDims, Scalar> m_tangentSpace;
    Scalar m_detInv;

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
class MultivariateTangentNormalConditioning {
};

}

#endif /* __MULTIVARIATE_TANGENT_NORMAL_H */
