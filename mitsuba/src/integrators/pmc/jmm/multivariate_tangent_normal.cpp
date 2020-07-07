#include "multivariate_tangent_normal.h"

namespace jmm {

template<int t_dims, int t_conditionalDims, typename Scalar>
constexpr int MultivariateTangentNormal<t_dims, t_conditionalDims, Scalar>::t_conditionDims;
template<int t_dims, int t_conditionalDims, typename Scalar>
constexpr int MultivariateTangentNormal<t_dims, t_conditionalDims, Scalar>::t_tangentDims;
template<int t_dims, int t_conditionalDims, typename Scalar>
constexpr int MultivariateTangentNormal<t_dims, t_conditionalDims, Scalar>::t_embeddingDims;
template<int t_dims, int t_conditionalDims, typename Scalar>
constexpr int MultivariateTangentNormal<t_dims, t_conditionalDims, Scalar>::t_jointTangentDims;
template<int t_dims, int t_conditionalDims, typename Scalar>
constexpr int MultivariateTangentNormal<t_dims, t_conditionalDims, Scalar>::t_jointEmbeddingDims;

template<int t_dims, int t_conditionalDims, typename Scalar>
bool MultivariateTangentNormal<t_dims, t_conditionalDims, Scalar>::set(
    const typename MultivariateTangentNormal<t_dims, t_conditionalDims, Scalar>::Vectord& mean,
    const typename MultivariateTangentNormal<t_dims, t_conditionalDims, Scalar>::Matrixd& cov
) {
    if(t_dims == 0) {
        return true;
    }

    m_mean = mean;
    m_zeroedOutMean <<
        m_mean.template topRows<t_conditionDims>(),
        EmbeddingVectord::Zero();
    m_tangentSpace = TangentSpace<t_dims, t_conditionalDims, Scalar>(m_mean);

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

    if(t_dims > t_embeddingDims) {
        precomputeConditioning();
    }

    Eigen::LLT<Matrixd> cholesky = m_cov.llt();
    if(cholesky.info() == Eigen::NumericalIssue) {
        std::cerr << "MATRIX NOT PD!" << "\n";
    }
    if(cholesky.info() != Eigen::Success) {
        std::stringstream ss;
        ss << "ERROR CREATING DISTRIBUTION WITH\nMEAN=\n" << mean.transpose() << ", AND COV=\n" << cov << "\n\n";
        std::cerr << ss.str();
        return false;
    }
    m_cholL = cholesky.matrixL();
    m_cholLInv = m_cholL.inverse();

    m_detInv = 1.f / m_cholL.determinant();
    return true;
}

template class MultivariateTangentNormal<3, 3, double>;
template class MultivariateTangentNormal<4, 3, double>;
template class MultivariateTangentNormal<6, 3, double>;

template class MultivariateTangentNormal<3, 3, float>;
template class MultivariateTangentNormal<4, 3, float>;
template class MultivariateTangentNormal<6, 3, float>;

}
