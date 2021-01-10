#ifndef __MIXTURE_MODEL_H
#define __MIXTURE_MODEL_H

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

#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "distribution.h"
#include "utils.h"
#include "multivariate_normal.h"
#include "multivariate_tangent_normal.h"
#include "multivariate_uniform.h"

#define FAIL_ON_ZERO_CDF 0
#define USE_MAX_KEEP 0

namespace jmm {

template<
    int t_dims,
    int t_components,
    int t_conditionalDims,
    typename Scalar_t,
    template<int, int, typename> class Component_t,
    template<int, int, typename> class Marginal_t
>
class MixtureModel : public Distribution<t_dims, Scalar_t> {
public:
    constexpr static int t_conditionDims = t_dims - t_conditionalDims;
    using Scalar = Scalar_t;

    using ConditionalDistribution = MixtureModel<
        t_conditionalDims,
        t_components,
        t_conditionalDims,
        Scalar,
        Component_t,
        Marginal_t
    >;

    using Component = Component_t<t_dims, t_conditionalDims, Scalar>;
    using MarginalComponent = Marginal_t<t_conditionDims, t_conditionDims, Scalar>;

    using Vectord = typename Component::Vectord;
    using Matrixd = typename Component::Matrixd;

    using ConditionalVectord = Eigen::Matrix<Scalar, t_conditionalDims, 1>;
    using ConditionalMatrixd = Eigen::Matrix<Scalar, t_conditionalDims, t_conditionalDims>;

    using ConditionVectord = Eigen::Matrix<Scalar, t_conditionDims, 1>;
    using ConditionMatrixd = Eigen::Matrix<Scalar, t_conditionDims, t_conditionDims>;

    MixtureModel() :
        m_components(t_components),
        m_weights(t_components),
        m_cdf(t_components),
        m_marginals(t_components) 
    { }

    Vectord sample(const std::function<Scalar()>& rng) const {
        size_t component_i = sampleDiscreteCdf(std::begin(m_cdf), std::begin(m_cdf) + m_lastIdx, rng());
        return m_components[component_i].sample(rng);
    }

    template<typename Vector>
    Scalar surfaceMarginalPdf(const Vector& sample) const {
        Scalar pdfAccum = 0.f;
        for(int component_i = 0; component_i < m_lastIdx; ++component_i) {
            if(m_weights[component_i] != 0) {
                pdfAccum += m_weights[component_i] * m_marginals[component_i].pdf(sample.template topRows<t_conditionDims>());
            }
        }
        return m_normalization * pdfAccum / m_surfaceIntegral;
    }

    Scalar surfacePdf(const Vectord& sample) const {
        Scalar pdfAccum = 0.f;
        for(int component_i = 0; component_i < m_lastIdx; ++component_i) {
            if(m_weights[component_i] != 0) {
                pdfAccum += m_weights[component_i] * m_components[component_i].pdf(sample);
            }
        }
        return m_normalization * pdfAccum / m_surfaceIntegral;
    }

    Scalar surfacePdf(const Vectord& sample, Scalar heuristicPdf) {
        return (1 - m_heuristicWeight) * surfacePdf(sample) + m_heuristicWeight * heuristicPdf;
    }

    template<typename Vector>
    Scalar marginalPdf(const Vector& sample) const {
        Scalar pdfAccum = 0.f;
        for(int component_i = 0; component_i < m_lastIdx; ++component_i) {
            if(m_weights[component_i] != 0) {
                pdfAccum += m_weights[component_i] * m_marginals[component_i].pdf(sample.template topRows<t_conditionDims>());
            }
        }
        return pdfAccum;
    }
    
    Scalar pdf(const Vectord& sample) const {
        Scalar pdfAccum = 0.f;
        for(int component_i = 0; component_i < m_lastIdx; ++component_i) {
            if(m_weights[component_i] != 0) {
                pdfAccum += m_weights[component_i] * m_components[component_i].pdf(sample);
            }
        }
        return pdfAccum;
    }

    jmm::aligned_vector<Component>& components() { return m_components; }
    jmm::aligned_vector<Scalar>& weights() { return m_weights; }

    void setMixtureThreshold(Scalar mixtureThreshold) { m_mixtureThreshold = mixtureThreshold; }

    int nComponents() const { return m_lastIdx; }
    void setNComponents(int lastIdx) {m_lastIdx = lastIdx;}

    Scalar heuristicWeight() const { return m_heuristicWeight; }
    void setHeuristicWeight(Scalar weight) { m_heuristicWeight = weight; }

    Scalar normalization() const { return m_normalization; }
    void setNormalization(Scalar normalization) { m_normalization = normalization; }

    Scalar surfaceIntegral() const { return m_surfaceIntegral; }
    void setSurfaceIntegral(Scalar surfaceIntegral) { m_surfaceIntegral = surfaceIntegral; }

    Scalar surfaceArea() const { return m_surfaceArea; }
    void setSurfaceArea(Scalar surfaceArea) { m_surfaceArea = surfaceArea; }

    Scalar modelError() const { return m_modelError; }
    void setModelError(Scalar modelError) { m_modelError = modelError; }

    void posteriorAndLog(
        const Vectord& sample,
        bool useHeuristic,
        Scalar heuristicPdf,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& pdf,
        // Scalar& marginalPdf,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& posterior,
        Eigen::Matrix<Scalar, Component::t_jointTangentDims, Eigen::Dynamic>& tangentVectors,
        Scalar& heuristicPosterior
    ) const {
        heuristicPosterior = 0.f;
        // marginalPdf = 0.f;
        typename Component::JointTangentVectord tangent;
        for(int component_i = 0; component_i < m_lastIdx; ++component_i) {
            pdf(component_i) = m_components[component_i].pdfAndLog(sample, tangent);
            if(!std::isfinite(pdf(component_i))) {
                std::cerr << "Infinite pdf: " << pdf(component_i) << ".\n";
            }
            posterior(component_i) = m_weights[component_i] * pdf(component_i);
            tangentVectors.col(component_i) = tangent;
            // pdf(component_i) /= m_surfaceIntegral;
        }

        Scalar sum = posterior.sum();
        if(useHeuristic) {
            sum = (1 - m_heuristicWeight) * sum + m_heuristicWeight * heuristicPdf;
        }
        
        const Scalar invSum = 1 / sum;
        if(std::isfinite(invSum)) {
            posterior *= invSum;
            if(useHeuristic) {
                posterior *= (1.f - m_heuristicWeight);
                heuristicPosterior = m_heuristicWeight * heuristicPdf * invSum;
                pdf = m_heuristicWeight * heuristicPdf + (1.f - m_heuristicWeight) * pdf.array();
            }
        } else {
            std::cerr << "Infinite or nan posterior sum = 1.f / " << sum << ", " << pdf.sum() << "\n";
            Scalar weightSum = 0.f;
            for(int component_i = 0; component_i < m_lastIdx; ++component_i) {
                weightSum += m_weights[component_i];
            }
            posterior.setZero();
            pdf.setZero();
            heuristicPosterior = 0.f;
        }
    }

    void posterior(
        const Vectord& sample,
        bool useHeuristic,
        Scalar heuristicPdf,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& posterior,
        Scalar& heuristicPosterior
    ) const {
        heuristicPosterior = 0.f;
        for(int component_i = 0; component_i < m_lastIdx; ++component_i) {
            Scalar pdf = m_components[component_i].pdf(sample);
            posterior(component_i) = m_weights[component_i] * pdf;
        }

        Scalar sum = posterior.sum();
        if(useHeuristic) {
            sum = (1 - m_heuristicWeight) * sum + m_heuristicWeight * heuristicPdf;
        }
        
        const Scalar invSum = 1 / sum;
        if(std::isfinite(invSum)) {
            posterior *= invSum;
            if(useHeuristic) {
                posterior *= (1.f - m_heuristicWeight);
                heuristicPosterior = m_heuristicWeight * heuristicPdf * invSum;
            }
        } else {
            std::cerr << "Infinite or nan posterior sum = 1.f / " << sum << "\n";
            posterior.setZero();
            heuristicPosterior = 0.f;
        }
    }

    // mutable std::atomic_ullong m_componentsPerSample;
    // mutable std::atomic_ullong m_samplesRequested;

    // void resetComponentsPerSample() { m_componentsPerSample.store(0); }
    // int componentsPerSample() { return m_componentsPerSample; }

    // void resetSamplesRequested() { m_samplesRequested.store(0); }
    // int samplesRequested() { return m_samplesRequested; }

    bool conditional(
        const ConditionVectord& condition,
        ConditionalDistribution& conditional,
        Scalar& heuristicConditionalWeight
    ) const {
        auto& conditionalDistributions = conditional.components();
        auto& conditionalWeights = conditional.weights();
        std::array<int, t_components> idcs;
        std::array<Scalar, t_components> weights;
        std::iota(std::begin(idcs), std::end(idcs), 0);

        Scalar totalMass = 0.f;
        bool isInside;
        for(int component_i = 0; component_i < m_lastIdx; ++component_i) {
            auto& marginal = m_marginals[component_i];
            // if(!marginal.isInsideAABB(condition)) {
            //     weights[component_i] = 0;
            //     continue;
            // }
            Scalar marginalPdf = marginal.pdf(condition, isInside);
            // if(!isInside) {
            //     weights[component_i] = 0;
            //     continue;
            // }
            weights[component_i] = m_weights[component_i] * marginalPdf;
            totalMass += weights[component_i];
        }
        Scalar totalMassCutoff = 0.99 * totalMass;

        std::sort(
            std::begin(idcs),
            std::begin(idcs) + m_lastIdx,
            [&weights](int idx1, int idx2) {
                return weights[idx1] > weights[idx2];
            }
        );

        Scalar accumMass = 0.f;
        int lastIdx;
        for(int component_i = 0; component_i < m_lastIdx; ++component_i) {
            auto& component = m_components[idcs[component_i]];
            conditionalWeights[component_i] = weights[idcs[component_i]];
            component.conditional(condition, conditionalDistributions[component_i]);

            accumMass += conditionalWeights[component_i];
            if(accumMass >= totalMassCutoff) {
                lastIdx = component_i + 1;
                break;
            }
        }

        conditional.setNComponents(lastIdx);
        const Scalar sum = std::accumulate(
            conditionalWeights.begin(),
            conditionalWeights.begin() + lastIdx,
            0.f
        );
        const Scalar invSum = 1 / sum;
        if(std::isfinite(invSum)) {
            heuristicConditionalWeight = m_heuristicWeight * 1.f * invSum;
            std::transform(
                conditionalWeights.begin(),
                conditionalWeights.begin() + lastIdx,
                conditionalWeights.begin(),
                [invSum](Scalar weight) { return weight * invSum; }
            );
        }

        return conditional.createCdf(true);
    }

    bool configure() {
        createMarginals();
        bool success = createCdf(true);
        if(!success) {
            return false;
        }
        return true;
    }

    void save(const std::string& filename) const {
        // make an archive
        std::ofstream ofs(filename.c_str());
        boost::archive::binary_oarchive oa(ofs);
        oa << BOOST_SERIALIZATION_NVP(*this);
    }

    void load(const std::string& filename) {
        // open the archive
        std::ifstream ifs(filename.c_str());
        boost::archive::binary_iarchive ia(ifs);
        ia >> BOOST_SERIALIZATION_NVP(*this);
    }

    bool createCdf(bool normalize=true) {
        return jmm::createCdf(std::begin(m_weights), std::begin(m_weights) + m_lastIdx, std::begin(m_cdf), normalize);
    }

    bool rotateTo(const Eigen::Matrix<Scalar, 3, 1>& mean, MixtureModel& rotated) {
        rotated.setNComponents(m_lastIdx);
        auto& rotatedComponents = rotated.components();
        auto& rotatedWeights = rotated.weights();
        for(int component_i = 0; component_i < m_lastIdx; ++component_i) {
            rotatedComponents[component_i] = m_components[component_i].rotateTo(mean);
            rotatedWeights[component_i] = m_weights[component_i];
        }
        rotated.createCdf(true);
        return true;
    }

    bool multiply(const MixtureModel& other, MixtureModel& product) {
        product.setNComponents(m_lastIdx * other.nComponents());
        int productComponent_i = 0;
        auto& productComponents = product.components();
        for(int component_i = 0; component_i < m_lastIdx; ++component_i) {
            for(int component_j = 0; component_j < other.nComponents(); ++component_j) {
                Scalar newWeight = 0.f;
                if(m_components[component_i].mean().dot(
                    other.m_components[component_j].mean()) < 0) {
                    continue;
                }
                m_components[component_i].multiply(
                    other.m_components[component_j],
                    newWeight,
                    productComponents[productComponent_i]
                );
                product.m_weights[productComponent_i] =
                    m_weights[component_i] *
                    other.m_weights[component_j] *
                    newWeight;
                ++productComponent_i;
            }
        }
        product.createCdf(true);
        return true;
    }

private:
    void createMarginals() {
        for(int i = 0; i < m_lastIdx; ++i) {
            m_components[i].marginal(m_marginals[i]);
        }
    }

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar  & m_components;
        ar  & m_weights;
        ar  & m_lastIdx;
        ar  & m_heuristicWeight;
        if(version > 0) {
            ar  & m_normalization;
        }
    }

    jmm::aligned_vector<Component> m_components;
    jmm::aligned_vector<Scalar> m_weights;
    jmm::aligned_vector<Scalar> m_cdf;
    jmm::aligned_vector<MarginalComponent> m_marginals;
    int m_lastIdx = 0;

    Scalar m_heuristicWeight = 0.5f;
    Scalar m_mixtureThreshold = 0.f; // 1.f / (Scalar) t_components;
    Scalar m_normalization = 1.f;
    Scalar m_surfaceIntegral = 1.f;
    Scalar m_surfaceArea = 0.f;
    Scalar m_modelError = 0.f;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}

// BOOST_CLASS_VERSION(jmm::MixtureModel<5, 1024, 2, double>, 1)

#endif /* __MIXTURE_MODEL_H */
