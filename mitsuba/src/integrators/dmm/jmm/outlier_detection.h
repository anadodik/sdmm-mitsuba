#ifndef __OUTLIER_DETECTION_H
#define __OUTLIER_DETECTION_H

#include <algorithm>

#include <omp.h>

#include "distribution.h"
#include "mixture_model.h"
#include "samples.h"
#include "sphere_volume.h"

#include "kdtree-eigen/kdtree_eigen.h"

#define FAIL_ON_ZERO_CDF 0
#define USE_MAX_KEEP 0

namespace jmm {
    template<typename Samples>
    bool isValidSample(const Samples& samples, int sample_i) {
        if(!std::isfinite(samples.weights(sample_i))) {
            std::cerr 
                << "inf or nan sample in outlier detection, id=" 
                << sample_i 
                << ", value="
                << samples.weights(sample_i) 
                << '\n';
            return false;
        }
        return samples.weights(sample_i) > 0 &&
            // samples.isDiffuse(sample_i) &&
            samples.discounts(sample_i) == 0;
    }

    template<
        int t_dims,
        int t_components,
        int t_conditionalDims,
        typename Scalar,
        template<int, int, typename> class Component,
        template<int, int, typename> class Marginal
    > void normalizeModel(
        Samples<t_dims, Scalar>& samples,
        MixtureModel<t_dims, t_components, t_conditionalDims, Scalar, Component, Marginal>& distribution
    ) {
        Scalar surfaceIntegral = 0.f, surfaceArea = 0.f, sampleMean = 0.f;
        #pragma omp parallel for reduction(+: surfaceIntegral, surfaceArea, sampleMean)
        for(int sample_i = 0; sample_i < samples.size(); ++sample_i) {
            Scalar pdf = 0.f, marginalPdf = 0.f;
            // assert(false); // TODO: next line is needed.
            pdf = distribution.pdf(samples.samples.col(sample_i));
            marginalPdf = distribution.marginalPdf(samples.samples.col(sample_i));
            // distribution.pdfAndMarginalPdfPrune(
            //     samples.samples.col(sample_i), pdf, marginalPdf
            // );
            surfaceIntegral += marginalPdf; // samples.stateDensities(sample_i);
            surfaceArea += 1.f; // / samples.stateDensities(sample_i);
            // Scalar cos = samples.normals.col(sample_i).transpose() * samples.samples.col(sample_i).bottomRows(3);
            // sampleMean += cos * samples.weights(sample_i);
            sampleMean += samples.weights(sample_i);
        }
        
        surfaceIntegral /= (Scalar) samples.size();
        surfaceArea /= (Scalar) samples.size();
        sampleMean /= (Scalar) samples.size();

        distribution.setSurfaceIntegral(surfaceIntegral);
        distribution.setSurfaceArea(surfaceArea);
        // distribution.setNormalization(sampleMean);
    }

    template<
        int t_dims,
        int t_components,
        int t_conditionalDims,
        typename Scalar,
        template<int, int, typename> class Component,
        template<int, int, typename> class Marginal
    > Eigen::Matrix<Scalar, Eigen::Dynamic, 1> sarsaError(
        Samples<t_dims, Scalar>& samples,
        const MixtureModel<t_dims, t_components, t_conditionalDims, Scalar, Component, Marginal>& distribution
    ) {
        std::cerr << "Computing sarsa error.\n";
        int nSamples = samples.size();
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> marginalPdfs(nSamples, 1), pdfs(nSamples, 1);

        Scalar surfaceIntegral = 0.f, sampleMean = 0.f;
        #pragma omp parallel for reduction(+: surfaceIntegral, sampleMean)
        for(int sample_i = 0; sample_i < nSamples; ++sample_i) {
            pdfs(sample_i) = marginalPdfs(sample_i) = 0;
            assert(false); // TODO: next line is needed.
            // distribution.pdfAndMarginalPdfPrune(
            //     samples.samples.col(sample_i), pdfs(sample_i), marginalPdfs(sample_i)
            // );
            surfaceIntegral += marginalPdfs(sample_i) / samples.stateDensities(sample_i);
            sampleMean += samples.weights(sample_i);
        }
        
        surfaceIntegral /= (Scalar) nSamples;
        sampleMean /= (Scalar) nSamples;
        
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> error(nSamples, 1);
        #pragma omp parallel for schedule(static)
        for(int sample_i = 0; sample_i < nSamples; ++sample_i) {
            if(!isValidSample(samples, sample_i)) {
                error(sample_i) = 0;
                continue;
            }
            Scalar reward = samples.rewards(sample_i) / sampleMean;
            if(pdfs(sample_i) == 0 || marginalPdfs(sample_i) == 0) {
                error(sample_i) = reward;
                continue;
            }
            Scalar pdf = pdfs(sample_i);
            Scalar conditionalPdf = pdf / marginalPdfs(sample_i);
            Scalar marginalNorm = surfaceIntegral;
            Scalar marginalPdf = marginalPdfs(sample_i) / marginalNorm;
            Scalar misPdf;
            if(samples.isDiffuse(sample_i)) {
                misPdf = 
                    distribution.heuristicWeight() * samples.heuristicPdfs(sample_i) +
                    (1.f - distribution.heuristicWeight()) * marginalPdf * conditionalPdf;
            } else {
                misPdf = marginalPdf * conditionalPdf;
            }
            error(sample_i) = reward - misPdf;
        }
        return error;
    }
    template<int t_dims, int t_conditionDims, typename Scalar>
    void estimateStateDensity(Samples<t_dims, Scalar>& samples) {
        using KDTree = kdt::KDTree<Scalar, kdt::EuclideanDistance<Scalar>>;
        int nSamples = samples.size();
        Eigen::Matrix<Scalar, t_conditionDims, Eigen::Dynamic> samplesCopy =
            samples.samples.topLeftCorner(t_conditionDims, nSamples);
        KDTree kdtree(samplesCopy, true);
        kdtree.setSorted(false);
        kdtree.setTakeRoot(false);
        kdtree.build();

        typename  KDTree::Matrix distsSqr;
        typename  KDTree::MatrixI idx;
        size_t knn = 15;
        kdtree.query(samplesCopy, knn, idx, distsSqr);

        Scalar knnNorm = (Scalar) (knn - 1) / (Scalar) (nSamples - 1);
        constexpr int manifoldDims = 2;

        #pragma omp parallel for
        for(int sample_i = 0; sample_i < nSamples; ++sample_i) {
            Scalar maxDistance = std::sqrt(distsSqr.col(sample_i).maxCoeff());
            if(maxDistance == 0.f) {
                std::cerr << "maxDistance=0 for sample id=" 
                    << sample_i 
                    << ", value="
                    << samples.weights(sample_i) 
                    << '\n';
                continue;
            }
            Scalar volume = 
                jmm::volume_norm<manifoldDims>::value *
                std::pow(maxDistance, manifoldDims);
            samples.stateDensities(sample_i) = knnNorm / volume;
            Scalar correctedWeight =
                samples.weights(sample_i) / samples.stateDensities(sample_i);

            if(!std::isfinite(correctedWeight)) {
                std::cerr << "spatially reweighted sample inf or nan, id=" 
                    << sample_i 
                    << ", value="
                    << samples.weights(sample_i)
                    << ", reweighted value="
                    << correctedWeight
                    << '\n';
                continue;
            }
            samples.weights(sample_i) = correctedWeight;
        }
        // Eigen::Matrix<Scalar, 1, Eigen::Dynamic> maxDistances = dists.colwise().maxCoeff();
        // Eigen::Matrix<Scalar, Eigen::Dynamic, 1> volume = 
        //     jmm::volume_norm<manifoldDims>::value * maxDistances.array().pow(manifoldDims);
        // Eigen::Matrix<Scalar, 1, Eigen::Dynamic> density = knnNorm / volume.array();

        // m_samples->stateDensities.topRows(nSamples) = density.transpose();
        // m_samples->weights.topRows(nSamples).array() /= m_samples->stateDensities.topRows(nSamples).array();
        // m_samples->samplingPdfs.topRows(nSamples).array() *= m_samples->stateDensities.topRows(nSamples).array();
    }


    // class OutlierDetection {
    //     constexpr static int t_dims = 5;
    //     constexpr static int t_components = 1024;
    //     constexpr static int t_conditionalDims = 2;
    //     using Scalar = float;

    //     using MM = jmm::MixtureModel<t_dims, t_components, t_conditionalDims>;
    //     using MMCond = jmm::MixtureModel<t_conditionalDims, t_components, 0>;
        
    //     using MMScalar = typename MM::Scalar;
    //     using Vectord = typename MM::Vectord;
    //     using Matrixd = typename MM::Matrixd;

    //     using ConditionalVectord = typename MMCond::Vectord;
    //     using ConditionalMatrixd = typename MMCond::Matrixd;


    //     void detectOutliers(
    //         const MM& distribution,
    //         const Samples<t_dims, Scalar>& samples,
    //         size_t knn,
    //         int minOutliers,
    //         Samples<t_dims, Scalar>& outliers
    //     ) {
    //         int nSamples = samples.size();
    //         Scalar totalWeight = samples.weights.topRows(nSamples).sum();
    //         using KDTree = kdt::KDTree<Scalar, kdt::EuclideanDistance<Scalar>>;
    //         Eigen::Matrix<Scalar, t_dims, Eigen::Dynamic> samplesCopy =
    //             samples.samples.topLeftCorner(t_dims, nSamples);
    //         KDTree kdtree(samplesCopy, true);
    //         kdtree.setSorted(false);
    //         kdtree.build();

    //         KDTree::Matrix dists;
    //         KDTree::MatrixI idx;
    //         kdtree.query(samplesCopy, knn, idx, dists);

    //         outliers.clear();
    //         outliers.reserve(nSamples);

    //         Scalar knnNorm = (Scalar) (knn - 1) / (Scalar) (nSamples);

    //         Eigen::Matrix<Scalar, 1, Eigen::Dynamic> maxDistances = dists.colwise().maxCoeff();
    //         Eigen::Matrix<Scalar, 1, Eigen::Dynamic> volume = volume_norm<t_dims>::value * maxDistances.array().pow(t_dims);
    //         Eigen::Matrix<Scalar, 1, Eigen::Dynamic> density = knnNorm * 1.f / volume.array();

    //         std::vector<std::pair<int, Scalar>> metricSort; metricSort.reserve(nSamples);
    //         for(int sample_i = 0; sample_i < nSamples; ++sample_i) {
    //             Scalar sum = 0, sum_sq = 0;
    //             for(int nn_i = 0; nn_i < knn; ++nn_i) {
    //                 Scalar weight = samples.weights(idx(nn_i, sample_i)) / totalWeight; // * (logFunctionPdf(idx(nn_i, sample_i)) - logLearnedPdf(idx(nn_i, sample_i)));
    //                 sum += weight;
    //                 sum_sq += weight * weight;
    //             }
    //             Scalar variance = (sum_sq / (Scalar) knn - sum * sum / ((Scalar) knn * knn)) * ((Scalar) knn) / (Scalar) knn * density(sample_i);
    //             Scalar normalizedSampleWeight = samples.weights(sample_i) / totalWeight;
    //             Scalar metric = variance * normalizedSampleWeight * normalizedSampleWeight;
    //             metricSort.emplace_back(sample_i, metric);
    //         }

    //         int keepForInit = (int) std::max(0.002f * (Scalar) outliers.size(), (Scalar) minOutliers + 1);
    //         std::sort(metricSort.begin(), metricSort.end(), 
    //                 [](const auto &a, const auto &b){ return a.second > b.second; });
    //         metricSort.erase(metricSort.begin() + keepForInit, metricSort.end());

    //         // TODO: we used to do some denoising here. Was it useful?
            
    //         // Scalar totalDenoisedWeight = 0.f;
    //         // Scalar outlierDenoisedWeight = 0.f;
    //         // for(int outlier_i = 0; outlier_i < (int) metricSort.size(); ++outlier_i) {
    //         //     int sample_i = metricSort[outlier_i].first;
    //         // }            
    //     }

    // };

}


#endif /* __OUTLIER_DETECTION_H */
