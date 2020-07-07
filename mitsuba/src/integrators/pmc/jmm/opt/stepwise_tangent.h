#ifndef __MIXTURE_MODEL_OPT_STEPWISE_TANGENT_H
#define __MIXTURE_MODEL_OPT_STEPWISE_TANGENT_H

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

#include "../kdtree-eigen/kdtree_eigen.h"

#include "../distribution.h"
#include "../mixture_model.h"
#include "../samples.h"
#include "../sphere_volume.h"

#include "util.h"
// #include "../gsl/gsl"

#define TANGENT_DEBUG 0
#define SPLIT_AND_MERGE 0


namespace jmm {

template<
    typename Scalar,
    int t_meanDims,
    int t_covDims
> struct SDMMParams {
    using Vectord = Eigen::Matrix<Scalar, t_meanDims, 1>;
    using Matrixd = Eigen::Matrix<Scalar, t_covDims, t_covDims>;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SDMMParams(int size) : size(size) {
        heuristicWeight = Scalar(0.f);
        weights.resize(size, Scalar(0.f));
        means.resize(size, Vectord::Zero());
        covs.resize(size, Matrixd::Zero());

        densitySum.resize(size, Scalar(0.f));
        logDensitySum.resize(size, Scalar(0.f));
        logPdfSum.resize(size, Scalar(0.f));
    }

    SDMMParams(SDMMParams&& other) = default;
    SDMMParams(const SDMMParams& other) = default;
    SDMMParams& operator=(SDMMParams&& other) = default;
    SDMMParams& operator=(const SDMMParams& other) = default;

    void setZero() {
        heuristicWeight = Scalar(0.f);
        std::fill(weights.begin(), weights.end(), Scalar(0.f));
        std::fill(means.begin(), means.end(), Vectord::Zero());
        std::fill(covs.begin(), covs.end(), Matrixd::Zero());

        std::fill(densitySum.begin(), densitySum.end(), Scalar(0.f));
        std::fill(logPdfSum.begin(), logPdfSum.end(), Scalar(0.f));
        std::fill(logDensitySum.begin(), logDensitySum.end(), Scalar(0.f));
    }

    void normalize(SDMMParams& out, Scalar invWeight) {
        auto normalize = [&](const auto& value) {
            return value * invWeight;
        };
        out.heuristicWeight = normalize(heuristicWeight);
        std::transform(weights.begin(), weights.end(), out.weights.begin(), normalize);
        std::transform(means.begin(), means.end(), out.means.begin(), normalize);
        std::transform(covs.begin(), covs.end(), out.covs.begin(), normalize);
    }

    SDMMParams& operator*=(Scalar multiplier) {
        auto multiply = [multiplier](auto& range) {
            std::transform(
                range.begin(),
                range.end(),
                range.begin(),
                [multiplier](const auto& value) { return value * multiplier; }
            );
        };
        heuristicWeight *= multiplier;
        multiply(weights);
        multiply(means);
        multiply(covs);

        return *this;
    }

    void sumProductInto(SDMMParams& out, Scalar multiplier) {
        auto sumProduct = [&](const auto& value, const auto& outValue) {
            return multiplier * value + outValue;
        };

        auto sumProductIntoRange = [&](const auto& range, auto& outRange) {
            std::transform(
                range.begin(),
                range.end(),
                outRange.begin(),
                outRange.begin(),
                sumProduct
            );
        };
        out.heuristicWeight = sumProduct(heuristicWeight, out.heuristicWeight);
        sumProductIntoRange(weights, out.weights);
        sumProductIntoRange(means, out.means);
        sumProductIntoRange(covs, out.covs);
    }

    void sumErrorStatsInto(SDMMParams& out) {
        auto sum = [&](const auto& value, const auto& outValue) {
            return value + outValue;
        };

        auto sumIntoRange = [&](const auto& range, auto& outRange) {
            std::transform(
                range.begin(),
                range.end(),
                outRange.begin(),
                outRange.begin(),
                sum
            );
        };
        sumIntoRange(densitySum, out.densitySum);
        sumIntoRange(logDensitySum, out.logDensitySum);
        sumIntoRange(logPdfSum, out.logPdfSum);
    }

    void calculateError(jmm::aligned_vector<Scalar>& error, Scalar nSamples) {
        if(error.size() != size) {
            error.resize(size, 0.f);
        }
        for(int component_i = 0; component_i < densitySum.size(); ++component_i) {
            // error[component_i] = (
            //     logDensitySum[component_i] -
            //     std::log(densitySum[component_i]) * densitySum[component_i] -
            //     logPdfSum[component_i]
            // ) / densitySum[component_i];
            // error[component_i] /= nSamples;
            error[component_i] = densitySum[component_i] / nSamples;
        }
    }

    int size;

    Scalar heuristicWeight;
    jmm::aligned_vector<Scalar> weights;
    jmm::aligned_vector<Vectord> means;
    jmm::aligned_vector<Matrixd> covs;

    jmm::aligned_vector<Scalar> densitySum;
    jmm::aligned_vector<Scalar> logDensitySum;
    jmm::aligned_vector<Scalar> logPdfSum;
};


template<
    int t_dims,
    int t_components,
    int t_conditionalDims,
    typename Scalar,
    template<int, int, typename> class Component_t,
    template<int, int, typename> class Marginal_t
> class StepwiseTangentEM {
protected:
    using MM = MixtureModel<t_dims, t_components, t_conditionalDims, Scalar, Component_t, Marginal_t>;
    using Vectord = typename MM::Vectord;
    using Component = typename MM::Component;
    using JointTangentVectord = typename Component::JointTangentVectord;
    using Matrixd = typename MM::Matrixd;
    using TangentSpaced = TangentSpace<t_dims, t_conditionalDims, Scalar>;
    constexpr static int t_statDims = Component::t_jointTangentDims;

    int iterationsRun;
    jmm::aligned_vector<int> iterationsRunForMixture;
    jmm::aligned_vector<bool> startedTraining;
    Scalar heuristicTotalWeight;
    jmm::aligned_vector<Scalar> totalWeightForMixture;
    
    SDMMParams<Scalar, t_statDims, t_statDims> statsGlobal;
    SDMMParams<Scalar, t_statDims, t_statDims> statsGlobalNormalized;
    SDMMParams<Scalar, t_statDims, t_statDims> newParams;

    jmm::aligned_vector<TangentSpaced> tangentSpacesNew;

    jmm::aligned_vector<Scalar> samplesPerComponentGlobal;
    jmm::aligned_vector<Scalar> samplesPerComponentInIterationGlobal;
    Scalar sampleCountGlobal;

    jmm::aligned_vector<Matrixd> bPriors;
    jmm::aligned_vector<Eigen::Matrix<Scalar, 3, 3>> bDepthPriors;

    Scalar alpha;
    Scalar bPrior;
    Scalar niPriorMinusOne;
    Scalar epsilon;
    bool decreasePrior;
    Scalar trainingBatch = 0;
    bool jacobianCorrection = true;
    int trainingCutoff = 32;
    
    Scalar minBPrior = 0; // 1e-8f;
    Scalar minNiPriorMinusOne = 0; // 1e-6;

public:

    // StepwiseTangentEM() : StepwiseTangentEM(0.9, 0.5, 6e-5, 1e-100, true) { }
        // alpha(0.9),
        // bPrior(0.5),
        // niPriorMinusOne(6e-5),
        // epsilon(1e-100),
        // decreasePrior(true) {}

    StepwiseTangentEM(
        Scalar alpha=0.9,
        Eigen::Matrix<Scalar, 5, 1> bPrior=Eigen::Matrix<Scalar, 5, 1>::Constant(1e-5),
        Scalar niPriorMinusOne=6e-5,
        Scalar epsilon=1e-100,
        bool decreasePrior=true
    ) :
        statsGlobal(t_components),
        statsGlobalNormalized(t_components),
        newParams(t_components),
        alpha(alpha),
        // bPrior(bPrior),
        niPriorMinusOne(niPriorMinusOne),
        epsilon(epsilon),
        decreasePrior(decreasePrior)
    {
        iterationsRun = 0;
        iterationsRunForMixture.resize(t_components, 0);
        heuristicTotalWeight = 0.f;
        totalWeightForMixture.resize(t_components, 0.f);
        startedTraining.resize(t_components, false);

        samplesPerComponentGlobal.resize(t_components, 0);
        samplesPerComponentInIterationGlobal.resize(t_components);
        sampleCountGlobal = 0;

        Matrixd bPriorMatrix = bPrior.asDiagonal();
        bPriors.resize(t_components, bPriorMatrix);
        bDepthPriors.resize(t_components, Eigen::Matrix<Scalar, 3, 3>::Identity() * epsilon);

        tangentSpacesNew.resize(t_components);
    };

    void setJacobianCorrection(bool on) {
        jacobianCorrection = on;
    }

    jmm::aligned_vector<Matrixd>& getBPriors() {
        return bPriors;
    }

    jmm::aligned_vector<Eigen::Matrix<Scalar, 3, 3>>& getBDepthPriors() {
        return bDepthPriors;
    }

    void calculateStats(
        MM& distribution,
        Samples<t_dims, Scalar>& samples,
        bool countSamples,
        SDMMParams<Scalar, t_statDims, t_statDims>& stats,
        jmm::aligned_vector<Scalar>& samplesPerComponent,
        Scalar weightSum
    ) {
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> posterior(distribution.nComponents(), 1);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> pdf(distribution.nComponents(), 1);
        Eigen::Matrix<
            Scalar, Component::t_jointTangentDims, Eigen::Dynamic
        > tangentVectors(Component::t_jointTangentDims, distribution.nComponents());

        Scalar weightNormalization = (Scalar) samples.size() / weightSum;

        #pragma omp for
        for(int sample_i = 0; sample_i < samples.size(); ++sample_i) {
            if(!isValidSample(samples, sample_i)) {
                continue;
            }
            if(samples.weights(sample_i) == 0) {
                continue;
            }
            Vectord sample = samples.samples.col(sample_i);
            Scalar heuristicPosterior = 0;
            bool useHeuristic = samples.isDiffuse(sample_i);
            
            distribution.posteriorAndLog(
                sample,
                useHeuristic,
                samples.heuristicPdfs(sample_i),
                pdf,
                posterior,
                tangentVectors,
                heuristicPosterior
            );

            stats.heuristicWeight += samples.weights(sample_i) * heuristicPosterior;
            const auto& components = distribution.components();
            
            for(int component_i = 0; component_i < distribution.nComponents(); ++component_i) {
                if(posterior(component_i) < 1e-10) {
                    // TODO: still calculate marginals and normalization
                    continue;
                }
				// Scalar weightAugmented = std::sqrt(samples.weights(sample_i));
                Scalar weight = samples.weights(sample_i) * posterior(component_i);
                #if TANGENT_DEBUG == 1
                if(weight == 0.f) {
                    std::cerr << "Zero weight * posterior: "
                        << samples.weights(sample_i) 
                        << " * "
                        << posterior(component_i)
                        << " from component "
                        << component_i
                        << ", with weight: "
                        << distribution.weights()[component_i]
                        << ", and covariance determinant: "
                        << components[component_i].cov().determinant()
                        << "\n";
                    continue;
                }
                #endif // TANGENT_DEBUG == 1

                stats.weights[component_i] += weight;
                stats.means[component_i] += weight * tangentVectors.col(component_i);
                stats.covs[component_i] += weight * tangentVectors.col(component_i) * tangentVectors.col(component_i).transpose();

                #if SPLIT_AND_MERGE == 1
                if(weight > 0 && pdf(component_i) > 0) {
                    Scalar samplingPdf = samples.samplingPdfs(sample_i);
                    Scalar Li = samples.weights(sample_i) * samplingPdf;
                    Scalar LiNormalized = Li * weightNormalization;
                    stats.densitySum[component_i] += 
                        LiNormalized * LiNormalized * posterior(component_i) / (pdf(component_i) * samplingPdf) - 1;
                    // stats.densitySum[component_i] += std::abs(weight * weightNormalization - pdf(component_i));
                    // stats.logDensitySum[component_i] += std::log(weight * weightNormalization);
                    // stats.logPdfSum[component_i] += weight * weightNormalization * std::log(pdf(component_i));
                }
                #endif // SPLIT_AND_MERGE == 1
            }
        }
    }

    void calculateStatsPrune(
        MM& distribution,
        Samples<t_dims, Scalar>& samples,
        bool countSamples,
        SDMMParams<Scalar, t_statDims, t_statDims>& stats,
        jmm::aligned_vector<Scalar>& samplesPerComponent
    ) {
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> posterior(distribution.nComponents(), 1);
        Eigen::Matrix<
            Scalar, Component::t_jointTangentDims, Eigen::Dynamic
        > tangentVectors(Component::t_jointTangentDims, distribution.nComponents());
        Eigen::Matrix<int, Eigen::Dynamic, 1> posteriorIndices(distribution.nComponents(), 1);
        int posteriorLastIdx;

        #pragma omp for
        for(int sample_i = 0; sample_i < samples.size(); ++sample_i) {
            if(!isValidSample(samples, sample_i)) {
                continue;
            }
            if(samples.weights(sample_i) == 0) {
                continue;
            }
            Vectord sample = samples.samples.col(sample_i);
            Scalar heuristicPosterior = 0;
            bool useHeuristic = samples.isDiffuse(sample_i);

            // distribution.posterior(
            //     sample,
            //     useHeuristic,
            //     samples.heuristicPdfs(sample_i),
            //     posterior,
            //     heuristicPosterior
            // );
            
            distribution.posteriorPruneAndLog(
                sample,
                useHeuristic,
                samples.heuristicPdfs(sample_i),
                posterior,
                tangentVectors,
                posteriorIndices,
                posteriorLastIdx,
                heuristicPosterior
            );

            stats.heuristicWeight += samples.weights(sample_i) * heuristicPosterior;
            const auto& components = distribution.components();
            
            for(int found_i = 0; found_i < posteriorLastIdx; ++found_i) {
                Scalar weight = samples.weights(sample_i) * posterior(found_i);
                int component_i = posteriorIndices(found_i);
                #if TANGENT_DEBUG == 1
                if(weight == 0.f) {
                    std::cerr << "Zero weight * posterior: "
                        << samples.weights(sample_i) 
                        << " * "
                        << posterior(component_i)
                        << " from component "
                        << component_i
                        << ", with weight: "
                        << distribution.weights()[component_i]
                        << ", and covariance determinant: "
                        << components[component_i].cov().determinant()
                        << "\n";
                    continue;
                }
                #endif // TANGENT_DEBUG == 1

                // JointTangentVectord tangentSample;
                // Scalar jacobian;
                // bool logSuccess = components[component_i].tangentSpace().log(sample, tangentSample, jacobian);
                // if(!logSuccess) {
                //     continue;
                // }

                // bool isInside = 
                //     components[component_i].isInside(samples.samples.col(sample_i), 0.95);

                // if(countSamples) {
                //     samplesPerComponent[component_i] += isInside ? 1 : 0;
                // }

                stats.weights[component_i] += weight;
                stats.means[component_i] += weight * tangentVectors.col(found_i);
                stats.covs[component_i] += weight * tangentVectors.col(found_i) * tangentVectors.col(found_i).transpose();
            }
        }
    }


    bool isValidSample(
        const Samples<t_dims, Scalar>& samples, int sample_i, bool warn=true
    ) {
        if(std::isfinite(samples.weights(sample_i))) {
            return true;
        }
        if(warn) {
            std::cerr 
                << "inf or nan sample, id=" 
                << sample_i 
                << ", value="
                << samples.weights(sample_i) 
                << '\n';
        }
        return false;
    }

    Scalar sumWeights(const Samples<t_dims, Scalar>& samples) {
        Scalar weightSum = 0.f;
        #pragma omp parallel num_threads(1)
        {
            #pragma omp for reduction(+: weightSum)
            for(int sample_i = 0; sample_i < samples.size(); ++sample_i) {
                if(!isValidSample(samples, sample_i)) {
                    continue;
                }
                weightSum += samples.weights(sample_i);
            }
        }
        return weightSum;
    }

    struct SDMMIndividualParams {
        Scalar weight;
        JointTangentVectord mean;
        Matrixd cov;

        Component distribution() {
            return Component(mean, cov);
        }
    };

    constexpr static Scalar weightSplitWeight = 0.5;

    std::pair<SDMMIndividualParams, SDMMIndividualParams> splitSVD(
        const SDMMIndividualParams& gaussian
    ) {
        constexpr static Scalar u = 0.5;
        constexpr static Scalar beta = 0.5;
        constexpr static int l = 0;
        constexpr static Scalar mean_j_const = std::sqrt((1 - weightSplitWeight) / weightSplitWeight) * u;
        constexpr static Scalar mean_k_const = std::sqrt(weightSplitWeight / (1 - weightSplitWeight)) * u;

        Scalar weight = gaussian.weight;
        Matrixd cov = gaussian.cov;

        const Eigen::JacobiSVD<Matrixd> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Matrixd A = svd.matrixU() * svd.singularValues().cwiseSqrt().asDiagonal();
        JointTangentVectord a_l = A.col(l);
        a_l.topRows(3).setZero();
        std::cerr << a_l.transpose() << "\n";

        Scalar weight_j = weightSplitWeight * weight;
        Scalar weight_k = (1 - weightSplitWeight) * weight;

        Scalar sqrt_k_over_j = std::sqrt(weight_k / weight_j);
        Scalar sqrt_j_over_k = std::sqrt(weight_j / weight_k);
        JointTangentVectord mean_j = gaussian.mean - mean_j_const * a_l;
        JointTangentVectord mean_k = gaussian.mean + mean_k_const * a_l;

        Matrixd cov_j = (1 - weightSplitWeight) / weightSplitWeight * cov + (
            (beta - beta * u * u - 1) / weightSplitWeight + 1
        ) * a_l * a_l.transpose();
        
        Matrixd cov_k = weightSplitWeight / (1 - weightSplitWeight) * cov + (
            (beta * u * u - beta - u * u) / (1 - weightSplitWeight) + 1
        ) * a_l * a_l.transpose();

        return {
            {weight_j, mean_j, cov_j},
            {weight_k, mean_k, cov_k}
        };
    }

    void splitStatsSVD(MM& distribution, int statIdx) {
        std::cerr << "Splitting component " << statIdx << ".\n";

        Scalar weight = distribution.weights()[statIdx];
        const Component& component = distribution.components()[statIdx];

        const auto& splits_pair = splitSVD(
            {weight, component.tangentMean(), component.cov()}
        );
        SDMMIndividualParams splits[2] = {splits_pair.first, splits_pair.second};

        std::cerr << "Increasing nComponents to " << distribution.nComponents() + 1 << '\n';
        assert(distribution.nComponents() + 1 < t_components);
        distribution.setNComponents(distribution.nComponents() + 1);

        int j = statIdx;
        int k = distribution.nComponents() - 1;

        const Scalar weightStat = statsGlobal.weights[statIdx];
        statsGlobal.weights[j] = weightSplitWeight * weightStat;
        statsGlobal.weights[k] = (1 - weightSplitWeight) * weightStat;

        for(int split_i = 0; split_i < 2; ++split_i) {
            int component_i = (split_i == 0) ? j : k;
            Scalar decreasedBPrior = bPrior / 
                (Scalar) (iterationsRunForMixture[component_i] + 1);
            // Vectord bPriorDiag = Vectord::Constant(decreasedBPrior);

            newParams.weights[component_i] = splits[split_i].weight;
            newParams.means[component_i] = splits[split_i].mean;
            newParams.covs[component_i] = splits[split_i].cov;
            // newParams.covs[component_i] += (
            //     bPriorDiag * 
            //     totalWeightForMixture[component_i] /
            //     statsGlobal.weights[component_i]
            // ).asDiagonal();

            
            distribution.weights()[component_i] = newParams.weights[component_i];
            std::cerr << "Setting component " << component_i << " to: " << newParams.weights[component_i] << ".\n";
            Vectord embeddedMean;
            Scalar expJacobianDet;
            bool success = distribution.components()[component_i].tangentSpace().exp(
                newParams.means[component_i],
                embeddedMean,
                expJacobianDet
            );
            assert(success);
            distribution.components()[component_i].set(
                embeddedMean, newParams.covs[component_i]
            );

            statsGlobal.means[component_i] =
                statsGlobal.weights[component_i] *
                newParams.means[component_i];

            statsGlobal.covs[component_i] =
                statsGlobal.weights[component_i] * (
                    newParams.covs[component_i] +
                    newParams.means[component_i] *
                    newParams.means[component_i].transpose()
                );
            
            // compareNewAndStats(component_i);
        }
        bool successfulCdfCreation = distribution.createCdf(true);
    }

    void optimize(
        MM& distribution,
        Samples<t_dims, Scalar>& samples,
        Scalar& maxError
    ) {
        int componentBegin = 0;
        int componentEnd = distribution.nComponents();

        // Sum up the weights
        Scalar weightSum = sumWeights(samples);
        if(weightSum == 0) {
            return;
        }

        #pragma omp parallel num_threads(1)
        {
            #if TANGENT_DEBUG == 1
            #pragma omp critical
            {
                std::cerr << "Optimizer threadID=" << omp_get_thread_num() << "\n";
            }
            #pragma omp single
            {
                std::cerr << "Weights sum: " << weightSum << "\n";
            }
            #endif

            jmm::aligned_vector<Scalar> eta_i(t_components);
            Scalar heuristicEta = 0;

            SDMMParams<Scalar, t_statDims, t_statDims> stats(t_components);
            jmm::aligned_vector<Scalar> samplesPerComponent(t_components);

            int iterations = 1;
            if(iterationsRun < 3) {
                iterations = 2;
            }

            for(int emIt = 0; emIt < iterations; ++emIt) {
                #pragma omp barrier
                #pragma omp single
                {
                    newParams.setZero();
                    std::fill(
                        samplesPerComponentInIterationGlobal.begin(),
                        samplesPerComponentInIterationGlobal.end(),
                        0.f
                    );
                }

                stats.setZero();
                std::fill(
                    samplesPerComponent.begin(), samplesPerComponent.end(), 0.f
                );

                #pragma omp barrier

                calculateStats(
                    distribution,
                    samples,
                    emIt == 0,
                    stats,
                    samplesPerComponent,
                    weightSum
                );

                #pragma omp barrier

                #pragma omp critical
                {
                    for(int component_i = componentBegin; component_i < componentEnd; ++component_i) {
                        samplesPerComponentGlobal[component_i] += samplesPerComponent[component_i];
                        samplesPerComponentInIterationGlobal[component_i] += samplesPerComponent[component_i];
                        sampleCountGlobal += samplesPerComponent[component_i];
                    }
                    #if TANGENT_DEBUG == 1
                    std::cerr
                        << "Thread ID="
                        << omp_get_thread_num()
                        << " finished calculating stats."
                        << " Sample count: "
                        << samplesPerComponent[0]
                        << "\n";
                    #endif
                }

                #pragma omp barrier

                Scalar learningRate = 0.2;
                heuristicEta = std::pow(learningRate * iterationsRun + 1, -alpha);
                for(int component_i = componentBegin; component_i < componentEnd; ++component_i) {
                    // If a mixture gets 0 samples in the first iteration,
                    // eta_i interpolate between 0 and the new ss. 
                    // If we add the samples per component before this method here,
                    // we risk that happening.

                    // Simply deciding eta_i based on iterationsRunForMixture also doesn't work,
                    // because then we don't accumulate anything from previous iterations since
                    // (iterationsRunForMixture + 1)^-1 = 1.
                    
                    // TODO: COULD BE THE PROBLEM! Turn off after 3rd iteration or so?
                    eta_i[component_i] = std::pow(learningRate * iterationsRun + 1, -alpha);
                }

                #pragma omp barrier

                #pragma omp single
                {
                    #if TANGENT_DEBUG == 1
                    std::cerr << "eta_i=" << heuristicEta << '\n';
                    #endif // TANGENT_DEBUG == 1

                    heuristicTotalWeight *= (1.f - heuristicEta);
                    heuristicTotalWeight += heuristicEta * weightSum;

                    for(int component_i = componentBegin; component_i < componentEnd; ++component_i) {
                        totalWeightForMixture[component_i] *= (1 - eta_i[component_i]);
                        totalWeightForMixture[component_i] += eta_i[component_i] * weightSum;
                    }
                }

                #pragma omp barrier
                #pragma omp single
                {
                    statsGlobal *= (1.f - heuristicEta);
                }
                #pragma omp barrier
                #pragma omp critical
                {
                    stats.sumProductInto(statsGlobal, heuristicEta);
                    #if SPLIT_AND_MERGE == 1
                    stats.sumErrorStatsInto(statsGlobal);
                    #endif // SPLIT_AND_MERGE == 1
                }
                #pragma omp barrier

                #pragma omp single
                {
                    auto& components = distribution.components();

                    // Normalize distribution.
                    distribution.setNormalization(
                        (1.f - heuristicEta) * distribution.normalization() +
                        heuristicEta * weightSum / (Scalar) samples.size()
                    );
                    Scalar invTotalWeight = 1.f / heuristicTotalWeight;
                    statsGlobal.normalize(statsGlobalNormalized, invTotalWeight);

                    int weakGaussiansCount = 0;
                    int degenerateWeightsCount = 0;
                    int degenerateGaussiansCount = 0;
                    int untrainedGaussiansCount = 0;

                    auto killComponent = [&](int component_i) -> void {
                        newParams.weights[component_i] = 0;
                        statsGlobal.weights[component_i] = 0;
                    };

                    Scalar invGlobalDecreaseFactor = 
                        1.f / Scalar(std::pow((Scalar) 3, (Scalar) std::min(trainingCutoff, iterationsRun)));
                    newParams.heuristicWeight =
                        niPriorMinusOne * invGlobalDecreaseFactor + statsGlobalNormalized.heuristicWeight;
                    
                    // #pragma omp for
                    for(int component_i = componentBegin; component_i < componentEnd; ++component_i) {                    
                        Scalar decreasedNiPriorMinusOne = niPriorMinusOne;
                        Scalar decreasedApriorMinusTwo = 100.f / (Scalar) distribution.nComponents();
                        Matrixd decreasedBPrior = decreasedApriorMinusTwo * bPriors[component_i];
                        Scalar invMixtureDecreaseFactor = 
                            1.f / std::pow((Scalar) 2, (Scalar) std::min(trainingCutoff, iterationsRun));

                        if(decreasePrior) {
                            decreasedBPrior = decreasedBPrior * invMixtureDecreaseFactor;
                            decreasedApriorMinusTwo = decreasedApriorMinusTwo * invMixtureDecreaseFactor;
                            decreasedNiPriorMinusOne = niPriorMinusOne * invGlobalDecreaseFactor;
                        }

                        Scalar invWeightStatGlobal = 1.f / statsGlobalNormalized.weights[component_i];
                        Scalar invMatrixNormalization = 1.f / (0.05 * decreasedApriorMinusTwo + statsGlobalNormalized.weights[component_i]);

                        // THIS SHOULD NEVER HAPPEN
                        assert(isfinite(statsGlobalNormalized.weights[component_i]));
                        assert(totalWeightForMixture[component_i] > 0);
                        assert(isfinite(invMatrixNormalization));

                        // Dead components should stay dead:
                        // Zero mixture weight means a zero posterior, a zero mean, and a zero covariance.
                        // Equivalently, a zero weight component will never get
                        if(distribution.weights()[component_i] == 0.f) {
                            ++degenerateWeightsCount;
                            newParams.weights[component_i] = 0.f;
                            continue;
                        }

                        if(!std::isfinite(invWeightStatGlobal)) {
                            ++weakGaussiansCount;
                            #if TANGENT_DEBUG == 1
                            std::cerr << "!isfinite(invWeightStatGlobal) = 1.f / " << statsGlobalNormalized.weights[component_i] << std::endl;
                            #endif // TANGENT_DEBUG == 1
                            newParams.weights[component_i] = decreasedNiPriorMinusOne + statsGlobalNormalized.weights[component_i];
                            continue;
                        }

                        // Only allow components to wake up in the first few iterations.
                        // Otherwise very likely to be garbage.
                        if(samplesPerComponentGlobal[component_i] < trainingBatch && iterationsRun < 3) {
                            ++untrainedGaussiansCount;
                            newParams.weights[component_i] = 
                                (iterationsRun < trainingCutoff) ?
                                decreasedNiPriorMinusOne + statsGlobalNormalized.weights[component_i] :
                                0;
                            continue;
                        }

                        newParams.weights[component_i] = decreasedNiPriorMinusOne + statsGlobalNormalized.weights[component_i];
                        newParams.means[component_i] = statsGlobalNormalized.means[component_i] * invWeightStatGlobal;
                        newParams.covs[component_i] = 
                            statsGlobalNormalized.covs[component_i] -
                            statsGlobalNormalized.means[component_i] * newParams.means[component_i].transpose();

                        auto dumpDebugInfo = [&](const std::string& error, int component_i) {
                            std::cerr 
                                << error << ": " << component_i << ":\n"
                                << "weightStatsGlobal=" << statsGlobal.weights[component_i] << "\n"
                                << "statsGlobalNormalized.weights=" << statsGlobalNormalized.weights[component_i] << "\n"
                                << "invWeightStatGlobal=" << invWeightStatGlobal << "\n"
                                << "invMatrixNormalization=" << invMatrixNormalization << "\n"
                                << newParams.covs[component_i]
                                << "\n decreasedBPrior:\n"
                                << decreasedBPrior
                                << "\n = decreasedAPrior: " << decreasedApriorMinusTwo << " * bPriors:\n"
                                << bPriors[component_i]
                                << "\n + bDepthPriors:\n"
                                << bDepthPriors[component_i]
                                << '\n';
                        };

                        #if TANGENT_DEBUG == 1
                        dumpDebugInfo("OPTIMIZATION DEBUG", component_i);
                        #endif // TANGENT_DEBUG == 1
                        
                        newParams.covs[component_i] += decreasedBPrior;
                        newParams.covs[component_i] *= invMatrixNormalization;
                        
                        if(t_dims == 6) {
                            newParams.covs[component_i].topLeftCorner(3, 3) += bDepthPriors[component_i];
                        }
                        
                        Vectord embeddedMean;
                        Scalar expJacobianDet;
                        bool success = components[component_i].tangentSpace().exp(
                            newParams.means[component_i],
                            embeddedMean,
                            expJacobianDet
                        );
                        assert(success);

                        Matrixd jointJacobian = Matrixd::Identity();
                        Matrixd jointInvJacobian = Matrixd::Identity();
                        if(jacobianCorrection) {
                            const TangentSpaced& oldTangentSpace =
                                components[component_i].tangentSpace();
                            TangentSpaced newTangentSpace(embeddedMean);

                            Eigen::Matrix<Scalar, 3, 2> expJacobian =
                                oldTangentSpace.expJacobian(
                                    newParams.means[component_i].bottomRows(2)
                                );

                            Eigen::Matrix<Scalar, 2, 3> logJacobian =
                                newTangentSpace.logJacobian(
                                    embeddedMean.bottomRows(3)
                                );
                            
                            Eigen::Matrix<Scalar, 2, 1> meanNewDir =
                                newParams.means[component_i].bottomRows(2).normalized();
                            Eigen::Matrix<Scalar, 2, 1> meanNewPerpDir;
                            meanNewPerpDir << -meanNewDir[1], meanNewDir[0];

                            Eigen::Matrix<Scalar, 2, 2> hackobian =
                                meanNewDir * meanNewDir.transpose() +
                                expJacobianDet * meanNewPerpDir * meanNewPerpDir.transpose();
                            jointJacobian.bottomRightCorner(2, 2) =
                                logJacobian * newTangentSpace.invRotation() *
                                oldTangentSpace.rotation() * expJacobian;
                            
                            #if TANGENT_DEBUG == 1
                            Eigen::Matrix<Scalar, 2, 3> invExpJacobian =
                                oldTangentSpace.logJacobian(
                                    embeddedMean.bottomRows(3)
                                );

                            Eigen::Matrix<Scalar, 3, 2> invLogJacobian =
                                newTangentSpace.expJacobian(
                                    {0.f, 0.f}
                                );

                            jointInvJacobian.bottomRightCorner(2, 2) =
                                invExpJacobian * oldTangentSpace.invRotation() *
                                newTangentSpace.rotation() * invLogJacobian;

                            std::cerr
                                << "EXP JACOBIAN VALIDATION:\n"
                                << expJacobian
                                << "\n vs \n"
                                << invExpJacobian
                                << "\n=\n"
                                << invExpJacobian * expJacobian
                                << "\n"

                                << "LOG JACOBIAN VALIDATION:\n"
                                << logJacobian
                                << "\n vs \n"
                                << invLogJacobian
                                << "\n"

                                << "JOINT JACOBIAN VALIDATION:\n"
                                << jointJacobian
                                << "\n vs \n"
                                << jointInvJacobian
                                << "\n vs \n"
                                << hackobian
                                << "\n"
                                
                                ;
                            #endif // TANGENT_DEBUG == 1
                        }

                        newParams.covs[component_i] =
                            jointJacobian * newParams.covs[component_i] * jointJacobian.transpose();

                        #if TANGENT_DEBUG == 1
                        std::cerr
                            << "Mean " << component_i << ": "
                            << newParams.means[component_i].transpose()
                            << ", embedded: " << embeddedMean.transpose()
                            << ", jacobian: " << jacobian
                            << ", newParams.covs det: " << newParams.covs[component_i].determinant()                        
                            << "\n";

                        std::cerr
                            << "Joint jacobian matrix det: "
                            << jointJacobian.determinant()
                            << " vs. "
                            << expJacobianDet
                            << '\n';
                        #endif // TANGENT_DEBUG == 1
                        
                        if(!isPositiveDefinite(newParams.covs[component_i])) {
                            dumpDebugInfo("Non-PD Matrix", component_i);
                            Eigen::Matrix<Scalar, 3, 3> spatial = 
                                newParams.covs[component_i].topLeftCorner(3, 3);
                            Eigen::Matrix<Scalar, 2, 2> directional =
                                newParams.covs[component_i].bottomRightCorner(2, 2);
                            if(!isPositiveDefinite(spatial)) {
                                std::cerr << "Non-PD Spatial Matrix:\n" << spatial << std::endl;
                            }
                            if(!isPositiveDefinite(directional)) {
                                std::cerr << "Non-PD Directional Matrix:\n" << directional << std::endl;
                            }
                            ++degenerateGaussiansCount;
                            newParams.weights[component_i] = 0.f;
                            continue;
                        }

                        components[component_i].set(
                            embeddedMean,
                            newParams.covs[component_i]
                        );

                        statsGlobalNormalized.covs[component_i] -= 
                            statsGlobalNormalized.means[component_i] * newParams.means[component_i].transpose();                        
                        
                        JointTangentVectord conditionMeanStat = statsGlobalNormalized.means[component_i];
                        conditionMeanStat.template bottomRows<Component::t_tangentDims>().setZero();
                        JointTangentVectord conditionMeanNew = newParams.means[component_i];
                        conditionMeanNew.template bottomRows<Component::t_tangentDims>().setZero();
                        statsGlobalNormalized.covs[component_i] += conditionMeanStat * conditionMeanNew.transpose();
                        statsGlobalNormalized.covs[component_i] =
                            jointJacobian * statsGlobalNormalized.covs[component_i] * jointJacobian.transpose();
                        
                        statsGlobal.covs[component_i] = statsGlobalNormalized.covs[component_i] * totalWeightForMixture[component_i];
                        statsGlobal.means[component_i].template bottomRows<Component::t_tangentDims>().setZero();
                    }

                    // Copy new distributions
                    #if TANGENT_DEBUG == 1
                    std::cerr << "weakGaussiansCount=" << weakGaussiansCount << '\n';
                    std::cerr << "degenerateWeightsCount=" << degenerateWeightsCount << '\n';
                    std::cerr << "degenerateGaussiansCount=" << degenerateGaussiansCount << '\n';
                    std::cerr << "untrainedGaussiansCount=" << untrainedGaussiansCount << '\n';
                    #endif // TANGENT_DEBUG == 1
                    
                    Scalar pdfNorm = std::accumulate(
                        std::begin(newParams.weights) + componentBegin,
                        std::begin(newParams.weights) + componentEnd,
                        0.f
                    ) + newParams.heuristicWeight;

                    #if TANGENT_DEBUG == 1
                    std::cerr << "heuristicWeightNew=" << heuristicWeightNew / pdfNorm << std::endl;
                    #endif // TANGENT_DEBUG == 1

                    jmm::normalizePdf(
                        std::begin(newParams.weights) + componentBegin,
                        std::begin(newParams.weights) + componentEnd
                    );
                    std::copy(
                        std::begin(newParams.weights) + componentBegin,
                        std::begin(newParams.weights) + componentEnd,
                        std::begin(distribution.weights()) + componentBegin
                    );

                    #if TANGENT_DEBUG == 1
                    std::cerr << "newParams.weights = [";
                    for(int component_i = componentBegin; component_i < componentEnd; ++component_i) {
                        std::cerr << newParams.weights[component_i] << ", ";
                    }
                    std::cerr << "]\n";
                    #endif // TANGENT_DEBUG == 1

                    // if(iterationsRun > 3) {
                    //     jmm::aligned_vector<Scalar> error(t_components);
                    //     statsGlobal.calculateError(error, samples.size());
                    //     maxError = *std::max_element(error.begin(), error.begin() + componentEnd);
                    //     distribution.setModelError(maxError);

                    //     for(int error_i = 0; error_i < componentEnd; ++error_i) {
                    //         if(error[error_i] > 100 && distribution.nComponents() < t_components - 1) {
                    //             splitStatsSVD(distribution, error_i);
                    //         }
                    //     }
                    // }

                    bool successfulCdfCreation = distribution.createCdf(false);
                    distribution.configure();
                    assert(successfulCdfCreation);

                    for(int component_i = componentBegin; component_i < componentEnd; ++component_i) {
                        if(samplesPerComponentGlobal[component_i] > 0) {
                            startedTraining[component_i] = true;
                        }
                    }
                    for(int component_i = componentBegin; component_i < componentEnd; ++component_i) {
                        if(samplesPerComponentGlobal[component_i] >= trainingBatch) {
                            ++iterationsRunForMixture[component_i];
                        }
                    }
                    ++iterationsRun;
                }

                #pragma omp barrier
            }
        }
    }
};

}

#endif /* __MIXTURE_MODEL_OPT_STEPWISE_TANGENT_H */
