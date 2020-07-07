#ifndef __MIXTURE_MODEL_OPT_STEPWISE_H
#define __MIXTURE_MODEL_OPT_STEPWISE_H

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

#include "util.h"


namespace jmm {

template<
    int t_dims,
    int t_components,
    int t_conditionalDims,
    typename Scalar,
    template<int, int, typename> class Component,
    template<int, int, typename> class Marginal
> class StepwiseEM {
protected:
    using MM = MixtureModel<t_dims, t_components, t_conditionalDims, Scalar, Component, Marginal>;
    using Vectord = typename MM::Vectord;
    using Matrixd = typename MM::Matrixd;

    int iterationsRun;
    std::vector<int, Eigen::aligned_allocator<int>> iterationsRunForMixture;
    std::vector<bool, Eigen::aligned_allocator<bool>> startedTraining;
    Scalar heuristicTotalWeight;
    std::vector<Scalar, Eigen::aligned_allocator<Scalar>> totalWeightForMixture;
    
    Scalar heuristicWeightStatGlobal;
    std::vector<Scalar, Eigen::aligned_allocator<Scalar>> weightsStatsGlobal;
    std::vector<Vectord, Eigen::aligned_allocator<Vectord>> meansStatsGlobal;
    std::vector<Matrixd, Eigen::aligned_allocator<Matrixd>> covsStatsGlobal;
    
    Scalar heuristicWeightStatGlobalNormalized;
    std::vector<Scalar, Eigen::aligned_allocator<Scalar>> weightsStatsGlobalNormalized;
    std::vector<Vectord, Eigen::aligned_allocator<Vectord>> meansStatsGlobalNormalized;
    std::vector<Matrixd, Eigen::aligned_allocator<Matrixd>> covsStatsGlobalNormalized;

    Scalar heuristicWeightNew;
    std::vector<Scalar, Eigen::aligned_allocator<Scalar>> weightsNew;
    std::vector<Vectord, Eigen::aligned_allocator<Vectord>> meansNew;
    std::vector<Matrixd, Eigen::aligned_allocator<Matrixd>> covsNew;

    std::vector<Scalar, Eigen::aligned_allocator<Scalar>> samplesPerComponentGlobal;
    std::vector<Scalar, Eigen::aligned_allocator<Scalar>> samplesPerComponentInIterationGlobal;
    Scalar sampleCountGlobal;

    std::vector<Matrixd, Eigen::aligned_allocator<Matrixd>> bPriors;
    std::vector<Eigen::Matrix<Scalar, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<Scalar, 3, 3>>> bDepthPriors;

    Scalar alpha;
    Scalar bPrior;
    Scalar niPriorMinusOne;
    Scalar epsilon;
    bool decreasePrior;
    Scalar trainingBatch = 15;
    bool batchModeOn = false;
    int trainingCutoff = 32;
    
    Scalar minBPrior = 0; // 1e-8f;
    Scalar minNiPriorMinusOne = 0; // 1e-6;

    boost::barrier barrier;
    boost::mutex mutex;

public:
    StepwiseEM(
        int nThreads,
        Scalar alpha=0.9,
        Scalar bPrior=6e-7,
        Scalar niPriorMinusOne=6e-5,
        Scalar epsilon=1e-100,
        bool decreasePrior=true
    ) : alpha(alpha),
        bPrior(bPrior),
        niPriorMinusOne(niPriorMinusOne),
        epsilon(epsilon),
        decreasePrior(decreasePrior),
        barrier(nThreads)
    {
        iterationsRun = 0;
        iterationsRunForMixture.resize(t_components, 0);
        heuristicTotalWeight = 0.f;
        totalWeightForMixture.resize(t_components, 0.f);
        startedTraining.resize(t_components, false);

        heuristicWeightStatGlobal = 0.f;
        weightsStatsGlobal.resize(t_components, 0.f);
        meansStatsGlobal.resize(t_components, Vectord::Zero());
        covsStatsGlobal.resize(t_components, Matrixd::Zero());

        weightsStatsGlobalNormalized.resize(t_components, 0.f);
        meansStatsGlobalNormalized.resize(t_components, Vectord::Zero());
        covsStatsGlobalNormalized.resize(t_components, Matrixd::Zero());

        weightsNew.resize(t_components, 0.f);
        meansNew.resize(t_components, Vectord::Zero());
        covsNew.resize(t_components, Matrixd::Zero());

        samplesPerComponentGlobal.resize(t_components, 0);
        samplesPerComponentInIterationGlobal.resize(t_components);
        sampleCountGlobal = 0;

        bPriors.resize(t_components, Matrixd::Zero());
        bDepthPriors.resize(t_components, Eigen::Matrix<Scalar, 3, 3>::Zero());
    };

    void setBatchMode(bool on) {
        batchModeOn = on;
    }

    std::vector<Matrixd, Eigen::aligned_allocator<Matrixd>>& getBPriors() { return bPriors; }
    std::vector<Eigen::Matrix<Scalar, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<Scalar, 3, 3>>>& getBDepthPriors() { return bDepthPriors; }

    void calculateStats(
        MM& distribution,
        const Samples<t_dims, Scalar>& samples,
        int begin,
        int end,
        bool countSamples,
        Scalar& heuristicWeightStat,
        std::vector<Scalar, Eigen::aligned_allocator<Scalar>>& weightsStats,
        std::vector<Vectord, Eigen::aligned_allocator<Vectord>>& meansStats,
        std::vector<Matrixd, Eigen::aligned_allocator<Matrixd>>& covsStats,
        std::vector<Scalar, Eigen::aligned_allocator<Scalar>>& samplesPerComponent
    ) {
        const int componentBegin = 0;
        const int componentEnd = distribution.nComponents();
        Eigen::Matrix<Scalar, t_components, 1> posterior;

        for(int sample_i = begin; sample_i < end; ++sample_i) {
            if(samples.weights(sample_i) == 0) {
                continue;
            }
            Vectord sample = samples.samples.col(sample_i);
            Scalar heuristicPosterior = 0;
            bool useHeuristic = samples.isDiffuse(sample_i);
            distribution.posterior(
                sample,
                useHeuristic,
                samples.heuristicPdfs(sample_i),
                posterior,
                heuristicPosterior
            );
            heuristicWeightStat += samples.weights(sample_i) * heuristicPosterior;
            const auto& components = distribution.getComponentDistributions();
            for(int component_i = componentBegin; component_i < componentEnd; ++component_i) {
                Scalar weight = samples.weights(sample_i) * posterior(component_i);
                if(countSamples) {
                    samplesPerComponent[component_i] +=
                        components[component_i].isInside(samples.samples.col(sample_i), 0.95) ? 1 : 0;
                } 
                weightsStats[component_i] += weight;
                meansStats[component_i] += weight * sample;
                covsStats[component_i] += weight * sample * sample.transpose();
            }
        }
    }

    void calculateStatsPrune(
        MM& distribution,
        Samples<t_dims, Scalar>& samples,
        int begin,
        int end,
        bool countSamples,
        Scalar& heuristicWeightStat,
        std::vector<Scalar, Eigen::aligned_allocator<Scalar>>& weightsStats,
        std::vector<Vectord, Eigen::aligned_allocator<Vectord>>& meansStats,
        std::vector<Matrixd, Eigen::aligned_allocator<Matrixd>>& covsStats,
        std::vector<Scalar, Eigen::aligned_allocator<Scalar>>& samplesPerComponent
    ) {
        const int componentBegin = 0;
        const int componentEnd = distribution.nComponents();

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> posterior(t_components, 1);
        Eigen::Matrix<int, Eigen::Dynamic, 1> posteriorIndices(t_components, 1);
        int posteriorLastIdx;

        for(int sample_i = begin; sample_i < end; ++sample_i) {
            if(!std::isfinite(samples.weights(sample_i))) {
                std::cerr 
                    << "inf or nan sample, id=" 
                    << sample_i 
                    << ", value="
                    << samples.weights(sample_i) 
                    << '\n';
                continue;
            }
            if(samples.weights(sample_i) == 0) {
                continue;
            }
            Vectord sample = samples.samples.col(sample_i);
            Scalar heuristicPosterior = 0;
            bool useHeuristic = samples.isDiffuse(sample_i);
            
            distribution.posteriorPrune(
                sample,
                useHeuristic,
                samples.heuristicPdfs(sample_i),
                posterior,
                posteriorIndices,
                posteriorLastIdx,
                heuristicPosterior
            );

            heuristicWeightStat += samples.weights(sample_i) * heuristicPosterior;
            const auto& components = distribution.getComponentDistributions();
            
            for(int found_i = 0; found_i < posteriorLastIdx; ++found_i) {
                Scalar weight = samples.weights(sample_i) * posterior(found_i);
                if(weight == 0.f) {
                    continue;
                }
                int component_i = posteriorIndices(found_i);
                bool isInside = 
                    components[component_i].isInside(samples.samples.col(sample_i), 0.95);

                if(countSamples) {
                    samplesPerComponent[component_i] += isInside ? 1 : 0;
                } 
                weightsStats[component_i] += weight;
                meansStats[component_i] += weight * sample;
                covsStats[component_i] += weight * sample * sample.transpose();
            }
        }
    }

    void optimize(
        MM& distribution,
        Samples<t_dims, Scalar>& samples,
        int iterations,
        int threadId,
        int begin,
        int end
    ) {
        int componentBegin = 0;
        int componentEnd = distribution.nComponents();
        
        std::vector<Scalar, Eigen::aligned_allocator<Scalar>> eta_i(t_components);

        Scalar heuristicEta = 0;
        Scalar heuristicWeightStat;
        std::vector<Scalar, Eigen::aligned_allocator<Scalar>> weightsStats(t_components);
        std::vector<Vectord, Eigen::aligned_allocator<Vectord>> meansStats(t_components);
        std::vector<Matrixd, Eigen::aligned_allocator<Matrixd>> covsStats(t_components);

        std::vector<Scalar, Eigen::aligned_allocator<Scalar>> samplesPerComponent(t_components);
        
        barrier.wait();

        Scalar weightSum = 0.f;
        for(int sample_i = begin; sample_i < end; ++sample_i) {
            if(!std::isfinite(samples.weights(sample_i))) {
                std::cerr 
                    << "inf or nan sample, id=" 
                    << sample_i 
                    << ", value="
                    << samples.weights(sample_i) 
                    << '\n';
                continue;
            }
            samples.weights(sample_i) /= (Scalar) samples.size();
            weightSum += samples.weights(sample_i);
        }
        barrier.wait();

        for(int emIt = 0; emIt < iterations; ++emIt) {
            barrier.wait();
            if(threadId == 0) {
                heuristicWeightNew = 0.f;
                std::fill(std::begin(weightsNew), std::end(weightsNew), 0.f);
                std::fill(std::begin(meansNew), std::end(meansNew), Vectord::Zero());
                std::fill(std::begin(covsNew), std::end(covsNew), Matrixd::Zero());
                std::fill(
                    std::begin(samplesPerComponentInIterationGlobal),
                    std::end(samplesPerComponentInIterationGlobal),
                    0
                );
            }

            barrier.wait();

            heuristicWeightStat = 0.f;
            std::fill(std::begin(weightsStats), std::end(weightsStats), 0.f);
            std::fill(std::begin(meansStats), std::end(meansStats), Vectord::Zero());
            std::fill(std::begin(covsStats), std::end(covsStats), Matrixd::Zero());

            std::fill(std::begin(samplesPerComponent), std::end(samplesPerComponent), 0);

            calculateStatsPrune(
                distribution,
                samples,
                begin,
                end,
                emIt == 0,
                heuristicWeightStat,
                weightsStats,
                meansStats,
                covsStats,
                samplesPerComponent
            );

            {
                boost::unique_lock<boost::mutex> lock(mutex);
                for(int component_i = componentBegin; component_i < componentEnd; ++component_i) {
                    samplesPerComponentGlobal[component_i] += samplesPerComponent[component_i];
                    samplesPerComponentInIterationGlobal[component_i] += samplesPerComponent[component_i];
                    sampleCountGlobal += samplesPerComponent[component_i];
                }
            }

            barrier.wait();

            Scalar learningRate = 0.1;
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

            barrier.wait();

            if(threadId == 0) {
                std::cerr << "eta_i=" << eta_i[componentBegin] << '\n';
                heuristicTotalWeight *= (1.f - heuristicEta);
                heuristicTotalWeight += heuristicEta * weightSum;

                for(int component_i = componentBegin; component_i < componentEnd; ++component_i) {
                    totalWeightForMixture[component_i] *= (1 - eta_i[component_i]);
                }
            }

            barrier.wait();

            {
                boost::unique_lock<boost::mutex> lock(mutex);
                for(int component_i = componentBegin; component_i < componentEnd; ++component_i) {
                    totalWeightForMixture[component_i] += eta_i[component_i] * weightSum;
                }
            }

            barrier.wait();

            if(threadId == 0) {
                std::cerr << "Optimizing, StepwiseEM iteration=" + std::to_string(emIt) + "\n";

                heuristicWeightStatGlobal *= (1.f - heuristicEta);
                heuristicWeightStatGlobal += heuristicEta * heuristicWeightStat; 
                for(int component_i = componentBegin; component_i < componentEnd; ++component_i) {
                    weightsStatsGlobal[component_i] *= (1.f - eta_i[component_i]);
                    meansStatsGlobal[component_i] *= (1.f - eta_i[component_i]);
                    covsStatsGlobal[component_i] *= (1.f - eta_i[component_i]);
                }
            }
            
            barrier.wait();

            {
                boost::unique_lock<boost::mutex> lock(mutex);
                for(int component_i = componentBegin; component_i < componentEnd; ++component_i) {
                    weightsStatsGlobal[component_i] += eta_i[componentBegin] * weightsStats[component_i];
                    meansStatsGlobal[component_i] += eta_i[componentBegin] * meansStats[component_i];
                    covsStatsGlobal[component_i] += eta_i[componentBegin] * covsStats[component_i];
                }
            }

            barrier.wait();

            if(threadId == 0) {
                distribution.setNormalization(heuristicTotalWeight);
                heuristicWeightStatGlobalNormalized = heuristicWeightStatGlobal / heuristicTotalWeight;
                for(int component_i = componentBegin; component_i < componentEnd; ++component_i) {
                    Scalar invTotalWeight = 1.f / totalWeightForMixture[component_i];
                    // std::cerr << "totalWeightForMixture[" << component_i << "] = " << totalWeightForMixture[component_i] << "\n"; 
                    weightsStatsGlobalNormalized[component_i] = weightsStatsGlobal[component_i] * invTotalWeight;
                    meansStatsGlobalNormalized[component_i] = meansStatsGlobal[component_i] * invTotalWeight;
                    covsStatsGlobalNormalized[component_i] = covsStatsGlobal[component_i] * invTotalWeight;
                }

                auto& componentDistributions = distribution.getComponentDistributions();
                int weakGaussiansCount = 0;
                int degenerateWeightsCount = 0;
                int degenerateGaussiansCount = 0;
                int untrainedGaussiansCount = 0;

                auto killComponent = [&](int component_i) -> void {
                    weightsNew[component_i] = 0;
                    weightsStatsGlobal[component_i] = 0;
                };

                Scalar invGlobalDecreaseFactor = 
                    1.f / Scalar(std::pow((Scalar) 3, (Scalar) std::min(trainingCutoff, iterationsRun)));
                heuristicWeightNew =
                    niPriorMinusOne * invGlobalDecreaseFactor + heuristicWeightStatGlobalNormalized;
                for(int component_i = componentBegin; component_i < componentEnd; ++component_i) {                    
                    Scalar decreasedNiPriorMinusOne = niPriorMinusOne;
                    Scalar decreasedApriorMinusTwo = 100.f / (Scalar) distribution.nComponents(); // 1e-1;
                    Matrixd decreasedBPrior = decreasedApriorMinusTwo * bPriors[component_i];
                    Scalar invMixtureDecreaseFactor = 
                        1.f / std::pow((Scalar) 2, (Scalar) std::min(trainingCutoff, iterationsRun));
                    if(decreasePrior) {
                        // Scalar invMixtureDecreaseFactor = 
                        //     1.f / ((Scalar) iterationsRunForMixture[component_i] + 1);
                        
                        decreasedBPrior = decreasedBPrior * invMixtureDecreaseFactor;
                        // decreasedBPrior.topLeftCorner(3, 3) += decreasedApriorMinusTwo * bDepthPriors[component_i];
                        decreasedApriorMinusTwo = decreasedApriorMinusTwo * invMixtureDecreaseFactor;
                        decreasedNiPriorMinusOne = niPriorMinusOne * invGlobalDecreaseFactor;
                        
                        // std::cerr << "bPrior = " << decreasedBPrior << '\n';
                        // std::cerr << "niPrior = " << decreasedNiPriorMinusOne << '\n';
                    }

                    Scalar invWeightStatGlobal = 1.f / weightsStatsGlobalNormalized[component_i];
                    Scalar invMatrixNormalization = 1.f / (1e-1 * decreasedApriorMinusTwo + weightsStatsGlobalNormalized[component_i]);

                    // THIS SHOULD NEVER HAPPEN
                    assert(isfinite(weightsStatsGlobalNormalized[component_i]));
                    assert(totalWeightForMixture[component_i] > 0);
                    assert(isfinite(invMatrixNormalization));

                    // Dead components should stay dead:
                    // Zero mixture weight means a zero posterior, a zero mean, and a zero covariance.
                    // Equivalently, a zero weight component will never get
                    if(distribution.getMixtureWeights()[component_i] == 0.f) {
                        ++degenerateWeightsCount;
                        weightsNew[component_i] = 0.f;
                        continue;
                    }

                    if(!std::isfinite(invWeightStatGlobal)) {
                        ++weakGaussiansCount;
                        // std::cerr << "!isfinite(invWeightStatGlobal) = 1.f / " << weightsStatsGlobalNormalized[component_i] << std::endl;
                        weightsNew[component_i] = decreasedNiPriorMinusOne + weightsStatsGlobalNormalized[component_i];
                        continue;
                    }

                    // Only allow components to wake up in the first few iterations.
                    // Otherwise very likely to be garbage.
                    if(samplesPerComponentGlobal[component_i] < trainingBatch && iterationsRun < 3) {
                        ++untrainedGaussiansCount;
                        weightsNew[component_i] = 
                            (iterationsRun < trainingCutoff) ?
                            decreasedNiPriorMinusOne + weightsStatsGlobalNormalized[component_i] :
                            0;
                        // meansNew[component_i] = meansStatsGlobalNormalized[component_i] * invWeightStatGlobal;
                        // std::cerr << "WEIGHTS NEW: " << weightsNew[component_i] << ", decreasedNiPriorMinusOne" << decreasedNiPriorMinusOne << std::endl;
                        // std::cerr << "iterationsRun" << iterationsRun << ", trainingCutoff" << trainingCutoff << std::endl;
                        continue;
                    }

                    weightsNew[component_i] = decreasedNiPriorMinusOne + weightsStatsGlobalNormalized[component_i];
                    meansNew[component_i] = meansStatsGlobalNormalized[component_i] * invWeightStatGlobal;
                    covsNew[component_i] = 
                        // covsStatsGlobalNormalized[component_i] * invWeightStatGlobal -
                        // meansNew[component_i] * meansNew[component_i].transpose();
                        covsStatsGlobalNormalized[component_i] -
                        meansStatsGlobalNormalized[component_i] * meansNew[component_i].transpose();

                    covsNew[component_i] += decreasedBPrior;
                    covsNew[component_i] *= invMatrixNormalization;
                    covsNew[component_i].topLeftCorner(3, 3) += bDepthPriors[component_i];
                    // covsNew[component_i] += (bPriorDiag * invWeightStatGlobal).asDiagonal();

                    if(!isPositiveDefinite(covsNew[component_i])) {
                        std::cerr 
                            << "Non-PD Matrix " << component_i << ":\n"
                            << "weightStatsGlobal=" << weightsStatsGlobal[component_i] << "\n"
                            << "weightsStatsGlobalNormalized=" << weightsStatsGlobalNormalized[component_i] << "\n"
                            << "invWeightStatGlobal=" << invWeightStatGlobal << "\n"
                            << "invMatrixNormalization=" << invMatrixNormalization << "\n"
                            << covsNew[component_i]
                            << "\n decreasedBPrior:\n"
                            << decreasedBPrior
                            << "\n = decreasedAPrior: " << decreasedApriorMinusTwo << " * bPriors:\n"
                            << bPriors[component_i]
                            << "\n + bDepthPriors:\n"
                            << bDepthPriors[component_i]
                            << '\n';
                        Eigen::Matrix<Scalar, 3, 3> spatial = 
                            covsNew[component_i].topLeftCorner(3, 3);
                        Eigen::Matrix<Scalar, 2, 2> directional =
                            covsNew[component_i].bottomRightCorner(2, 2);
                        if(!isPositiveDefinite(spatial)) {
                            std::cerr << "Non-PD Spatial Matrix:\n" << spatial << std::endl;
                        }
                        if(!isPositiveDefinite(directional)) {
                            std::cerr << "Non-PD Directional Matrix:\n" << directional << std::endl;
                        }
                        ++degenerateGaussiansCount;
                        weightsNew[component_i] = 0.f;
                        continue;
                    }
                    
                    // makePositiveDefinite(covsNew[component_i], bPriorDiag(0) * invWeightStatGlobal, component_i);
                    
                    componentDistributions[component_i].set(meansNew[component_i], covsNew[component_i]);
                }

                // Copy new distributions
                std::cerr << "weakGaussiansCount=" << weakGaussiansCount << '\n';
                std::cerr << "degenerateWeightsCount=" << degenerateWeightsCount << '\n';
                std::cerr << "degenerateGaussiansCount=" << degenerateGaussiansCount << '\n';
                std::cerr << "untrainedGaussiansCount=" << untrainedGaussiansCount << '\n';
                
                Scalar pdfNorm = std::accumulate(
                    std::begin(weightsNew) + componentBegin,
                    std::begin(weightsNew) + componentEnd,
                    0.f
                ) + heuristicWeightNew;
                distribution.setHeuristicWeight(heuristicWeightNew / pdfNorm);

                jmm::normalizePdf(
                    std::begin(weightsNew) + componentBegin,
                    std::begin(weightsNew) + componentEnd
                );
                std::copy(
                    std::begin(weightsNew) + componentBegin,
                    std::begin(weightsNew) + componentEnd,
                    std::begin(distribution.getMixtureWeights()) + componentBegin
                );

                // std::cerr << "weightsNew = [";
                // for(int component_i = componentBegin; component_i < componentEnd; ++component_i) {
                //     std::cerr << weightsNew[component_i] << ", ";
                // }
                // std::cerr << "]\n";

                bool successfulCdfCreation = distribution.createCdf(false);
                distribution.configure(true);
                assert(successfulCdfCreation);

                if(!batchModeOn) {
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
                
            }

            barrier.wait();
        }
    }
};

}

#endif /* __MIXTURE_MODEL_OPT_STEPWISE_H */
