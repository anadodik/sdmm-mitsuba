#ifndef __MIXTURE_MODEL_OPT_BATCH_H
#define __MIXTURE_MODEL_OPT_BATCH_H

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
> class BatchEM {
private:
    using MM = MixtureModel<t_dims, t_components, t_conditionalDims, Scalar, Component, Marginal>;
    using Vectord = typename MM::Vectord;
    using Matrixd = typename MM::Matrixd;

    Scalar heuristicWeightStatGlobal;
    std::array<Scalar, t_components> weightsStatsGlobal;
    std::array<Vectord, t_components> meansStatsGlobal;
    std::array<Matrixd, t_components> covsStatsGlobal;

    Scalar heuristicWeightNew;
    std::array<Scalar, t_components> weightsNew;
    std::array<Vectord, t_components> meansNew;
    std::array<Matrixd, t_components> covsNew;

    Scalar weightsSum = 0;

    Scalar bPrior;
    Scalar niPriorMinusOne;
    Scalar epsilon;

    constexpr static bool USE_BAYESIAN = true;

    boost::barrier barrier;
    boost::mutex mutex;

    void normalizeWeights(
        Samples<t_dims, Scalar>& samples,
        int threadId,
        int begin,
        int end
    ) {
        if(threadId == 0) {
            weightsSum = 0;
        }
        
        barrier.wait();

        Scalar weightsSumLocal = samples.weights.middleRows(begin, end - begin).sum();
        {
            boost::unique_lock<boost::mutex> lock(mutex);
            weightsSum += weightsSumLocal;
        }

        barrier.wait();
        Scalar invWeightsSum = 1 / weightsSum;
        samples.weights.middleRows(begin, end - begin) *= invWeightsSum;
        barrier.wait();
    }

public:
    BatchEM(int nThreads, Scalar bPrior=1e-7, Scalar niPriorMinusOne=6e-5, Scalar epsilon=1e-14)
    : barrier(nThreads), bPrior(bPrior), niPriorMinusOne(niPriorMinusOne), epsilon(epsilon) { };
    
    void optimize(
        MM& distribution,
        Samples<t_dims, Scalar>& samples,
        int iterations,
        int threadId,
        int begin,
        int end
    ) {
        barrier.wait();
        normalizeWeights(samples, threadId, begin, end);

        Scalar heuristicWeightStat;
        std::array<Scalar, t_components> weightsStats;
        std::array<Vectord, t_components> meansStats;
        std::array<Matrixd, t_components> covsStats;

        Eigen::Matrix<Scalar, t_components, 1> posterior;

        for(int emIt = 0; emIt < iterations; ++emIt) {
            // m_timer->reset();

            barrier.wait();
            if(threadId == 0) {
                std::cerr << "Optimizing, BatchEM iteration=" + std::to_string(emIt) + "\n";

                heuristicWeightStatGlobal = 0.f;
                std::fill(std::begin(weightsStatsGlobal), std::end(weightsStatsGlobal), 0.f);
                std::fill(std::begin(meansStatsGlobal), std::end(meansStatsGlobal), Vectord::Zero());
                std::fill(std::begin(covsStatsGlobal), std::end(covsStatsGlobal), Matrixd::Zero());

                heuristicWeightNew = 0.f;
                std::fill(std::begin(weightsNew), std::end(weightsNew), 0.f);
                std::fill(std::begin(meansNew), std::end(meansNew), Vectord::Zero());
                std::fill(std::begin(covsNew), std::end(covsNew), Matrixd::Zero());
            }

            heuristicWeightStat = 0.f;
            std::fill(std::begin(weightsStats), std::end(weightsStats), 0.f);
            std::fill(std::begin(meansStats), std::end(meansStats), Vectord::Zero());
            std::fill(std::begin(covsStats), std::end(covsStats), Matrixd::Zero());

            barrier.wait();

            const int activeComponents = distribution.nComponents();
            for(int sample_i = begin; sample_i < end; ++sample_i) {
                Vectord sample = samples.samples.col(sample_i);
                Scalar heuristicPosterior = 1;
                bool useHeuristic = samples.isDiffuse(sample_i);
                distribution.posterior(
                    sample,
                    useHeuristic,
                    samples.heuristicPdfs(sample_i),
                    posterior,
                    heuristicPosterior
                );
                
                heuristicWeightStat += samples.weights(sample_i) * heuristicPosterior;
                for(int component_i = 0; component_i < activeComponents; ++component_i) {
                    Scalar weight = samples.weights(sample_i) * posterior(component_i);
                    weightsStats[component_i] += weight;
                    meansStats[component_i] += weight * sample;
                    covsStats[component_i] += weight * sample * sample.transpose();
                }
            }
            {
                boost::unique_lock<boost::mutex> lock(mutex);
                heuristicWeightStatGlobal += heuristicWeightStat;
                for(int component_i = 0; component_i < activeComponents; ++component_i) {
                    weightsStatsGlobal[component_i] += weightsStats[component_i];
                    meansStatsGlobal[component_i] += meansStats[component_i];
                    covsStatsGlobal[component_i] += covsStats[component_i];
                }
            }
            barrier.wait();

            if(threadId == 0) {
                distribution.setNormalization(weightsSum);
                Vectord bPriorDiag = Vectord::Constant(bPrior);
                heuristicWeightNew = heuristicWeightStatGlobal + niPriorMinusOne;
                for(int component_i = 0; component_i < activeComponents; ++component_i) {
                    Scalar invWeightStatGlobal = 1.f / weightsStatsGlobal[component_i];
                    if(weightsStatsGlobal[component_i] == 0 || std::isnan(invWeightStatGlobal)) {
                        continue;
                    }
                    weightsNew[component_i] = weightsStatsGlobal[component_i];
                    if(USE_BAYESIAN) {
                        weightsNew[component_i] += niPriorMinusOne;
                    }
                    meansNew[component_i] = meansStatsGlobal[component_i] * invWeightStatGlobal;
                    covsNew[component_i] = (covsStatsGlobal[component_i]
                        - meansStatsGlobal[component_i] * meansNew[component_i].transpose())
                        * invWeightStatGlobal;
                    if(USE_BAYESIAN) {
                        covsNew[component_i] += (bPriorDiag * invWeightStatGlobal).asDiagonal();
                        jmm::makePositiveDefinite(covsNew[component_i], bPriorDiag(0) * invWeightStatGlobal, component_i);
                    }
                }
                heuristicWeightNew /= std::accumulate(
                    std::begin(weightsNew),
                    std::begin(weightsNew) + activeComponents,
                    0.f
                ) + heuristicWeightNew;

                distribution.setHeuristicWeight(heuristicWeightNew);
                std::cerr << "heuristicWeightNew=" << distribution.heuristicWeight() << std::endl;

                jmm::normalizePdf(std::begin(weightsNew), std::begin(weightsNew) + activeComponents);

                // Copy new distributions
                std::copy(std::begin(weightsNew), std::end(weightsNew), std::begin(distribution.getMixtureWeights()));
                auto& componentDistributions = distribution.getComponentDistributions();
                for(int component_i = 0; component_i < activeComponents; ++component_i) {
                    Scalar invWeightStatGlobal = 1.f / weightsStatsGlobal[component_i];
                    if(weightsStatsGlobal[component_i] == 0 || std::isnan(invWeightStatGlobal)) {
                        continue;
                    }
                    componentDistributions[component_i].set(meansNew[component_i], covsNew[component_i]);
                }
                bool successfulCdfCreation = distribution.createCdf(false);
                assert(successfulCdfCreation);
            }
            barrier.wait();
            // avgEMIterationTime.incrementBase();
            // avgEMIterationTime += m_timer->lap();
        }
    }
};

}

#endif /* __MIXTURE_MODEL_OPT_BATCH_H */
