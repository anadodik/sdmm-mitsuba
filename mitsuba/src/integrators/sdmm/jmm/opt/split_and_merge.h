#ifndef __MIXTURE_MODEL_OPT_SPLIT_AND_MERGE_H
#define __MIXTURE_MODEL_OPT_SPLIT_AND_MERGE_H

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
#include "../outlier_detection.h"

#include "util.h"


namespace jmm {

template<
    int t_dims,
    int t_components,
    int t_conditionalDims,
    typename Scalar,
    template<int, int, typename> class Component_t,
    template<int, int, typename> class Marginal_t
> class SplitAndMergeEM : public StepwiseEM<t_dims, t_components, t_conditionalDims, Scalar, Component_t, Marginal_t> {
public:
    using MM = MixtureModel<t_dims, t_components, t_conditionalDims, Scalar, Component_t, Marginal_t>;
    using Component = typename MM::Component;
    using Vectord = typename MM::Vectord;
    using Matrixd = typename MM::Matrixd;
    using BaseEM = StepwiseEM<t_dims, t_components, t_conditionalDims, Scalar, Component_t, Marginal_t>;

    SplitAndMergeEM(
        int nThreads,
        Scalar alpha=0.9,
        Scalar bPrior=6e-7,
        Scalar niPriorMinusOne=6e-5,
        Scalar epsilon=1e-14,
        bool decreasePrior=true,
        int nSplits=1
    )
    : BaseEM(nThreads, alpha, bPrior, niPriorMinusOne, epsilon, decreasePrior), nSplits(nSplits) {
        std::cerr << nThreads << '\n'
            << alpha << '\n';
    };

    struct GaussianParams {
        Scalar weight;
        Vectord mean;
        Matrixd cov;

        Component distribution() {
            return Component(mean, cov);
        }
    };

    GaussianParams merge(
        const GaussianParams& gaussian_i,
        const GaussianParams& gaussian_j
    ) {
        const Scalar weight_i = gaussian_i.weight;
        const Scalar weight_j = gaussian_j.weight;
        const Vectord& mean_i = gaussian_i.mean;
        const Vectord& mean_j = gaussian_j.mean;
        const Matrixd& cov_i = gaussian_i.cov;
        const Matrixd& cov_j = gaussian_j.cov;

        Scalar weight = weight_i + weight_j;
        Vectord mean = (
            weight_i * mean_i + weight_j * mean_j
        ) / weight;
        Vectord mean_i_centered = mean_i - mean;
        Vectord mean_j_centered = mean_i - mean;

        Matrixd cov = (
            weight_i * (cov_i + mean_i_centered * mean_i_centered.transpose()) +
            weight_j * (cov_j + mean_j_centered * mean_j_centered.transpose())
        ) / weight;

        return {weight, mean, cov};
    }


    void mergeStats(MM& distribution, int stat_i, int stat_j) {
        std::cerr << "Merging components (" << stat_i << ", " << stat_j << ").\n";
        assert(distribution.nComponents() > 1);

        Scalar weight_i = distribution.getMixtureWeights()[stat_i];
        Scalar weight_j = distribution.getMixtureWeights()[stat_j];
        const Component& component_i = distribution.getComponentDistributions()[stat_i];
        const Component& component_j = distribution.getComponentDistributions()[stat_j];

        GaussianParams merged = merge(
            {weight_i, component_i.mean(), component_i.cov()},
            {weight_j, component_j.mean(), component_j.cov()}
        );
        
        this->weightsNew[stat_i] = merged.weight;
        this->meansNew[stat_i] = merged.mean;
        this->covsNew[stat_i] = merged.cov;
        
        distribution.getMixtureWeights()[stat_i] = this->weightsNew[stat_i];
        std::cerr << "Setting component " << stat_i << " to: " << this->weightsNew[stat_i] << ".\n";
        distribution.getComponentDistributions()[stat_i].set(
            this->meansNew[stat_i], this->covsNew[stat_i]
        );

        this->weightsStatsGlobal[stat_i] =
            this->weightsStatsGlobal[stat_i] +
            this->weightsStatsGlobal[stat_j];

        this->meansStatsGlobal[stat_i] = 
            this->weightsStatsGlobal[stat_i] *
            this->meansNew[stat_i];

        this->covsStatsGlobal[stat_i] =
            this->weightsStatsGlobal[stat_i] * (
                this->covsNew[stat_i] +
                this->meansNew[stat_i] *
                this->meansNew[stat_i].transpose()
            );
        
        // compareNewAndStats(stat_i);

        std::swap(
            distribution.getComponentDistributions()[stat_j],
            distribution.getComponentDistributions()[distribution.nComponents() - 1]
        );

        distribution.setNComponents(distribution.nComponents() - 1);
        bool successfulCdfCreation = distribution.createCdf(true);
    }

    std::pair<GaussianParams, GaussianParams> splitCholesky(
        const GaussianParams& gaussian
    ) {
        Scalar weight = gaussian.weight;

        const Eigen::LLT<Matrixd>& cholesky = gaussian.cov.llt();
        Vectord a_l = Matrixd(cholesky.matrixL()).col(l);

        Scalar weight_j = weightSplitWeight * weight;
        Scalar weight_k = (1 - weightSplitWeight) * weight;

        Scalar sqrt_k_over_j = std::sqrt(weight_k / weight_j);
        Scalar sqrt_j_over_k = std::sqrt(weight_j / weight_k);
        Vectord mean_j = gaussian.mean - sqrt_k_over_j * u * a_l;
        Vectord mean_k = gaussian.mean + sqrt_j_over_k * u * a_l;
        
        Eigen::DiagonalMatrix<Scalar, t_dims> diag_j;
        diag_j.diagonal().setConstant(sqrt_k_over_j);
        diag_j.diagonal()(l) = std::sqrt(
            beta * (1 - u * u) * weight / weight_j
        );
        Matrixd L_j = Matrixd(cholesky.matrixL()) * diag_j;

        Eigen::DiagonalMatrix<Scalar, t_dims> diag_k;
        diag_k.diagonal().setConstant(sqrt_j_over_k);
        diag_k.diagonal()(l) = std::sqrt(
            (1 - beta) * (1 - u * u) * weight / weight_k
        );
        Matrixd L_k = Matrixd(cholesky.matrixL()) * diag_k;

        return {
            {weight_j, mean_j, L_j * L_j.transpose()},
            {weight_k, mean_k, L_k * L_k.transpose()}
        };
    }

    std::pair<GaussianParams, GaussianParams> splitSVD(
        const GaussianParams& gaussian
    ) {
        Scalar weight = gaussian.weight;
        Matrixd cov = gaussian.cov;

        const Eigen::JacobiSVD<Matrixd> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Matrixd A = svd.matrixU() * svd.singularValues().cwiseSqrt().asDiagonal();
        Vectord a_l = A.col(l);
        std::cerr << a_l.transpose() << "\n";

        Scalar weight_j = weightSplitWeight * weight;
        Scalar weight_k = (1 - weightSplitWeight) * weight;

        Scalar sqrt_k_over_j = std::sqrt(weight_k / weight_j);
        Scalar sqrt_j_over_k = std::sqrt(weight_j / weight_k);
        Vectord mean_j = gaussian.mean - mean_j_const * a_l;
        Vectord mean_k = gaussian.mean + mean_k_const * a_l;

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

        Scalar weight = distribution.getMixtureWeights()[statIdx];
        const Component& component = distribution.getComponentDistributions()[statIdx];

        const auto& splits_pair = splitSVD(
            {weight, component.mean(), component.cov()}
        );
        GaussianParams splits[2] = {splits_pair.first, splits_pair.second};

        const Scalar totalWeight = this->totalWeightForMixture[statIdx];
        const Scalar weightStat = this->weightsStatsGlobal[statIdx];

        std::cerr << "Increasing nComponents to " << distribution.nComponents() + 1 << '\n';
        assert(distribution.nComponents() + 1 < t_components);
        distribution.setNComponents(distribution.nComponents() + 1);

        int j = statIdx;
        int k = distribution.nComponents() - 1;

        this->weightsStatsGlobal[j] = weightSplitWeight * weightStat;
        this->weightsStatsGlobal[k] = (1 - weightSplitWeight) * weightStat;

        for(int split_i = 0; split_i < 2; ++split_i) {
            int component_i = (split_i == 0) ? j : k;
            Scalar decreasedBPrior = this->bPrior / 
                (Scalar) (this->iterationsRunForMixture[component_i] + 1);
            Vectord bPriorDiag = Vectord::Constant(decreasedBPrior);

            this->totalWeightForMixture[component_i] = totalWeight;
            this->weightsNew[component_i] = splits[split_i].weight;
            this->meansNew[component_i] = splits[split_i].mean;
            this->covsNew[component_i] = splits[split_i].cov;
            this->covsNew[component_i] += (
                bPriorDiag * 
                this->totalWeightForMixture[component_i] /
                this->weightsStatsGlobal[component_i]
            ).asDiagonal();

            
            distribution.getMixtureWeights()[component_i] = this->weightsNew[component_i];
            std::cerr << "Setting component " << component_i << " to: " << this->weightsNew[component_i] << ".\n";
            distribution.getComponentDistributions()[component_i].set(
                this->meansNew[component_i], this->covsNew[component_i]
            );

            this->meansStatsGlobal[component_i] =
                this->weightsStatsGlobal[component_i] *
                this->meansNew[component_i];

            this->covsStatsGlobal[component_i] =
                this->weightsStatsGlobal[component_i] * (
                    this->covsNew[component_i] +
                    this->meansNew[component_i] *
                    this->meansNew[component_i].transpose()
                );
            
            // compareNewAndStats(component_i);
        }
        bool successfulCdfCreation = distribution.createCdf(true);
    }

    void compareNewAndStats(int component_i) {
        Scalar invTotalWeight = 1.f / this->totalWeightForMixture[component_i];
        
        Scalar weightsStatNormalized =
            this->weightsStatsGlobal[component_i] * invTotalWeight;
        Vectord meansStatNormalized =
            this->meansStatsGlobal[component_i] * invTotalWeight;
        Matrixd covsStatNormalized =
            this->covsStatsGlobal[component_i] * invTotalWeight;

        Scalar invWeightStat = 1.f / weightsStatNormalized;
        Vectord meanNew = meansStatNormalized * invWeightStat;
        Matrixd covNew = covsStatNormalized * invWeightStat - meanNew * meanNew.transpose();

        Vectord meanDiff = (this->meansNew[component_i] - meanNew).cwiseAbs();
        Matrixd covDiff = (this->covsNew[component_i] - covNew).cwiseAbs();
        std::cerr << meanDiff << "\n";
        std::cerr << covDiff << "\n";
    }

    struct MergeScore {
        int component_i;
        int component_j;
        Scalar score;
    };

    std::vector<MergeScore> posteriorDotProduct(
        MM& distribution,
        Samples<t_dims, Scalar>& samples
    ) {
        int nSamples = samples.size();
        int nComponents = distribution.nComponents();

        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> posteriors(nSamples, nComponents);
        Eigen::Matrix<Scalar, t_components, 1> posterior;
        for(int sample_i = 0; sample_i < nSamples; ++sample_i) {
            Vectord sample = samples.samples.col(sample_i);
            Scalar heuristicPosterior;
            distribution.posterior(
                sample,
                samples.isDiffuse(sample_i),
                samples.heuristicPdfs(sample_i),
                posterior,
                heuristicPosterior
            );

            assert(posterior.topRows(nComponents).array().isFinite().all());
            posteriors.row(sample_i) =
                samples.weights(sample_i) * posterior.topRows(nComponents).transpose();
        }

        posteriors.rowwise() -= posteriors.colwise().mean();
        Eigen::Matrix<Scalar, 1, Eigen::Dynamic> norms = posteriors.colwise().norm();
        std::cerr << 
            "Nonzero posterior norms " << (norms.array() != 0.f).count() <<
            " out of " << nComponents << " samples.\n";
        posteriors.array().rowwise() *= (norms.array() + 1e-15).inverse();

        std::vector<MergeScore> scores;
        scores.reserve(nComponents * (nComponents + 1) / 2);

        const Scalar threshold = 0.8;
        for(int component_i = 0; component_i < nComponents - 1; ++component_i) {
            for(int component_j = component_i + 1; component_j < nComponents; ++component_j) {
                Scalar score = posteriors.col(component_i).transpose() * posteriors.col(component_j);
                // assert(posteriors.col(component_i).array().isFinite().all());
                assert(std::isfinite(score));
                if(score < threshold) {
                    continue;
                }
                scores.push_back({component_i, component_j, score});
            }
        }
        if(scores.size() == 0) {
            std::cerr << "No merging candidates.\n";
        }
        std::sort(scores.begin(), scores.end(), 
            [](const MergeScore& score_1, const MergeScore& score_2) {
                return score_1.score < score_2.score;
            }
        );

        return scores;
    }

    Eigen::Matrix<Scalar, 1, Eigen::Dynamic> sumPosteriorOverSamples(
        MM& distribution,
        const Samples<t_dims, Scalar>& samples
    ) {
        int nSamples = samples.size();
        int nComponents = distribution.nComponents();

        Eigen::Matrix<Scalar, t_components, 1> posterior;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> posteriorSum(nComponents, 1);
        posteriorSum.setZero();
        for(int sample_i = 0; sample_i < nSamples; ++sample_i) {
            if(samples.weights(sample_i) == 0.f) {
                continue;
            }
            Vectord sample = samples.samples.col(sample_i);
            Scalar heuristicPosterior;
            distribution.posterior(
                sample,
                samples.isDiffuse(sample_i),
                samples.heuristicPdfs(sample_i),
                posterior,
                heuristicPosterior
            );
            posteriorSum += samples.weights(sample_i) * posterior.topRows(nComponents);
        }
        return posteriorSum;
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> kernelLocalKLDivergence(
        MM& distribution,
        const Samples<t_dims, Scalar>& samples
    ) {
        using KDTree = typename kdt::KDTree<Scalar, kdt::EuclideanDistance<Scalar>>;
        using DistMatrix = typename KDTree::Matrix;
        using IdxMatrix = typename KDTree::MatrixI;

        int nSamples = samples.size();
        int nComponents = distribution.nComponents();

        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> posteriors(nSamples, nComponents);
        Eigen::Matrix<Scalar, t_components, 1> posterior;
        for(int sample_i = 0; sample_i < nSamples; ++sample_i) {
            Vectord sample = samples.samples.col(sample_i);
            Scalar heuristicPosterior;
            distribution.posterior(
                sample,
                samples.isDiffuse(sample_i),
                samples.heuristicPdfs(sample_i),
                posterior,
                heuristicPosterior
            );

            assert(posterior.topRows(nComponents).array().isFinite().all());
            posteriors.row(sample_i) =
                samples.weights(sample_i) * posterior.topRows(nComponents).transpose();
        }

        KDTree kdtree(samples.samples, true);
        kdtree.setTakeRoot(true);
        kdtree.build();

        DistMatrix dists;
        IdxMatrix idx;
        size_t knn = 10;
        kdtree.query(samples.samples, knn, idx, dists);

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> maxDistances =
            dists.transpose().rowwise().maxCoeff();
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> volume = 
            volume_norm<t_dims>::value * maxDistances.array().pow(t_dims);
        assert(volume.rows() == nSamples);
        
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> localDensity(nSamples, nComponents);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> localDensityNormalization(nComponents, 1);

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> normalizedWeights =
            samples.weights.topRows(nSamples) /
            samples.weights.topRows(samples.size()).mean();

        localDensityNormalization.setZero();
        localDensity.setZero();
        for(int sample_i = 0; sample_i < nSamples; ++sample_i) {
            for(int component_i = 0; component_i < nComponents; ++component_i) {
                for(int nn_i = 0; nn_i < knn; ++nn_i) {
                    int sample_j = idx(nn_i, sample_i);
                    localDensity(sample_i, component_i) +=
                        normalizedWeights(sample_j) * posteriors(sample_j, component_i);
                }
                localDensityNormalization(component_i) +=
                    normalizedWeights(sample_i) * posteriors(sample_i, component_i);
            }
        }
        
        localDensity.array().rowwise() /= localDensityNormalization.transpose().array();
        localDensity.array().colwise() /= volume.array();

        auto& components = distribution.getComponentDistributions();
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> divergence(nComponents, 1);
        divergence.setConstant(-std::numeric_limits<Scalar>::infinity());
        for(int sample_i = 0; sample_i < nSamples; ++sample_i) {
            if(samples.weights(sample_i) == 0.f) {
                continue;
            }
            Vectord sample = samples.samples.col(sample_i);
            for(int component_i = 0; component_i < nComponents; ++component_i) {
                Scalar density = localDensity(sample_i, component_i);
                Scalar pdf = 
                    distribution.heuristicWeight() * samples.heuristicPdfs(sample_i) + 
                    (1 - distribution.heuristicWeight()) * components[component_i].pdf(sample);
                if(!std::isfinite(density) ||
                    density == 0.f ||
                    pdf == 0.f ||
                    localDensityNormalization(component_i) == 0.f
                ) {
                    continue;
                }
                
                Scalar kl = density * (std::log(density) - std::log(pdf));

                if(!std::isfinite(divergence(component_i))) {
                    divergence(component_i) = 0.f;
                }
                divergence(component_i) += kl;
            }
        }
        return divergence;
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> denoisedLocalKLDivergence(
        MM& distribution,
        const Samples<t_dims, Scalar>& samples
    ) {
        using KDTree = typename kdt::KDTree<Scalar, kdt::EuclideanDistance<Scalar>>;
        using DistMatrix = typename KDTree::Matrix;
        using IdxMatrix = typename KDTree::MatrixI;

        int nSamples = samples.size();
        int nComponents = distribution.nComponents();

        KDTree kdtree(samples.samples, true);
        kdtree.setTakeRoot(true);
        kdtree.build();

        DistMatrix dists;
        IdxMatrix idx;
        size_t knn = 10;
        kdtree.query(samples.samples, knn, idx, dists);

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> maxDistances =
            dists.transpose().rowwise().maxCoeff();
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> volume = 
            volume_norm<t_dims>::value * maxDistances.array().pow(t_dims);
        assert(volume.rows() == nSamples);

        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> posteriors(nSamples, nComponents);
        Eigen::Matrix<Scalar, t_components, 1> posterior;
        for(int sample_i = 0; sample_i < nSamples; ++sample_i) {
            Vectord sample = samples.samples.col(sample_i);
            Scalar heuristicPosterior;
            distribution.posterior(
                sample,
                samples.isDiffuse(sample_i),
                samples.heuristicPdfs(sample_i),
                posterior,
                heuristicPosterior
            );

            assert(posterior.topRows(nComponents).array().isFinite().all());
            posteriors.row(sample_i) =
                samples.weights(sample_i) * posterior.topRows(nComponents).transpose();
        }
    
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> localDensity(nSamples, nComponents);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> localDensityNormalization(nComponents, 1);

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> normalizedWeights =
            samples.weights.topRows(nSamples) /
            samples.weights.topRows(samples.size()).mean();

        localDensityNormalization.setZero();
        localDensity.setZero();
        for(int sample_i = 0; sample_i < nSamples; ++sample_i) {
            for(int component_i = 0; component_i < nComponents; ++component_i) {
                for(int nn_i = 0; nn_i < knn; ++nn_i) {
                    int sample_j = idx(nn_i, sample_i);
                    localDensity(sample_i, component_i) +=
                        normalizedWeights(sample_j) * posteriors(sample_j, component_i);
                }
                localDensityNormalization(component_i) +=
                    normalizedWeights(sample_i) * posteriors(sample_i, component_i);
            }
        }
        
        localDensity.array().rowwise() /= localDensityNormalization.transpose().array();
        localDensity.array().colwise() /= volume.array();

        auto& components = distribution.getComponentDistributions();
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> divergence(nComponents, 1);
        divergence.setConstant(-std::numeric_limits<Scalar>::infinity());
        for(int sample_i = 0; sample_i < nSamples; ++sample_i) {
            if(samples.weights(sample_i) == 0.f) {
                continue;
            }
            Vectord sample = samples.samples.col(sample_i);
            for(int component_i = 0; component_i < nComponents; ++component_i) {
                Scalar density = localDensity(sample_i, component_i);
                Scalar pdf = 
                    distribution.heuristicWeight() * samples.heuristicPdfs(sample_i) + 
                    (1 - distribution.heuristicWeight()) * components[component_i].pdf(sample);
                if(!std::isfinite(density) ||
                    density == 0.f ||
                    pdf == 0.f ||
                    localDensityNormalization(component_i) == 0.f
                ) {
                    continue;
                }
                
                Scalar kl = density * (std::log(density) - std::log(pdf));

                if(!std::isfinite(divergence(component_i))) {
                    divergence(component_i) = 0.f;
                }
                divergence(component_i) += kl;
            }
        }
        return divergence;
    }


    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> partialLogLikelihood(
        MM& distribution,
        const Samples<t_dims, Scalar>& samples
    ) {
        std::cerr << "Computing partial log-likelihood for each mixture.\n";
        int nSamples = samples.size();
        int nComponents = distribution.nComponents();
        std::cerr << "nSamples=" << nSamples << '\n';
        std::cerr << "nComponents=" << nComponents << '\n';

        auto& components = distribution.getComponentDistributions();
        auto& weights = distribution.getMixtureWeights();

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> pll(nComponents, 1);
        pll.setConstant(-std::numeric_limits<Scalar>::infinity());
        Eigen::Matrix<Scalar, t_components, 1> posterior;
        Scalar heuristicPosterior;

        for(int sample_i = 0; sample_i < nSamples; ++sample_i) {
            if(samples.weights(sample_i) == 0.f) {
                continue;
            }

            Vectord sample = samples.samples.col(sample_i);
            distribution.posterior(
                sample,
                samples.isDiffuse(sample_i),
                samples.heuristicPdfs(sample_i),
                posterior,
                heuristicPosterior
            );
            for(int component_i = 0; component_i < nComponents; ++component_i) {
                Scalar componentPdf = components[component_i].pdf(sample);
                if(weights[component_i] == 0.f || componentPdf == 0.f) {
                    continue;
                }

                if(!std::isfinite(pll(component_i))) {
                    pll(component_i) = 0.f;
                }

                pll(component_i) -= samples.weights(sample_i) * posterior(component_i) * (
                    std::log(weights[component_i]) + std::log(componentPdf)
                );
            }
        }
        return pll;
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> localError(
        MM& distribution,
        const Samples<t_dims, Scalar>& samples,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& error
    ) {
        int nSamples = samples.size();
        int nComponents = distribution.nComponents();
        std::cerr << "nSamples=" << nSamples << '\n';
        std::cerr << "nComponents=" << nComponents << '\n';

        auto& components = distribution.getComponentDistributions();
        auto& weights = distribution.getMixtureWeights();

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> posteriorSum =
            sumPosteriorOverSamples(distribution, samples);

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> mixtureErrors(nComponents, 1);
        mixtureErrors.setConstant(-std::numeric_limits<Scalar>::infinity());
        
        Eigen::Matrix<Scalar, t_components, 1> posterior;
        Scalar heuristicPosterior;
        for(int sample_i = 0; sample_i < nSamples; ++sample_i) {
            if(samples.weights(sample_i) == 0.f || error(sample_i) == 0.f) {
                continue;
            }
            Vectord sample = samples.samples.col(sample_i);
            distribution.posterior(
                sample,
                samples.isDiffuse(sample_i),
                samples.heuristicPdfs(sample_i),
                posterior,
                heuristicPosterior
            );
            for(int component_i = 0; component_i < nComponents; ++component_i) {
                if(posteriorSum(component_i) == 0.f || posterior(component_i) == 0.f) {
                    continue;
                }
                
                Scalar localError = 
                    error(sample_i) * posterior(component_i) / posteriorSum(component_i);
                mixtureErrors(component_i) += localError;
            }
        }
        return mixtureErrors.array() / nSamples;
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> localKLDivergence(
        MM& distribution,
        const Samples<t_dims, Scalar>& samples
    ) {
        std::cerr << "Computing local KL Divergence for each mixture.\n";
        int nSamples = samples.size();
        int nComponents = distribution.nComponents();
        std::cerr << "nSamples=" << nSamples << '\n';
        std::cerr << "nComponents=" << nComponents << '\n';

        auto& components = distribution.getComponentDistributions();
        auto& weights = distribution.getMixtureWeights();

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> posteriorSum =
            sumPosteriorOverSamples(distribution, samples);

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> divergence(nComponents, 1);
        divergence.setConstant(-std::numeric_limits<Scalar>::infinity());
        Eigen::Matrix<Scalar, t_components, 1> posterior;
        Scalar heuristicPosterior;

        for(int sample_i = 0; sample_i < nSamples; ++sample_i) {
            if(samples.weights(sample_i) == 0.f) {
                continue;
            }
            Vectord sample = samples.samples.col(sample_i);
            distribution.posterior(
                sample,
                samples.isDiffuse(sample_i),
                samples.heuristicPdfs(sample_i),
                posterior,
                heuristicPosterior
            );
            for(int component_i = 0; component_i < nComponents; ++component_i) {
                if(posteriorSum(component_i) == 0.f || posterior(component_i) == 0.f) {
                    // std::cerr << "posteriorSum(" << component_i << ")=" <<
                    //     posteriorSum(component_i) << std::endl;
                    // std::cerr << "posterior(" << component_i << ")=" <<
                    //     posterior(component_i) << std::endl; 
                    continue;
                }
                
                Scalar localDensity = 
                    samples.weights(sample_i) *
                    posterior(component_i) / 
                    posteriorSum(component_i);
                Scalar pdf = 
                    distribution.heuristicWeight() * samples.heuristicPdfs(sample_i) + 
                    (1 - distribution.heuristicWeight()) * components[component_i].pdf(sample);
                Scalar kl = localDensity * (
                    std::log(localDensity) - std::log(pdf)
                );
                if(!std::isfinite(divergence(component_i))) {
                    divergence(component_i) = 0.f;
                }
                divergence(component_i) += kl;
            }
        }
        return divergence.array() / nSamples;
    }

    void optimize(
        MM& distribution,
        Samples<t_dims, Scalar>& samples,
        int iterations,
        int threadId,
        int begin,
        int end
    ) {
        BaseEM::optimize(distribution, samples, iterations, threadId, begin, end);
        if(!splittingOn) {
            return;
        }

        this->barrier.wait();

        if(threadId == 0) {
            // std::vector<MergeScore> merge_criteria = posteriorDotProduct(distribution, samples);
            // while(!merge_criteria.empty()) {
            //     MergeScore merge_score = merge_criteria.back();
            //     mergeStats(distribution, merge_score.component_i, merge_score.component_j);

            //     // TODO: recalculate only when empty
            //     merge_criteria = posteriorDotProduct(distribution, samples);
            // }

            
            const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& error = sarsaError(samples, distribution);
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> splitMetric =
                localError(distribution, samples, error);
            // for(int component_i = 0; component_i < distribution.nComponents(); ++component_i) {
            //     splitMetric(component_i) *=
            //         distribution.getComponentDistributions()[component_i].cov().determinant();
            // }
            std::vector<int> split_idcs(splitMetric.rows());
            std::iota(split_idcs.begin(), split_idcs.end(), 0);
            std::sort(split_idcs.begin(), split_idcs.end(), [&splitMetric](int i, int j) {
                return splitMetric(i) > splitMetric(j);
            });

            int maxSplits = std::min(nSplits, distribution.nComponents());
            for(int index_i = 0; index_i < maxSplits; ++index_i) {
                if(distribution.nComponents() == t_components - 1) {
                    break;
                }
                int split_i = split_idcs[index_i];
                std::cerr << "Splitting " << split_i;
                Scalar metric = splitMetric(split_i);
                std::cerr << " with score " << metric << ".";
                splitStatsSVD(distribution, split_i);
                std::cerr << "Done splitting component.";
            }
        }
        this->barrier.wait();

        BaseEM::optimize(distribution, samples, iterations, threadId, begin, end);
    }

    void setSplittingOn(bool on) {
        splittingOn = on;
    }

private:
    constexpr static Scalar weightSplitWeight = 0.5;
    constexpr static Scalar u = 0.5;
    constexpr static Scalar beta = 0.5;
    constexpr static int l = 0;
    constexpr static Scalar mean_j_const = std::sqrt((1 - weightSplitWeight) / weightSplitWeight) * u;
    constexpr static Scalar mean_k_const = std::sqrt(weightSplitWeight / (1 - weightSplitWeight)) * u;

    bool splittingOn = true;
    int nSplits;
};

}

#endif /* __MIXTURE_MODEL_OPT_SPLIT_AND_MERGE_H */
