#ifndef __MIXTURE_MODEL_INIT_H
#define __MIXTURE_MODEL_INIT_H

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
#include <boost/thread.hpp>
#include <boost/thread/barrier.hpp>

#include "distribution.h"
#include "mixture_model.h"
#include "samples.h"


namespace jmm {
    template<
        int t_dims,
        int t_components,
        int t_conditionalDims,
        typename Scalar,
        template<int, int, typename> class Component_t,
        template<int, int, typename> class Marginal_t
    > void uniformRandomInit(
        MixtureModel<t_dims, t_components, t_conditionalDims, Scalar, Component_t, Marginal_t>& distribution,
        const std::function<Scalar()>& rng,
        Scalar covInitDiag
    ) {
        using MM = MixtureModel<t_dims, t_components, t_conditionalDims, Scalar, Component_t, Marginal_t>;
        using Component = typename MM::Component;
        using Vectord = typename MM::Vectord;
        using Matrixd = typename MM::Matrixd;

        auto& components = distribution.components();
        auto& weights = distribution.weights();

        Matrixd cov = Matrixd::Identity() * covInitDiag;
        for(int component_i = 0; component_i < distribution.nComponents(); ++component_i) {
            Vectord mean;
            for(int dim_i = 0; dim_i < t_dims; ++dim_i) {
                mean(dim_i) = rng();
            }
            components[component_i] = Component(mean, cov);
            weights[component_i] = 1.f / distribution.nComponents();
        }
        if(t_components > distribution.nComponents()) {
            std::fill(std::begin(weights) + distribution.nComponents(), std::end(weights), 0);
        }
        bool success = distribution.configure();
        assert(success);
    }

    template<typename Scalar>
    Eigen::Matrix<Scalar, 3, 1> canonicalToCosineHemisphere(
        const Eigen::Matrix<Scalar, 2, 1>& canonical
    ) {
        Scalar heightSqr = canonical(0);
        Scalar radius = std::sqrt(1 - heightSqr);
        Scalar theta = 2 * M_PI * canonical(1);
        return {radius * std::cos(theta), radius * std::sin(theta), std::sqrt(heightSqr)};
    }


    enum class SphereSide {
        Top,
        Bottom,
        Both
    };

    constexpr static double NORMAL_DISTANCE_TRHESHOLD = 0.2;

    template<
        int t_dims,
        int t_components,
        int t_conditionalDims,
        typename Scalar,
        template<int, int, typename> class Component_t,
        template<int, int, typename> class Marginal_t
    > void uniformHemisphereInit(
        MixtureModel<
            t_dims,
            t_components,
            t_conditionalDims,
            Scalar,
            Component_t,
            Marginal_t
        >& distribution,
        jmm::aligned_vector<
            typename Component_t<t_dims, t_conditionalDims, Scalar>::Matrixd
        >& bPriors,
        jmm::aligned_vector<Eigen::Matrix<Scalar, 3, 3>>& bDepthPriors,
        const std::function<Scalar()>& rng,
        int nComponents,
        Scalar depthPrior,
        Scalar minAllowedSpatialDistance,
        const Samples<t_dims, Scalar>& samples,
        std::vector<SphereSide>& sides,
        bool kMeansPlusPlus
    ) {
        std::cerr << "Initializing.\n";
        if(t_dims - t_conditionalDims > 3) {
            std::cerr << "Need to fix initialization first.\n";
            assert(t_dims - t_conditionalDims == 3);
        }
        using MM = MixtureModel<t_dims, t_components, t_conditionalDims, Scalar, Component_t, Marginal_t>;
        using Component = typename MM::Component;
        using Vectord = typename MM::Vectord;
        using Matrixd = typename MM::Matrixd;

        auto& components = distribution.components();
        auto& weights = distribution.weights();

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> metric(samples.size());
        metric = samples.weights.topRows(samples.size()).array().max(1e-3).min(3);
        
        // assert((samples.stateDensities.topRows(samples.size()).array() > 0).all());
        // metric.array() /= samples.stateDensities.topRows(samples.size()).array();

        Eigen::Matrix<Scalar, 3, Eigen::Dynamic> positions, normals;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> curvatures;

        positions.resize(Eigen::NoChange, nComponents);
        normals.resize(Eigen::NoChange, nComponents);
        curvatures.resize(nComponents, Eigen::NoChange);

        if(kMeansPlusPlus) {
            kMeansPPInit(
                samples,
                metric,
                sides,
                rng,
                nComponents,
                positions,
                normals,
                curvatures,
                true
            );
        } else {
            positions = samples.samples.topRows(3).leftCols(nComponents);
            normals = samples.normals.leftCols(nComponents);
            curvatures = samples.curvatures.leftCols(nComponents);
        }
        
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> minSpatialDistances(nComponents);
        minSpatialDistances.setConstant(std::numeric_limits<Scalar>::infinity());
        Scalar minAllowedSpatialDistanceSqr = minAllowedSpatialDistance * minAllowedSpatialDistance;
        for(int position_i = 0; position_i < nComponents; ++position_i) {
            for(int position_j = 0; position_j < nComponents; ++position_j) {
                if(position_i == position_j) {
                    continue;
                }
                Scalar normalDot = samples.normals.col(position_i).transpose() * normals.col(position_j);
                // Scalar normalDistance = std::acos(
                //     std::min(Scalar(1), std::max(Scalar(-1), normalDot))                    
                // ) / M_PI;
                Scalar distSqr = (positions.col(position_i) - positions.col(position_j)).squaredNorm();
                if(
                    // normalDistance < NORMAL_DISTANCE_TRHESHOLD &&
                    distSqr < minSpatialDistances(position_i)
                ) {
                    minSpatialDistances(position_i) = distSqr;
                }
            }
            if(minSpatialDistances(position_i) == std::numeric_limits<Scalar>::infinity()) {
                minSpatialDistances(position_i) = minAllowedSpatialDistanceSqr;
            } else {
                minSpatialDistances(position_i) = 
                    std::max(minAllowedSpatialDistanceSqr, minSpatialDistances(position_i));
            }
        }

        static const boost::math::chi_squared chiSquared(t_dims);
        static const Scalar containedMass = 0.90;
        static const Scalar maxRadiusSqr = (Scalar) boost::math::quantile(chiSquared, containedMass);
        static const Scalar maxRadius = std::sqrt(maxRadiusSqr);
        std::cerr << "Gaussian " << containedMass << " mass radius: " << maxRadius << '\n';
        Scalar depthPriorSqr = depthPrior * depthPrior;
        std::cerr << "3D depth initialized to " << depthPriorSqr / maxRadiusSqr << '\n';

        int component_i = 0;
        Scalar nThetas = 3;
        Scalar nPhis = 4;
        Scalar directionalInit = 1.f / (nThetas * nPhis);
        // Scalar nThetas = 3;
        // Scalar nPhis = 4;
        distribution.setNComponents(nComponents * nThetas * nPhis);
        for(int position_i = 0; position_i < nComponents; ++position_i) {
            Eigen::Matrix<Scalar, 3, 1> p = positions.col(position_i);
            Eigen::Matrix<Scalar, 3, 1> n = normals.col(position_i);

            Eigen::Matrix<Scalar, 3, 1> s, t;
            coordinateSystem(n, s, t);

            Scalar spatialRadiusSqr = 0.5 * minSpatialDistances(position_i);
            Eigen::Matrix<Scalar, 3, 3> manifoldCov =
                s * s.transpose() * spatialRadiusSqr / maxRadiusSqr +
                t * t.transpose() * spatialRadiusSqr / maxRadiusSqr;
            
            Eigen::Matrix<Scalar, 3, 3> depthCov =
                n * n.transpose() * depthPriorSqr / maxRadiusSqr;

            Eigen::Matrix<Scalar, 2, 2> directionalCov = Eigen::Matrix<Scalar, 2, 2>::Zero();
            if(t_conditionalDims == 2) {
                directionalCov(0, 0) = directionalInit;
                directionalCov(1, 1) = std::sqrt(1.f / 2.f) * directionalInit;
            } else if(t_conditionalDims == 3) {
                directionalCov(0, 0) = 2 * M_PI * directionalInit;
                directionalCov(1, 1) = 2 * M_PI * directionalInit;
            }
            Matrixd cov = Matrixd::Identity();
            cov.topLeftCorner(3, 3) = manifoldCov + depthCov;
            cov.bottomRightCorner(2, 2) = directionalCov;

            Matrixd bPrior = Matrixd::Identity();
            bPrior.topLeftCorner(3, 3) <<
                s * s.transpose() * 1e-4 +
                t * t.transpose() * 1e-4 +
                n * n.transpose() * 1e-4;
            bPrior.bottomRightCorner(2, 2).diagonal().setConstant(1e-5); // directionalCov * 0.5;

            Vectord mean;
            mean.topRows(3) = p;

            bool isTwoSided = false; // sides[position_i] == SphereSide::Both;
            Scalar theta = (isTwoSided) ? -0.5 * M_PI : 0;
            Scalar totalNThetas = (isTwoSided) ? 2 * nThetas : nThetas;
            if(std::abs(curvatures(position_i)) > 1e-4) {
                totalNThetas++;
            }
            for(int theta_i = 0; theta_i < totalNThetas; ++theta_i) {
                Scalar rn = (rng() - 0.5) * 2e-1;
                theta += 0.5 * M_PI / (nThetas + 1) + rn;
                Scalar cosTheta = std::cos(theta);
                Scalar phi = 0;
                for(int phi_i = 0; phi_i < nPhis; ++phi_i) {
                    rn = (rng() - 0.5) * 1e-1;
                    phi += 2 * M_PI / nPhis + rn;
                    
                    const Scalar sinTheta = std::sqrt(1 - cosTheta * cosTheta);
                    Scalar sinPhi, cosPhi;
                    sincos(phi, &sinPhi, &cosPhi);

                    Eigen::Matrix<Scalar, 3, 1> direction;
                    direction << sinTheta * cosPhi, sinTheta * sinPhi, cosTheta; 
                    direction = s * direction(0) + t * direction(1) + n * direction(2);
                    if(t_conditionalDims == 3) {
                        mean.bottomRows(t_conditionalDims) = direction;
                    } else if(t_conditionalDims == 2) {
                        mean.bottomRows(t_conditionalDims) = dirToCanonical(direction);
                    }

                    components[component_i] = Component(mean, cov);
                    weights[component_i] = 1.f / distribution.nComponents();

                    bPriors[component_i] = bPrior;
                    bDepthPriors[component_i].setZero();
                    // std::cerr << "Curvature: " << curvatures(position_i) << "\n";
                    // if(std::abs(curvatures(position_i)) < 1e-4) {
                    //     bDepthPriors[component_i] += 
                    //         n * n.transpose() * 5e-8;
                    // } else {
                    //     // std::cerr << "Large curvature init\n";
                    //     bDepthPriors[component_i] +=
                    //         n * n.transpose() * 1e-5;
                    // }

                    ++component_i;
                }
            }
        }
        distribution.setNComponents(component_i);
        if(t_components > distribution.nComponents()) {
            std::fill(std::begin(weights) + distribution.nComponents(), std::end(weights), 0);
        }
        bool success = distribution.configure();
        assert(success);
    }

    template<int t_dims, typename Scalar>
    void kMeansPPInit(
        const Samples<t_dims, Scalar>& samples,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& metric,
        const std::vector<SphereSide>& sides,
        const std::function<Scalar()>& rng,
        int nComponents,
        Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& positions,
        Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& normals,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& curvatures,
        bool useInitWeightsForMeans
    ) {
        if(!useInitWeightsForMeans) {
            metric.topRows(samples.size()).setOnes();
        }
        
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> cdf(samples.size());
        positions.resize(Eigen::NoChange, nComponents);
        normals.resize(Eigen::NoChange, nComponents);
        curvatures.resize(nComponents, Eigen::NoChange);

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> minDistances, minSpatialDistances, minSpatialNormalDistances;
        Eigen::Matrix<int, Eigen::Dynamic, 1> minDistanceIdcs;
        minDistances.resize(samples.size());
        minDistances.setConstant(std::numeric_limits<Scalar>::infinity());
        minSpatialDistances.resize(samples.size());
        minSpatialDistances.setConstant(std::numeric_limits<Scalar>::infinity());
        minSpatialNormalDistances.resize(samples.size());
        minSpatialNormalDistances.setConstant(std::numeric_limits<Scalar>::infinity());
        minDistanceIdcs.resize(samples.size());

        const Scalar minAllowedDistance = 2e-2;
        for(int position_i = 0; position_i < nComponents; ++position_i) {
            // Make sampling CDF
            if(position_i == 0) {
				assert(cdf.rows() == metric.rows());
                cdf = metric;
            } else {
                cdf = minDistances.array().pow(5);
                int remainingPositions = 0;
                for(int sample_i = 0; sample_i < samples.size(); ++sample_i) {
                    if(
                        minSpatialNormalDistances(sample_i) < NORMAL_DISTANCE_TRHESHOLD &&
                        minSpatialDistances(sample_i) < minAllowedDistance * minAllowedDistance
                    ) {
                        cdf(sample_i) = 0;
                    } else {
                        ++remainingPositions;
                    }
                }
                jmm::normalizePdf(cdf);
                cdf.array() *= metric.array();
                std::cerr << "Reminaining positions: " << remainingPositions << "\n";
            }
            bool success = jmm::createCdfEigen(cdf, cdf, true);
			Scalar rngSample = rng();
            if(!success) {
                std::cerr <<
					"Could not create discrete CDF for initialization, "
					"using uniform CDF.\n";
				cdf.setOnes();
				cdf /= (Scalar) cdf.rows();
            	jmm::createCdfEigen(cdf, cdf, true);
            }
            int sampled = jmm::sampleDiscreteCdf(cdf, rngSample);
            positions.col(position_i) = samples.samples.col(sampled).topRows(3); 
            normals.col(position_i) = samples.normals.col(sampled);
            // if(sides[sampled] == SphereSide::Bottom) {
            //     normals.col(position_i) = -normals.col(position_i);
            // }
            curvatures(position_i) = samples.curvatures(sampled);

            // Compute min distance for all samples to any component mean so far
            #pragma omp parallel for
            for(int sample_i = 0; sample_i < samples.size(); ++sample_i){
                Scalar normalDot = samples.normals.col(sample_i).transpose() * normals.col(position_i);
                Scalar normalDistance = std::acos(
                    std::min(Scalar(1), std::max(Scalar(-1), normalDot))                    
                ) / M_PI;
                assert(std::isfinite(normalDistance));
                Scalar spatialDistanceSqr = (
                    samples.samples.col(sample_i).topRows(3) - positions.col(position_i)
                ).squaredNorm();
                Scalar distSqr = spatialDistanceSqr + normalDistance * normalDistance;
                // Scalar distSqr = (
                //     samples.samples.col(sample_i).topRows(3) - positions.col(position_i)
                // ).squaredNorm();
                if(distSqr < minDistances(sample_i)){
                    minDistances(sample_i) = distSqr;
                    minDistanceIdcs(sample_i) = position_i;
                }
                if(
                    normalDistance < NORMAL_DISTANCE_TRHESHOLD &&
                    spatialDistanceSqr < minSpatialDistances(sample_i)
                ){
                    minSpatialNormalDistances(sample_i) = normalDistance;
                    minSpatialDistances(sample_i) = spatialDistanceSqr;
                }
            } 
        }
    }

    template<
        int t_dims,
        int t_components,
        int t_conditionalDims,
        typename Scalar,
        template<int, int, typename> class Component,
        template<int, int, typename> class Marginal
    > class KmeansPlusPlusInit {
    private:
        using MM = MixtureModel<t_dims, t_components, t_conditionalDims, Scalar, Component, Marginal>;
        using Vectord = typename MM::Vectord;
        using Matrixd = typename MM::Matrixd;

        Scalar m_covInitDiag;
        bool m_useInitWeightsForMixture;
        bool m_useInitWeightsForMeans;

        void initMeans(
            const Samples<t_dims, Scalar>& samples,
            const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& metric,
            const std::function<Scalar()>& rng,
            int nComponents,
            std::array<Eigen::Matrix<Scalar, t_dims, 1>, t_components> &means,
            std::vector<std::pair<Scalar, int>> &distances
        ) {
            if(!m_useInitWeightsForMeans) {
                metric.topRows(samples.size()).setOnes();
            }

            std::vector<Scalar> cdf(samples.size(), 0);
            for(int component_i = 0; component_i < nComponents; ++component_i) {
                // Make sampling CDF
                for(int sample_i = 0; sample_i < samples.size(); ++sample_i){
                    if(component_i == 0) {
                        cdf[sample_i] = metric(sample_i);
                    } else {
                        cdf[sample_i] = distances[sample_i].first * metric(sample_i);
                    }
                }
                bool success = jmm::createCdf(std::begin(cdf), std::end(cdf), std::begin(cdf));
                if(!success) {
                    std::cerr << "Could not create discrete CDF for initialization.\n";
                }
                int sampled = jmm::sampleDiscreteCdf(std::begin(cdf), std::end(cdf), rng());
                means[component_i] = samples.samples.col(sampled); 

                // Compute min distance for all samples to any component mean so far
                for(int sample_i = 0; sample_i < samples.size(); ++sample_i){
                    // Scalar greatCircleDistance = std::acos(
                    //     dot(canonicalToDir(sorted_data[j].point.bottomRows(t_conditionalDims)),
                    //     canonicalToDir(means[component_i].bottomRows(t_conditionalDims)))
                    // );
                    // Scalar euclidianDistance = (
                    //     sorted_data[j].point.topRows(t_conditionDims) -
                    //     means[component_i].topRows(t_conditionDims)
                    // ).squaredNorm();
                    // Scalar d2 = euclidianDistance + greatCircleDistance * greatCircleDistance;
                    Scalar distSqr = (samples.samples.col(sample_i) - means[component_i]).squaredNorm();
                    if(distSqr < distances[sample_i].first){
                        distances[sample_i].first = distSqr;
                        distances[sample_i].second = component_i;
                    }
                } 
            }
        }

        // Compute covariance matrix and mixture weight for every component
        void initCovsAndWeights(
            const Samples<t_dims, Scalar>& samples,
            const std::array<Vectord, t_components> &means,
            int nComponents,
            std::array<Matrixd, t_components> &covs, 
            std::array<Scalar, t_components> &weights
        ) {
            Matrixd covInit = Matrixd::Identity() * m_covInitDiag;
            
            for(int component_i = 0; component_i < nComponents; ++component_i){
                covs[component_i] = covInit;
                weights[component_i] = 1;
            }

            bool pdfSuccess = jmm::normalizePdf(
                std::begin(weights), std::begin(weights) + nComponents
            );
            if(!pdfSuccess) {
                std::cerr << "All zero entries in mixture weights!\n";
            }
        }
        
    public:

        KmeansPlusPlusInit(
            Scalar covInitDiag,
            bool useInitWeightsForMixture,
            bool useInitWeightsForMeans
        ) : m_covInitDiag(covInitDiag),
            m_useInitWeightsForMixture(useInitWeightsForMixture),
            m_useInitWeightsForMeans(useInitWeightsForMeans) { }

        void init(
            MM& distribution,
            const Samples<t_dims, Scalar>& samples,
            const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& metric,
            const std::function<Scalar()>& rng
        ) {
            int nComponents = distribution.nComponents();
            auto& components = distribution.components();
            auto& weights = distribution.weights();

            std::array<Scalar, t_components> weightsNew;
            std::array<Vectord, t_components> meansNew;
            std::array<Matrixd, t_components> covsNew;

            weightsNew.fill(0);
            meansNew.fill(Vectord::Zero());
            covsNew.fill(Matrixd::Constant(0));
                        
            std::vector<std::pair<Scalar, int>> distances(
                samples.size(),
                std::make_pair<Scalar, int>(std::numeric_limits<Scalar>::infinity(), -1)
            );

            // k-means++ initialization. 
            std::cerr << "Performing k-means++.\n";
            initMeans(samples, metric, rng, nComponents, meansNew, distances);

            // Run k-means on data
            // std::cerr << "Running k-means.\n";
            // kmeans(means, distances, sorted_data);

            // compute covariance and mixture weight
            std::cerr << "Initializing covariances and weights.\n";
            initCovsAndWeights(samples, meansNew, nComponents, covsNew, weightsNew);
                
            // create inital GMM based on initial means, covariances and weights
            for(int component_i = 0; component_i < nComponents; ++component_i) {
                components[component_i] =
                    jmm::MultivariateNormal<t_dims, t_conditionalDims, Scalar>(
                        meansNew[component_i], covsNew[component_i]
                    );
                if(m_useInitWeightsForMixture){
                    weights[component_i] = weightsNew[component_i];
                } else{
                    weights[component_i] = 1.f / ((Scalar) nComponents); 
                }
            }
            jmm::normalizePdf(std::begin(weights), std::begin(weights) + nComponents);
            distribution.setNComponents(nComponents);
        }
    };
}

#endif /* __MIXTURE_MODEL_INIT_H */
