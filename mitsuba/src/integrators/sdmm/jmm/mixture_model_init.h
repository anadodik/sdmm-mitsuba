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

#include <fmt/format.h>

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

    constexpr static double NORMAL_DISTANCE_TRHESHOLD = 0.2 * 0.2;
    constexpr static double SPATIAL_DISTANCE_THRESHOLD = 2e-2 * 2e-2;

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
        bool kMeansPlusPlus
    ) {
        // std::cerr << "Initializing.\n";
        if(t_dims - t_conditionalDims > 3) {
            // std::cerr << "Need to fix initialization first.\n";
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

        positions.resize(Eigen::NoChange, nComponents);
        normals.resize(Eigen::NoChange, nComponents);

        if(kMeansPlusPlus) {
            kMeansPPInit(
                samples,
                metric,
                rng,
                nComponents,
                positions,
                normals
            );
        } else {
            positions = samples.samples.topRows(3).leftCols(nComponents);
            normals = samples.normals.leftCols(nComponents);
        }
        

        static const boost::math::chi_squared chiSquared(t_dims);
        static const Scalar containedMass = 0.90;
        static const Scalar maxRadiusSqr = (Scalar) boost::math::quantile(chiSquared, containedMass);

        const Scalar widthVarSqr = 0.5 * minAllowedSpatialDistance * minAllowedSpatialDistance / maxRadiusSqr;
        const Scalar depthVarSqr = depthPrior * depthPrior / maxRadiusSqr;

        // std::cerr << "Gaussian " << containedMass << " mass radius: " << maxRadius << '\n';
        // std::cerr << "3D depth initialized to " << depthPriorSqr / maxRadiusSqr << '\n';

        int component_i = 0;
        Scalar nThetas = 2;
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

            Eigen::Matrix<Scalar, 3, 3> manifoldCov =
                s * s.transpose() * widthVarSqr +
                t * t.transpose() * widthVarSqr;
            
            Eigen::Matrix<Scalar, 3, 3> depthCov =
                n * n.transpose() * depthVarSqr;

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
            // std::cout << fmt::format("OG: n={}, s={}, t={}\ncov=\n{}\n", n.transpose(), s.transpose(), t.transpose(), cov);

            Matrixd bPrior = Matrixd::Identity();
            bPrior.topLeftCorner(3, 3) <<
                s * s.transpose() * 1e-4 +
                t * t.transpose() * 1e-4 +
                n * n.transpose() * 1e-4;
            bPrior.bottomRightCorner(2, 2).diagonal().setConstant(1e-5); // directionalCov * 0.5;

            Vectord mean;
            mean.topRows(3) = p;

            Scalar theta = 0;
            for(int theta_i = 0; theta_i < nThetas; ++theta_i) {
                Scalar rn = (rng() - 0.5) * 2e-1;
                theta += 0.5 * M_PI / (nThetas + 1) + rn;
                const Scalar cosTheta = std::cos(theta);
                const Scalar sinTheta = std::sqrt(1 - cosTheta * cosTheta);

                Scalar phi = 0;
                for(int phi_i = 0; phi_i < nPhis; ++phi_i) {
                    rn = (rng() - 0.5) * 1e-1;
                    phi += 2 * M_PI / nPhis + rn;
                    
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
                    bDepthPriors[component_i] += 
                        n * n.transpose() * 1e-6;
                    ++component_i;
                }
            }
        }
        // distribution.setNComponents(component_i);
        if(t_components > distribution.nComponents()) {
            std::fill(std::begin(weights) + distribution.nComponents(), std::end(weights), 0);
        }
        bool success = distribution.configure();
        assert(success);
        // std::cerr << "Finished initializing.\n";
        // std::cerr << "Initialized " << distribution.nComponents() << " components.\n";
    }

    template<int t_dims, typename Scalar>
    void kMeansPPInit(
        const Samples<t_dims, Scalar>& samples,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& metric,
        const std::function<Scalar()>& rng,
        int nComponents,
        Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& positions,
        Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& normals
    ) {
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> cdf(samples.size());
        positions.resize(Eigen::NoChange, nComponents);
        normals.resize(Eigen::NoChange, nComponents);

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> minDistances, minSpatialDistances, minSpatialNormalDistances;
        minDistances.resize(samples.size());
        minDistances.setConstant(std::numeric_limits<Scalar>::infinity());
        minSpatialDistances.resize(samples.size());
        minSpatialDistances.setConstant(std::numeric_limits<Scalar>::infinity());
        minSpatialNormalDistances.resize(samples.size());
        minSpatialNormalDistances.setConstant(std::numeric_limits<Scalar>::infinity());

        for(int position_i = 0; position_i < nComponents; ++position_i) {
            // Make sampling CDF
            int remainingPositions = cdf.size();
            if(position_i == 0) {
				assert(cdf.rows() == metric.rows());
                cdf = metric;
            } else {
                cdf = minDistances.array().pow(5);
                remainingPositions = 0;
                for(int sample_i = 0; sample_i < samples.size(); ++sample_i) {
                    if(
                        minSpatialNormalDistances(sample_i) < NORMAL_DISTANCE_TRHESHOLD &&
                        minSpatialDistances(sample_i) < SPATIAL_DISTANCE_THRESHOLD
                    ) {
                        cdf(sample_i) = 0;
                    } else {
                        ++remainingPositions;
                    }
                }
                // std::cerr << "Reminaining positions: " << remainingPositions << "/" << cdf.size() << ".\n";
            }
            bool cdfSuccess = remainingPositions > 0;
            if(cdfSuccess) {
                // jmm::normalizePdf(cdf);
                cdf.array() *= metric.array();
                cdfSuccess = cdfSuccess && jmm::createCdfEigen(cdf, cdf, true);
            }
            if(!cdfSuccess) {
                std::cerr <<
					"Could not create discrete CDF for initialization, "
					"using uniform CDF.\n";
				cdf.setOnes();
				cdf /= (Scalar) cdf.rows();
            	jmm::createCdfEigen(cdf, cdf, true);
            }

			Scalar rngSample = rng();
            int sampled = jmm::sampleDiscreteCdf(cdf, rngSample);
            positions.col(position_i) = samples.samples.col(sampled).topRows(3); 
            normals.col(position_i) = samples.normals.col(sampled);

            // Compute min distance for all samples to any component mean so far
            for(int sample_i = 0; sample_i < samples.size(); ++sample_i){
                Scalar normalDot = samples.normals.col(sample_i).transpose() * normals.col(position_i);
                Scalar normalDistance = std::acos(
                    std::min(Scalar(1), std::max(Scalar(-1), normalDot))                    
                ) / M_PI;
                Scalar normalDistanceSqr = normalDistance * normalDistance;
                assert(std::isfinite(normalDistance));
                Scalar spatialDistanceSqr = (
                    samples.samples.col(sample_i).topRows(3) - positions.col(position_i)
                ).squaredNorm();
                Scalar distSqr = spatialDistanceSqr + normalDistanceSqr;
                if(distSqr < minDistances(sample_i)){
                    minDistances(sample_i) = distSqr;
                }
                if(
                    normalDistanceSqr < NORMAL_DISTANCE_TRHESHOLD &&
                    spatialDistanceSqr < minSpatialDistances(sample_i)
                ){
                    minSpatialNormalDistances(sample_i) = normalDistanceSqr;
                    minSpatialDistances(sample_i) = spatialDistanceSqr;
                }
            } 
        }
    }
}

#endif /* __MIXTURE_MODEL_INIT_H */
