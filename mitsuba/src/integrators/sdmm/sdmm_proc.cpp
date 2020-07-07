/*
   This file is part of Mitsuba, a physically based rendering system.

   Copyright (c) 2007-2014 by Wenzel Jakob and others.

   Mitsuba is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License Version 3
   as published by the Free Software Foundation.

   Mitsuba is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program. If not, see <http://www.gnu.org/licenses/>.
   */

#include "sdmm_proc.h"

#include "blob_writer.h"

#include <mitsuba/core/statistics.h>
#include <mitsuba/core/sfcurve.h>
#include <mitsuba/bidir/util.h>
#include <mitsuba/render/rectwu.h>
#include <mitsuba/core/lock.h>
#include <mitsuba/bidir/path.h>
#include <mitsuba/bidir/edge.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/kdtree.h>
#include "../../subsurface/bluenoise.h"

#include <iterator>

#include <boost/thread.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/thread/synchronized_value.hpp>

#define DUMP_DISTRIB 0
// #define INIT_DEBUG
// #define INIT_DETERMINISTIC
// #define VISUALIZE_GAUSSIANS
// #define GAUSSIAN_INIT_TEST //disable for rendering!!!! This is only to visualize intitial gaussian distribution!!!

#define RUN_KNN 1

#define DENOISE 0

#define COMPUTE_KL 0

#define LEARN_COSINE 0

MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLength("SDMM path tracer", "Average path length", EAverage);

static StatsCounter avgIterationTime("SDMM Profiling (s)", "Average duration (s): render iteration", EAverage);
static StatsCounter avgPyramidBuildTime("SDMM Profiling (s)", "Average duration (s): pyramid build time", EAverage);
static StatsCounter avgDensityEstimationTime("SDMM Profiling (s)", "Average duration (s): density estimation", EAverage);
static StatsCounter avgEMIterationTime("SDMM Profiling (s)", "Average duration (s): EM iteration", EAverage);
static StatsCounter avgPosteriorTime("SDMM Profiling (s)", "Average duration (s): posterior calculation", EAverage);

class SDMMRenderer : public WorkProcessor {
    constexpr static int t_dims = SDMMProcess::t_dims;
    constexpr static int t_conditionalDims = SDMMProcess::t_conditionalDims;
    constexpr static int t_conditionDims = SDMMProcess::t_conditionDims;
    constexpr static int t_initComponents = SDMMProcess::t_initComponents;
    constexpr static int t_components = SDMMProcess::t_components;
    constexpr static bool USE_BAYESIAN = SDMMProcess::USE_BAYESIAN;

    using MM = typename SDMMProcess::MM;
    using HashGridType = typename SDMMProcess::HashGridType;
    using GridCell = typename SDMMProcess::GridCell;
	using GridKeyVector = typename HashGridType::Vectord;
    using MMDiffuse = typename SDMMProcess::MMDiffuse;
    using MMCond = typename MM::ConditionalDistribution;
    using StepwiseEMType = typename SDMMProcess::StepwiseEMType;
    
    using MMScalar = typename MM::Scalar;
    using Vectord = typename MM::Vectord;
    using Matrixd = typename MM::Matrixd;

    using ConditionalVectord = typename MMCond::Vectord;
    using ConditionalMatrixd = typename MMCond::Matrixd;

    using RenderingSamplesType = typename SDMMProcess::RenderingSamplesType;

    using FeatureVectord = Point3f;
    using KDNode = SimpleKDNode<FeatureVectord, Float>;
public:
	SDMMRenderer(
        const SDMMConfiguration &config,
        std::shared_ptr<MM> distribution,
        std::shared_ptr<HashGridType> grid,
        std::shared_ptr<MMDiffuse> diffuseDistribution,
        std::shared_ptr<RenderingSamplesType> samples,
        int iteration
    ) :
        m_config(config),
        m_distribution(distribution),
        m_grid(grid),
        m_diffuseDistribution(diffuseDistribution),
        m_samples(samples),
        m_iteration(iteration)
    { }

	SDMMRenderer(Stream *stream, InstanceManager *manager)
	: WorkProcessor(stream, manager), m_config(stream) { }

	virtual ~SDMMRenderer() { }

	void serialize(Stream *stream, InstanceManager *manager) const {
		m_config.serialize(stream);
	}

	ref<WorkUnit> createWorkUnit() const {
		return new RectangularWorkUnit();
	}

	ref<WorkResult> createWorkResult() const {
		return new SDMMWorkResult(m_config, m_rfilter.get(), Vector2i(m_config.blockSize));
	}

    Vectord sampleUniformVector(ref<Sampler> sampler) const {
        Vectord uniformSample;
        for(int dim_i = 0; dim_i < t_dims; ++dim_i) {
            uniformSample(dim_i) = math::clamp(sampler->next1D(), 0.f, 1.f);
        }
        return uniformSample;
    }

	void prepare() {
		Scene *scene = static_cast<Scene *>(getResource("scene"));
		m_scene = new Scene(scene);
		m_sampler = static_cast<Sampler *>(getResource("sampler"));
        m_rng = [samplerCopy = m_sampler]() mutable { return samplerCopy->next1D(); };
		m_sensor = static_cast<Sensor *>(getResource("sensor"));
		m_rfilter = m_sensor->getFilm()->getReconstructionFilter();
		m_scene->removeSensor(scene->getSensor());
		m_scene->addSensor(m_sensor);
		m_scene->setSensor(m_sensor);
		m_scene->setSampler(m_sampler);
		m_scene->wakeup(NULL, m_resources);
        m_scene->initializeBidirectional();
	}

	void process(const WorkUnit *workUnit, WorkResult *workResult, const bool &stop) {
        const RectangularWorkUnit *rect = static_cast<const RectangularWorkUnit *>(workUnit);
        SDMMWorkResult *result = static_cast<SDMMWorkResult *>(workResult);

		fs::path destinationFile = m_scene->getDestinationFile();
        m_cameraMatrix = m_sensor->getWorldTransform()->eval(0).getMatrix();

        PerspectiveCamera* perspectiveCamera = dynamic_cast<PerspectiveCamera*>(&(*m_sensor));
        if (perspectiveCamera) {
            m_fieldOfView = perspectiveCamera->getXFov();
        }
        m_sceneAabb = m_scene->getAABBWithoutCamera();

        if(m_config.maxDepth != 2 && m_config.savedSamplesPerPath <= 1) {
            std::cerr << "WARNING: ONLY SAVING ONE VERTEX PER PATH!\n";
            throw std::runtime_error( "WARNING: ONLY SAVING ONE VERTEX PER PATH!\n");
        }

#ifdef MTS_DEBUG_FP
        enableFPExceptions();
#endif

        result->setSize(rect->getSize());
        result->setOffset(rect->getOffset());
        result->clear();
        m_hilbertCurve.initialize(TVector2<int>(rect->getSize()));
        auto& points = m_hilbertCurve.getPoints();

#ifdef MTS_DEBUG_FP
        disableFPExceptions();
#endif

        Float diffScaleFactor = 1.0f /
            std::sqrt((Float) m_sampler->getSampleCount());

        bool needsApertureSample = m_sensor->needsApertureSample();
        bool needsTimeSample = m_sensor->needsTimeSample();

        MMCond conditional;
        MMCond diffuseConditional;
        MMCond rotatedMaterialConditional;
        MMCond productConditional;

        {
            MMScalar diffuseHeuristicWeight;
            typename MMDiffuse::ConditionVectord diffuseCondition;
            diffuseCondition.setOnes();
            // m_diffuseDistribution->conditional(
            //     diffuseCondition, diffuseConditional, diffuseHeuristicWeight
            // );
            // std::cerr <<
            //     "Mean before rotation: " <<
            //     diffuseConditional.components()[0].mean().transpose() <<
            //     ".\n";
        }

        const auto aabb_extents = m_scene->getAABBWithoutCamera().getExtents();
        m_spatialNormalization = std::max(
            aabb_extents[0], std::max(aabb_extents[1], aabb_extents[2])
        );

        RadianceQueryRecord rRec(m_scene, m_sampler);
        Point2 apertureSample(0.5f);
        Float timeSample = 0.5f;
        RayDifferential sensorRay;

        uint32_t queryType = RadianceQueryRecord::ESensorRay;

        if (!m_sensor->getFilm()->hasAlpha()) /* Don't compute an alpha channel if we don't have to */
            queryType &= ~RadianceQueryRecord::EOpacity;

#if DUMP_DISTRIB == 1
        std::ofstream jsonDumpFile("dumps/full.json");
#endif

        m_timer = new Timer();

        for(
            int sampleInIteration = 0;
            sampleInIteration < (int) m_config.samplesPerIteration;
            ++sampleInIteration
        ) {
            bool allSamplesZero = true;
            m_timer->reset();
            for (size_t pixel_i = 0; pixel_i < points.size(); ++pixel_i) {
                Point2i offset =
                    Point2i(m_hilbertCurve[pixel_i]) +
                    Vector2i(rect->getOffset());
                m_sampler->generate(offset);
                if (stop)
                    break;

                Point2 samplePos(Point2(offset) + Vector2(m_sampler->next2D()));
                rRec.newQuery(queryType, m_sensor->getMedium());  

                Spectrum spec;

                if (needsApertureSample)
                    apertureSample = m_sampler->next2D();
                if (needsTimeSample)
                    timeSample = m_sampler->next1D();
                spec = m_sensor->sampleRayDifferential(
                    sensorRay, samplePos, apertureSample, timeSample);

                sensorRay.scaleDifferential(diffScaleFactor);

                Vectord firstSample = Vectord::Constant(0);
                bool savedSample = false;
                spec *= Li(
                    sensorRay,
                    rRec,
                    *m_distribution,
                    conditional,
                    diffuseConditional,
                    productConditional,
                    rotatedMaterialConditional,
                    samplePos,
                    firstSample,
                    savedSample
                );
                result->averagePathLength += rRec.depth;
                result->pathCount++;

                if(spec.max() != 0.f) {
                    allSamplesZero = false;
                }

                if(!spec.isValid()) {
                    std::cerr << spec.toString() << " INVALID\n";
                    return;
                }
                result->putSample(samplePos, spec);
            }

            avgIterationTime.incrementBase();
            avgIterationTime += m_timer->lap();

            continue;

            if(allSamplesZero) {
                continue;
            }

#if DENOISE == 1
            barrier->wait();
            if(m_threadId == 0){
                std::cerr << "Denoising.\n";
                result->clearDenoised();
                denoise(result, *m_distribution);
                result->dumpDenoised(
                    samplesInThisIteration,
                    destinationFile.parent_path(),
                    destinationFile.stem(),
                    m_iteration
                );
            }
            barrier->wait();
#endif
        }
	}

    template<int conditionalDims>
    static typename std::enable_if<conditionalDims == 2, Vector3>::type
    canonicalToDir(const Eigen::Matrix<MMScalar, conditionalDims, 1>& p) {
        const Float cosTheta = 2 * p.x() - 1;
        const Float phi = 2 * M_PI * p.y();

        const Float sinTheta = sqrt(1 - cosTheta * cosTheta);
        Float sinPhi, cosPhi;
        math::sincos(phi, &sinPhi, &cosPhi);

        return {sinTheta * cosPhi, sinTheta * sinPhi, cosTheta};
    }

    template<int conditionalDims>
    static typename std::enable_if<conditionalDims == 3, Vector3>::type
    canonicalToDir(const Eigen::Matrix<MMScalar, conditionalDims, 1>& p) {
        return {p.x(), p.y(), p.z()};
    }

    template<int conditionalDims>
    static typename std::enable_if<conditionalDims == 2, MMScalar>::type
    canonicalToDirInvJacobian() {
        return 4 * M_PI;
    }

    template<int conditionalDims>
    static typename std::enable_if<conditionalDims == 3, MMScalar>::type
    canonicalToDirInvJacobian() {
        return 1;
    }
    
    template<int conditionalDims>
    static typename std::enable_if<
        conditionalDims == 2, Eigen::Matrix<MMScalar, conditionalDims, 1>
    >::type dirToCanonical(const Vector& d) {
        if (!std::isfinite(d.x) || !std::isfinite(d.y) || !std::isfinite(d.z)) {
            return {0, 0};
        }

        const Float cosTheta = std::min(std::max(d.z, -1.0f), 1.0f);
        Float phi = std::atan2(d.y, d.x);
        while (phi < 0)
            phi += 2.0 * M_PI;

        return {(cosTheta + 1) / 2, phi / (2 * M_PI)};
    }
    
    template<int conditionalDims>
    static typename std::enable_if<
        conditionalDims == 3, Eigen::Matrix<MMScalar, conditionalDims, 1>
    >::type dirToCanonical(const Vector& d) {
        return {d.x, d.y, d.z};
    }

    template<int conditionalDims>
    static typename std::enable_if<conditionalDims == 2, MMScalar>::type
    dirToCanonicalInvJacobian() {
        return INV_FOURPI;
    }

    template<int conditionalDims>
    static typename std::enable_if<conditionalDims == 3, MMScalar>::type
    dirToCanonicalInvJacobian() {
        return 1;
    }

    void createCondition(Vectord& sample, const Intersection& its, int depth) const {
        const auto aabb_min = m_sceneAabb.min;

        if(t_conditionDims >= 3) {
            sample.template topRows<3>() <<
                (its.p[0] - aabb_min[0]) / m_spatialNormalization,
                (its.p[1] - aabb_min[1]) / m_spatialNormalization,
                (its.p[2] - aabb_min[2]) / m_spatialNormalization
            ;
        }
        if(t_conditionDims >= 4) {
            // sample.template segment<1>(3) <<
            //     its.time
            // ;
            sample.template segment<1>(3) <<
                (Float) depth / (Float) m_config.maxDepth;
            ;
        }
        if(t_conditionDims >= 5) {
            bool transmissive = its.isValid() ? (its.getBSDF()->getType() & BSDF::ETransmission) : false;

            Vector n = its.shFrame.n;
            if (!transmissive && dot(its.shFrame.n, its.shFrame.toWorld(its.wi)) < 0) {
                n = -n;
            }
            auto canonicalNormal = dirToCanonical<2>(n);
    
            sample.template segment<2>(3) <<
                canonicalNormal[0],
                canonicalNormal[1]
            ;
        }
        if(t_conditionDims >= 7) {
            auto canonicalWi = dirToCanonical<2>(its.shFrame.toWorld(its.wi));
            sample.template segment<2>(5) <<
                canonicalWi[0],
                canonicalWi[1];
        }
    }

    Spectrum sampleSurface(
        const BSDF* bsdf,
        const Scene* scene,
        const Intersection& its,
        BSDFSamplingRecord& bRec,
        Float& pdf,
        Float& bsdfPdf,
        Float& gmmPdf,
        MMScalar& heuristicConditionalWeight,
        Vectord& sample,
        RadianceQueryRecord& rRec,
        const MM& distribution,
        MMCond& conditional,
        MMCond& materialConditional,
        MMCond& rotatedMaterialConditional,
        MMCond& productConditional
    ) const {
        createCondition(sample, its, rRec.depth);
        gmmPdf = bsdfPdf = pdf = 0.f;

        Spectrum bsdfWeight;

        const auto type = bsdf->getType();

        if((type & BSDF::EDelta) == (type & BSDF::EAll)) {
            heuristicConditionalWeight = 1.0f;
            Spectrum result = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
            pdf = bsdfPdf;
            return result;
        }

		Eigen::Matrix<MMScalar, 3, 1> normal;
		normal << its.shFrame.n.x, its.shFrame.n.y, its.shFrame.n.z;
		if(Frame::cosTheta(its.wi) < 0) {
			normal = -normal;
		}

		GridKeyVector key;
		jmm::buildKey(sample, normal, key);
        auto gridCell = m_grid->find(key);
        // if(gridCell == nullptr) {
        //     if(m_iteration > 0) {
        //         std::cerr <<
        //             "sampleSurface: Could not find matching cell with key " <<
        //             key.transpose() <<
        //             ", position " <<
        //             sample.transpose() <<
        //             ", normal " <<
        //             normal.transpose() <<
        //             "\n";
        //     }
        // }

        if(m_iteration == 0 || gridCell == nullptr || gridCell->distribution.nComponents() == 0) {
            heuristicConditionalWeight = 1.0f;
            Spectrum result = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
            pdf = bsdfPdf;
            sample.template bottomRows<t_conditionalDims>() = dirToCanonical<t_conditionalDims>(
                bRec.its.shFrame.toWorld(bRec.wo)
            );
            return result;
        }

        bool validConditional;
        if(m_config.useHierarchical) {
            validConditional = gridCell->distribution.conditional(
                sample.template topRows<t_conditionDims>(),
                conditional,
                heuristicConditionalWeight
            );
        } else {
            validConditional = gridCell->distribution.conditional(
                sample.template topRows<t_conditionDims>(),
                conditional,
                heuristicConditionalWeight
            );
        }

        MMCond* samplingConditional = nullptr;

        if(
            m_config.sampleProduct &&
            validConditional &&
            (type & BSDF::EDiffuseReflection) == (type & BSDF::EAll)
        ) {
            std::cerr << "Calculating product!\n";
            materialConditional.rotateTo(normal, rotatedMaterialConditional);
            conditional.multiply(rotatedMaterialConditional, productConditional);
            samplingConditional = &productConditional;
        } else {
            samplingConditional = &conditional;
        }
        
        heuristicConditionalWeight = 0.5;
        if(!validConditional) {
            heuristicConditionalWeight = 1.0f;
        }

        if(rRec.nextSample1D() <= heuristicConditionalWeight) {
            bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
            pdf = bsdfPdf;
            if(bsdfWeight.isZero()) {
                return Spectrum{0.f};
            }
            sample.template bottomRows<t_conditionalDims>() = dirToCanonical<t_conditionalDims>(
                bRec.its.shFrame.toWorld(bRec.wo)
            );
            if(!validConditional || bRec.sampledType & BSDF::EDelta) {
                gmmPdf = 0.f;
                pdf *= heuristicConditionalWeight;
                return bsdfWeight / heuristicConditionalWeight;
            } 

            bsdfWeight *= pdf;
        } else {
            ConditionalVectord condVec = samplingConditional->sample(m_rng);
            sample.template bottomRows<t_conditionalDims>() << condVec;

            if(
                (t_conditionalDims == 2 && !jmm::isInUnitHypercube(condVec)) ||
                (t_conditionalDims == 3 && condVec.isZero())
             ) {
                return Spectrum{0.f};
            }

            bRec.wo = bRec.its.toLocal(canonicalToDir(condVec));
            bsdfWeight = bsdf->eval(bRec);
        }
        
        pdf = pdfSurface(
            bsdf,
            bRec,
            bsdfPdf,
            gmmPdf,
            *samplingConditional,
            heuristicConditionalWeight,
            sample.template bottomRows<t_conditionalDims>()
        );
        
        if(pdf == 0) {
            return Spectrum{0.f};
        }
        return bsdfWeight / pdf;
    }

    Float pdfSurface(
        const BSDF* bsdf,
        const BSDFSamplingRecord& bRec,
        Float& bsdfPdf,
        Float& gmmPdf,
        const MMCond& conditional,
        const Float heuristicConditionalWeight,
        const ConditionalVectord& sample
    ) const {
        auto type = bsdf->getType();
        if ((type & BSDF::EDelta) == (type & BSDF::EAll)) {
            return bsdf->pdf(bRec);
        }

        bsdfPdf = bsdf->pdf(bRec);
        if (bsdfPdf <= 0 || !std::isfinite(bsdfPdf)) {
            return 0;
        }
        gmmPdf = conditional.pdf(sample) * dirToCanonicalInvJacobian<t_conditionalDims>();

        return
            heuristicConditionalWeight * bsdfPdf +
            (1 - heuristicConditionalWeight) * gmmPdf;
    }
    
    Spectrum Li(
        const RayDifferential &r,
        RadianceQueryRecord &rRec,
        const MM& distribution,
        MMCond& conditional,
        MMCond& materialConditional,
        MMCond& rotatedMaterialConditional,
        MMCond& productConditional,
        const Point2& samplePos,
        Vectord& firstSample,
        bool& savedSample
    ) const {
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        MediumSamplingRecord mRec;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        Float eta = 1.0f;

        constexpr static Float initialHeuristicWeight = 1.f;
        const Float heuristicWeight = 
            (m_iteration == 0) ? initialHeuristicWeight : distribution.heuristicWeight();

        struct Vertex {
            Vectord canonicalSample;
            Spectrum weight;
            Spectrum functionValue;
            Spectrum throughput;
            Spectrum directIllumination;
            Float heuristicWeight;
            Float heuristicPdf;
            Float samplingPdf;
            Float learnedPdf;
            Eigen::Matrix<MMScalar, 3, 1> normal;
            bool isDiffuse;
            MMScalar curvature;

            void record(const Spectrum& radiance) {
                for(int ch = 0; ch < 3; ++ch) {
                    if(throughput[ch] > Epsilon) {
                        weight[ch] += radiance[ch] / (throughput[ch] * samplingPdf);
                        functionValue[ch] += radiance[ch] / throughput[ch];
                    }
                }
            }
        };

        std::array<Vertex, 100> vertices;
        assert(std::max(m_config.maxDepth + 1, m_config.savedSamplesPerPath) < 100);
        int depth = 0;

        auto recordRadiance = [&](const Spectrum& radiance, bool recordDirect=true) {
            Li += radiance;

            if(depth == 0) {
                return;
            }

            if(recordDirect) {
                for(int ch = 0; ch < 3; ++ch) {
                    if(vertices[depth - 1].throughput[ch] > Epsilon) {
                        vertices[depth - 1].directIllumination[ch] +=
                            radiance[ch] / vertices[depth - 1].throughput[ch];
                    }
                }
            }
            for(int i = 0; i < depth; ++i) {
                vertices[i].record(radiance);
            }
        };

        /* Perform the first ray intersection (or ignore if the
           intersection has already been provided). */
        rRec.rayIntersect(ray);

        Spectrum throughput(1.0f);
        Vectord canonicalSample;
        bool scattered = false;

            // = std::max(minHeuristicWeight, initialHeuristicWeight * std::pow(0.6f, (Float) m_iteration));
        while (rRec.depth <= m_config.maxDepth || m_config.maxDepth < 0) {
            Eigen::Matrix<MMScalar, 3, 1> normal;
            normal << its.shFrame.n.x, its.shFrame.n.y, its.shFrame.n.z;

            assert(depth < (int) vertices.size() - 1);

            /* Sample
                tau(x, y) (Surface integral). This happens with probability mRec.pdfFailure
                Account for this and multiply by the proper per-color-channel transmittance.
            */
            if (!its.isValid()) {
                /* If no intersection could be found, possibly return
                    attenuated radiance from a background luminaire */
                if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
                    && (true || scattered)) {
                    Spectrum value = throughput * scene->evalEnvironment(ray);
                    if (rRec.medium)
                        value *= rRec.medium->evalTransmittance(ray, rRec.sampler);
                    recordRadiance(value);
                }

                break;
            }

            /* Possibly include emitted radiance if requested */
            if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
                && (true || scattered)) {
                recordRadiance(throughput * its.Le(-ray.d));
            }

            /* Include radiance from a subsurface integrator if requested */
            if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance)) {
                recordRadiance(throughput * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth));
            }

            if (rRec.depth >= m_config.maxDepth && m_config.maxDepth != -1)
                break;

            /* Prevent light leaks due to the use of shading normals */
            Float wiDotGeoN = -dot(its.geoFrame.n, ray.d),
                    wiDotShN  = Frame::cosTheta(its.wi);
            if (wiDotGeoN * wiDotShN < 0 && false)
                break;

            /* ==================================================================== */
            /*                          Luminaire sampling                          */
            /* ==================================================================== */

            const BSDF *bsdf = its.getBSDF(ray);
            DirectSamplingRecord dRec(its);

#ifdef NEE
            /* Estimate the direct illumination if this is requested */
            if ((rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) &&
                (bsdf->getType() & BSDF::ESmooth)) {
                int interactions = m_config.maxDepth - rRec.depth - 1;

                Spectrum value = scene->sampleAttenuatedEmitterDirect(
                        dRec, its, rRec.medium, interactions,
                        rRec.nextSample2D(), rRec.sampler);

                if (!value.isZero()) {
                    const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

                    /* Evaluate BSDF * cos(theta) */
                    BSDFSamplingRecord bRec(its, its.toLocal(dRec.d));
                    const Spectrum bsdfVal = bsdf->eval(bRec);

                    Float woDotGeoN = dot(its.geoFrame.n, dRec.d);

                    /* Prevent light leaks due to the use of shading normals */
                    if (!bsdfVal.isZero() && (!m_strictNormals ||
                        woDotGeoN * Frame::cosTheta(bRec.wo) > 0)) {
                        /* Calculate prob. of having generated that direction
                            using BSDF sampling */
                        Float bsdfPdf = (emitter->isOnSurface()
                                && dRec.measure == ESolidAngle)
                                ? bsdf->pdf(bRec) : (Float) 0.0f;

                        /* Weight using the power heuristic */
                        const Float weight = miWeight(dRec.pdf, bsdfPdf);
                        Li += throughput * value * bsdfVal * weight;
                    }
                }
            }
#endif

            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */

            /* Sample BSDF * cos(theta) */
            BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
            Float misPdf, heuristicPdf, gmmPdf;
            MMScalar heuristicConditionalWeight;
            Spectrum bsdfWeight = sampleSurface(
                bsdf,
                scene,
                its,
                bRec,
                misPdf,
                heuristicPdf,
                gmmPdf,
                heuristicConditionalWeight,
                canonicalSample,
                rRec,
                distribution,
                conditional,
                materialConditional,
                rotatedMaterialConditional,
                productConditional
            );

            bool cacheable = !(bRec.sampledType & BSDF::EDelta);
            if(Frame::cosTheta(its.wi) < 0) {
                normal = -normal;
            }

            // if(m_iteration > 30 && !(bsdf->getType() & BSDF::EGlossy)) {
            //     GridKeyVector key;
            //     jmm::buildKey(canonicalSample, normal, key);
            //     typename HashGridType::AABB sampleAABB;
            //     auto cell = m_grid->find(key, sampleAABB);
            //     if(cell != nullptr) {
            //         Li +=
            //             throughput *
            //             bsdfWeight *
            //             Spectrum(
            //                 cell->distribution.modelError()
            //                 // cell->distribution.surfacePdf(canonicalSample, heuristicPdf)
            //             );
            //         break;
            //     }
            // }

            Float meanCurvature = 0, gaussianCurvature = 0;
            its.shape->getCurvature(its, meanCurvature, gaussianCurvature);
            
            if(bsdfWeight.isZero()) {
                if(!savedSample) {
                    firstSample = canonicalSample;
                    savedSample = true;
                }
                break;
            }

            /* Prevent light leaks due to the use of shading normals */
            const Vector wo = its.toWorld(bRec.wo);
            Float woDotGeoN = dot(its.geoFrame.n, wo);
            if (woDotGeoN * Frame::cosTheta(bRec.wo) <= 0 && false)
                break;

            /* Trace a ray in this direction */
            ray = Ray(its.p, wo, ray.time);

            /* Keep track of the throughput, medium, and relative
                refractive index along the path */
            throughput *= bsdfWeight;
            eta *= bRec.eta;
            if (its.isMediumTransition())
                rRec.medium = its.getTargetMedium(ray.d);

            /* Handle index-matched medium transitions specially */
            if (bRec.sampledType == BSDF::ENull) {
                if (!(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                    break;
                rRec.type = scattered ? RadianceQueryRecord::ERadianceNoEmission
                    : RadianceQueryRecord::ERadiance;
                scene->rayIntersect(ray, its);
                rRec.depth++;
                continue;
            }

            Spectrum value(0.0f);
            rayIntersectAndLookForEmitter(scene, rRec.sampler, rRec.medium,
                m_config.maxDepth - rRec.depth - 1, ray, its, dRec, value);

            /* If a luminaire was hit, estimate the local illumination and
                weight using the power heuristic */
            if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) {
#ifdef NEE
                const Float emitterPdf = (!(bRec.sampledType & BSDF::EDelta) && !value.isZero()) ?
                    scene->pdfEmitterDirect(dRec) : 0;
#else
                const Float emitterPdf = 0;
#endif
                Float weight = 1; // miWeight(misPdf, emitterPdf);
                if(!value.isZero()) {
                    recordRadiance(throughput * value * weight, !cacheable);
                }

                if(cacheable) {
                    Float invPdf = 1.f / misPdf;
                    Spectrum incomingRadiance = value * weight;
                    if(!savedSample) {
                        firstSample = canonicalSample;
                        savedSample = true;
                    }

                    if (misPdf > 0 && std::isfinite(invPdf)) {
                        bool isDiffuse = !(bsdf->getType() & BSDF::EGlossy);
                        //  || (
                        //     bsdf->getDiffuseReflectance(its).max() >
                        //     bsdf->getSpecularReflectance(its).max()
                        // );
                        
                        vertices[depth] = Vertex{
                            canonicalSample,
                            incomingRadiance * invPdf,
                            incomingRadiance,
                            throughput,
                            incomingRadiance,
                            heuristicConditionalWeight,
                            heuristicPdf,
                            misPdf,
                            gmmPdf,
                            normal,
                            isDiffuse,
                            meanCurvature
                        };

                        ++depth;
                    }
                }
            }

            /* ==================================================================== */
            /*                         Indirect illumination                        */
            /* ==================================================================== */

            /* Stop if indirect illumination was not requested */
            if (!(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                break;

            rRec.type = RadianceQueryRecord::ERadianceNoEmission;

            if (rRec.depth++ >= m_config.rrDepth) {
                /* Russian roulette: try to keep path weights equal to one,
                   while accounting for the solid angle compression at refractive
                   index boundaries. Stop with at least some probability to avoid
                   getting stuck (e.g. due to total internal reflection) */

                Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
                if (rRec.nextSample1D() >= q)
                    break;
                throughput /= q;
            }

            scattered = true;
        }

        avgPathLength.incrementBase();
        avgPathLength += rRec.depth;

        if(depth == 0) {
            return Li;
        }

        auto push_back_vertex = [&](RenderingSamplesType& samples, int d) {
            Eigen::Matrix<MMScalar, 3, 1> color;
            color <<
                vertices[d].weight[0],
                vertices[d].weight[1],
                vertices[d].weight[2];
            MMScalar discount = 0;
            samples.push_back_synchronized(
                vertices[d].canonicalSample,
                // vertices[d].functionValue.max() * canonicalToDirInvJacobian<t_conditionalDims>(),
                vertices[d].samplingPdf * canonicalToDirInvJacobian<t_conditionalDims>(),
                vertices[d].learnedPdf * canonicalToDirInvJacobian<t_conditionalDims>(),
                vertices[d].heuristicPdf * canonicalToDirInvJacobian<t_conditionalDims>(),
                vertices[d].heuristicWeight,
                vertices[d].weight.average(),
                color,

                vertices[d].isDiffuse,
                vertices[d].normal,
                vertices[d].curvature,

                vertices[d].directIllumination.max() * canonicalToDirInvJacobian<t_conditionalDims>(), // reward
                discount
            );
        };
        
        typename MM::ConditionVectord offset;
		int firstSaved = std::max(depth - m_config.savedSamplesPerPath, 0);
        for(int d = depth - 1; d >= firstSaved; --d) {
            Eigen::Matrix<MMScalar, 3, 1> position = vertices[d].
                canonicalSample.template topRows<3>();
            GridKeyVector key;
            jmm::buildKey(position, vertices[d].normal, key);
            typename HashGridType::AABB sampleAABB;
            auto sampleCell = m_grid->find(key, sampleAABB);
            if(sampleCell != nullptr) {
                push_back_vertex(sampleCell->samples, d);
            } else {
                push_back_vertex(*m_samples, d);
                continue;
            }

            int nJitters = ((d == depth - 1) ? 1 : 0);
            for(int jitter_i = 0; jitter_i < nJitters; ++jitter_i) {
                offset <<
                    rRec.sampler->next1D() - 0.5,
                    rRec.sampler->next1D() - 0.5,
                    rRec.sampler->next1D() - 0.5;
                offset.array() *= sampleAABB.diagonal().template topRows<3>().array();

                Eigen::Matrix<MMScalar, 3, 1> jitteredPosition =
                    position.array() + offset.array();
                GridKeyVector key;
                jmm::buildKey(jitteredPosition, vertices[d].normal, key);
                typename HashGridType::AABB aabb;
                auto gridCell = m_grid->find(key, aabb);
                if(gridCell == nullptr || (aabb.min().array() == sampleAABB.min().array()).all()) {
                    continue;
                } else {
                    push_back_vertex(gridCell->samples, d);
                }
            }
        }

        return Li;
    }

    /**
     * This function is called by the recursive ray tracing above after
     * having sampled a direction from a BSDF/phase function. Due to the
     * way in which this integrator deals with index-matched boundaries,
     * it is necessarily a bit complicated (though the improved performance
     * easily pays for the extra effort).
     *
     * This function
     *
     * 1. Intersects 'ray' against the scene geometry and returns the
     *    *first* intersection via the '_its' argument.
     *
     * 2. It checks whether the intersected shape was an emitter, or if
     *    the ray intersects nothing and there is an environment emitter.
     *    In this case, it returns the attenuated emittance, as well as
     *    a DirectSamplingRecord that can be used to query the hypothetical
     *    sampling density at the emitter.
     *
     * 3. If current shape is an index-matched medium transition, the
     *    integrator keeps on looking on whether a light source eventually
     *    follows after a potential chain of index-matched medium transitions,
     *    while respecting the specified 'maxDepth' limits. It then returns
     *    the attenuated emittance of this light source, while accounting for
     *    all attenuation that occurs on the wya.
     */
    void rayIntersectAndLookForEmitter(const Scene *scene, Sampler *sampler,
            const Medium *medium, int maxInteractions, Ray ray, Intersection &_its,
            DirectSamplingRecord &dRec, Spectrum &value) const {
        Intersection its2, *its = &_its;
        Spectrum transmittance(1.0f);
        bool surface = false;
        int interactions = 0;

        while (true) {
            surface = scene->rayIntersect(ray, *its);

            if (medium)
                transmittance *= medium->evalTransmittance(Ray(ray, 0, its->t), sampler);

            if (surface && (interactions == maxInteractions ||
                !(its->getBSDF()->getType() & BSDF::ENull) ||
                its->isEmitter())) {
                /* Encountered an occluder / light source */
                break;
            }

            if (!surface)
                break;

            if (transmittance.isZero())
                return;

            if (its->isMediumTransition())
                medium = its->getTargetMedium(ray.d);

            Vector wo = its->shFrame.toLocal(ray.d);
            BSDFSamplingRecord bRec(*its, -wo, wo, ERadiance);
            bRec.typeMask = BSDF::ENull;
            transmittance *= its->getBSDF()->eval(bRec, EDiscrete);

            ray.o = ray(its->t);
            ray.mint = Epsilon;
            its = &its2;

            if (++interactions > 100) { /// Just a precaution..
                Log(EWarn, "rayIntersectAndLookForEmitter(): round-off error issues?");
                return;
            }
        }

        if (surface) {
            /* Intersected something - check if it was a luminaire */
            if (its->isEmitter()) {
                dRec.setQuery(ray, *its);
                value = transmittance * its->Le(-ray.d);
            }
        } else {
            /* Intersected nothing -- perhaps there is an environment map? */
            const Emitter *env = scene->getEnvironmentEmitter();

            if (env && env->fillDirectSamplingRecord(dRec, ray))
                value = transmittance * env->evalEnvironment(RayDifferential(ray));
        }
    }

    inline Float miWeight(Float pdfA, Float pdfB) const {
        pdfA *= pdfA; pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }

    /* 
    void denoise(SDMMWorkResult* result, const MM& distribution) {
        using Scalar = MMScalar;
        using KDTree = typename kdt::KDTree<Scalar, kdt::EuclideanDistance<Scalar>>;
        using DistMatrix = typename KDTree::Matrix;
        using IdxMatrix = typename KDTree::MatrixI;

        int nSamples = m_samples->size();
        int nComponents = distribution.nComponents();

        KDTree kdtree(m_samples->samples.topLeftCorner(t_dims, nSamples), true);
        kdtree.setTakeRoot(true);
        kdtree.build();

        DistMatrix dists;
        IdxMatrix idx;
        size_t knn = 10;
        kdtree.query(m_samples->samples.topLeftCorner(t_dims, nSamples), knn, idx, dists);

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> maxDistances =
            dists.transpose().rowwise().maxCoeff();
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> volume = 
            jmm::volume_norm<t_dims>::value * maxDistances.array().pow(t_dims);
        assert(volume.rows() == nSamples);
        
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> localDensity(nSamples, 1);
        
        // Eigen::Matrix<Scalar, Eigen::Dynamic, 1> normalizedWeights =
        //    m_samples->weights.topRows(nSamples) /
        //    m_samples->weights.topRows(m_samples->size()).mean();

        localDensity.setZero();
        Scalar localDensityNormalization = 0.f;
        for(int sample_i = 0; sample_i < nSamples; ++sample_i) {
            for(int nn_i = 0; nn_i < (int) knn; ++nn_i) {
                int sample_j = idx(nn_i, sample_i);
                localDensity(sample_i) += m_samples->weights(sample_j);
            }
            localDensityNormalization += m_samples->weights(sample_i);
        }
        
        localDensity.array() /= volume.array() * localDensityNormalization;

        for(int sample_i = 0; sample_i < nSamples; ++sample_i) {
            if(m_samples->depths(sample_i) == 0) {
                result->putDenoisedSample(
                    m_samples->sensorPositions[sample_i],
                    m_samples->throughputs[sample_i] *
                    m_samples->samplingPdfs[sample_i] *
                    Spectrum(localDensity(sample_i))
                    // m_samples->spatialDensity(sample_i) *
                    // Spectrum(m_samples->weights(sample_i))
                );
            }
        }

        // int nSamples = m_samples->size();
        // using KDTree = kdt::KDTree<MMScalar, kdt::EuclideanDistance<MMScalar>>;
        // constexpr static int filteringDims = t_dims + 3;
        // constexpr static MMScalar featureSensitivity = 0.6;
        // Eigen::Matrix<MMScalar, t_conditionDims, Eigen::Dynamic> features;
        // features.conservativeResize(Eigen::NoChange, nSamples);
        // features.setZero();
        // features.topLeftCorner(t_conditionDims, nSamples) = m_samples->samples.topLeftCorner(t_conditionDims, nSamples);
        //     // (0.5 * m_samples->normals.topLeftCorner(3, nSamples) + 0.5) / featureSensitivity;

        // KDTree kdtree(features, true);
        // kdtree.setSorted(true);
        // kdtree.setTakeRoot(false);
        // kdtree.build();

        // KDTree::Matrix dists;
        // KDTree::MatrixI idx;
        // constexpr static size_t knn = 100;
        // kdtree.query(features, knn, idx, dists);

        // MMScalar sigma = 0.01;
        // for(int sample_i = 0; sample_i < nSamples; ++sample_i) {
        //     Eigen::Matrix<MMScalar, filteringDims, 1> sampleFeatures;
        //     sampleFeatures << m_samples->samples.col(sample_i), m_samples->normals.col(sample_i);
        //     Spectrum denoised(0.f);
        //     MMScalar normalization = 0.f;
        //     for(int nn_i = 0; nn_i < knn; ++nn_i) {
        //         int nn_idx = idx(nn_i, sample_i);
        //         if(nn_idx == -1) {
        //             continue;
        //         }
        //         Eigen::Matrix<MMScalar, filteringDims, 1> neighborFeatures;
        //         neighborFeatures <<
        //             m_samples->samples.col(nn_idx).topRows(3),
        //             canonicalToDir(m_samples->samples.col(nn_idx).bottomRows(2)),
        //             (0.5 * m_samples->normals.col(nn_idx).array() + 0.5);
        //         MMScalar distanceSqr = (sampleFeatures - neighborFeatures).squaredNorm();
        //         MMScalar weight = std::exp(-distanceSqr / sigma * sigma);
        //         denoised += weight * m_samples->Lis[nn_idx];
        //         normalization += weight;
        //     }

        //     if(normalization > 0) {
        //         denoised /= normalization;
        //     } else {
        //         denoised = Spectrum(0.f);
        //     }

        //     m_samples->denoisedWeights(sample_i) = denoised.max();
        //     if(m_samples->depths(sample_i) == 0) {
        //         result->putDenoisedSample(
        //             m_samples->sensorPositions[sample_i],
        //             m_samples->throughputs[sample_i] *
        //             m_samples->samplingPdfs[sample_i] *
        //             denoised
        //             // m_samples->spatialDensity(sample_i) *
        //             // Spectrum(m_samples->weights(sample_i))
        //         );
        //     }
        // }
    }
    */

        
    void getShapesAndBsdfs(std::vector<Shape*>& shapes, std::vector<BSDF*>& bsdfs) {        
        auto& sceneShapes = m_scene->getShapes();
        for (size_t i = 0; i < sceneShapes.size(); ++i) {
            Shape* shape = sceneShapes[i].get();
            if (!shape || !shape->getBSDF()) {
                continue;
            }
            BSDF* bsdf = shape->getBSDF();

            auto bsdfType = shape->getBSDF()->getType();
            if ((bsdfType & BSDF::EDiffuseReflection) || (bsdfType & BSDF::EGlossyReflection)) {
                shapes.push_back(shape);
                bsdfs.push_back(bsdf);
            }
        }

        for (size_t i = 0; i < shapes.size(); ++i) {
            Shape* s = shapes[i];
            if (s->isCompound()) {
                int j = 0;
                Shape* child = s->getElement(j);
                while (child != nullptr) {
                    shapes.emplace_back(child);
                    child = s->getElement(++j);
                }
            }
        }        
    }

    template<typename Optimizer>
    void perMeshBlueNoiseInit(MM& distribution, Optimizer em) {
        MMScalar spatialComponents = 85;
        std::vector<Shape*> shapes;
        std::vector<BSDF*> bsdfs;
        getShapesAndBsdfs(shapes, bsdfs);

        MMScalar totalArea = 0;
        for(auto shape : shapes) {
            totalArea += shape->getSurfaceArea();
        }

        jmm::Samples<t_dims, MMScalar> samples;
        m_samples->reserve(4 * spatialComponents);
        m_samples->setSize(spatialComponents);
        std::vector<jmm::SphereSide> sides;
        sides.reserve(spatialComponents);

        const auto aabb_min = m_sceneAabb.min;
        int point_i = 0;
        for(auto shape : shapes) {
            AABB shapeAABB = shape->getAABB();
            MMScalar radius = std::sqrt(shape->getSurfaceArea()) * 0.20;
            AABB aabb;
            Float sa;
            std::vector<Shape*> bnShapes;
            bnShapes.push_back(shape);
            ref<PositionSampleVector> points = new PositionSampleVector();
            blueNoisePointSet(m_scene, bnShapes, radius, points, sa, aabb, nullptr);

            Log(EInfo, "Generated " SIZE_T_FMT " blue-noise points.", points->size());

            for(int sample_i = 0; sample_i < points->size(); ++sample_i) {
                m_samples->samples.col(point_i).topRows(3) <<
                    ((*points)[sample_i].p.x - aabb_min[0]) / m_spatialNormalization,
                    ((*points)[sample_i].p.y - aabb_min[1]) / m_spatialNormalization,
                    ((*points)[sample_i].p.z - aabb_min[2]) / m_spatialNormalization
                ;

                m_samples->normals.col(point_i) <<
                    (*points)[sample_i].n.x,
                    (*points)[sample_i].n.y,
                    (*points)[sample_i].n.z
                ;
                jmm::SphereSide side;
                auto type = shape->getBSDF()->getType();
                if((type | BSDF::EFrontSide) && (type | BSDF::EBackSide)) {
                    sides.push_back(jmm::SphereSide::Both);
                } else if(type | BSDF::EFrontSide) {
                    sides.push_back(jmm::SphereSide::Top);
                } else if(type | BSDF::EBackSide) {
                    sides.push_back(jmm::SphereSide::Bottom);
                }
                ++point_i;
            }
        }
        spatialComponents = std::min((int) spatialComponents, point_i);
        m_samples->setSize(spatialComponents);

        auto& bPriors = em.getBPriors();
        auto& bDepthPriors = em.getBDepthPriors();
        jmm::uniformHemisphereInit(
            distribution,
            bPriors,
            bDepthPriors,
            m_rng,
            spatialComponents,
            // 1e-2,
            // (MMScalar) std::sqrt(0.5) * radius / m_spatialNormalization,
            1e-4,
            samples,
            sides,
            false,
            true
        );
    }

    
    template<typename Optimizer>
    void hammersleyInit(MM& distribution, Optimizer em) {
        MMScalar spatialComponents = 40;
        std::vector<Shape*> shapes;
        std::vector<BSDF*> bsdfs;
        getShapesAndBsdfs(shapes, bsdfs);

        ref<Sampler> shapeSampler;
        MMScalar totalArea = 0;
        for(auto shape : shapes) {
            totalArea += shape->getSurfaceArea();
        }

        jmm::Samples<t_dims, MMScalar> samples;
        m_samples->reserve(4 * spatialComponents);
        m_samples->setSize(spatialComponents);
        std::vector<jmm::SphereSide> sides;
        sides.reserve(spatialComponents);

        const auto aabb_min = m_sceneAabb.min;
        int point_i = 0;
        for(auto shape : shapes) {
            int nSamples = spatialComponents * shape->getSurfaceArea() / totalArea;
            nSamples = std::max(nSamples, 8);
            auto properties = Properties("hammersley");
            properties.setInteger("sampleCount", nSamples);
            properties.setInteger("dimension", 2);
            properties.setInteger("scramble", -1);
            shapeSampler = static_cast<Sampler*>(
                PluginManager::getInstance()->createObject(MTS_CLASS(Sampler), properties)
            );
            shapeSampler->generate({-1, -1});
            nSamples = shapeSampler->getSampleCount();
            PositionSamplingRecord pRec;
            for(int sample_i = 0; sample_i < nSamples; ++sample_i) {
                shape->samplePosition(pRec, shapeSampler->next2D());
                shapeSampler->advance();
                
                m_samples->samples.col(point_i).topRows(3) <<
                    (pRec.p.x - aabb_min[0]) / m_spatialNormalization,
                    (pRec.p.y - aabb_min[1]) / m_spatialNormalization,
                    (pRec.p.z - aabb_min[2]) / m_spatialNormalization
                ;

                m_samples->normals.col(point_i) <<
                    pRec.n.x,
                    pRec.n.y,
                    pRec.n.z
                ;
                jmm::SphereSide side;
                auto type =shape->getBSDF()->getType();
                if((type | BSDF::EFrontSide) && (type | BSDF::EBackSide)) {
                    sides.push_back(jmm::SphereSide::Both);
                } else if(type | BSDF::EFrontSide) {
                    sides.push_back(jmm::SphereSide::Top);
                } else if(type | BSDF::EBackSide) {
                    sides.push_back(jmm::SphereSide::Bottom);
                }
                ++point_i;
            }
        }
        spatialComponents = point_i;


        auto& bPriors = em.getBPriors();
        auto& bDepthPriors = em.getBDepthPriors();
        jmm::uniformHemisphereInit(
            distribution,
            bPriors,
            bDepthPriors,
            m_rng,
            spatialComponents,
            // 1e-2,
            // (MMScalar) std::sqrt(0.5) * radius / m_spatialNormalization,
            1e-4,
            samples,
            sides,
            false,
            true
        );
    }

    template<typename Optimizer>
    void blueNoiseInit(MM& distribution, Optimizer em) {
        const auto aabb_extents = m_scene->getAABBWithoutCamera().getExtents();
        Float radius = std::min(aabb_extents[0], std::min(aabb_extents[1], aabb_extents[2])) / 100.f;
        std::cerr << "Spatial radius: " << radius << std::endl; 

        std::vector<Shape*> shapes;
        std::vector<BSDF*> bsdfs;
        getShapesAndBsdfs(shapes, bsdfs);

        AABB aabb;
        Float sa;
        ref<PositionSampleVector> points = new PositionSampleVector();
        blueNoisePointSet(m_scene, shapes, radius, points, sa, aabb, nullptr);

        Log(EInfo, "Generated " SIZE_T_FMT " blue-noise points.", points->size());

        jmm::Samples<t_dims, MMScalar> samples;
        m_samples->reserve(points->size());
        m_samples->setSize(points->size());
        std::vector<jmm::SphereSide> sides(points->size(), jmm::SphereSide::Both);

        const auto aabb_min = m_sceneAabb.min;
        for(int point_i = 0; point_i < points->size(); ++point_i) {
            m_samples->samples.col(point_i).topRows(3) <<
                ((*points)[point_i].p.x - aabb_min[0]) / m_spatialNormalization,
                ((*points)[point_i].p.y - aabb_min[1]) / m_spatialNormalization,
                ((*points)[point_i].p.z - aabb_min[2]) / m_spatialNormalization
            ;

            m_samples->normals.col(point_i) <<
                (*points)[point_i].n.x,
                (*points)[point_i].n.y,
                (*points)[point_i].n.z
            ;
            jmm::SphereSide side;
            auto type = bsdfs[(*points)[point_i].shapeIndex]->getType();
            if((type | BSDF::EFrontSide) && (type | BSDF::EBackSide)) {
                sides[point_i] = jmm::SphereSide::Both;
            } else if(type | BSDF::EFrontSide) {
                sides[point_i] = jmm::SphereSide::Top;
            } else if(type | BSDF::EBackSide) {
                sides[point_i] = jmm::SphereSide::Bottom;
            }
        }

        auto& bPriors = em.getBPriors();
        auto& bDepthPriors = em.getBDepthPriors();
        jmm::uniformHemisphereInit(
            distribution,
            bPriors,
            bDepthPriors,
            m_rng,
            80,
            // 1e-2,
            // (MMScalar) std::sqrt(0.5) * radius / m_spatialNormalization,
            1e-4,
            samples,
            sides,
            true,
            true
        );
    }
    
    void dumpScene(const fs::path& path) {
        cout << "Dumping scene description to " << path.string() << endl;

        auto& sceneShapes = m_scene->getShapes();
        std::vector<Shape*> shapes;
        for (size_t i = 0; i < sceneShapes.size(); ++i) {
            Shape* shape = sceneShapes[i].get();
            if (!shape || !shape->getBSDF()) {
                continue;
            }

            auto bsdfType = shape->getBSDF()->getType();
            if ((bsdfType & BSDF::EDiffuseReflection) || (bsdfType & BSDF::EGlossyReflection)) {
                shapes.push_back(shape);
            }
        }

        for (size_t i = 0; i < shapes.size(); ++i) {
            Shape* s = shapes[i];
            if (s->isCompound()) {
                int j = 0;
                Shape* child = s->getElement(j);
                while (child != nullptr) {
                    shapes.emplace_back(child);
                    child = s->getElement(++j);
                }
            }
        }

        m_meshes.clear();
        for (Shape* s : shapes) {
            if (s->isCompound()) {
                continue;
            }
            ref<TriMesh> mesh = s->createTriMesh();
            if (mesh) {
                m_meshes.emplace_back(mesh);
            }
        }

        {
            std::cout << "Generating blue noise m_samples->\n";
            std::vector<Shape*> blueNoiseShapes;
            blueNoiseShapes.insert(std::begin(blueNoiseShapes), std::begin(shapes), std::end(shapes));
            m_blueNoisePoints = placePointsBlueNoise(m_scene, blueNoiseShapes);
        }

        BlobWriter blob(path.string());

        blob << (float) m_fieldOfView;

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                blob << (float) m_cameraMatrix(i, j);
            }
        }

        blob << (float) m_sceneAabb.min[0];
        blob << (float) m_sceneAabb.min[1];
        blob << (float) m_sceneAabb.min[2];

        blob << (float) m_spatialNormalization;

        blob << static_cast<uint64_t>(m_blueNoisePoints->size());
        for (auto p : m_blueNoisePoints->get()) {
            blob << p.p.x << p.p.y << p.p.z << p.n.x << p.n.y << p.n.z << p.shapeIndex;

            Intersection its;
            its.p = p.p;
            its.shFrame.n = p.n;

            Vectord sample;
            createCondition(sample, its, 0);
            for(int dim_i = 0; dim_i < t_conditionDims; ++dim_i) {
                blob << sample(dim_i);
            }
        }

        bool allMeshesHaveNormals = true;
        for (auto& mesh : m_meshes) {
            allMeshesHaveNormals = allMeshesHaveNormals && mesh->hasVertexNormals();
        }
        m_dumpMesh = m_dumpMesh && allMeshesHaveNormals;
        blob << (int32_t)(m_dumpMesh ? 1 : 0);
        if (m_dumpMesh) {
            // _all_ indices
            uint64_t totalTriangleCount = 0;
            for (auto& mesh : m_meshes) {
                totalTriangleCount += mesh->getTriangleCount();
            }

            blob << totalTriangleCount;

            uint64_t currentVertex = 0;
            for (auto& mesh : m_meshes) {
                // Indices
                size_t triangleCount = mesh->getTriangleCount();

                const Triangle* triangles = mesh->getTriangles();
                for (size_t i = 0; i < triangleCount; ++i) {
                    blob
                        << (triangles[i].idx[0] + currentVertex)
                        << (triangles[i].idx[1] + currentVertex)
                        << (triangles[i].idx[2] + currentVertex);
                }

                currentVertex += mesh->getVertexCount();
            }

            blob << currentVertex;

            // _all_ vertices
            for (auto& mesh : m_meshes) {
                // Vertices
                size_t vertexCount = mesh->getVertexCount();

                SAssert(mesh->hasVertexNormals());
                const Point* vertices = mesh->getVertexPositions();
                const Normal* normals = mesh->getVertexNormals();
                for (size_t i = 0; i < vertexCount; ++i) {
                    blob
                        << vertices[i].x << vertices[i].y << vertices[i].z
                        << normals[i].x << normals[i].y << normals[i].z;
                }
            }
        }
    }

    ref<PositionSampleVector> placePointsBlueNoise(const Scene *scene, const std::vector<Shape*> shapes) {
        Log(EInfo, "Generating blue-noise points throughout the scene.");

        ref<PositionSampleVector> points = new PositionSampleVector();

        Float actualRadius = m_sceneAabb.getExtents().length() / 500;
        if (m_dumpMesh) {
            // Make points even higher-res when dumping meshed, because
            // we want to _seriously_ visualize them.
            actualRadius /= 4;
        }

        AABB aabb;
        Float sa;
        blueNoisePointSet(m_scene, shapes, actualRadius, points, sa, aabb, nullptr);

        Log(EInfo, "Generated " SIZE_T_FMT " blue-noise points.", points->size());

        return points;
    }

	ref<WorkProcessor> clone() const {
		return new SDMMRenderer(
            m_config, m_distribution, m_grid, m_diffuseDistribution, m_samples, m_iteration
        );
	}

	MTS_DECLARE_CLASS()

private:
	ref<Scene> m_scene;
	ref<Sensor> m_sensor;
	ref<Sampler> m_sampler;
    std::function<MMScalar()> m_rng;
	ref<ReconstructionFilter> m_rfilter;
	MemoryPool m_pool;
	SDMMConfiguration m_config;
    int m_threadId = -1;
	HilbertCurve2D<int> m_hilbertCurve;
    int m_iteration;
    ref<Timer> m_timer;
    Float m_spatialNormalization;

    Matrix4x4 m_cameraMatrix;
    AABB m_sceneAabb;
    Float m_fieldOfView = 50;
    std::vector<ref<TriMesh>> m_meshes;
    bool m_dumpMesh = true;
    ref<PositionSampleVector> m_blueNoisePoints;

    static std::deque<jmm::Samples<t_dims, MMScalar>> prioritySamples;
    static Eigen::Matrix<MMScalar, t_conditionDims, 1> m_sampleMean;
    static Eigen::Matrix<MMScalar, t_conditionDims, 1> m_sampleStd;
    static std::unique_ptr<StepwiseEMType> stepwiseEM;

    std::shared_ptr<MM> m_distribution;
    std::shared_ptr<HashGridType> m_grid;
    std::shared_ptr<MMDiffuse> m_diffuseDistribution;
    std::shared_ptr<RenderingSamplesType> m_samples;
};

std::deque<jmm::Samples<SDMMProcess::t_dims, SDMMRenderer::MMScalar>> SDMMRenderer::prioritySamples;
std::unique_ptr<typename SDMMRenderer::StepwiseEMType> SDMMRenderer::stepwiseEM;

Eigen::Matrix<SDMMRenderer::MMScalar, SDMMProcess::t_conditionDims, 1> SDMMRenderer::m_sampleMean;
Eigen::Matrix<SDMMRenderer::MMScalar, SDMMProcess::t_conditionDims, 1> SDMMRenderer::m_sampleStd;

/* ==================================================================== */
/*                           Parallel process                           */
/* ==================================================================== */

constexpr int SDMMProcess::t_dims;
constexpr int SDMMProcess::t_conditionalDims;
constexpr int SDMMProcess::t_conditionDims;
constexpr int SDMMProcess::t_initComponents;
constexpr int SDMMProcess::t_components;
constexpr bool SDMMProcess::USE_BAYESIAN;

SDMMProcess::SDMMProcess(
    const RenderJob *parent,
    RenderQueue *queue,
    const SDMMConfiguration &config,
    std::shared_ptr<MM> distribution,
    std::shared_ptr<HashGridType> grid,
    std::shared_ptr<MMDiffuse> diffuseDistribution,
    std::shared_ptr<RenderingSamplesType> samples,
    int iteration
) :
    BlockedRenderProcess(parent, queue, config.blockSize),
    m_config(config),
    m_distribution(distribution),
    m_grid(grid),
    m_diffuseDistribution(diffuseDistribution),
    m_samples(samples),
    m_iteration(iteration)
{
    m_refreshTimer = new Timer();
    // barrier = std::make_unique<boost::barrier>(m_config.populations);
}

ref<WorkProcessor> SDMMProcess::createWorkProcessor() const {
    ref<WorkProcessor> renderer = new SDMMRenderer(
        m_config, m_distribution, m_grid, m_diffuseDistribution, m_samples, m_iteration
    );
    return renderer;
}

void SDMMProcess::develop() {
    LockGuard lock(m_resultMutex);
    Bitmap *bitmap = const_cast<Bitmap *>(
        m_result->getImageBlock()->getBitmap()
    );
    ref<Bitmap> converted = bitmap->convert(
        Bitmap::ESpectrum, Bitmap::EFloat, 1.0f, 1.0f
    );
    m_film->addBitmap(
        converted,
        1.0f / (
            Float(m_config.sampleCount) / Float(m_config.samplesPerIteration)
        )
    );
    m_refreshTimer->reset();
    m_queue->signalRefresh(m_parent);
}

void SDMMProcess::processResult(const WorkResult *wr, bool cancelled) {
    if (cancelled)
        return;
    const SDMMWorkResult *result = static_cast<const SDMMWorkResult *>(wr);
    ImageBlock *block = const_cast<ImageBlock *>(result->getImageBlock());
    LockGuard lock(m_resultMutex);
    m_progress->update(++m_resultCount);

    m_result->put(result);
    m_result->averagePathLength += result->averagePathLength;
    m_result->pathCount += result->pathCount;

    m_queue->signalWorkEnd(m_parent, result->getImageBlock(), false);
}

void SDMMProcess::bindResource(const std::string &name, int id) {
    BlockedRenderProcess::bindResource(name, id);
    if (name == "sensor") {
        m_result = new SDMMWorkResult(m_config, NULL, m_film->getCropSize());
        m_result->clear();
    }
}

MTS_IMPLEMENT_CLASS(SDMMRenderer, false, WorkProcessor)
MTS_IMPLEMENT_CLASS(SDMMProcess, false, BlockedImageProcess)

MTS_NAMESPACE_END
#pragma GCC diagnostic pop
