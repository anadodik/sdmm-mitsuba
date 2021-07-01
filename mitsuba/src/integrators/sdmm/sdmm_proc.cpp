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
#include <mutex>

#include <sdmm/distributions/sdmm_product.h>

MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLength("SDMM path tracer", "Average path length", EAverage);
static StatsCounter avgInvalidSamples("SDMM path tracer", "Average proportion of discarded samples.", EAverage);

class SDMMRenderer : public WorkProcessor {
    constexpr static int t_dims = SDMMProcess::t_dims;
    constexpr static int t_conditionalDims = SDMMProcess::t_conditionalDims;
    constexpr static int t_conditionDims = SDMMProcess::t_conditionDims;
    constexpr static int t_components = SDMMProcess::t_components;

    using Scalar = typename SDMMProcess::Scalar;
    using Embedded = typename SDMMProcess::Embedded;
    using ConditionalEmbedded = typename SDMMProcess::ConditionalEmbedded;
    using MarginalEmbedded = typename SDMMProcess::MarginalEmbedded;

    using SDMMContext = typename SDMMProcess::SDMMContext;
    using Accelerator = typename SDMMProcess::Accelerator;
    using AcceleratorPoint = typename Accelerator::Point;
    using AcceleratorAABB = typename Accelerator::AABB;

public:
    SDMMRenderer(
        const SDMMConfiguration &config,
        Accelerator* accelerator,
        int iteration,
        bool collect_data
    ) :
        m_config(config),
        m_accelerator(accelerator),
        m_iteration(iteration),
        m_collect_data(collect_data)
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

    void prepare() {
        Scene *scene = static_cast<Scene *>(getResource("scene"));
        m_scene = new Scene(scene);
        m_sampler = static_cast<Sampler *>(getResource("sampler"));
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

        m_sceneAabb = m_scene->getAABBWithoutCamera();
        const auto aabb_extents = m_sceneAabb.getExtents();
        m_spatialNormalization = std::max(
            aabb_extents[0], std::max(aabb_extents[1], aabb_extents[2])
        );

        if(m_config.maxDepth != 2 && m_config.savedSamplesPerPath <= 1) {
            std::cerr << "WARNING: ONLY SAVING ONE VERTEX PER PATH!\n";
            throw std::runtime_error( "WARNING: ONLY SAVING ONE VERTEX PER PATH!\n");
        }

        result->setSize(rect->getSize());
        result->setOffset(rect->getOffset());
        result->clear();

        m_hilbertCurve.initialize(TVector2<int>(rect->getSize()));
        auto& points = m_hilbertCurve.getPoints();

        Float diffScaleFactor = 1.0f /
            std::sqrt((Float) m_sampler->getSampleCount());

        bool needsApertureSample = m_sensor->needsApertureSample();
        bool needsTimeSample = m_sensor->needsTimeSample();

        RadianceQueryRecord rRec(m_scene, m_sampler);
        Point2 apertureSample(0.5f);
        Float timeSample = 0.5f;
        RayDifferential sensorRay;

        uint32_t queryType = RadianceQueryRecord::ESensorRay;

        if (!m_sensor->getFilm()->hasAlpha()) /* Don't compute an alpha channel if we don't have to */
            queryType &= ~RadianceQueryRecord::EOpacity;

        for(
            int sampleInIteration = 0;
            sampleInIteration < (int) m_config.samplesPerIteration;
            ++sampleInIteration
        ) {
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

                spec *= Li(sensorRay, rRec);
                result->averagePathLength += rRec.depth;
                result->pathCount++;
                result->putSample(samplePos, spec);
            }
        }
    }

    MarginalEmbedded toMarginalEmbedded(const Point& p) const {
        const auto aabb_min = m_sceneAabb.min;
        return MarginalEmbedded{
            (p[0] - aabb_min[0]) / m_spatialNormalization,
            (p[1] - aabb_min[1]) / m_spatialNormalization,
            (p[2] - aabb_min[2]) / m_spatialNormalization};
    }

    ConditionalEmbedded toConditionalEmbedded(const Vector3& v) const {
        return ConditionalEmbedded{v[0], v[1], v[2]};
    }

    Embedded toJointEmbedded(const Point& p, const Vector3 v) const {
        const auto aabb_min = m_sceneAabb.min;
        return Embedded{
            (p[0] - aabb_min[0]) / m_spatialNormalization,
            (p[1] - aabb_min[1]) / m_spatialNormalization,
            (p[2] - aabb_min[2]) / m_spatialNormalization,
            v[0], v[1], v[2]};
    }

    void setLearnedBSDF(
        const Intersection& its,
        const BSDFSamplingRecord& bRec,
        unsigned int bsdfType
    ) const {
        using RotationMatrix = SDMMProcess::ConditionalSDMM::TangentSpace::MatrixS;

        size_t n_slices = enoki::slices(learned_bsdf);
        if((bsdfType & BSDF::EDiffuseReflection) == (bsdfType & BSDF::EAll)) {
            Normal normal = its.shFrame.n;
            if(Frame::cosTheta(bRec.wi) < 0) {
                normal = -normal;
            }
            sdmm::linalg::Vector<float, 3> world_mean(
                (float) normal[0], (float) normal[1], (float) normal[2]
            );
            enoki::slice(learned_bsdf.tangent_space, 0).set_mean(world_mean);
        } else {
            sdmm::linalg::Vector<float, 3> wi(
                bRec.wi[0], bRec.wi[1], bRec.wi[2]
            );
            RotationMatrix to_world_space(
                its.shFrame.s[0], its.shFrame.t[0], its.shFrame.n[0],
                its.shFrame.s[1], its.shFrame.t[1], its.shFrame.n[1],
                its.shFrame.s[2], its.shFrame.t[2], its.shFrame.n[2]
            );
            auto&& ts = enoki::packet(learned_bsdf.tangent_space, 0);
            auto&& cs = ts.coordinate_system;
            ts.rotate_to_wo(wi);
            cs.from = to_world_space * cs.from;
            cs.to = cs.to * sdmm::linalg::transpose(to_world_space);
            ts.mean = to_world_space * ts.mean;
        }
    }

    Spectrum sampleSurface(
        const BSDF* bsdf,
        const Scene* scene,
        const Intersection& its,
        BSDFSamplingRecord& bRec,
        Float& pdf,
        Scalar& bsdfSamplingFraction,
        Embedded& sample,
        RadianceQueryRecord& rRec
    ) const {
        MarginalEmbedded marginalEmbedded = toMarginalEmbedded(its.p);
        ConditionalEmbedded conditionalEmbedded;
        pdf = 0.f;

        Spectrum bsdfWeight;

        const auto type = bsdf->getType();

        if((type & BSDF::EDelta) == (type & BSDF::EAll)) {
            bsdfSamplingFraction = 1.0f;
            Spectrum result = bsdf->sample(bRec, pdf, rRec.nextSample2D());
            return result;
        }

        SDMMContext* sdmmContext = nullptr;
        if(m_iteration != 0) {
            sdmmContext = m_accelerator->find(marginalEmbedded);
        }
        if(m_iteration == 0 || sdmmContext == nullptr || !sdmmContext->initialized) {
            bsdfSamplingFraction = 1.0f;
            Spectrum result = bsdf->sample(bRec, pdf, rRec.nextSample2D());
            Vector3 worldWo = bRec.its.shFrame.toWorld(bRec.wo);
            conditionalEmbedded = toConditionalEmbedded(worldWo);
            sample = toJointEmbedded(its.p, worldWo);
            return result;
        }

        bool validConditional = false;
        bool learnedBSDFFound = false;
        if(m_config.sampleProduct || m_config.bsdfOnly) {
            learnedBSDFFound = bsdf->getDMM(bRec, learned_bsdf);
        }
        bool usingProduct = !m_config.bsdfOnly && m_config.sampleProduct && learnedBSDFFound;
        bool usingLearnedBSDF = m_config.bsdfOnly && learnedBSDFFound;

        if(usingProduct || usingLearnedBSDF) {
            setLearnedBSDF(its, bRec, type);
        }

        if(!usingLearnedBSDF) {
            if(enoki::slices(conditional) != enoki::slices(sdmmContext->conditioner)) {
                enoki::set_slices(conditional, enoki::slices(sdmmContext->conditioner));
            }
            validConditional = sdmm::create_conditional(sdmmContext->conditioner, marginalEmbedded, conditional);
        }

        if(!usingLearnedBSDF && validConditional && usingProduct) {
            auto product_success = sdmm::product(conditional, learned_bsdf, product);
            if(enoki::none(product_success)) {
                spdlog::info("product unsuccessful={}", product_success);
                usingProduct = false;
            }
        }

        bsdfSamplingFraction = 0.5;
        if(usingLearnedBSDF) {
            bsdfSamplingFraction = 0.f;
        } else if(usingProduct) {
            bsdfSamplingFraction = 0.3;
        } else if(!validConditional) {
            bsdfSamplingFraction = 1.0f;
        }

        if(!validConditional || (!usingLearnedBSDF && rRec.nextSample1D() <= bsdfSamplingFraction)) {
            bsdfWeight = bsdf->sample(bRec, pdf, rRec.nextSample2D());
            if(bsdfWeight.isZero()) {
                return Spectrum{0.f};
            }
            Vector3 worldWo = bRec.its.shFrame.toWorld(bRec.wo);
            conditionalEmbedded = toConditionalEmbedded(worldWo);
            sample = toJointEmbedded(its.p, worldWo);
            if(!validConditional || bRec.sampledType & BSDF::EDelta) {
                pdf *= bsdfSamplingFraction;
                return bsdfWeight / bsdfSamplingFraction;
            }

            bsdfWeight *= pdf;
        } else {
            float inv_jacobian;
            if(usingLearnedBSDF) {
                learned_bsdf.sample(
                    sdmm_rng, conditionalEmbedded, inv_jacobian, tangent_sample
                );
            } else if(usingProduct) {
                product.sample(
                    sdmm_rng, conditionalEmbedded, inv_jacobian, tangent_sample
                );
            } else {
                conditional.sample(
                    sdmm_rng, conditionalEmbedded, inv_jacobian, tangent_sample
                );
            }

            auto length = enoki::norm(enoki::tail<3>(conditionalEmbedded));
            if(inv_jacobian != 0 && enoki::any(enoki::abs(length - 1) >= 1e-5)) {
                std::cerr << fmt::format(
                    "length=({}, {}, {})\nfrom={}\n",
                    conditionalEmbedded,
                    tangent_sample,
                    inv_jacobian,
                    conditional.tangent_space.coordinate_system.from
                );
            }

            avgInvalidSamples.incrementBase();
            if(inv_jacobian == 0) {
                avgInvalidSamples += 1;
                return Spectrum(0.f);
            }

            if(conditionalEmbedded == 0) {
                return Spectrum{0.f};
            }

            Vector3 worldWo(
                (float) conditionalEmbedded.coeff(0),
                (float) conditionalEmbedded.coeff(1),
                (float) conditionalEmbedded.coeff(2)
            );
            bRec.wo = bRec.its.toLocal(worldWo);
            sample = toJointEmbedded(its.p, worldWo);
            bsdfWeight = bsdf->eval(bRec);
        }

        if(usingLearnedBSDF) {
            pdf = pdfSurface(
                bsdf,
                bRec,
                learned_bsdf,
                bsdf_posterior,
                sdmmContext->conditioner,
                bsdfSamplingFraction,
                conditionalEmbedded
            );
        } else if(usingProduct) {
            pdf = pdfSurface(
                bsdf,
                bRec,
                product,
                posterior,
                sdmmContext->conditioner,
                bsdfSamplingFraction,
                conditionalEmbedded
            );
        } else {
            pdf = pdfSurface(
                bsdf,
                bRec,
                conditional,
                posterior,
                sdmmContext->conditioner,
                bsdfSamplingFraction,
                conditionalEmbedded
            );
        }

        if(pdf == 0) {
            return Spectrum{0.f};
        }
        return bsdfWeight / pdf;
    }

    template<typename DMM, typename Value>
    Float pdfSurface(
        const BSDF* bsdf,
        const BSDFSamplingRecord& bRec,
        DMM& conditional,
        Value& posterior,
        SDMMProcess::Conditioner& conditioner,
        const Float bsdfSamplingFraction,
        const ConditionalEmbedded& sample
    ) const {
        if(enoki::slices(posterior) != enoki::slices(conditional)) {
            enoki::set_slices(posterior, enoki::slices(conditional));
        }
        auto type = bsdf->getType();
        if (bsdfSamplingFraction == 1.0f || (type & BSDF::EDelta) == (type & BSDF::EAll)) {
            return bsdf->pdf(bRec);
        }

        Float bsdfPdf = bsdf->pdf(bRec);
        if (bsdfPdf <= 0 || !std::isfinite(bsdfPdf)) {
            return 0;
        }
        enoki::vectorize_safe(
            VECTORIZE_WRAP_MEMBER(posterior),
            conditional,
            sample,
            posterior
        );
        Float sdmmPdf = enoki::hsum_nested(posterior);
        if(!std::isfinite(sdmmPdf)) {
            std::cerr << fmt::format(
                "pdf={}\n"
                "posterior={}\n"
                "sdmmPdf={}\n"
                "bsdfPdf={}\n"
                "sample={}\n"

                "distribution=\n"

                "marginal_weight={}\n"
                "marginal_cov={}\n"
                "marginal_cov_sqrt={}\n"

                "cond_weight={}\n"
                "to={}\n"
                "rotated={}\n"
                "mean={}\n"
                "cond_cov={}\n"
                "cov_sqrt={}\n"
                "",
                sdmmPdf,
                posterior,
                sdmmPdf,
                bsdfPdf,
                sample,

                conditioner.marginal.weight.pmf,
                conditioner.marginal.cov,
                conditioner.marginal.cov_sqrt,

                conditional.weight.pmf,
                conditional.tangent_space.coordinate_system.to,
                conditional.tangent_space.coordinate_system.to * sample,
                conditional.tangent_space.mean,
                conditional.cov,
                conditional.cov_sqrt
            );
        }

        return
            bsdfSamplingFraction * bsdfPdf +
            (1 - bsdfSamplingFraction) * sdmmPdf;
    }

    Spectrum Li(
        const RayDifferential &r,
        RadianceQueryRecord &rRec
    ) const {
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        MediumSamplingRecord mRec;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        Float eta = 1.0f;

        constexpr static Float initialHeuristicWeight = 1.f;
        const Float heuristicWeight =
            (m_iteration == 0) ? initialHeuristicWeight : 0.5;

        struct Vertex {
            Spectrum weight;
            Spectrum throughput;
            Float samplingPdf;
            sdmm::embedded_s_t<SDMMProcess::Data> point;
            sdmm::Vector<float, 3> position;
            sdmm::normal_s_t<SDMMProcess::Data> sdmm_normal;

            void record(const Spectrum& radiance) {
                for(int ch = 0; ch < 3; ++ch) {
                    if(throughput[ch] > Epsilon) {
                        weight[ch] += radiance[ch] / (throughput[ch] * samplingPdf);
                    }
                }
            }
        };

        std::array<Vertex, 10> vertices;
        assert(std::max(m_config.maxDepth + 1, m_config.savedSamplesPerPath) < 100);
        int depth = 0;

        auto recordRadiance = [&](const Spectrum& radiance) {
            Li += radiance;

            if(depth == 0) {
                return;
            }
            for(int i = 0; i < depth; ++i) {
                vertices[i].record(radiance);
            }
        };

        /* Perform the first ray intersection (or ignore if the
           intersection has already been provided). */
        rRec.rayIntersect(ray);

        Spectrum throughput(1.0f);
        Embedded sample;
        bool scattered = false;

            // = std::max(minHeuristicWeight, initialHeuristicWeight * std::pow(0.6f, (Float) m_iteration));
        while (rRec.depth <= m_config.maxDepth || m_config.maxDepth < 0) {
            Normal normal = its.shFrame.n;

            assert(depth <= (int) vertices.size() - 1);

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
            if (wiDotGeoN * wiDotShN < 0 && m_config.strictNormals)
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
            Float surfacePdf;
            Scalar bsdfSamplingFraction;
            Spectrum bsdfWeight = sampleSurface(
                bsdf,
                scene,
                its,
                bRec,
                surfacePdf,
                bsdfSamplingFraction,
                sample,
                rRec
            );
            if(!bsdfWeight.isValid()) {
                std::cerr << "!bsdfWeight.isValid()\n";
                return Li;
            }

            bool cacheable = !(bRec.sampledType & BSDF::EDelta);
            if(Frame::cosTheta(bRec.wi) < 0) {
                normal = -normal;
            }

            if(bsdfWeight.isZero()) {
                break;
            }

            /* Prevent light leaks due to the use of shading normals */
            const Vector wo = its.toWorld(bRec.wo);
            Float woDotGeoN = dot(its.geoFrame.n, wo);
            if (woDotGeoN * Frame::cosTheta(bRec.wo) <= 0 && m_config.strictNormals)
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
                Float weight = 1; // miWeight(surfacePdf, emitterPdf);
                if(!value.isZero()) {
                    recordRadiance(throughput * value * weight);
                }

                if(cacheable) {
                    Float invPdf = 1.f / surfacePdf;
                    Spectrum incomingRadiance = value * weight;

                    if (surfacePdf > 0 && std::isfinite(invPdf)) {
                        bool isDiffuse = !(bsdf->getType() & BSDF::EGlossy);

                        vertices[depth] = Vertex{
                            incomingRadiance * invPdf,
                            throughput,
                            surfacePdf,
                            sample,
                            sdmm::Vector<float, 3>{
                                sample.coeff(0), sample.coeff(1), sample.coeff(2)
                            },
                            sdmm::normal_s_t<SDMMProcess::Data>{
                                (float) normal[0], (float) normal[1], (float) normal[2]
                            }
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

        if(depth == 0 || !m_collect_data || Li.max() == 0) {
            return Li;
        }

        auto push_back_data = [&](SDMMProcess::SDMMContext& context, int d, bool addToStats=true) {
            Float averageWeight = vertices[d].weight.average();
            {
                std::lock_guard lock(context.mutex_wrapper.mutex);
                if(addToStats) {
                    context.stats.push_back(vertices[d].position);
                }
                context.data.push_back(
                    vertices[d].point,
                    vertices[d].sdmm_normal,
                    averageWeight
                );
            }
        };

        sdmm::Vector<float, 3> offset;
        int firstSaved = std::max(depth - m_config.savedSamplesPerPath, 0);
        for(int d = depth - 1; d >= firstSaved; --d) {
            Float averageWeight = vertices[d].weight.average();
            if(!m_collect_data || !sdmm::is_valid_sample(averageWeight)) {
                continue;
            }
            const AcceleratorPoint& position = vertices[d].position;
            AcceleratorAABB sampleAABB;
            auto sampleContext = m_accelerator->find(position, sampleAABB);
            if(sampleContext != nullptr) {
                push_back_data(*sampleContext, d);
            } else {
                std::cerr << "ERROR: COULD NOT FIND CELL FOR SAMPLE." << std::endl;
                throw std::runtime_error("ERROR: COULD NOT FIND CELL FOR SAMPLE.");
                continue;
            }

            int nJitters = 1; // ((d == depth - 1) ? 1 : 0);
            for(int jitter_i = 0; jitter_i < nJitters; ++jitter_i) {
                offset = sdmm::Vector<float, 3>{
                    rRec.sampler->next1D() - 0.5f,
                    rRec.sampler->next1D() - 0.5f,
                    rRec.sampler->next1D() - 0.5f
                };
                auto diagonal = sampleAABB.diagonal();
                offset *= diagonal;

                sdmm::Vector<float, 3> jitteredPosition = position + offset;
                AcceleratorAABB aabb;
                auto sdmmContext = m_accelerator->find(jitteredPosition, aabb);
                if(sdmmContext == nullptr || enoki::all(aabb.min == sampleAABB.min)) {
                    continue;
                } else {
                    push_back_data(*sdmmContext, d, false);
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

    ref<WorkProcessor> clone() const {
        return new SDMMRenderer(
            m_config, m_accelerator, m_iteration, m_collect_data
        );
    }

    MTS_DECLARE_CLASS()

private:
    ref<Scene> m_scene;
    ref<Sensor> m_sensor;
    ref<Sampler> m_sampler;
    ref<ReconstructionFilter> m_rfilter;

    SDMMConfiguration m_config;
    HilbertCurve2D<int> m_hilbertCurve;
    int m_iteration;
    bool m_collect_data;
    Float m_spatialNormalization;

    AABB m_sceneAabb;
    Float m_fieldOfView = 50;

    mutable SDMMProcess::ConditionalSDMM conditional;
    mutable BSDF::DMM learned_bsdf;
    mutable SDMMProcess::ConditionalSDMM product;
    mutable SDMMProcess::RNG sdmm_rng;
    mutable sdmm::embedded_s_t<SDMMProcess::ConditionalSDMM> embedded_sample;
    mutable sdmm::tangent_s_t<SDMMProcess::ConditionalSDMM> tangent_sample;

    mutable SDMMProcess::Value posterior;
    mutable BSDF::Value bsdf_posterior;

    Accelerator* m_accelerator;
};

/* ==================================================================== */
/*                           Parallel process                           */
/* ==================================================================== */

constexpr int SDMMProcess::t_dims;
constexpr int SDMMProcess::t_conditionalDims;
constexpr int SDMMProcess::t_conditionDims;
constexpr int SDMMProcess::t_components;

SDMMProcess::SDMMProcess(
    const RenderJob *parent,
    RenderQueue *queue,
    const SDMMConfiguration &config,
    Accelerator* accelerator,
    int iteration,
    bool collect_data
) :
    BlockedRenderProcess(parent, queue, config.blockSize),
    m_config(config),
    m_accelerator(accelerator),
    m_iteration(iteration),
    m_collect_data(collect_data)
{
    m_refreshTimer = new Timer();
    // barrier = std::make_unique<boost::barrier>(m_config.populations);
}

ref<WorkProcessor> SDMMProcess::createWorkProcessor() const {
    ref<WorkProcessor> renderer = new SDMMRenderer(
        m_config, m_accelerator, m_iteration, m_collect_data
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

    m_result->put(result);
    m_result->averagePathLength += result->averagePathLength;
    m_result->pathCount += result->pathCount;

    m_queue->signalWorkEnd(m_parent, result->getImageBlock(), false);

    {
        LockGuard lock(m_resultMutex);
        m_progress->update(++m_resultCount);
    }

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
