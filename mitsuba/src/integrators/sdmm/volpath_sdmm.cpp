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

#include <fstream>
#include <memory>

#include <nlohmann/json.hpp>

#include <mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/bidir/util.h>
#include <mitsuba/bidir/path.h>
#include <mitsuba/bidir/edge.h>
#include "../../subsurface/bluenoise.h"

#include <tev/ThreadPool.h>
#include "sdmm_config.h"
#include "sdmm_proc.h"
#include "mesh.h"

using json = nlohmann::json;

MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLength("SDMM volumetric path tracer", "Average path length", EAverage);

class SDMMVolumetricPathTracer : public Integrator {
    using Scalar = typename SDMMProcess::Scalar;

    using SDMMContext = typename SDMMProcess::SDMMContext;
    using Accelerator = typename SDMMProcess::Accelerator;
    using AcceleratorNode = typename Accelerator::Node;
    using AcceleratorPoint = typename Accelerator::Point;

public:
    SDMMVolumetricPathTracer(const Properties &props) : Integrator(props) {
		/* Load the parameters / defaults */
		m_config.maxDepth = props.getInteger("maxDepth", -1);
		m_config.blockSize = props.getInteger("blockSize", -1);
		m_config.rrDepth = props.getInteger("rrDepth", 5);
		m_config.strictNormals = props.getBoolean("strictNormals", true);
		m_config.sampleDirect = props.getBoolean("sampleDirect", true);
		m_config.showWeighted = props.getBoolean("showWeighted", false);
		m_config.samplesPerIteration = props.getInteger("samplesPerIteration", 8);
		m_config.useHierarchical = props.getBoolean("useHierarchical", true);
		m_config.sampleProduct = props.getBoolean("sampleProduct", false);
		m_config.bsdfOnly = props.getBoolean("bsdfOnly", false);
		m_config.alpha = props.getFloat("alpha", 0.5f);
		m_config.batchIterations = props.getInteger("batchIterations", 0);
		m_config.initIterations = props.getInteger("initIterations", 3);
		m_config.enablePER = props.getBoolean("enablePER", false);
		m_config.replayBufferLength = props.getInteger("replayBufferLength", 3);
		m_config.resampleProportion = props.getFloat("resampleProportion", 0.1);
		m_config.decreasePrior = props.getBoolean("decreasePrior", true);
		m_config.correctStateDensity = props.getBoolean("correctStateDensity", true);
		m_config.savedSamplesPerPath = props.getInteger("savedSamplesPerPath", 8);

		m_config.useInit = props.getBoolean("useInit", true);
		m_config.useInitCovar = props.getBoolean("useInitCovar", false);
		m_config.useInitWeightsForMeans = props.getBoolean("useInitWeightsForMeans", true);
		m_config.useInitWeightsForMixture = props.getBoolean("useInitWeightsForMixture", false);
		m_config.initKMeansSwapTolerance = props.getFloat("initKMeansSwapTolerance", 1.f); //one means disabled

        m_config.dump();

		if (m_config.rrDepth <= 0)
			Log(EError, "'rrDepth' must be set to a value greater than zero!");

		if (m_config.maxDepth <= 0 && m_config.maxDepth != -1)
			Log(EError, "'maxDepth' must be set to -1 (infinite) or a value greater than zero!");
    }

    /// Unserialize from a binary data stream
    SDMMVolumetricPathTracer(Stream *stream, InstanceManager *manager)
    : Integrator(stream, manager) {
        m_config = SDMMConfiguration(stream);
    }

    void serialize(Stream *stream, InstanceManager *manager) const  override{
        Integrator::serialize(stream, manager);
        m_config.serialize(stream);
    }

	bool preprocess(const Scene *scene, RenderQueue *queue,
			const RenderJob *job, int sceneResID, int sensorResID,
			int samplerResID) override {
		Integrator::preprocess(scene, queue, job, sceneResID,
				sensorResID, samplerResID);

		if (scene->getSubsurfaceIntegrators().size() > 0)
			Log(EError, "Subsurface integrators are not supported "
				"by the SDMM path tracer!");

		return true;
	}

    void cancel() override {
		Scheduler::getInstance()->cancel(m_process);
    }

    void saveCheckpoint(const fs::path& experimentPath, int iteration) {
        fs::path checkpointsDir = experimentPath / "checkpoints";
        if(iteration == 0 && (!fs::is_directory(checkpointsDir) || !fs::exists(checkpointsDir))) {
            fs::create_directories(checkpointsDir);
        }
        fs::path distributionPath = checkpointsDir / fs::path(
            formatString("model_%05i.asdmm", iteration)
        );
        sdmm::save_json(m_accelerator, distributionPath.string());
    }

    void initializeSDMMContext(SDMMProcess::SDMMContext* context, Scalar maxDiagonal) {
        constexpr static int n_spatial_components = SDMMProcess::NComponents / 8;

        if(context->data.size < 2 * n_spatial_components || enoki::slices(context->sdmm) > 0) {
            return;
        }
        float spatial_distance = 3 * maxDiagonal / n_spatial_components;
        sdmm::initialize(context->sdmm, context->em, context->data, context->rng, n_spatial_components, spatial_distance);
        enoki::set_slices(context->conditioner, enoki::slices(context->sdmm));
        sdmm::prepare(context->conditioner, context->sdmm);
    }

    bool canBeOptimized(const AcceleratorNode& node) {
        const auto& context = node.value;
        return (
            node.is_leaf &&
            // node.depth >= 8 &&
            context != nullptr &&
            context->data.size >= 32
        );
    }

    void optimize() {
        constexpr static int t_conditionDims = SDMMProcess::t_conditionDims;
        int splitThreshold = 16000;

        std::cerr << "Splitting samples.\n";
        m_accelerator->split(splitThreshold);

        auto& nodes = m_accelerator->data();
        std::vector<size_t> node_idcs;
        node_idcs.reserve(nodes.size());
        for(size_t context_i = 0; context_i < nodes.size(); ++context_i) {
            auto& context = nodes[context_i].value;
            if(!canBeOptimized(nodes[context_i])) {
                continue;
            }
            node_idcs.push_back(context_i);
        }

        std::cerr << "Optimizing guiding distribution: " << node_idcs.size() << " distributions in tree.\n";
        m_thread_pool->parallelFor(0, (int) node_idcs.size(), [&](int i){
            size_t context_i = node_idcs[(size_t) i];
            auto& context = nodes[context_i].value;
            initializeSDMMContext(context.get(), enoki::hmax(nodes[context_i].aabb.diagonal()));
            if(enoki::slices(context->sdmm) == 0) {
                return;
            }

            sdmm::em_step(context->sdmm, context->em, context->data);
            enoki::set_slices(context->conditioner, enoki::slices(context->sdmm));
            sdmm::prepare(context->conditioner, context->sdmm);

            context->data.clear();
        });
    }

    template<typename AABB>
    void getAABB(
        AABB& aabb,
        const Point& aabb_min,
        const Point& aabb_max,
        Scalar spatialNormalization
    ) {
        using PointAABB = typename AABB::Point;
        PointAABB epsilon = enoki::full<PointAABB>(1e-5f);
        aabb.min = enoki::zero<PointAABB>();
        aabb.max = PointAABB(
            aabb_max[0] - aabb_min[0],
            aabb_max[1] - aabb_min[1],
            aabb_max[2] - aabb_min[2]
        );
        aabb.max /= spatialNormalization;
        aabb.min -= epsilon;
        aabb.max += epsilon;
    }

	bool render(
        Scene *scene,
        RenderQueue *queue,
        const RenderJob *job,
        int sceneResID,
        int sensorResID,
        int samplerResID
    ) override {

	spdlog::info("Max packet size={}", enoki::max_packet_size);
        m_scene = scene;
		ref<Scheduler> scheduler = Scheduler::getInstance();
		ref<Sensor> sensor = scene->getSensor();
		const Film *film = sensor->getFilm();
		size_t sampleCount = scene->getSampler()->getSampleCount();
		size_t seed = scene->getSampler()->getSeed();
		size_t nCores = scheduler->getCoreCount();

        // Thread::initializeOpenMP(nCores);
        m_thread_pool = std::make_unique<tev::ThreadPool>(nCores);

		Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " samples, " SIZE_T_FMT
			" %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y,
			sampleCount, nCores, nCores == 1 ? "core" : "cores");

        m_config.blockSize = scene->getBlockSize();
		m_config.cropSize = film->getCropSize();
        m_config.sampleCount = sampleCount;
		m_config.rngSeed = seed;
        if(sampleCount % m_config.samplesPerIteration != 0) {
            Log(EWarn,
                "sampleCount % m_config.samplesPerIteration "
                "(" SIZE_T_FMT " % " SIZE_T_FMT ") != 0\n",
                sampleCount,
                m_config.samplesPerIteration
            );
        }
		m_config.dump();

        fs::path destinationFile = scene->getDestinationFile();

        const int nPixels = m_config.cropSize.x * m_config.cropSize.y;
        m_maxSamplesSize =
            nPixels * m_config.samplesPerIteration * m_config.savedSamplesPerPath;

        const auto scene_aabb = m_scene->getAABBWithoutCamera();
        const auto aabb_min = scene_aabb.min;
        const auto aabb_max = scene_aabb.max;
        const auto aabb_extents = scene_aabb.getExtents();
        Float spatialNormalization = std::max(
            aabb_extents[0], std::max(aabb_extents[1], aabb_extents[2])
        );
        json scene_norm;
        scene_norm["scene_min"] = {
            aabb_min[0], aabb_min[1], aabb_min[2]
        };
        scene_norm["spatial_norm"] = spatialNormalization;
        std::ofstream sceneNormFile(
            (destinationFile.parent_path() / "scene_norm.json").string()
        );
        sceneNormFile << std::setw(4) << scene_norm << std::endl;

        typename Accelerator::AABB aabb;
        getAABB(aabb, aabb_min, aabb_max, spatialNormalization);
        m_accelerator = std::make_unique<Accelerator>(
            aabb, std::make_unique<SDMMContext>(m_maxSamplesSize)
        );
        m_accelerator->split_to_depth(3);

        bool success = true;

        // dumpScene(destinationFile.parent_path() / "scene.vio");

        Float totalElapsedSeconds = 0.f;
        auto stats = json::array();

        int originalSamplesPerIteration = m_config.samplesPerIteration;
        int iteration = 0;
        for(
            int samplesRendered = 0;
            samplesRendered < sampleCount;
            samplesRendered += m_config.samplesPerIteration
        ) {
            // if(samplesRendered == 0) {
            //     m_config.samplesPerIteration = 2;
            // } else {
            //     m_config.samplesPerIteration = originalSamplesPerIteration;
            // }
            m_still_training = samplesRendered < m_config.sampleCount / 3;

            // if(!m_still_training) {
            //     m_config.samplesPerIteration = sampleCount - samplesRendered;
            // }

            std::cerr <<
                "Render iteration " + std::to_string(iteration) + ".\n";

            ref<SDMMProcess> process = new SDMMProcess(
                job,
                queue,
                m_config,
                m_accelerator.get(),
                iteration,
                m_still_training
            );
            m_process = process;

            process->bindResource("scene", sceneResID);
            process->bindResource("sensor", sensorResID);
            process->bindResource("sampler", samplerResID);

            ref<Timer> timer = new Timer();
            scheduler->schedule(process);
            scheduler->wait(process);

            m_process = NULL;
            process->develop();

            success = success && (process->getReturnStatus() == ParallelProcess::ESuccess);
            if(!success) {
                break;
            }

            if(m_still_training) {
                optimize();
            }

            auto workResult = process->getResult();

            Float elapsedSeconds = timer->getSeconds();
            totalElapsedSeconds += elapsedSeconds;
            Float meanPathLength = workResult->averagePathLength / (float) workResult->pathCount;

            stats.push_back({
                {"iteration", iteration},
                {"elapsed_seconds", elapsedSeconds},
                {"total_elapsed_seconds", totalElapsedSeconds},
                {"mean_path_length", meanPathLength},
                {"spp", m_config.samplesPerIteration},
                {"total_spp", samplesRendered + m_config.samplesPerIteration}
            });

            workResult->dumpIndividual(
                m_config.samplesPerIteration,
                iteration,
                destinationFile.parent_path(),
                timer->lap()
            );
            // saveCheckpoint(destinationFile.parent_path(), iteration);
            ++iteration;
        }

        std::ofstream statsOutFile(
            (destinationFile.parent_path() / "stats.json").string()
        );

        statsOutFile << std::setw(4) << stats << std::endl;

		return success;
	}

    void dumpScene(const fs::path& path) {
        std::cerr << "Dumping scene description to " << path.string() << endl;

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

        std::vector<ref<TriMesh>> meshes;
        for (Shape* s : shapes) {
            if (s->isCompound()) {
                continue;
            }
            ref<TriMesh> mesh = s->createTriMesh();
            if (mesh) {
                meshes.emplace_back(mesh);
            }
        }

        // blob << (float) m_fieldOfView;

        // for (int i = 0; i < 4; ++i) {
        //     for (int j = 0; j < 4; ++j) {
        //         blob << (float) m_cameraMatrix(i, j);
        //     }
        // }

        // blob << (float) m_sceneAabb.min[0];
        // blob << (float) m_sceneAabb.min[1];
        // blob << (float) m_sceneAabb.min[2];

        // blob << (float) m_spatialNormalization;

        bool allMeshesHaveNormals = true;
        for (auto& mesh : meshes) {
            allMeshesHaveNormals = allMeshesHaveNormals && mesh->hasVertexNormals();
        }
        std::cerr << "All meshes have normals: " << allMeshesHaveNormals << ".\n";
        assert(allMeshesHaveNormals);

        vio::Scene vio_scene;
        for (auto& mesh : meshes) {
            SAssert(mesh->hasVertexNormals());

            auto vio_mesh = std::make_shared<vio::Mesh>();
            // std::cerr << "Added mesh!\n";

            size_t triangleCount = mesh->getTriangleCount();
            // std::cerr << "triangleCount=" << triangleCount << "!.\n";
            vio_mesh->indices().resize(3, triangleCount);
            // std::cerr << "Resized indices!\n";

            size_t vertexCount = mesh->getVertexCount();
            vio_mesh->positions().resize(3, vertexCount);
            vio_mesh->normals().resize(3, vertexCount);
            // std::cerr << "Resized positions and normals!\n";

            // Indices
            const Triangle* triangles = mesh->getTriangles();
            for (size_t i = 0; i < triangleCount; ++i) {
                vio_mesh->indices().col(i) <<
                    triangles[i].idx[0],
                    triangles[i].idx[1],
                    triangles[i].idx[2];
            }
            std::cerr << "Added indices!.\n";

            // Vertices and normals
            const Point* vertices = mesh->getVertexPositions();
            const Normal* normals = mesh->getVertexNormals();

            for (size_t i = 0; i < vertexCount; ++i) {
                vio_mesh->positions().col(i) <<
                    vertices[i].x, vertices[i].y, vertices[i].z;
                vio_mesh->normals().col(i) <<
                    normals[i].x, normals[i].y, normals[i].z;
            }
            // std::cerr << "Added positions and normals!\n";
            vio_scene.meshes().push_back(vio_mesh);
        }

        vio_scene.save(path.string());
    }

    MTS_DECLARE_CLASS()
private:
    ref<ParallelProcess> m_process;
    SDMMConfiguration m_config;

    uint32_t m_maxSamplesSize;
    bool m_still_training = false;

    Scene* m_scene;
    std::unique_ptr<Accelerator> m_accelerator;
    std::unique_ptr<tev::ThreadPool> m_thread_pool;
};

MTS_IMPLEMENT_CLASS_S(SDMMVolumetricPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(SDMMVolumetricPathTracer, "SDMM volumetric path tracer");
MTS_NAMESPACE_END
