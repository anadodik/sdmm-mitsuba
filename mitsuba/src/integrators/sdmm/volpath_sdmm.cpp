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

#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wpedantic"

#include "jmm/mixture_model.h"
#include "jmm/mixture_model_init.h"
#include "jmm/mixture_model_opt.h"
// #include "jmm/outlier_detection.h"

#include <nlohmann/json.hpp>

#include "jmm/kdtree-eigen/kdtree_eigen.h"

#include "mesh.h"

#pragma GCC diagnostic pop

#include <mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/bidir/util.h>
#include <mitsuba/bidir/path.h>
#include <mitsuba/bidir/edge.h>
#include "../../subsurface/bluenoise.h"

#include "sdmm_config.h"
#include "sdmm_proc.h"

MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLength("SDMM volumetric path tracer", "Average path length", EAverage);

class SDMMVolumetricPathTracer : public Integrator {
    using Scalar = typename SDMMProcess::Scalar;
    using SamplesType = typename jmm::Samples<SDMMProcess::t_dims, Scalar>;

    using GridCell = typename SDMMProcess::GridCell;
    using HashGridType = typename SDMMProcess::HashGridType;
	using GridKeyVector = typename SDMMProcess::HashGridType::Vectord;
        
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

    template<typename JMM, typename SDMM, size_t... Indices>
    void copy_means(JMM& jmm, SDMM& sdmm, std::index_sequence<Indices...>) {
        size_t NComponents = jmm.nComponents();
        for(size_t component_i = 0; component_i < NComponents; ++component_i) {
            enoki::slice(sdmm.tangent_space, component_i).set_mean(
                sdmm::embedded_s_t<SDMM>(
                    jmm.components()[component_i].mean()(Indices)...
                )
            );
        }
    }

    template<typename JMM, typename SDMM, size_t... Indices>
    void copy_covs(JMM& jmm, SDMM& sdmm, std::index_sequence<Indices...>) {
        size_t NComponents = jmm.nComponents();
        for(size_t component_i = 0; component_i < NComponents; ++component_i) {
            enoki::slice(sdmm.cov, component_i) = sdmm::matrix_s_t<SDMM>(
                jmm.components()[component_i].cov()(Indices)...
            );
        }
    }

    template<typename JMM, typename SDMM>
    void copy_sdmm(JMM& jmm, SDMM& sdmm) {
        size_t NComponents = jmm.nComponents();
        enoki::set_slices(sdmm, NComponents);
        copy_means(jmm, sdmm, std::make_index_sequence<SDMM::Embedded::Size>{});
        copy_covs(jmm, sdmm, std::make_index_sequence<SDMM::Tangent::Size * SDMM::Tangent::Size>{});
        for(size_t component_i = 0; component_i < NComponents; ++component_i) {
            enoki::slice(sdmm.weight.pmf, component_i) = jmm.weights()[component_i];
        }
        bool prepare_success = sdmm::prepare_vectorized(sdmm);
        assert(prepare_success);
    }

    void saveCheckpoint(const fs::path& experimentPath, int iteration) {
        fs::path checkpointsDir = experimentPath / "checkpoints";
        if(iteration == 0 && (!fs::is_directory(checkpointsDir) || !fs::exists(checkpointsDir))) {
            fs::create_directories(checkpointsDir);
        }
        // fs::path samplesPath = checkpointsDir / fs::path(
        //     formatString("samples_%05i.jmms", iteration)
        // );
        // fs::path distributionPath = checkpointsDir / fs::path(
        //     formatString("model_%05i.jmm", iteration)
        // );
        // m_samples->save(samplesPath.string());
    }

    GridCell initCell() {
        using JointSDMM = typename SDMMProcess::JointSDMM;
        constexpr static size_t NComponents = SDMMProcess::NComponents;
        Eigen::Matrix<Scalar, 5, 1> bPrior;// bPrior << 1e-3, 1e-3, 1e-3, 1e-5, 1e-5;
        GridCell cell;
        cell.samples.reserve(m_maxSamplesSize);
        cell.optimizer = SDMMProcess::StepwiseEMType(
            m_config.alpha, bPrior, 1.f / NComponents
        );
        cell.data.reserve(m_maxSamplesSize);
        cell.em = enoki::zero<SDMMProcess::EM>(SDMMProcess::NComponents);

        float weight_prior = 1.f / NComponents;
        float cov_prior_strength = 5.f / NComponents;
        sdmm::matrix_t<JointSDMM> cov_prior = enoki::zero<sdmm::matrix_t<JointSDMM>>(NComponents);
        for(size_t slice_i = 0; slice_i < NComponents; ++slice_i) {
            enoki::slice(cov_prior, slice_i) = sdmm::matrix_s_t<JointSDMM>(
                2e-3, 0, 0, 0, 0,
                0, 2e-3, 0, 0, 0,
                0, 0, 2e-3, 0, 0,
                0, 0, 0, 2e-4, 0,
                0, 0, 0, 0, 2e-4
            );
        }
        cell.em.set_priors(weight_prior, cov_prior_strength, cov_prior);
        return cell;
    }

    void initializeHashGridComponents() {
        constexpr static int nSpatialComponents = SDMMProcess::NComponents / 8;
        constexpr static Scalar depthPrior = 3e-2;
        // Scalar cellSize = m_grid->cellPositionSize()(0);
        for(auto& node : *m_grid) {
            if(!node.isLeaf) {
                continue;
            }
            HashGridType::AABB aabb = node.aabb;
            // for(std::shared_ptr<GridCell> cell : node.normalsGrid) {
            auto& cell = node.value;
            {
                if(cell == nullptr) {
                    continue;
                }
                if(
                    cell->samples.size() < 2 * nSpatialComponents ||
                    enoki::slices(cell->sdmm) > 0
                ) {
                    continue;
                }
                std::cerr << "Initializing grid cell.\n";
                std::function<Scalar()> rng = 
                    [samplerCopy = m_sampler]() mutable -> Scalar {
                        return samplerCopy->next1D();
                    };
                std::vector<jmm::SphereSide> sides(
                    cell->samples.size(), jmm::SphereSide::Top
                );
                auto& bPriors = cell->optimizer.getBPriors();
                auto& bDepthPriors = cell->optimizer.getBDepthPriors();
                jmm::uniformHemisphereInit(
                    cell->distribution,
                    bPriors,
                    bDepthPriors,
                    rng,
                    nSpatialComponents,
                    depthPrior,
                    3 * node.aabb.diagonal().maxCoeff() / (Scalar) nSpatialComponents,
                    cell->samples,
                    sides,
                    true
                );
                copy_sdmm(cell->distribution, cell->sdmm);
                cell->em.depth_prior = enoki::zero<decltype(cell->em.depth_prior)>(
                    cell->distribution.nComponents()
                );
                for(size_t prior_i = 0; prior_i < cell->distribution.nComponents(); ++prior_i) {
                    for(size_t r = 0; r < 3; ++r) {
                        for(size_t c = 0; c < 3; ++c) {
                            enoki::slice(cell->em.depth_prior, prior_i)(r, c) =
                                bDepthPriors[prior_i](r, c);
                        }
                    }
                }
                enoki::set_slices(cell->conditioner, enoki::slices(cell->sdmm));
                sdmm::prepare(cell->conditioner, cell->sdmm);
            }
        }
    }

    void optimizeHashGrid(int iteration) {
        using ConditionVectord = typename SDMMProcess::MM::ConditionVectord;
        constexpr static int t_conditionDims = SDMMProcess::t_conditionDims;
        int splitThreshold = 16000;

        std::cerr << "Splitting samples.\n";
        m_grid->split(splitThreshold);

        initializeHashGridComponents();
        auto& nodes = m_grid->data();

        std::cerr << "Optimizing guiding distribution: " << nodes.size() << " nodes in tree.\n";

        #pragma omp parallel for
        for(int cell_i = 0; cell_i < nodes.size(); ++cell_i) {
            if(!nodes[cell_i].isLeaf) {
                continue;
            }

            auto& cell = nodes[cell_i].value;
            if(cell == nullptr) {
                continue;
            }

            if(cell->data.size < 35 || enoki::slices(cell->sdmm) == 0) {
                continue;
            }

            sdmm::em_step(cell->sdmm, cell->em, cell->data);
            enoki::set_slices(cell->conditioner, enoki::slices(cell->sdmm));
            sdmm::prepare(cell->conditioner, cell->sdmm);

            cell->data.clear();
        }
        /*
        std::ofstream gridOut("grid.csv");
        for(int cell_i = 0; cell_i < nodes.size(); ++cell_i) {
            if(nodes[cell_i].value == nullptr) {
                continue;
            }
            Eigen::Matrix<Scalar, 3, 1> center = nodes[cell_i].aabb.center();
            gridOut <<
                center(0) <<
                "," <<
                center(1) <<
                "," <<
                center(2) <<
                "," <<
                nodes[cell_i].value->error <<
                "\n";
        }
        gridOut.close();
        */
    }

    template<typename AABBVec, int dims>
    void getAABB(
        Eigen::AlignedBox<Scalar, dims>& aabb,
        const AABBVec& aabb_min,
        const AABBVec& aabb_max,
        Scalar spatialNormalization
    );

    template<typename AABBVec>
    void getAABB(
        Eigen::AlignedBox<Scalar, 3>& aabb,
        const AABBVec& aabb_min,
        const AABBVec& aabb_max,
        Scalar spatialNormalization
    ) {
        Eigen::Matrix<Scalar, 3, 1> epsilon;
        epsilon.setConstant(Eigen::NumTraits<Scalar>::dummy_precision());
        aabb.min() << 0, 0, 0;
        aabb.max() <<
            aabb_max[0] - aabb_min[0],
            aabb_max[1] - aabb_min[1],
            aabb_max[2] - aabb_min[2];
        aabb.max().template topRows<3>() /= spatialNormalization;
        aabb.min() -= epsilon;
        aabb.max() += epsilon;
    }

    template<typename AABBVec>
    void getAABB(
        Eigen::AlignedBox<Scalar, 6>& aabb,
        const AABBVec& aabb_min,
        const AABBVec& aabb_max,
        Scalar spatialNormalization
    ) {
        Eigen::Matrix<Scalar, 6, 1> epsilon;
        epsilon.setConstant(Eigen::NumTraits<Scalar>::dummy_precision());
        aabb.min() << 0, 0, 0, -1, -1, -1;
        aabb.max() <<
            aabb_max[0] - aabb_min[0],
            aabb_max[1] - aabb_min[1],
            aabb_max[2] - aabb_min[2],
            1,
            1,
            1;
        aabb.max().template topRows<3>() /= spatialNormalization;
        aabb.min() -= epsilon;
        aabb.max() += epsilon;
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

        Thread::initializeOpenMP(nCores);

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
                std::to_string(sampleCount),
                std::to_string(m_config.samplesPerIteration)
            );
        }
		m_config.dump();

        const int nPixels = m_config.cropSize.x * m_config.cropSize.y;
        m_maxSamplesSize =
            nPixels * m_config.samplesPerIteration * m_config.savedSamplesPerPath;

        Properties props("independent");
        props.setInteger(
            "sampleCount", SDMMProcess::t_initComponents * SDMMProcess::t_dims
        );
        props.setInteger("dimension", SDMMProcess::t_dims);
        props.setInteger("seed", m_config.rngSeed);
        m_sampler = static_cast<Sampler*>(
            PluginManager::getInstance()->createObject(
                MTS_CLASS(Sampler), props
            )
        );

        const auto scene_aabb = m_scene->getAABBWithoutCamera();
        const auto aabb_min = scene_aabb.min;
        const auto aabb_max = scene_aabb.max;
        const auto aabb_extents = scene_aabb.getExtents();
        Float spatialNormalization = std::min(
            aabb_extents[0], std::min(aabb_extents[1], aabb_extents[2])
        );

        Eigen::AlignedBox<Scalar, HashGridType::dims> aabb;
        getAABB(aabb, aabb_min, aabb_max, spatialNormalization);

		// GridKeyVector cellSize;
		// Eigen::Matrix<Scalar, 3, 1> cellPositionSize;
		// cellPositionSize << 2e-1, 2e-1, 2e-1;
		// // cellPositionSize << aabb_extents[0], aabb_extents[1], aabb_extents[2];
		// // cellPositionSize *= 2e-1 / spatialNormalization;
		// Eigen::Matrix<Scalar, 2, 1> cellNormalsSize;
		// cellNormalsSize << 0.2, 0.2; // 0.3333334; // 0.5;
		// // Eigen::Matrix<Scalar, 3, 1> cellNormalsSize;
		// // cellNormalsSize << 1.1, 2.2, 2.2; // 1, 1, 1;
		// jmm::buildKey(cellPositionSize, cellNormalsSize, cellSize);
		// GridKeyVector gridOrigin;
		// gridOrigin.setZero();
        // // gridOrigin.template bottomRows<3>() << -1.1, -1.1, -1.1;
        // // m_grid = std::make_shared<HashGridType>(
        // //     cellSize,
        // //     gridOrigin
        // // );
        std::cerr << "Making SNTree.\n";
        m_grid = std::make_shared<HashGridType>(
            aabb,
            initCell()
        );
        m_grid->split_to_depth(2);

        m_diffuseDistribution = std::make_shared<SDMMProcess::MMDiffuse>();
        // m_diffuseDistribution->load("/home/anadodik/sdmm/scenes/cornell-box/diffuse.jmm");

        bool success = true;
        fs::path destinationFile = scene->getDestinationFile();
        
        dumpScene(destinationFile.parent_path() / "scene.vio");

        Float totalElapsedSeconds = 0.f;
        using json = nlohmann::json;
        auto stats = json::array();
        for(
            int samplesRendered = 0;
            samplesRendered < sampleCount;
            samplesRendered += m_config.samplesPerIteration
        ) {
            int iteration = samplesRendered / m_config.samplesPerIteration;
            m_still_training = samplesRendered <= m_config.sampleCount / 2;

            if(!m_still_training) {
                m_config.samplesPerIteration = sampleCount - samplesRendered;
            }

            std::cerr << 
                "Render iteration " + std::to_string(iteration) + ".\n";

            ref<SDMMProcess> process = new SDMMProcess(
                job,
                queue,
                m_config,
                m_grid,
                m_diffuseDistribution,
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

            #if SDMM_DEBUG == 1
                process->getResult()->dump(m_config, path.parent_path(), path.stem());
            #endif

            success = success && (process->getReturnStatus() == ParallelProcess::ESuccess);
            if(!success) {
                break;
            }

            if(m_still_training) {
                optimizeHashGrid(iteration);
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

    int m_maxSamplesSize;
    bool m_still_training = false;

    ref<Sampler> m_sampler;
    Scene* m_scene;
    std::shared_ptr<typename SDMMProcess::MMDiffuse> m_diffuseDistribution;
    std::shared_ptr<HashGridType> m_grid;
};

MTS_IMPLEMENT_CLASS_S(SDMMVolumetricPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(SDMMVolumetricPathTracer, "SDMM volumetric path tracer");
MTS_NAMESPACE_END
