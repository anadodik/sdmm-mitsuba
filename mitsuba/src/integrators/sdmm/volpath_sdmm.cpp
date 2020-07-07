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
//
#include "jmm/nlohmann/json.hpp"

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
		m_config.sampleDirect = props.getBoolean("sampleDirect", true);
		m_config.showWeighted = props.getBoolean("showWeighted", false);
		m_config.samplesPerIteration = props.getInteger("samplesPerIteration", 8);
		m_config.useHierarchical = props.getBoolean("useHierarchical", true);
		m_config.sampleProduct = props.getBoolean("sampleProduct", false);
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

    void initializeJMM() {
        m_distribution->setNComponents(SDMMProcess::t_initComponents);
        std::function<Scalar()> rng = 
            [samplerCopy = m_sampler]() mutable -> Scalar {
                return samplerCopy->next1D();
            };
        std::vector<jmm::SphereSide> sides(
            m_samples->size(), jmm::SphereSide::Top
        );
        auto& bPriors = m_optimizer->getBPriors();
        auto& bDepthPriors = m_optimizer->getBDepthPriors();
        jmm::uniformHemisphereInit(
            *m_distribution,
            bPriors,
            bDepthPriors,
            rng,
            1, // 80, 170, 250, 360
            Scalar(3e-2),
            Scalar(5e-4),
            *m_samples,
            sides,
            true
        );

        bool success = m_distribution->configure();
        assert(success);
    }

    void saveCheckpoint(const fs::path& experimentPath, int iteration) {
        fs::path checkpointsDir = experimentPath / "checkpoints";
        if(iteration == 0 && (!fs::is_directory(checkpointsDir) || !fs::exists(checkpointsDir))) {
            fs::create_directories(checkpointsDir);
        }
        fs::path samplesPath = checkpointsDir / fs::path(
            formatString("samples_%05i.jmms", iteration)
        );
        fs::path distributionPath = checkpointsDir / fs::path(
            formatString("model_%05i.jmm", iteration)
        );
        m_samples->save(samplesPath.string());
        m_distribution->save(distributionPath.string());
    }

    void mergeReplayBufferSamples(int iteration) {
        m_mergedSamples.clear();
        int prioritySampleCount = 0;
        for(auto& samplesBatch : m_replayBuffer) {
            prioritySampleCount += samplesBatch.size();
        }
        m_mergedSamples.reserve(m_samples->size() + prioritySampleCount);

        m_mergedSamples.push_back(*m_samples);
        for(auto& samplesBatch : m_replayBuffer) {
            m_mergedSamples.push_back(samplesBatch);
        }
        
        Scalar weightsSum = 0.f;
        #pragma omp parallel for reduction(+: weightsSum)
        for(int sample_i = 0; sample_i < m_mergedSamples.size(); ++sample_i) {
            m_mergedSamples.weights(sample_i) =
                std::min(Scalar(1e-3), m_mergedSamples.weights(sample_i));
            weightsSum += m_mergedSamples.weights(sample_i);
        }

        #pragma omp parallel for 
        for(int sample_i = 0; sample_i < m_mergedSamples.size(); ++sample_i) {
            m_mergedSamples.weights(sample_i) /= weightsSum;
        }
    }

    void resetEM() {
        // if(m_iteration < m_config.batchIterations) {
        //     std::unique_ptr<StepwiseEMType> resetEM = std::make_unique<StepwiseEMType>(
        //         m_config.alpha, 1e-5, 1.f / (Scalar) t_initComponents
        //     );

        //     resetEM->getBPriors() = stepwiseEM->getBPriors();
        //     resetEM->getBDepthPriors() = stepwiseEM->getBDepthPriors();
        //     std::swap(stepwiseEM, resetEM);
        // }
    }

    void addSamplesToReplayBuffer(int iteration, Float nIterations) {
        if(!m_config.enablePER || iteration < m_config.initIterations || iteration > 7) {
            return;
        }
        std::cerr << "Adding samples to replay buffer.\n";
        int remainingSamplesCount =
            (int) (m_samples->size() * m_config.resampleProportion);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> randomUniform(
            remainingSamplesCount, 1
        );
        for(int sample_i = 0; sample_i < remainingSamplesCount; ++sample_i) {
            randomUniform(sample_i) = m_sampler->next1D();
        }
        m_replayBuffer.emplace_back();
        auto& sampleBatch = m_replayBuffer.back();
        sampleBatch = std::move(m_samples->prioritizedSample(
            jmm::sarsaError(*m_samples, *m_distribution),
            randomUniform,
            (iteration - m_config.initIterations) / 
            (nIterations - m_config.initIterations)
        ));
        if(m_replayBuffer.size() > m_config.replayBufferLength) {
            m_replayBuffer.pop_front();
        }
    }

    // void initializeHashGrid() {
    //     const int nPixels = m_config.cropSize.x * m_config.cropSize.y;
    //     for(int sample_i = 0; sample_i < m_samples->size(); ++sample_i) {
	// 		GridKeyVector key;
	// 		jmm::buildKey(
    //             m_samples->samples.col(sample_i),
    //             m_samples->normals.col(sample_i),
    //             key
    //         );
    //         if(auto found = m_grid->find(key); found == nullptr) {
    //             m_grid->insert(key, initCell());
    //         } else {
    //             found->samples.push_back(*m_samples, sample_i);
    //         }
    //     }
    //     std::cerr << "Number of grid cells: " << m_grid->size() << std::endl;
    // }

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

    GridCell initCell() {
        Eigen::Matrix<Scalar, 5, 1> bPrior;// bPrior << 1e-3, 1e-3, 1e-3, 1e-5, 1e-5;
        GridCell cell;
        cell.samples.reserve(m_maxSamplesSize);
        cell.optimizer = SDMMProcess::StepwiseEMType(
            m_config.alpha, bPrior, 1e-3
        );
        return cell;
    }

    void blueNoiseHashGridInit() {
        using ConditionVectord = typename SDMMProcess::MM::ConditionVectord;
        const auto scene_aabb = m_scene->getAABBWithoutCamera();
        const auto aabb_min = scene_aabb.min;
        const auto aabb_extents = scene_aabb.getExtents();
        Float spatialNormalization = std::min(
            aabb_extents[0], std::min(aabb_extents[1], aabb_extents[2])
        );
        Float radius = spatialNormalization / 100.f;
        std::cerr << "Spatial radius: " << radius << std::endl; 

        std::vector<Shape*> shapes;
        std::vector<BSDF*> bsdfs;
        getShapesAndBsdfs(shapes, bsdfs);

        AABB aabb;
        Float sa;
        ref<PositionSampleVector> points = new PositionSampleVector();
        blueNoisePointSet(m_scene, shapes, radius, points, sa, aabb, nullptr);

        Log(EInfo, "Generated " SIZE_T_FMT " blue-noise points.", points->size());

        for(int point_i = 0; point_i < points->size(); ++point_i) {
            // TODO: initialize on both sides if transmissive.
            Eigen::Matrix<Scalar, 3, 1> position, normal;
            position <<
                ((*points)[point_i].p.x - aabb_min[0]) / spatialNormalization,
                ((*points)[point_i].p.y - aabb_min[1]) / spatialNormalization,
                ((*points)[point_i].p.z - aabb_min[2]) / spatialNormalization
            ;
            normal <<
                (*points)[point_i].n.x,
                (*points)[point_i].n.y,
                (*points)[point_i].n.z
            ;
			GridKeyVector key;
			jmm::buildKey(position, normal, key);
            if(auto found = m_grid->find(key); found != nullptr) {
                continue;
            }
            // std::cerr << "Inserting into grid: " << m_grid->size() << std::endl;
            m_grid->insert(key, initCell());
        }
        std::cerr << "Number of grid cells: " << m_grid->size() << std::endl;
    }

    void initializeHashGridComponents() {
        constexpr static int nSpatialComponents = 3;
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
                    cell->distribution.nComponents() > 0
                ) {
                    continue;
                }
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
            }
        }
    }

    void optimizeHashGrid(int iteration) {
        std::cerr << "Splitting samples.\n";
        using ConditionVectord = typename SDMMProcess::MM::ConditionVectord;
        constexpr static int t_conditionDims = SDMMProcess::t_conditionDims;
        int splitThreshold = 16000;

        if(iteration * m_config.samplesPerIteration > m_config.sampleCount / 2) {
            return;
        }

        m_grid->split(splitThreshold);

        initializeHashGridComponents();
        auto& nodes = m_grid->data();

        std::cerr << "Optimizing guiding distribution: " << nodes.size() << " nodes in tree.\n";

        #pragma omp parallel for
        for(int cell_i = 0; cell_i < nodes.size(); ++cell_i) {
            if(!nodes[cell_i].isLeaf) {
                continue;
            }
            // for(std::shared_ptr<GridCell> cell : nodes[cell_i].normalsGrid) {
            auto& cell = nodes[cell_i].value;
            {
                if(cell == nullptr) {
                    continue;
                }

                if(cell->samples.size() < 20) {
                    continue;
                }
                    
                if(cell->distribution.nComponents() == 0) {
                    cell->samples.clear();
                    continue;
                }
                // Eigen::Matrix<Scalar, Eigen::Dynamic, 1> randomUniform(m_gridSamples[cell_i]->size(), 1);
                // for(int sample_i = 0; sample_i < m_gridSamples[cell_i]->size(); ++sample_i) {
                //     randomUniform(sample_i) = m_sampler->next1D();
                // }
                // m_gridSamples[cell_i]->russianRoulette(randomUniform, 100, false);
                // cell->samples.clear();
                Scalar error;
                // jmm::normalizeModel(cell->samples, cell->distribution);
                cell->optimizer.optimize(cell->distribution, cell->samples, error);
                cell->error = error;
                cell->samples.clear();
            }
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

        m_distribution = std::make_shared<SDMMProcess::MM>();
        m_diffuseDistribution = std::make_shared<SDMMProcess::MMDiffuse>();
        // m_diffuseDistribution->load("/home/anadodik/sdmm/scenes/cornell-box/diffuse.jmm");
        m_optimizer = std::make_shared<SDMMProcess::StepwiseEMType>(
            m_config.alpha,
            Eigen::Matrix<Scalar, 5, 1>::Constant(1e-5),
            1.f / Scalar(SDMMProcess::t_initComponents)
        );

        m_samples = std::make_shared<SDMMProcess::RenderingSamplesType>();
        m_samples->reserve(m_maxSamplesSize);
        m_samples->clear();
        // blueNoiseHashGridInit();

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
            std::cerr << 
                "Render iteration " + std::to_string(iteration) + ".\n";

            m_samples->clear();
            ref<SDMMProcess> process = new SDMMProcess(
                job,
                queue,
                m_config,
                m_distribution,
                m_grid,
                m_diffuseDistribution,
                m_samples,
                iteration
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

            // if(m_config.correctStateDensity) {
            //     jmm::estimateStateDensity<
            //         SDMMProcess::t_dims,
            //         SDMMProcess::t_conditionDims,
            //         Scalar
            //     >(*m_samples);
            //     // m_samples->stateDensities.topRows(m_samples->size()).setOnes();
            // } else {
            //     m_samples->stateDensities.topRows(m_samples->size()).setOnes();
            // }

            if(iteration == 0) {
                // initializeJMM();
                // saveCheckpoint(destinationFile.parent_path(), 0);
            }
            // initializeHashGrid();

            if(samplesRendered + m_config.samplesPerIteration < sampleCount) {
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

            // mergeReplayBufferSamples(iteration);
            // int emIterations = 1;
            // if(iteration < m_config.initIterations) {
            //     emIterations = 2;
            // }
            // m_optimizer->optimize(*m_distribution, *m_samples, emIterations);
            // addSamplesToReplayBuffer(iteration, sampleCount / (Float) m_config.samplesPerIteration);
            // saveCheckpoint(destinationFile.parent_path(), iteration + 1);
            // m_distribution->resetSamplesRequested();
            // m_distribution->resetComponentsPerSample();
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

    ref<Sampler> m_sampler;
    Scene* m_scene;
    std::shared_ptr<typename SDMMProcess::MM> m_distribution;
    std::shared_ptr<typename SDMMProcess::MMDiffuse> m_diffuseDistribution;
    std::shared_ptr<typename SDMMProcess::RenderingSamplesType> m_samples;
    std::deque<SamplesType> m_replayBuffer;
    SamplesType m_mergedSamples;

    std::shared_ptr<typename SDMMProcess::StepwiseEMType> m_optimizer;
    std::shared_ptr<HashGridType> m_grid;
};

MTS_IMPLEMENT_CLASS_S(SDMMVolumetricPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(SDMMVolumetricPathTracer, "SDMM volumetric path tracer");
MTS_NAMESPACE_END
