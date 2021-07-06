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

using json = nlohmann::json;

MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLength("SDMM volumetric path tracer", "Average path length", EAverage);
static StatsCounter avgInvalidSamples("SDMM path tracer", "Average proportion of discarded samples.", EAverage);

class SDMMVolumetricPathTracer : public Integrator {
    using Scalar = typename SDMMProcess::Scalar;

    using SDMMContext = typename SDMMProcess::SDMMContext;
    using Accelerator = typename SDMMProcess::Accelerator;
    using AcceleratorNode = typename Accelerator::Node;
    using AcceleratorPoint = typename Accelerator::Point;

public:
    SDMMVolumetricPathTracer(const Properties &props) : Integrator(props) {
        /* Load the parameters / defaults */
        m_config.strictNormals = props.getBoolean("strictNormals", true);
        m_config.maxDepth = props.getInteger("maxDepth", -1);
        m_config.rrDepth = props.getInteger("rrDepth", 5);
        m_config.blockSize = props.getInteger("blockSize", -1);

        m_config.samplesPerIteration = props.getInteger("samplesPerIteration", 8);
        m_config.sampleProduct = props.getBoolean("sampleProduct", false);
        m_config.bsdfOnly = props.getBoolean("bsdfOnly", false);
        m_config.savedSamplesPerPath = props.getInteger("savedSamplesPerPath", 8);

        m_config.flushDenormals = props.getBoolean("flushDenormals", true);
        m_config.optimizeAsync = props.getBoolean("optimizeAsync", true);
        // TODO: make a SDMMFactory for different configurations:
        //       24 components
        //       16 components
        //       Directional
        //       OffsetDirectional

        // m_config.sampleDirect = props.getBoolean("sampleDirect", true);
        // m_config.alpha = props.getFloat("alpha", 0.5f);
        // m_config.correctSpatialDensity = props.getBoolean("correctSpatialDensity", true);

        m_config.dump();

        if (m_config.rrDepth <= 0)
            Log(EError, "'rrDepth' must be set to a value greater than zero!");

        if (m_config.maxDepth <= 0 && m_config.maxDepth != -1)
            Log(EError, "'maxDepth' must be set to -1 (infinite) or a value greater than zero!");

        if (m_config.maxDepth != m_config.rrDepth)
            Log(EError, "'maxDepth' must match 'rrDepth' for the SDMM integrator!");

        spdlog::info("Max packet size={}", enoki::max_packet_size);
        if(m_config.flushDenormals) {
            enoki::set_flush_denormals(true);
        }
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
        if(!fs::is_directory(checkpointsDir) || !fs::exists(checkpointsDir)) {
            fs::create_directories(checkpointsDir);
        }
        fs::path distributionPath = checkpointsDir / fs::path(
            formatString("model_%05i.asdmm", iteration)
        );
        sdmm::save_json(m_accelerator, distributionPath.string());
    }

    void initializeSDMMContext(SDMMProcess::SDMMContext* context, Scalar maxDiagonal) {
        constexpr static int n_spatial_components = SDMMProcess::NComponents / 8;
        float spatial_distance = 3 * maxDiagonal / n_spatial_components;
        sdmm::initialize(context->sdmm, context->em, context->training_data, context->rng, n_spatial_components, spatial_distance);
        enoki::set_slices(context->conditioner, enoki::slices(context->sdmm));
        // sdmm::prepare(context->conditioner, context->sdmm);
    }

    bool canBeOptimized(const AcceleratorNode& node) {
        const auto& context = node.value;
        return (
            node.is_leaf &&
            context != nullptr &&
            context->data.size >= 16
        );
    }

    struct TreeStats {
        int samples_count = 0;
        int leaf_nodes_count = 0;
        int active_nodes_count = 0;
        int max_depth = 0;
        int optimized_nodes_count = 0;
    };

    TreeStats compute_tree_stats() {
        auto& nodes = m_accelerator->data();
        TreeStats tree_stats;
        for(size_t context_i = 0; context_i < nodes.size(); ++context_i) {
            if(nodes[context_i].is_leaf) {
                ++tree_stats.leaf_nodes_count;
                if(nodes[context_i].value != nullptr) {
                    ++tree_stats.active_nodes_count;
                }
                tree_stats.samples_count += nodes[context_i].value->data.size;
                if(tree_stats.max_depth < nodes[context_i].depth) {
                    tree_stats.max_depth = nodes[context_i].depth;
                }
            }
            if(canBeOptimized(nodes[context_i])) {
                ++tree_stats.optimized_nodes_count;
            }
        }
        return tree_stats;
    }

    TreeStats optimize_async_run() {
        spdlog::info("Splitting samples.");
        if (m_accelerator->leaf_nodes() <= 2048) {
            m_accelerator->split(m_splitThreshold);
        }

        auto& nodes = m_accelerator->data();
        m_node_idcs.reserve(nodes.size());
        m_node_idcs.clear();

        TreeStats tree_stats;
        for(size_t context_i = 0; context_i < nodes.size(); ++context_i) {
            if(nodes[context_i].is_leaf) {
                ++tree_stats.leaf_nodes_count;
                if(nodes[context_i].value != nullptr) {
                    ++tree_stats.active_nodes_count;
                }
                tree_stats.samples_count += nodes[context_i].value->data.size;
                if(tree_stats.max_depth < nodes[context_i].depth) {
                    tree_stats.max_depth = nodes[context_i].depth;
                }
            }
            if(!canBeOptimized(nodes[context_i])) {
                continue;
            }
            m_node_idcs.push_back(context_i);
        }
        tree_stats.optimized_nodes_count = m_node_idcs.size();

        if(m_node_idcs.size() == 0) {
            return tree_stats;
        } else {
            optimization_running = true;
        }

        m_thread_pool->parallelFor(0, (int) m_node_idcs.size(), [&nodes, this](int i){
            size_t context_i = m_node_idcs[(size_t) i];
            auto& context = nodes[context_i].value;
            std::swap(context->data, context->training_data);
            // context->stats.clear();
        });

        m_thread_pool->parallelForNoWait(0, (int) m_node_idcs.size(), [&nodes, this](int i){
            size_t context_i = m_node_idcs[(size_t) i];
            auto& context = nodes[context_i].value;
            if(enoki::slices(context->sdmm) == 0) {
                initializeSDMMContext(context.get(), enoki::hmax(nodes[context_i].aabb.diagonal()));
            }
            sdmm::em_step(context->sdmm, context->em, context->training_data);
            context->training_data.clear();
            // context->training_data.clear_stats();
            context->update_ready = true;
        });
        return tree_stats;
    }

    void optimize_async_wait_and_update() {
        m_thread_pool->waitUntilFinished();
        optimization_running = false;

        auto& nodes = m_accelerator->data();
        m_thread_pool->parallelFor(0, (int) m_node_idcs.size(), [&nodes, this](int i){
            size_t context_i = m_node_idcs[(size_t) i];
            auto& context = nodes[context_i].value;
            if(context->update_ready) {
                enoki::set_slices(context->conditioner, enoki::slices(context->sdmm));
                sdmm::prepare(context->conditioner, context->sdmm);
                context->update_ready = false;
                context->initialized = true;
            }
        });
    }

    void optimize() {
        spdlog::info("Splitting samples.");
        m_accelerator->split(m_splitThreshold);

        auto& nodes = m_accelerator->data();
        m_node_idcs.reserve(nodes.size());
        m_node_idcs.clear();
        int total_n_samples = 0;
        int max_depth = 0;
        for(size_t context_i = 0; context_i < nodes.size(); ++context_i) {
            if(nodes[context_i].is_leaf) {
                total_n_samples += nodes[context_i].value->data.size;
                if(max_depth < nodes[context_i].depth) {
                    max_depth = nodes[context_i].depth;
                }
            }
            if(!canBeOptimized(nodes[context_i])) {
                continue;
            }
            m_node_idcs.push_back(context_i);
        }

        std::cerr << "Total number of samples: " << total_n_samples << ".\n";
        std::cerr << "Maximum node depth: " << max_depth << ".\n";
        std::cerr << "Optimizing guiding distribution: " << m_node_idcs.size() << " distributions in tree.\n";

        m_thread_pool->parallelFor(0, (int) m_node_idcs.size(), [&nodes, this](int i){
            size_t context_i = m_node_idcs[(size_t) i];
            auto& context = nodes[context_i].value;
            std::swap(context->data, context->training_data);
            if(enoki::slices(context->sdmm) == 0) {
                initializeSDMMContext(context.get(), enoki::hmax(nodes[context_i].aabb.diagonal()));
            }
            sdmm::em_step(context->sdmm, context->em, context->training_data);
            enoki::set_slices(context->conditioner, enoki::slices(context->sdmm));
            sdmm::prepare(context->conditioner, context->sdmm);
            context->training_data.clear();
            context->data.clear();
            context->initialized = true;
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
        m_scene = scene;
        ref<Scheduler> scheduler = Scheduler::getInstance();
        ref<Sensor> sensor = scene->getSensor();
        const Film *film = sensor->getFilm();
        size_t sampleCount = scene->getSampler()->getSampleCount();
        size_t nCores = scheduler->getCoreCount();

        m_thread_pool = std::make_unique<tev::ThreadPool>((int) nCores);

        Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " samples, " SIZE_T_FMT
            " %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y,
            sampleCount, nCores, nCores == 1 ? "core" : "cores");

        m_config.imageSize = film->getCropSize();
        m_config.blockSize = scene->getBlockSize();
        m_config.sampleCount = sampleCount;
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

        const int nPixels = m_config.imageSize.x * m_config.imageSize.y;
        m_maxSamplesSize = 2000000;
            // nPixels * m_config.samplesPerIteration * m_config.savedSamplesPerPath;
        std::cerr << "Maximum number of samples possible: " << m_maxSamplesSize << "\n";

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
        std::cerr << "Splitting to depth...\n";
        m_accelerator->split_to_depth(3);
        std::cerr << "Done splitting to depth.\n";

        // dumpScene(destinationFile.parent_path() / "scene.vio");

        Float totalElapsedSeconds = 0.f;
        auto stats = json::array();
        int iteration = 0;
        bool success = true;
        for(
            int samplesRendered = 0;
            samplesRendered < sampleCount;
            samplesRendered += m_config.samplesPerIteration
        ) {
            m_still_training = !m_config.bsdfOnly && samplesRendered < m_config.sampleCount / 3;

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

            TreeStats tree_stats;
            if(m_config.optimizeAsync) {
                if(optimization_running) {
                    optimize_async_wait_and_update();
                }
                if(m_still_training) {
                    tree_stats = optimize_async_run();
                } else {
                    tree_stats = compute_tree_stats();
                }
            } else {
                if(m_still_training) {
                    optimize();
                }
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
                {"total_spp", samplesRendered + m_config.samplesPerIteration},

                {"samples_count", tree_stats.samples_count},
                {"leaf_nodes_count", tree_stats.leaf_nodes_count},
                {"max_depth", tree_stats.max_depth},
                {"optimized_nodes_count", tree_stats.optimized_nodes_count},
                {"active_nodes_count", tree_stats.active_nodes_count}
            });

            workResult->dumpIndividual(
                m_config.samplesPerIteration,
                iteration,
                destinationFile.parent_path(),
                timer->lap()
            );
            ++iteration;
        }

        std::ofstream statsOutFile(
            (destinationFile.parent_path() / "stats.json").string()
        );
        statsOutFile << std::setw(4) << stats << std::endl;
        saveCheckpoint(destinationFile.parent_path(), iteration);

        return success;
    }

    MTS_DECLARE_CLASS()
private:
    ref<ParallelProcess> m_process;
    SDMMConfiguration m_config;

    uint32_t m_maxSamplesSize;
    bool m_still_training = false;
    std::vector<size_t> m_node_idcs;
    bool optimization_running = false;
    int m_splitThreshold = 16000;

    Scene* m_scene;
    std::unique_ptr<Accelerator> m_accelerator;
    std::unique_ptr<tev::ThreadPool> m_thread_pool;
};

MTS_IMPLEMENT_CLASS_S(SDMMVolumetricPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(SDMMVolumetricPathTracer, "SDMM volumetric path tracer");
MTS_NAMESPACE_END
