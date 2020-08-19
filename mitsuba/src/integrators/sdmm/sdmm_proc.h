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

#if !defined(__SDMM_PROC_H)
#define __SDMM_PROC_H

#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wpedantic"

#include "jmm/mixture_model.h"
#include "jmm/mixture_model_init.h"
#include "jmm/mixture_model_opt.h"
#include "jmm/outlier_detection.h"

#include "jmm/kdtree-eigen/kdtree_eigen.h"

#pragma GCC diagnostic pop

#include <sdmm/distributions/sdmm.h>
#include <sdmm/distributions/sdmm_conditioner.h>
#include <sdmm/opt/em.h>
#include <sdmm/spaces/directional.h>
#include <sdmm/spaces/euclidian.h>
#include <sdmm/spaces/spatio_directional.h>
#include <enoki/random.h>

#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/renderjob.h>
#include <mitsuba/core/bitmap.h>
#include "sdmm_wr.h"
#include "sdmm_config.h"

#include "jmm/hash_grid.h"
#include "jmm/stree.h"
#include "jmm/sntree.h"

#include <deque>

MTS_NAMESPACE_BEGIN

template<int t_dims, typename Scalar>
using RenderingSamples = jmm::Samples<t_dims, Scalar>;

/* ==================================================================== */
/*                           Parallel process                           */
/* ==================================================================== */

class SDMMProcess : public BlockedRenderProcess {
public:
    constexpr static int t_dims = 6;
    constexpr static int t_conditionalDims = 3;
    constexpr static int t_conditionDims = t_dims - t_conditionalDims;
    constexpr static int t_initComponents = 24;
    constexpr static int t_components = 24;
    constexpr static bool USE_BAYESIAN = true;
    using Scalar = double;

    using MM = jmm::MixtureModel<
        t_dims,
        t_components,
        t_conditionalDims,
        Scalar,
        jmm::MultivariateTangentNormal,
        jmm::MultivariateNormal
    >;

    using MMDiffuse = jmm::MixtureModel<
        4,
        t_components,
        3,
        Scalar,
        jmm::MultivariateTangentNormal,
        jmm::MultivariateNormal
    >;

    using StepwiseEMType = jmm::StepwiseTangentEM<
        t_dims,
        t_components,
        t_conditionalDims,
        Scalar,
        jmm::MultivariateTangentNormal,
        jmm::MultivariateNormal
    >;

    using RenderingSamplesType = RenderingSamples<t_dims, typename MM::Scalar>;

    using MMCond = typename MM::ConditionalDistribution;
    
    using MMScalar = typename MM::Scalar;
    using Vectord = typename MM::Vectord;
    using Matrixd = typename MM::Matrixd;

    using ConditionalVectord = typename MMCond::Vectord;
    using ConditionalMatrixd = typename MMCond::Matrixd;

    constexpr static size_t PacketSize = 8;
    constexpr static size_t JointSize = 5;
    constexpr static size_t MarginalSize = 3;
    constexpr static size_t ConditionalSize = 2;
    constexpr static int NSamples = 1;
    constexpr static int NComponents = 24;
    static_assert(NComponents == t_components);
    static_assert(JointSize == MarginalSize + ConditionalSize);

    using Packet = enoki::Packet<float, PacketSize>;
    using Value = enoki::DynamicArray<Packet>;

    using JointTangentSpace = sdmm::SpatioDirectionalTangentSpace<
        sdmm::Vector<Value, JointSize + 1>, sdmm::Vector<Value, JointSize>
    >;
    using JointSDMM = sdmm::SDMM<
        sdmm::Matrix<Value, JointSize>, JointTangentSpace
    >;
    using MarginalTangentSpace = sdmm::EuclidianTangentSpace<
        sdmm::Vector<Value, MarginalSize>, sdmm::Vector<Value, MarginalSize>
    >;
    using MarginalSDMM = sdmm::SDMM<
        sdmm::Matrix<Value, MarginalSize>, MarginalTangentSpace
    >;
    using ConditionalTangentSpace = sdmm::DirectionalTangentSpace<
        sdmm::Vector<Value, ConditionalSize + 1>, sdmm::Vector<Value, ConditionalSize>
    >;
    using ConditionalSDMM = sdmm::SDMM<
        sdmm::Matrix<Value, ConditionalSize>, ConditionalTangentSpace
    >;

    using Conditioner = sdmm::SDMMConditioner<
        JointSDMM, MarginalSDMM, ConditionalSDMM
    >;

    using RNG = enoki::PCG32<float, NSamples>;

    using Data = sdmm::Data<JointSDMM>;

    using EM = sdmm::EM<JointSDMM>;

    struct MutexWrapper {
        MutexWrapper() = default;
        ~MutexWrapper() = default;
        MutexWrapper(const MutexWrapper& mutex_wrapper) { };
        MutexWrapper(MutexWrapper&& mutex_wrapper) { };
        MutexWrapper& operator=(const MutexWrapper& mutex_wrapper) { };
        MutexWrapper& operator=(MutexWrapper&& mutex_wrapper) { };
        std::mutex mutex;
    };

    struct GridCell {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        MM distribution;
        RenderingSamplesType samples;
        StepwiseEMType optimizer;
        Scalar error;
        
        JointSDMM sdmm;
        Conditioner conditioner;
        RNG rng;

        Data data;
        EM em;
        MutexWrapper mutex_wrapper;
    };

    using HashGridType = jmm::STree<
        Scalar, 3, GridCell
    >;
	using GridKeyVector = typename HashGridType::Vectord;

    SDMMProcess(
        const RenderJob *parent,
        RenderQueue *queue,
        const SDMMConfiguration &config,
        std::shared_ptr<HashGridType> grid,
        std::shared_ptr<MMDiffuse> diffuseDistribution,
        int iteration,
        bool collect_data
    );

    inline const SDMMWorkResult *getResult() const { return m_result.get(); }

    /// Develop the image
    void develop();

    /* ParallelProcess impl. */
    void processResult(const WorkResult *wr, bool cancelled);
    ref<WorkProcessor> createWorkProcessor() const;
    void bindResource(const std::string &name, int id);

    MTS_DECLARE_CLASS()
protected:
    /// Virtual destructor
    virtual ~SDMMProcess() { }
private:
    ref<SDMMWorkResult> m_result;
    Float m_averagePathLength = 0.f;
    int m_pathCount = 0;

    ref<Timer> m_refreshTimer;
    SDMMConfiguration m_config;

    std::shared_ptr<HashGridType> m_grid;
    std::shared_ptr<MMDiffuse> m_diffuseDistribution;
    int m_iteration;
    bool m_collect_data;
};

MTS_NAMESPACE_END

#endif /* __SDMM_PROC */
