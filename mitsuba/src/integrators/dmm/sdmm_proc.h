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

#include <Eigen/Dense>

#include <sdmm/distributions/sdmm.h>
#include <sdmm/distributions/dmm_context.h>
#include <sdmm/opt/em.h>
#include <sdmm/opt/init.h>
#include <sdmm/spaces/directional.h>
#include <sdmm/spaces/euclidian.h>
#include <sdmm/spaces/spatio_directional.h>
#include <sdmm/accelerators/dmm_spatial_tree.h>
#include <enoki/random.h>

#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/renderjob.h>
#include <mitsuba/core/bitmap.h>


#include "sdmm_wr.h"
#include "sdmm_config.h"

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                           Parallel process                           */
/* ==================================================================== */

class SDMMProcess : public BlockedRenderProcess {
public:
    constexpr static int t_dims = 6;
    constexpr static int t_conditionalDims = 3;
    constexpr static int t_conditionDims = t_dims - t_conditionalDims;
    constexpr static int t_components = 24;
    using Scalar = float;
    using Vectord = Eigen::Matrix<Scalar, t_dims, 1>;
    using ConditionalVectord = Eigen::Matrix<Scalar, t_conditionalDims, 1>;

    constexpr static size_t PacketSize = 8;
    constexpr static size_t JointSize = 5;
    constexpr static size_t MarginalSize = 3;
    constexpr static size_t ConditionalSize = 2;
    constexpr static int NComponents = t_components;
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
    using RNG = enoki::PCG32<float, 1>;
    using Data = sdmm::Data<JointSDMM>;
    using EM = sdmm::EM<JointSDMM>;

    using SDMMContext = sdmm::SDMMContext<JointSDMM, ConditionalSDMM, RNG>;
    using Accelerator = sdmm::accelerators::STree<Scalar, 3, SDMMContext>;

    SDMMProcess(
        const RenderJob *parent,
        RenderQueue *queue,
        const SDMMConfiguration &config,
        Accelerator* accelerator,
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

    Accelerator* m_accelerator;
    int m_iteration;
    bool m_collect_data;
};

MTS_NAMESPACE_END

#endif /* __SDMM_PROC */
