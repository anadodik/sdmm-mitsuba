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

#if !defined(__PMC_PROC_H)
#define __PMC_PROC_H

#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wpedantic"

#include "jmm/mixture_model.h"
#include "jmm/mixture_model_init.h"
#include "jmm/mixture_model_opt.h"
#include "jmm/outlier_detection.h"

#include "jmm/kdtree-eigen/kdtree_eigen.h"

#pragma GCC diagnostic pop

#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/renderjob.h>
#include <mitsuba/core/bitmap.h>
#include "pmc_wr.h"
#include "pmc_config.h"

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

class PMCProcess : public BlockedRenderProcess {
public:
    constexpr static int t_dims = 6;
    constexpr static int t_conditionalDims = 3;
    constexpr static int t_conditionDims = t_dims - t_conditionalDims;
    constexpr static int t_initComponents = 36;
    constexpr static int t_components = 36;
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
        typename MM::Scalar,
        jmm::MultivariateTangentNormal,
        jmm::MultivariateNormal
    >;

    using RenderingSamplesType = RenderingSamples<t_dims, typename MM::Scalar>;

    struct GridCell {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        MM distribution;
        RenderingSamplesType samples;
        StepwiseEMType optimizer;
        Scalar error;
    };

    using HashGridType = jmm::STree<
        Scalar, 3, GridCell
    >;
	using GridKeyVector = typename HashGridType::Vectord;

    using MMCond = typename MM::ConditionalDistribution;
    
    using MMScalar = typename MM::Scalar;
    using Vectord = typename MM::Vectord;
    using Matrixd = typename MM::Matrixd;

    using ConditionalVectord = typename MMCond::Vectord;
    using ConditionalMatrixd = typename MMCond::Matrixd;

    PMCProcess(
        const RenderJob *parent,
        RenderQueue *queue,
        const PMCConfiguration &config,
        std::shared_ptr<MM> distribution,
        std::shared_ptr<HashGridType> grid,
        std::shared_ptr<MMDiffuse> diffuseDistribution,
        std::shared_ptr<RenderingSamplesType> samples,
        int iteration
    );

    inline const PMCWorkResult *getResult() const { return m_result.get(); }

    /// Develop the image
    void develop();

    /* ParallelProcess impl. */
    void processResult(const WorkResult *wr, bool cancelled);
    ref<WorkProcessor> createWorkProcessor() const;
    void bindResource(const std::string &name, int id);

    MTS_DECLARE_CLASS()
protected:
    /// Virtual destructor
    virtual ~PMCProcess() { }
private:
    ref<PMCWorkResult> m_result;
    Float m_averagePathLength = 0.f;
    int m_pathCount = 0;

    ref<Timer> m_refreshTimer;
    PMCConfiguration m_config;

    std::shared_ptr<MM> m_distribution;
    std::shared_ptr<HashGridType> m_grid;
    std::shared_ptr<MMDiffuse> m_diffuseDistribution;
    std::shared_ptr<RenderingSamplesType> m_samples;
    int m_iteration;
};

MTS_NAMESPACE_END

#endif /* __PMC_PROC */