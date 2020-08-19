#ifndef __STREE_H
#define __STREE_H

#include <vector>

#include <Eigen/Geometry>

namespace jmm {

template<typename Scalar, int t_dims, typename Value>
struct STreeNode  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using AABB = Eigen::AlignedBox<Scalar, t_dims>;
    using Vectord = Eigen::Matrix<Scalar, t_dims, 1>;
    using Position = Eigen::Matrix<Scalar, 3, 1>;
    using Normal = Eigen::Matrix<Scalar, 3, 1>;
    STreeNode() : isLeaf(true), axis(0), children{0, 0}, value(nullptr) { }

    template<typename Vector>
    auto contains(const Vector& v) -> bool {
        return 
            v.x() > aabb.min()(0) &&
            v.y() > aabb.min()(1) &&
            v.z() > aabb.min()(2) &&
            v.x() < aabb.max()(0) &&
            v.y() < aabb.max()(1) &&
            v.z() < aabb.max()(2);
    }

    Value* find(
        const Vectord& point,
        const jmm::aligned_vector<STreeNode>& nodes,
        AABB& foundAABB
    ) const {
        // if(!aabb.contains(point)) {
        //     return nullptr;
        // }

        if(isLeaf) {
            foundAABB = aabb;
            return value.get();
        }

        int foundChild = -1;
        for(int child_i = 0; child_i < 2; ++child_i) {
            const int child_idx = children[child_i];
            if(child_i == 0) {
                if(nodes[child_idx].aabb.min()(axis) < point(axis)) {
                    foundChild = child_idx;
                    break;
                }
            } else {
                foundChild = child_idx;
            }
        }
        assert(foundChild != -1);
        if(
            auto found = nodes[foundChild].find(point, nodes, foundAABB);
            found != nullptr
        ) {
            return found;
        }
        return nullptr;
    }

    bool isLeaf = true;
    int axis = 0;
    uint32_t idx = 0;
    std::array<uint32_t, 2> children;
    // std::array<std::array<std::shared_ptr<Value>, 2>, 2> values;
    std::unique_ptr<Value> value;
    AABB aabb;
};

template<typename Scalar, int t_dims, typename Value>
class STree {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    constexpr static int dims = t_dims;
    using AABB = Eigen::AlignedBox<Scalar, t_dims>;
    using Vectord = Eigen::Matrix<Scalar, t_dims, 1>;
    using Position = Eigen::Matrix<Scalar, 3, 1>;
    using Normal = Eigen::Matrix<Scalar, 3, 1>;

    STree(const AABB& aabb, const Value& value) {
        m_aabb = aabb;

        // Enlarge AABB to turn it into a cube. This has the effect
        // of nicer hierarchical subdivisions.
        Vectord size = m_aabb.max() - m_aabb.min();
        m_aabb.max() = m_aabb.min() + Vectord::Constant(size.maxCoeff());

        m_nodes.emplace_back();
        m_nodes[0].aabb = m_aabb;
        m_nodes[0].value = std::make_unique<Value>(value);
    }

    Value* find(const Vectord& point) const {
        AABB aabb;
        return m_nodes[0].find(point, m_nodes, aabb);
    }

    Value* find(const Vectord& point, AABB& aabb) const {
        return m_nodes[0].find(point, m_nodes, aabb);
    }

    void insert(const Vectord& location, const Value& value) {
        if(m_nodes[0].value != nullptr) {
            return;
        }
        m_nodes[0].value = std::make_unique<Value>(value);
    }

    auto begin() {
        return m_nodes.begin();
    }

    auto end() {
        return m_nodes.end();
    }

    auto size() {
        return m_nodes.size();
    }

    const auto& data() {
        return m_nodes;
    }

    std::pair<int, Scalar> getSplitLocation(int node_i) {
        auto& data = m_nodes[node_i].value->data;
        auto mean_point = data.mean_point;
        auto mean_sqr_point = data.mean_sqr_point;
        auto var_point = (mean_sqr_point - enoki::sqr(mean_point / data.size) * data.size) / (float) (data.size - 1);
        mean_point /= data.size;
        mean_sqr_point /= data.size;

        size_t max_var_i = 0;
        for(size_t var_i = 0; var_i < 3; ++var_i) {
            assert(std::isfinite(var(var_i)));
            if(var_point.coeff(var_i) > var_point.coeff(max_var_i)) {
                max_var_i = var_i;
            }
        }

        Vectord aabb_min = m_nodes[node_i].aabb.min();
        Vectord aabb_diagonal = m_nodes[node_i].aabb.diagonal();
        // std::cerr <<
        // "var=" << var_point << 
        // ", mean=" << mean_point <<
        // ", data size=" << data.size << 
        // ", aabb_min=" << aabb_min.transpose() << 
        // ", aabb_diag=" << aabb_diagonal.transpose() <<
        // "\n";
        Scalar location = (mean_point.coeff(max_var_i) - aabb_min(max_var_i)) / aabb_diagonal(max_var_i);
        return {max_var_i, location};
    }

    STreeNode<Scalar, t_dims, Value> createChildNode(int node_i, int child_i, Scalar splitLocation) {
        STreeNode<Scalar, t_dims, Value> child;

        // Set correct parameters for child node
        child.isLeaf = true;
        int axis = m_nodes[node_i].axis;
        child.axis = (axis + 1) % t_dims;
        child.aabb = m_nodes[node_i].aabb;
        if(child_i == 0) {
            child.aabb.min()(axis) += splitLocation * child.aabb.diagonal()(axis);
        } else {
            child.aabb.max()(axis) -= (1.f - splitLocation) * child.aabb.diagonal()(axis);
        }

        auto childValue = std::make_unique<Value>();

        if(enoki::slices(m_nodes[node_i].value->sdmm) == 0) {
            childValue->distribution = m_nodes[node_i].value->distribution;
            childValue->optimizer = m_nodes[node_i].value->optimizer;
            childValue->samples.reserve(m_nodes[node_i].value->samples.capacity());
        }

        childValue->sdmm = m_nodes[node_i].value->sdmm;
        childValue->conditioner = m_nodes[node_i].value->conditioner;
        childValue->em = m_nodes[node_i].value->em;
        childValue->data.reserve(m_nodes[node_i].value->data.capacity);

        for(int sample_i = 0; sample_i < m_nodes[node_i].value->data.size; ++sample_i) {
            if(enoki::slices(m_nodes[node_i].value->sdmm) == 0) {
                Position position = 
                    m_nodes[node_i].value->samples.samples.col(sample_i).template topRows<3>();
                Normal normal =
                    m_nodes[node_i].value->samples.normals.col(sample_i);
                Vectord point;
                if(dims == 3) {
                    point << position;
                } else {
                    point << position, normal;
                }
                if(child.aabb.contains(point)) {
                    childValue->samples.push_back(
                        m_nodes[node_i].value->samples, sample_i
                    );
                }
            }
            if(child.contains(enoki::slice(m_nodes[node_i].value->data.point, sample_i))) {
                childValue->data.push_back(
                    enoki::slice(m_nodes[node_i].value->data, sample_i)
                );
            }
        }
        child.value = std::move(childValue);

        return child;
    }

    void split_to_depth(int maxDepth) {
        split_to_depth_recurse(0, 0, maxDepth);
    }

    void split_to_depth_recurse(uint32_t node_i, int depth, int maxDepth) {
        std::cerr << "Nodes size: " << m_nodes.size() << ", depth: " << depth << "\n";

        int maxAxis = (depth == 0) ? t_dims : 3;
        int nextDepth = (m_nodes[node_i].axis == maxAxis - 1) ? (depth + 1) : depth;
        if(!m_nodes[node_i].isLeaf) {
            for (int child_i = 0; child_i < 2; ++child_i) {
                split_to_depth_recurse(
                    m_nodes[node_i].children[child_i], nextDepth, maxDepth
                );
            }
            return;
        }

        if(depth >= maxDepth) {
            return;
        }

        m_nodes[node_i].isLeaf = false;
        for (int child_i = 0; child_i < 2; ++child_i) {
            // Create node
            STreeNode<Scalar, t_dims, Value> child =
                createChildNode(node_i, child_i, 0.5);

            // Insert child into vector
            uint32_t child_idx = m_nodes.size();
            child.idx = child_idx;
            m_nodes.push_back(std::move(child));
            m_nodes[node_i].children[child_i] = child_idx;
        }

        m_nodes[node_i].value->data = enoki::zero<decltype(Value::data)>(0);
        m_nodes[node_i].value->em = enoki::zero<decltype(Value::em)>(0);
        m_nodes[node_i].value = nullptr;

        for (int child_i = 0; child_i < 2; ++child_i) {
            int child_idx = m_nodes[node_i].children[child_i];
            split_to_depth_recurse(child_idx, nextDepth, maxDepth);
        }
    }

    void split(int splitThreshold) {
        split_recurse(0, splitThreshold);
    }

    void split_recurse(uint32_t node_i, int splitThreshold) {
        if(!m_nodes[node_i].isLeaf) {
            for (int child_i = 0; child_i < 2; ++child_i) {
                split_recurse(m_nodes[node_i].children[child_i], splitThreshold);
            }
            return;
        }

        if(m_nodes[node_i].value->data.size > splitThreshold) {
            m_nodes[node_i].isLeaf = false;
            std::pair<int, Scalar> split = getSplitLocation(node_i);
            m_nodes[node_i].axis = split.first;
            for (int child_i = 0; child_i < 2; ++child_i) {
                STreeNode<Scalar, t_dims, Value> child =
                    createChildNode(node_i, child_i, split.second);

                // Insert child into vector
                uint32_t child_idx = m_nodes.size();
                child.idx = child_idx;
                m_nodes.push_back(std::move(child));
                m_nodes[node_i].children[child_i] = child_idx;
            }
            m_nodes[node_i].value = nullptr;
            for (int child_i = 0; child_i < 2; ++child_i) {
                int child_idx = m_nodes[node_i].children[child_i];
                split_recurse(child_idx, splitThreshold);
            }
        }
        // std::cerr << "Split node " << idx << ", children ids = " << children[0] << ", " << children[1] << std::endl;
    }

    void set_initialized(bool initialized_) { initialized = initialized_; }

private:
    jmm::aligned_vector<STreeNode<Scalar, t_dims, Value>> m_nodes;
    AABB m_aabb;
    bool initialized = false;
};

}
#endif // __STREE_H
