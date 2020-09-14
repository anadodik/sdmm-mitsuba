#ifndef __STREE_H
#define __STREE_H

#include <vector>

#include <Eigen/Geometry>

namespace sdmm::linalg {

template<typename Scalar, int Size>
struct AABB {
    using Point = sdmm::Vector<Scalar, Size>;

    AABB() {}
    AABB(Point min_, Point max_) : min(min_), max(max_) { }

    template<typename PointIn>
    auto contains(const PointIn& point) -> bool {
        bool result = enoki::all(point > min) && enoki::all(point < max);
        return result;
    }

    auto diagonal() -> Point {
        return max - min;
    }

    Point min;
    Point max;
};

}

namespace jmm {

template<typename Scalar, int Size, typename Value>
struct STreeNode  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using AABB = sdmm::linalg::AABB<Scalar, Size>;
    using Point = typename AABB::Point;

    using Position = Eigen::Matrix<Scalar, 3, 1>;
    using Normal = Eigen::Matrix<Scalar, 3, 1>;
    STreeNode() : isLeaf(true), axis(0), children{0, 0}, value(nullptr), depth(-1) { }

    Value* find(
        const Point& point,
        const jmm::aligned_vector<STreeNode>& nodes,
        AABB& foundAABB
    ) const {
        if(isLeaf) {
            foundAABB = aabb;
            // spdlog::info(
            //     "aabb.min={}, aabb.max={}, data_aabb.min={}, data_aabb.max={}", 
            //     aabb.min, aabb.max, data_aabb.min, data_aabb.max
            // );
            return value.get();
        }

        int foundChild = -1;
        for(int child_i = 0; child_i < 2; ++child_i) {
            const int child_idx = children[child_i];
            if(child_i == 0) {
                if(nodes[child_idx].aabb.min.coeff(axis) < point.coeff(axis)) {
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
    AABB data_aabb;
    int depth;
};

template<typename Scalar, int Size, typename Value>
class STree {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Node = STreeNode<Scalar, Size, Value>;
    using AABB = typename Node::AABB;
    using Point = typename AABB::Point;

    using Vectord = Eigen::Matrix<Scalar, Size, 1>;
    using Position = Eigen::Matrix<Scalar, 3, 1>;
    using Normal = Eigen::Matrix<Scalar, 3, 1>;

    STree(const AABB& aabb, const Value& value) : m_aabb(aabb) {
        // Enlarge AABB to turn it into a cube. This has the effect
        // of nicer hierarchical subdivisions.
        Point diagonal = m_aabb.diagonal();
        m_aabb.max = m_aabb.min + enoki::full<Point>(enoki::hmax(diagonal));

        m_nodes.emplace_back();
        m_nodes[0].aabb = m_aabb;
        m_nodes[0].value = std::make_unique<Value>(value);
    }

    Value* find(const Vectord& point) const {
        AABB aabb;
        Point point_(point(0), point(1), point(2));
        return m_nodes[0].find(point_, m_nodes, aabb);
    }

    Value* find(const Vectord& point, AABB& aabb) const {
        Point point_(point(0), point(1), point(2));
        return m_nodes[0].find(point_, m_nodes, aabb);
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
        auto var_point = (mean_sqr_point - enoki::sqr(mean_point / data.stats_size) * data.stats_size) / (float) (data.stats_size - 1);
        mean_point /= (float) data.stats_size;
        mean_sqr_point /= (float) data.stats_size;

        size_t max_var_i = 0;
        for(size_t var_i = 0; var_i < 3; ++var_i) {
            assert(std::isfinite(var(var_i)));
            if(var_point.coeff(var_i) > var_point.coeff(max_var_i)) {
                max_var_i = var_i;
            }
        }

        Point aabb_min = m_nodes[node_i].aabb.min;
        Point aabb_diagonal = m_nodes[node_i].aabb.diagonal();
        // std::cerr <<
        //     "var=" << var_point << 
        //     ", mean=" << mean_point <<
        //     ", data size=" << data.stats_size << 
        //     ", aabb_min=" << aabb_min << 
        //     ", aabb_diag=" << aabb_diagonal <<
        //     "\n";
        Scalar location = (mean_point.coeff(max_var_i) - aabb_min.coeff(max_var_i)) / aabb_diagonal.coeff(max_var_i);
        return {max_var_i, location};
    }

    STreeNode<Scalar, Size, Value> createChildNode(int node_i, int child_i, Scalar splitLocation) {
        STreeNode<Scalar, Size, Value> child;

        // Set correct parameters for child node
        child.isLeaf = true;
        int axis = m_nodes[node_i].axis;
        child.axis = (axis + 1) % Size;
        child.aabb = m_nodes[node_i].aabb;
        child.data_aabb = m_nodes[node_i].data_aabb;
        if(child_i == 0) {
            child.aabb.min.coeff(axis) += splitLocation * child.aabb.diagonal().coeff(axis);
            child.data_aabb.min.coeff(axis) = child.aabb.min.coeff(axis);
        } else {
            child.aabb.max.coeff(axis) -= (1.f - splitLocation) * child.aabb.diagonal().coeff(axis);
            child.data_aabb.max.coeff(axis) = child.aabb.max.coeff(axis);
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
                Point point;
                Position position = 
                    m_nodes[node_i].value->samples.samples.col(sample_i).template topRows<3>();
                point = Point(position(0), position(1), position(2));
                if(child.aabb.contains(point)) {
                    childValue->samples.push_back(
                        m_nodes[node_i].value->samples, sample_i
                    );
                }
            }

            Point point(
                enoki::slice(m_nodes[node_i].value->data.point.coeff(0), sample_i),
                enoki::slice(m_nodes[node_i].value->data.point.coeff(1), sample_i),
                enoki::slice(m_nodes[node_i].value->data.point.coeff(2), sample_i)
            );
            if(child.aabb.contains(point)) {
                childValue->data.push_back(
                    enoki::slice(m_nodes[node_i].value->data, sample_i)
                );
            }
        }
        child.value = std::move(childValue);

        return child;
    }

    void split_to_depth(int maxDepth) {
        split_to_depth_recurse(0, 0, maxDepth, 0);
    }

    void split_to_depth_recurse(uint32_t node_i, int depth, int maxDepth, int recursion_depth) {
        std::cerr << "Nodes size: " << m_nodes.size() << ", depth: " << depth << "\n";

        int maxAxis = (depth == 0) ? Size : 3;
        int nextDepth = (m_nodes[node_i].axis == maxAxis - 1) ? (depth + 1) : depth;
        if(!m_nodes[node_i].isLeaf) {
            for (int child_i = 0; child_i < 2; ++child_i) {
                split_to_depth_recurse(
                    m_nodes[node_i].children[child_i], nextDepth, maxDepth, recursion_depth + 1
                );
            }
            return;
        }

        if(depth >= maxDepth) {
            return;
        }

        m_nodes[node_i].isLeaf = false;
        for (int child_i = 0; child_i < 2; ++child_i) {
            m_nodes[node_i].data_aabb = m_nodes[node_i].aabb;
            // Create node
            STreeNode<Scalar, Size, Value> child =
                createChildNode(node_i, child_i, 0.5);

            // Insert child into vector
            uint32_t child_idx = m_nodes.size();
            child.idx = child_idx;
            child.depth = recursion_depth;

            m_nodes.push_back(std::move(child));
            m_nodes[node_i].children[child_i] = child_idx;
        }

        m_nodes[node_i].value->data = enoki::zero<decltype(Value::data)>(0);
        m_nodes[node_i].value->em = enoki::zero<decltype(Value::em)>(0);
        m_nodes[node_i].value = nullptr;

        for (int child_i = 0; child_i < 2; ++child_i) {
            int child_idx = m_nodes[node_i].children[child_i];
            split_to_depth_recurse(child_idx, nextDepth, maxDepth, recursion_depth + 1);
        }
    }

    void split(int splitThreshold) {
        split_recurse(0, splitThreshold, 0);
    }

    void split_recurse(uint32_t node_i, int splitThreshold, int recursion_depth) {
        if(!m_nodes[node_i].isLeaf) {
            for (int child_i = 0; child_i < 2; ++child_i) {
                split_recurse(m_nodes[node_i].children[child_i], splitThreshold, recursion_depth + 1);
            }
            return;
        }

        if(m_nodes[node_i].value->data.stats_size > splitThreshold) {
            m_nodes[node_i].data_aabb = AABB(
                m_nodes[node_i].value->data.min_position,
                m_nodes[node_i].value->data.max_position
            );

            m_nodes[node_i].isLeaf = false;
            std::pair<int, Scalar> split = getSplitLocation(node_i);
            m_nodes[node_i].axis = split.first;
            for (int child_i = 0; child_i < 2; ++child_i) {
                STreeNode<Scalar, Size, Value> child =
                    createChildNode(node_i, child_i, split.second);

                // Insert child into vector
                uint32_t child_idx = m_nodes.size();
                child.idx = child_idx;
                child.depth = recursion_depth;
                m_nodes.push_back(std::move(child));
                m_nodes[node_i].children[child_i] = child_idx;
            }
            m_nodes[node_i].value->data.clear_stats();
            m_nodes[node_i].value->data.clear();
            m_nodes[node_i].value = nullptr;
            for (int child_i = 0; child_i < 2; ++child_i) {
                int child_idx = m_nodes[node_i].children[child_i];
                split_recurse(child_idx, splitThreshold, recursion_depth + 1);
            }
        }
        // std::cerr << "Split node " << idx << ", children ids = " << children[0] << ", " << children[1] << std::endl;
    }

    void set_initialized(bool initialized_) { initialized = initialized_; }

private:
    jmm::aligned_vector<STreeNode<Scalar, Size, Value>> m_nodes;
    AABB m_aabb;
    bool initialized = false;
};

}
#endif // __STREE_H
