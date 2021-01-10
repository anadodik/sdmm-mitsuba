#ifndef __SNTREE_H
#define __SNTREE_H

#include <vector>

#include <Eigen/Geometry>

namespace jmm {

template<typename Scalar, typename Value, int rows=4, int cols=5>
class NGridNode {
public:
    constexpr static int gridSize = rows * cols;
    NGridNode() : invCellSize(rows, cols) { }
    NGridNode(const Value& value) :
        invCellSize(rows, cols)
    {
        for(int grid_i = 0; grid_i < gridSize; ++grid_i) {
            std::cerr << "Making shared_ptr " << grid_i << ".\n";
            values[grid_i] = std::make_shared<Value>(value);
        }
    }

    std::shared_ptr<Value>& operator[](int i) { return values[i]; }
    Eigen::Vector2i toCellIdx(const Eigen::Matrix<Scalar, 2, 1>& normal) const {
        Eigen::Vector2i idx =
            (normal.array() * invCellSize.array()).floor().template cast<int>();
        idx(0) = std::min(idx(0), rows - 1);
        idx(1) = std::min(idx(1), cols - 1);
        return idx;
    }

    std::shared_ptr<Value> getValue(const Eigen::Matrix<Scalar, 2, 1>& normal) const {
        Eigen::Vector2i idx = toCellIdx(normal);
        return values[idx(0) * cols + idx(1)];
    }

    auto begin() const { return values.begin(); }
    auto end() const { return values.end(); }

    void clearValues() {
        for(int grid_i = 0; grid_i < gridSize; ++grid_i) {
            values[grid_i] = nullptr;
        }
    }

private:
    std::array<std::shared_ptr<Value>, rows * cols> values;
    Eigen::Matrix<Scalar, 2, 1> invCellSize;
};

template<typename Scalar, int t_dims, typename Value>
struct SNTreeNode  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using AABB = Eigen::AlignedBox<Scalar, 3>;
    using Vectord = Eigen::Matrix<Scalar, 3, 1>;
    SNTreeNode() : isLeaf(true), axis(0), children{0, 0} { }

    std::shared_ptr<Value> find(
        const Vectord& point,
        const jmm::aligned_vector<SNTreeNode>& nodes,
        AABB& foundAABB
    ) const {
        if(!aabb.contains(point.template topRows<3>())) {
            return nullptr;
        }

        if(isLeaf) {
            foundAABB = aabb;
            return normalsGrid.getValue(point.template bottomRows<2>());
        }

        for(int child_i = 0; child_i < 2; ++child_i) {
            if(
                auto found = nodes[children[child_i]].find(point, nodes, foundAABB);
                found != nullptr
            ) {
                return found;
            }
        }
        return nullptr;
    }

    bool isLeaf = true;
    int axis = 0;
    uint32_t idx = 0;
    std::array<uint32_t, 2> children;
    NGridNode<Scalar, Value> normalsGrid;
    AABB aabb;
};

template<typename Scalar, int t_dims, typename Value>
class SNTree {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using AABB = Eigen::AlignedBox<Scalar, 3>;
    using Vectord = Eigen::Matrix<Scalar, 3, 1>;

    SNTree(const AABB& aabb, const Value& value) {
        m_aabb = aabb;

        // Enlarge AABB to turn it into a cube. This has the effect
        // of nicer hierarchical subdivisions.
        Vectord size = m_aabb.max() - m_aabb.min();
        m_aabb.max() = m_aabb.min() + Vectord::Constant(size.maxCoeff());

        m_nodes.emplace_back();
        m_nodes[0].aabb = m_aabb;
        m_nodes[0].normalsGrid = NGridNode<Scalar, Value>(value);
    }

    std::shared_ptr<Value> find(const Vectord& point) const {
        AABB aabb;
        return m_nodes[0].find(point, m_nodes, aabb);
    }

    std::shared_ptr<Value> find(const Vectord& point, AABB& aabb) const {
        return m_nodes[0].find(point, m_nodes, aabb);
    }

    void insert(const Vectord& location, const Value& value) {
        m_nodes[0].normalsGrid = NGridNode<Scalar, Value>(value);
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
        Vectord mean = Vectord::Zero();
        Vectord var = Vectord::Zero();
        int totalNSamples = 0;
        for(auto& value : m_nodes[node_i].normalsGrid) {
            if(value == nullptr) {
                continue;
            }
            auto& samples = value->samples;
            totalNSamples += value->samples.size();
            mean += samples.samples.topLeftCorner(3, samples.size()).rowwise().sum();
            var.array() += samples.samples.topLeftCorner(3, samples.size()).array().square().rowwise().sum();
        }
        std::cerr << "Split mean: " << mean.transpose() << ", variance: " << var.transpose() << "totalNSamples: " << totalNSamples << "\n";
        mean /= (Scalar) totalNSamples;
        var /= (Scalar) totalNSamples;
        var = var.array() - mean.array().square();
        int maxVar = 0;
        for(int var_i = 0; var_i < 3; ++var_i) {
            assert(std::isfinite(var(var_i)));
            if(var(var_i) > var(maxVar)) {
                maxVar = var_i;
            }
        }

        Vectord aabb_min = m_nodes[node_i].aabb.min();
        Vectord aabb_diagonal = m_nodes[node_i].aabb.diagonal();
        Vectord splitLocation = (mean - aabb_min).array() / aabb_diagonal.array();
        Scalar location = splitLocation(maxVar);
        return {maxVar, location}; // std::min(std::max(location, 0.2), 0.8)};
    }

    SNTreeNode<Scalar, t_dims, Value> createChildNode(
        int node_i, int child_i, Scalar splitLocation
    ) {
        SNTreeNode<Scalar, t_dims, Value> child;

        // Set correct parameters for child node
        child.isLeaf = true;
        int axis = m_nodes[node_i].axis;
        child.axis = (axis + 1) % 3;
        child.aabb = m_nodes[node_i].aabb;
        if(child_i == 0) {
            child.aabb.min()(axis) += splitLocation * child.aabb.diagonal()(axis);
        } else {
            child.aabb.max()(axis) -= (1.f - splitLocation) * child.aabb.diagonal()(axis);
        }

        for(int grid_i = 0; grid_i < NGridNode<Scalar, Value>::gridSize; ++grid_i) {
            if(m_nodes[node_i].normalsGrid[grid_i] == nullptr) {
                continue;
            }
            auto childValue = std::make_shared<Value>();
            childValue->distribution = m_nodes[node_i].normalsGrid[grid_i]->distribution;
            childValue->optimizer = m_nodes[node_i].normalsGrid[grid_i]->optimizer;
            childValue->samples.reserve(m_nodes[node_i].normalsGrid[grid_i]->samples.capacity());
            for(int sample_i = 0; sample_i < m_nodes[node_i].normalsGrid[grid_i]->samples.size(); ++sample_i) {
                Vectord point = 
                    m_nodes[node_i].normalsGrid[grid_i]->samples.samples.col(sample_i).template topRows<3>();
                if(child.aabb.contains(point)) {
                    childValue->samples.push_back(m_nodes[node_i].normalsGrid[grid_i]->samples, sample_i);
                }
            }
            child.normalsGrid[grid_i] = childValue;
        }

        return child;
    }

    void split_to_depth(int maxDepth) {
        std::cerr << "Splitting to depth " << maxDepth << ".\n";
        split_to_depth_recurse(0, 0, maxDepth);
        std::cerr << "Done splitting.\n";
    }

    void split_to_depth_recurse(uint32_t node_i, int depth, int maxDepth) {
        std::cerr << "Nodes size: " << m_nodes.size() << ", depth: " << depth << "\n";
        int nextDepth = (m_nodes[node_i].axis == 3 - 1) ? (depth + 1) : depth;
        if(!m_nodes[node_i].isLeaf) {
            for (int child_i = 0; child_i < 2; ++child_i) {
                split_to_depth_recurse(
                    m_nodes[node_i].children[child_i], nextDepth, maxDepth
                );
            }
            return;
        }

        if(depth < maxDepth) {
            m_nodes[node_i].isLeaf = false;
            for (int child_i = 0; child_i < 2; ++child_i) {
                // Create node
                SNTreeNode<Scalar, t_dims, Value> child =
                    createChildNode(node_i, child_i, 0.5);

                // Insert child into vector
                uint32_t child_idx = m_nodes.size();
                child.idx = child_idx;
                m_nodes.push_back(child);
                m_nodes[node_i].children[child_i] = child_idx;
            }
            m_nodes[node_i].normalsGrid.clearValues();
            for (int child_i = 0; child_i < 2; ++child_i) {
                split_to_depth_recurse(
                    m_nodes[node_i].children[child_i], nextDepth, maxDepth
                );
            }
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

        int minNSamples = 0; // std::numeric_limits<int>::max();
        for(std::shared_ptr<Value> value : m_nodes[node_i].normalsGrid) {
            if(value == nullptr) {
                continue;
            }
            minNSamples = std::max(minNSamples, value->samples.size());
        }
        std::cerr << "Splitting " << minNSamples << ".\n";
        if(minNSamples > splitThreshold) {
            m_nodes[node_i].isLeaf = false;
            std::pair<int, Scalar> split = getSplitLocation(node_i);
            m_nodes[node_i].axis = split.first;
            for (int child_i = 0; child_i < 2; ++child_i) {
                SNTreeNode<Scalar, t_dims, Value> child =
                    createChildNode(node_i, child_i, split.second);

                // Insert child into vector
                uint32_t child_idx = m_nodes.size();
                child.idx = child_idx;
                m_nodes.push_back(child);
                m_nodes[node_i].children[child_i] = child_idx;
            }
            m_nodes[node_i].normalsGrid.clearValues();
            for (int child_i = 0; child_i < 2; ++child_i) {
                int child_idx = m_nodes[node_i].children[child_i];
                split_recurse(child_idx, splitThreshold);
            }
        }
        // std::cerr << "Split node " << idx << ", children ids = " << children[0] << ", " << children[1] << std::endl;
    }
    
private:

    jmm::aligned_vector<SNTreeNode<Scalar, t_dims, Value>> m_nodes;
    AABB m_aabb;
};


}
#endif // __SNTREE_H
