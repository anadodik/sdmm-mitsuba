#ifndef __HASH_GRID_H
#define __HASH_GRID_H

#include <tuple>
#include <utility>

#include <Eigen/Geometry>
#include "folly/container/F14Map.h"
#include "folly/Hash.h"

template<int t_dims>
struct TupleKey;

template<int t_dims>
struct TupleKey {
    using KeyType = decltype(
        std::tuple_cat(
            std::declval<std::tuple<int>>(),
            std::declval<typename TupleKey<t_dims - 1>::KeyType>()
        )
    );
};

template<>
struct TupleKey<1> {
    using KeyType = std::tuple<int>;
};

template<typename Scalar, int t_dims, typename Value>
class HashGrid {
public:
    using Key = typename TupleKey<t_dims>::KeyType;
    using HashMap = folly::F14FastMap<Key, uint32_t, folly::hasher<Key>>;
    using Vectord = Eigen::Matrix<Scalar, t_dims, 1>;
    using Vectori = Eigen::Matrix<int, t_dims, 1>;
    using Iterator = typename HashMap::iterator;

    HashGrid(const Vectord& cellSize, const Vectord& gridStart) :
        m_cellSize(cellSize),
        m_gridStart(gridStart),
        m_invCellSize(1.f / cellSize.array()) { }

    auto insert(const Vectord& location, const Value& value) {
        const Key key = toCellIdx<t_dims>(location);
        m_data.push_back(std::make_shared<Value>(value));
        return m_map.insert({key, m_data.size() - 1});
    }

    std::shared_ptr<Value> find(const Vectord& location) {
        if(auto found = m_map.find(toCellIdx<t_dims>(location)); found != m_map.end()) {
            return m_data[found->second];
        } else {
            return nullptr;
        }
    }

    auto end() {
        return m_data.end();
    }

    auto begin() {
        return m_data.begin();
    }

    auto size() {
        return m_data.size();
    }

    const auto& data() { return m_data; }

    Vectord cellSize() const { return m_cellSize; }

	Eigen::Matrix<Scalar, 3, 1> cellPositionSize() const {
		return m_cellSize.template topRows<3>();
	}

private:
	template<int dims>
    FOLLY_ALWAYS_INLINE std::enable_if_t<dims == 3, Key> toCellIdx(const Vectord& vector) {
        Vectori cell = ((vector - m_gridStart).array() * m_invCellSize.array()).floor().template cast<int>();
        return {cell(0), cell(1), cell(2)};
    }

	template<int dims>
    FOLLY_ALWAYS_INLINE std::enable_if_t<dims == 5, Key> toCellIdx(const Vectord& vector) {
        Vectori cell = ((vector - m_gridStart).array() * m_invCellSize.array()).floor().template cast<int>();
        return {cell(0), cell(1), cell(2), cell(3), cell(4)};
    }

	template<int dims>
    FOLLY_ALWAYS_INLINE std::enable_if_t<dims == 6, Key> toCellIdx(const Vectord& vector) {
        Vectori cell = ((vector - m_gridStart).array() * m_invCellSize.array()).floor().template cast<int>();
        return {cell(0), cell(1), cell(2), cell(3), cell(4), cell(5)};
    }

    Vectord m_cellSize;
    Vectord m_gridStart;
    Vectord m_invCellSize;
    HashMap m_map;
    std::vector<std::shared_ptr<Value>> m_data;
};

/*
template<typename Scalar, int t_dims, typename Value>
class UniformGrid {
public:
    using Vectord = Eigen::Matrix<Scalar, t_dims, 1>;
    using Vectori = Eigen::Matrix<int, t_dims, 1>;

    UniformGrid(const Vectord& cellSize, const Vectord& gridStart) : // , const Vectord& gridEnd) :
        m_cellSize(cellSize),
        m_gridStart(gridStart),
        m_invCellSize(1.f / cellSize.array())
    {
        m_data.resize(1000001);
    }

    auto insert(const Vectord& location, const Value& value) {
        return m_data[sub2ind(location)] = {sub(location), value};
    }

    auto find(const Vectord& location) {
        int ind = sub2ind(location);
        if(m_data[ind].second.samples == nullptr) {
            return m_data.end();
        }
        return m_data.begin() + ind; 
    }

    auto end() {
        return m_data.end();
    }

    auto begin() {
        return m_data.begin();
    }

    auto size() {
        return m_data.size();
    }

    Vectord cellSize() const { return m_cellSize; }

	Eigen::Matrix<Scalar, 3, 1> cellPositionSize() const {
		return m_cellSize.template topRows<3>();
	}

private:
    FOLLY_ALWAYS_INLINE Vectori sub(const Vectord& vector) {
        return ((vector - m_gridStart).array() * m_invCellSize.array()).template cast<int>();
    }

    FOLLY_ALWAYS_INLINE int sub2ind(const Vectord& vector) {
        Vectori s = sub(vector);
        Vectori size;
        size.setConstant(100);
        s = s.array().min(size.array() - 1);
        return sub2ind(s, size);
    }

    int sub2ind(const Vectori& sub, const Vectori& sizes) {
      int skipSize = 1;
      int result = 0;

      for(int i = 0; i < t_dims; ++i) {
         result += sub[i] * skipSize;
         skipSize *= sizes[i];
      }

      return result;

   }

    std::vector<std::pair<Vectori, Value>> m_data;
    Vectord m_cellSize;
    Vectord m_invCellSize;
    Vectord m_gridStart;
};
*/

#endif // __HASH_GRID_H
