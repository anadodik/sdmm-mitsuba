#ifndef MESH_H
#define MESH_H

#include <Eigen/Dense>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/shared_ptr_helper.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "jmm/eigen_boost_serialization.h"

// https://eigen.tuxfamily.org/bz/show_bug.cgi?id=1676
namespace boost { namespace serialization {
   struct U;  // forward-declaration for Bug 1676
} } // boost::serialization

namespace Eigen { namespace internal {
  // Workaround for bug 1676
  template<>
  struct traits<boost::serialization::U> {enum {Flags=0};};
} }

namespace vio {

class Mesh {
public:
    using IndicesMatrix = Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic>;

    Mesh() {
    }

    virtual ~Mesh() {
    }

    const IndicesMatrix& indices() const { return m_indices; }
    IndicesMatrix& indices() { return m_indices; }

    const Eigen::MatrixXf& positions() const { return m_positions; }
    Eigen::MatrixXf& positions() { return m_positions; }

    const Eigen::MatrixXf& normals() const { return m_normals; }
    Eigen::MatrixXf& normals() { return m_normals; }

    const Eigen::MatrixXf& colors() const { return m_colors; }
    Eigen::MatrixXf& colors() { return m_colors; }

    const Eigen::MatrixXf& uvs() const { return m_uvs; }
    Eigen::MatrixXf& uvs() { return m_uvs; }

    int triangle_count() { return m_indices.cols(); }

    void save(const std::string& filename) const {
        std::ofstream ofs(filename.c_str());
        boost::archive::binary_oarchive oa(ofs);
        oa << BOOST_SERIALIZATION_NVP(*this);
    }

    void load(const std::string& filename) {
        std::ifstream ifs(filename.c_str());
        boost::archive::binary_iarchive ia(ifs);
        ia >> BOOST_SERIALIZATION_NVP(*this);
    }

protected:

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & m_indices;
        ar & m_positions;
        ar & m_normals;
        ar & m_colors;
        ar & m_uvs;
    }

    IndicesMatrix m_indices;
    Eigen::MatrixXf m_positions;
    Eigen::MatrixXf m_normals;
    Eigen::MatrixXf m_colors;
    Eigen::MatrixXf m_uvs;
};

class Scene {
public:
    void save(const std::string& filename) const {
        std::ofstream ofs(filename.c_str());
        boost::archive::binary_oarchive oa(ofs);
        oa << BOOST_SERIALIZATION_NVP(*this);
    }

    void load(const std::string& filename) {
        std::ifstream ifs(filename.c_str());
        boost::archive::binary_iarchive ia(ifs);
        ia >> BOOST_SERIALIZATION_NVP(*this);
    }

    const std::vector<std::shared_ptr<Mesh>>& meshes() const { return m_meshes; }
    std::vector<std::shared_ptr<Mesh>>& meshes() { return m_meshes; }

protected:

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & m_meshes;
    }

    std::vector<std::shared_ptr<Mesh>> m_meshes;
};

}

#endif /* MESH_H */
