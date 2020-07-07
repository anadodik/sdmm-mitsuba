#ifndef __UTILS_H
#define __UTILS_H

#include <algorithm>
#include <numeric>

#include "fastonebigheader.h"

namespace jmm {

// TODO: create DiscreteDistribution

inline float tsqrtf(float s) {
    return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set1_ps(s)));
}

inline float approx_acos(float x) {
    float negate = float(x < 0);
    x = abs(x);
    float ret = -0.0187293;
    ret = ret * x;
    ret = ret + 0.0742610;
    ret = ret * x;
    ret = ret - 0.2121144;
    ret = ret * x;
    ret = ret + 1.5707288;
    ret = ret * tsqrtf(1.0-x);
    ret = ret - 2 * negate * ret;
    return negate * 3.14159265358979 + ret;
}

template<typename Scalar>
struct Coordinates {
    template<typename Derived>
    Coordinates(const Eigen::MatrixBase<Derived>& n) {
        static_assert(Derived::RowsAtCompileTime == 3);
        static_assert(Derived::ColsAtCompileTime == 1);
        to.row(2) << n.transpose();
        if (std::abs(n(0)) > std::abs(n(1))) {
            Scalar invLen = 1.0f / tsqrtf(n(0) * n(0) + n(2) * n(2));
            to.row(0) << n(2) * invLen, 0.0f, -n(0) * invLen;
        } else {
            Scalar invLen = 1.0f / tsqrtf(n(1) * n(1) + n(2) * n(2));
            to.row(0) << 0.0f, n(2) * invLen, -n(1) * invLen;
        }
        to.row(1) = to.row(0).cross(n);
    }

    Eigen::Matrix<Scalar, 3, 3> to;
    Eigen::Matrix<Scalar, 3, 1> n;
};

template<typename Scalar>
void coordinateSystem(
    const Eigen::Matrix<Scalar, 3, 1> &n,
    Eigen::Matrix<Scalar, 3, 1> &s,
    Eigen::Matrix<Scalar, 3, 1> &t
) {
    if (std::abs(n(0)) > std::abs(n(1))) {
        Scalar invLen = 1.0f / std::sqrt(n(0) * n(0) + n(2) * n(2));
        s << n(2) * invLen, 0.0f, -n(0) * invLen;
    } else {
        Scalar invLen = 1.0f / std::sqrt(n(1) * n(1) + n(2) * n(2));
        s << 0.0f, n(2) * invLen, -n(1) * invLen;
    }
    t = s.cross(n);
}


template<typename ForwardIterator>
inline static bool normalizePdf(ForwardIterator begin, ForwardIterator end) {
    using ValueType = typename std::iterator_traits<ForwardIterator>::value_type;
    ValueType sum = std::accumulate(begin, end, ValueType{});
#if FAIL_ON_ZERO_CDF
    assert(sum != ValueType{});
#endif
    if(sum == ValueType{}) {
        return false;
    }
    std::transform(begin, end, begin, [sum](ValueType weight) { return weight / sum; });
    return true;
}

template<typename ForwardIterator, typename Scalar=typename std::iterator_traits<ForwardIterator>::value_type>
inline static bool normalizePdf(ForwardIterator begin, ForwardIterator end, Scalar& additional) {
    using ValueType = typename std::iterator_traits<ForwardIterator>::value_type;
    ValueType sum = std::accumulate(begin, end, ValueType{}) + additional;
#if FAIL_ON_ZERO_CDF
    assert(sum != ValueType{});
#endif
    if(sum == ValueType{}) {
        return false;
    }
    std::transform(begin, end, begin, [sum](ValueType weight) { return weight / sum; });
    additional /= sum;
    return true;
}

template<typename ForwardIterator, typename OutputIterator>
inline bool createCdf(ForwardIterator begin, ForwardIterator end, OutputIterator outBegin, bool normalize=true) {
    if(normalize) {
        if(!normalizePdf(begin, end)) {
            return false;
        }
    }
    std::partial_sum(begin, end, outBegin);
    return true;
}

template<typename ForwardIterator, typename Scalar>
inline size_t sampleDiscreteCdf(ForwardIterator begin, ForwardIterator end, Scalar uniformSample) {
    assert(uniformSample >= 0.f && uniformSample < 1.f);
    auto lower_it = std::lower_bound(begin, end, uniformSample);
    if(lower_it == end) {
        --lower_it;
        while(*lower_it == *(lower_it - 1) && lower_it > begin) {
            --lower_it;
        }
    }
    return std::distance(begin, lower_it);
}


template<typename Derived>
inline static bool normalizePdf(Eigen::MatrixBase<Derived>& pdf) {
    using Scalar = typename Derived::Scalar;
    auto sum = pdf.sum();
#if FAIL_ON_ZERO_CDF
    assert(sum != Scalar{});
#endif
    if(sum == Scalar{}) {
        return false;
    }
    pdf.array() /= sum;
    return true;
}

template<typename Derived>
inline bool createCdfEigen(Eigen::MatrixBase<Derived>& pdf, Eigen::MatrixBase<Derived>& cdf, bool normalize=true) {
    if(normalize) {
        if(!normalizePdf(pdf)) {
            return false;
        }
    }
    if(cdf.rows() != pdf.rows()) {
        cdf.resize(pdf.rows(), 1);
    }

    cdf(0) = pdf(0);
    for(int sample_i = 1; sample_i < pdf.rows(); ++sample_i) {
        cdf(sample_i) = cdf(sample_i - 1) + pdf(sample_i);
    }
    return true;
}

template<typename Derived>
inline int lower_bound(const Eigen::MatrixBase<Derived>& cdf, typename Derived::Scalar value)
{
    int it;
    int count, step;
    int first = 0, last = cdf.rows();
    count = last - first;
 
    while (count > 0) {
        it = first; 
        step = count / 2; 
        it += step;
        if (cdf(it) < value) {
            first = ++it; 
            count -= step + 1; 
        }
        else
            count = step;
    }
    return first;
}

template<typename Derived>
inline int sampleDiscreteCdf(Eigen::MatrixBase<Derived>& cdf, typename Derived::Scalar uniformSample) {
    assert(uniformSample >= 0.f && uniformSample < 1.f);
    int lower_it = lower_bound(cdf, uniformSample);
    if(lower_it == cdf.rows()) {
        --lower_it;
        while(cdf(lower_it) == cdf(lower_it - 1) && lower_it > 0) {
            --lower_it;
        }
    }
    return lower_it;
}

template<typename Scalar>
inline Eigen::Matrix<Scalar, 3, 1> canonicalToDir(const Eigen::Matrix<Scalar, 2, 1>& p) {
    const Scalar cosTheta = 2 * p.x() - 1;
    const Scalar phi = 2 * M_PI * p.y();

    const Scalar sinTheta = std::sqrt(1 - cosTheta * cosTheta);
    Scalar sinPhi, cosPhi;
    sincos(phi, &sinPhi, &cosPhi);

    Eigen::Matrix<Scalar, 3, 1> result;
    result << sinTheta * cosPhi, sinTheta * sinPhi, cosTheta; 

    return result;
}

template<typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 2, 1> dirToCanonical(const Eigen::MatrixBase<Derived>& d) {
	using Scalar = typename Derived::Scalar;
    if (!std::isfinite(d(0)) || !std::isfinite(d(1)) || !std::isfinite(d(2))) {
        return {0, 0};
    }

    const Scalar cosTheta = std::min(std::max(d(2), Scalar(-1)), Scalar(1));
    Scalar phi = std::atan2(d(1), d(0));
    while (phi < 0)
        phi += 2.0 * M_PI;

    return {(cosTheta + 1) / 2, phi / (2 * M_PI)};
}

template<int dims_n, int dims_k>
struct KeyHas3DNormal { constexpr static bool value = false; };

template<>
struct KeyHas3DNormal<3, 5> { constexpr static bool value = true; };

template<int dims_n, int dims_k>
struct KeyHas2DNormal { constexpr static bool value = false; };

template<>
struct KeyHas2DNormal<2, 5> { constexpr static bool value = true; };

template<typename DerivedS, typename DerivedN, typename DerivedK>
inline std::enable_if_t<DerivedK::RowsAtCompileTime == 3, void> buildKey(
	const Eigen::MatrixBase<DerivedS>& sample,
	const Eigen::MatrixBase<DerivedN>& normal,
	Eigen::MatrixBase<DerivedK>& key
) {
	key << sample.template topRows<3>();
}

template<typename DerivedS, typename DerivedN, typename DerivedK>
inline std::enable_if_t<DerivedK::RowsAtCompileTime == 6, void> buildKey(
	const Eigen::MatrixBase<DerivedS>& sample,
	const Eigen::MatrixBase<DerivedN>& normal,
	Eigen::MatrixBase<DerivedK>& key
) {
    // using Normal = Eigen::Matrix<typename DerivedN::Scalar, 3, 1>;
    // Normal normalNormalized = (normal + Normal::Ones()) * 0.5;
	key << sample.template topRows<3>(), normal;
}

template<typename DerivedS, typename DerivedN, typename DerivedK>
inline std::enable_if_t<
	KeyHas3DNormal<DerivedN::RowsAtCompileTime, DerivedK::RowsAtCompileTime>::value,
	void
> buildKey(
	const Eigen::MatrixBase<DerivedS>& sample,
	const Eigen::MatrixBase<DerivedN>& normal,
	Eigen::MatrixBase<DerivedK>& key
) {
    // assert(std::abs(normal.norm() - 1) < 1e-4);
    Eigen::Matrix<typename DerivedN::Scalar, 2, 1> n = jmm::dirToCanonical(normal);
    assert(std::isfinite(n(0)) && std::isfinite(n(1)));
    assert((n.array() >= 0).all() && (n.array() <= 1).all());
	key <<
		sample.template topRows<3>(),
		n;
}

template<typename DerivedS, typename DerivedN, typename DerivedK>
inline std::enable_if_t<
	KeyHas2DNormal<DerivedN::RowsAtCompileTime, DerivedK::RowsAtCompileTime>::value,
	void
> buildKey(
	const Eigen::MatrixBase<DerivedS>& sample,
	const Eigen::MatrixBase<DerivedN>& normal,
	Eigen::MatrixBase<DerivedK>& key
) {
	key <<
		sample.template topRows<3>(),
		normal;
}

template<typename T>
using aligned_vector = std::vector<T, Eigen::aligned_allocator<T>>;

}

#endif // __UTILS_H
