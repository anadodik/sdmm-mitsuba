#ifndef __SPHERE_VOLUME_H
#define __SPHERE_VOLUME_H

#include "distribution.h"

namespace jmm {

template<int t_dims>
struct volume_norm {};

template<>
struct volume_norm<2> { constexpr static double value = M_PI; };
template<>
struct volume_norm<3> { constexpr static double value = 4.f * M_PI / 3.f; };
template<>
struct volume_norm<4> { constexpr static double value = M_PI * M_PI / 2.f; };
template<>
struct volume_norm<5> { constexpr static double value = 8.f * M_PI * M_PI / 15.f; };
template<>
struct volume_norm<6> { constexpr static double value = M_PI * M_PI * M_PI / 6.f; };
template<>
struct volume_norm<7> { constexpr static double value = 16.f * M_PI * M_PI * M_PI / 105.f; };
template<>
struct volume_norm<9> { constexpr static double value = 32.f * M_PI * M_PI * M_PI * M_PI / 945.f; };

}

#endif // __SPHERE_VOLUME_H