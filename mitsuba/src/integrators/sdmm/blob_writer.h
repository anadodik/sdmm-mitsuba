/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2017-2018 by ETH Zurich, Thomas Mueller.

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

#pragma once

#include <fstream>

#include <mitsuba/mitsuba.h>

MTS_NAMESPACE_BEGIN

class BlobWriter {
public:
    BlobWriter(const std::string& filename)
        : f(filename, std::ios::out | std::ios::binary) {
    }

    template <typename T>
    typename std::enable_if<std::is_pod<T>::value, BlobWriter&>::type
        operator<<(T element) {
        write(&element, 1);
        return *this;
    }

    template <typename T>
    typename std::enable_if<!std::is_pod<T>::value, BlobWriter&>::type operator<<(const T& vec) {
        *this << static_cast<uint64_t>(vec.size());
        for (const auto& c : vec) {
            *this << c;
        }
        return *this;
    }

    // CAUTION: This function may break down on big-endian architectures.
    //          The ordering of bytes has to be reverted then.
    template <typename T>
    void write(T* src, size_t size) {
        f.write(reinterpret_cast<const char*>(src), size * sizeof(T));
    }

private:
    std::ofstream f;
};

MTS_NAMESPACE_END
