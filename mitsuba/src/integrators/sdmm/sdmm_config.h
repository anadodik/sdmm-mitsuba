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

#if !defined(__SDMM_CONFIG_H)
#define __SDMM_CONFIG_H

#define SDMM_DEBUG 0

#include <mitsuba/mitsuba.h>

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                         Configuration storage                        */
/* ==================================================================== */

/**
 * \brief Stores all configuration parameters of the
 * bidirectional path tracer
 */
struct SDMMConfiguration {
    Vector2i imageSize;

    bool strictNormals;
    int blockSize;
    int maxDepth;
    int rrDepth;
    int sampleCount;

    size_t samplesPerIteration;
    bool sampleProduct;
    bool bsdfOnly;
    int savedSamplesPerPath;

    bool flushDenormals;
    bool optimizeAsync;

    // bool sampleDirect;
    // Float alpha;
    // bool correctSpatialDensity;

    inline SDMMConfiguration() { }

    inline SDMMConfiguration(Stream *stream) {
        imageSize = Vector2i(stream);

        strictNormals = stream->readBool();
        maxDepth = stream->readInt();
        rrDepth = stream->readInt();
        blockSize = stream->readInt();
        sampleCount = stream->readInt();

        samplesPerIteration = stream->readSize();
        sampleProduct = stream->readBool();
        bsdfOnly = stream->readBool();
        savedSamplesPerPath = stream->readInt();

        flushDenormals = stream->readBool();
        optimizeAsync = stream->readBool();

        // sampleDirect = stream->readBool();
        // alpha = stream->readFloat();
        // correctSpatialDensity = stream->readBool();
    }

    inline void serialize(Stream *stream) const {
        imageSize.serialize(stream);

        stream->writeBool(strictNormals);
        stream->writeInt(maxDepth);
        stream->writeInt(rrDepth);
        stream->writeInt(blockSize);
        stream->writeInt(sampleCount);
        
        stream->writeSize(samplesPerIteration);
        stream->writeBool(sampleProduct);
        stream->writeBool(bsdfOnly);
        stream->writeInt(savedSamplesPerPath);

        stream->writeBool(flushDenormals);
        stream->writeBool(optimizeAsync);

        // stream->writeBool(sampleDirect);
        // stream->writeFloat(alpha);
        // stream->writeBool(correctSpatialDensity);
    }

    void dump() const {
#define LOG_TYPE EInfo
        SLog(LOG_TYPE, "   SDMM path tracer configuration:");
        SLog(LOG_TYPE, "   Image size                  : %ix%i",
            imageSize.x, imageSize.y);
        SLog(LOG_TYPE, "   Block size                  : %i", blockSize);
        SLog(LOG_TYPE, "   Maximum path depth          : %i", maxDepth);
        SLog(LOG_TYPE, "   Russian roulette depth      : %i", rrDepth);
        SLog(LOG_TYPE, "   Use strict normals          : %s",
            strictNormals ? "yes" : "no");

        SLog(LOG_TYPE, "   Sample count                : %i", sampleCount);
        SLog(LOG_TYPE, "   Number of samples per iter. : " SIZE_T_FMT, samplesPerIteration);
        SLog(LOG_TYPE, "   Saved samples per path      : %i", savedSamplesPerPath);
        SLog(LOG_TYPE, "   Use product sampling        : %s",
            sampleProduct ? "yes" : "no");
        SLog(LOG_TYPE, "   Only sample learned BSDF    : %s",
            bsdfOnly ? "yes" : "no");

        SLog(LOG_TYPE, "   Flushing denromal floats    : %s",
            flushDenormals ? "yes" : "no");
        SLog(LOG_TYPE, "   Async optimization enabled  : %s",
            optimizeAsync ? "yes" : "no");
        // SLog(LOG_TYPE, "   Direct sampling strategies  : %s",
        //  sampleDirect ? "yes" : "no");
        // SLog(LOG_TYPE, "   Correct spatial density       : %s",
        //  correctSpatialDensity ? "yes" : "no");
#undef LOG_TYPE
    }
};

MTS_NAMESPACE_END

#endif /* __SDMM_CONFIG  */
