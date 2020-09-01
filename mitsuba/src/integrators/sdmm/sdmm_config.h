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
	int maxDepth, blockSize, borderSize;
    int sampleCount;
	bool strictNormals;
	bool sampleDirect;
	bool showWeighted;
	size_t samplesPerIteration;
    bool useHierarchical;
    bool sampleProduct;
    bool bsdfOnly;
    Float alpha;
    int batchIterations;
    int initIterations;
    bool enablePER;
    int replayBufferLength;
    Float resampleProportion;
    bool decreasePrior;
    bool correctStateDensity;
    int savedSamplesPerPath;

	bool useInit;
	bool useInitCovar;
	bool useInitWeightsForMeans;
	bool useInitWeightsForMixture;
	float initKMeansSwapTolerance;
	int rngSeed;

	Vector2i cropSize;
	int rrDepth;

	inline SDMMConfiguration() { }

	inline SDMMConfiguration(Stream *stream) {
		maxDepth = stream->readInt();
		blockSize = stream->readInt();
		strictNormals = stream->readBool();
		sampleDirect = stream->readBool();
		sampleCount = stream->readInt();
		showWeighted = stream->readBool();
		samplesPerIteration = stream->readSize();
		cropSize = Vector2i(stream);
		rrDepth = stream->readInt();
        useHierarchical = stream->readBool();
        sampleProduct = stream->readBool();
        bsdfOnly = stream->readBool();
        alpha = stream->readFloat();
        batchIterations = stream->readInt();
        initIterations = stream->readInt();
        enablePER = stream->readBool();
        replayBufferLength = stream->readInt();
        resampleProportion = stream->readFloat();
        decreasePrior = stream->readBool();
        correctStateDensity = stream->readBool();
        savedSamplesPerPath = stream->readInt();

		useInit = stream->readBool();
		useInitCovar = stream->readBool();
		useInitWeightsForMeans = stream->readBool();
		useInitWeightsForMixture = stream->readBool();
		initKMeansSwapTolerance = stream->readFloat();
		rngSeed = stream->readInt();
	}

	inline void serialize(Stream *stream) const {
		stream->writeInt(maxDepth);
		stream->writeInt(blockSize);
		stream->writeBool(strictNormals);
		stream->writeBool(sampleDirect);
		stream->writeInt(sampleCount);
		stream->writeBool(showWeighted);
		cropSize.serialize(stream);
		stream->writeInt(rrDepth);
        stream->writeBool(useHierarchical);
        stream->writeBool(sampleProduct);
        stream->writeBool(bsdfOnly);
        stream->writeFloat(alpha);
        stream->writeInt(batchIterations);
        stream->writeInt(initIterations);
        stream->writeInt(enablePER);
        stream->writeBool(enablePER);
        stream->writeInt(replayBufferLength);
        stream->writeFloat(resampleProportion);
        stream->writeBool(decreasePrior);
        stream->writeBool(correctStateDensity);
        stream->writeInt(savedSamplesPerPath);

		stream->writeBool(useInit);
		stream->writeBool(useInitCovar);
		stream->writeBool(useInitWeightsForMeans);
		stream->writeBool(useInitWeightsForMixture);
		stream->writeFloat(initKMeansSwapTolerance);
		stream->writeInt(rngSeed);
	}

	void dump() const {
#define LOG_TYPE EInfo
		SLog(LOG_TYPE, "   SDMM path tracer configuration:");
		SLog(LOG_TYPE, "   Maximum path depth          : %i", maxDepth);
		SLog(LOG_TYPE, "   Image size                  : %ix%i",
			cropSize.x, cropSize.y);
		SLog(LOG_TYPE, "   Use strict normals          : %s",
			strictNormals ? "yes" : "no");
		SLog(LOG_TYPE, "   Direct sampling strategies  : %s",
			sampleDirect ? "yes" : "no");
		SLog(LOG_TYPE, "   Sample count                : %i", sampleCount);
		SLog(LOG_TYPE, "   Russian roulette depth      : %i", rrDepth);
        SLog(LOG_TYPE, "   Block size                  : %i", blockSize);
		SLog(LOG_TYPE, "   Number of samples per iter. : " SIZE_T_FMT, samplesPerIteration);
		SLog(LOG_TYPE, "   Use hierarchical Gaussians  : %s",
			useHierarchical ? "yes" : "no");
		SLog(LOG_TYPE, "   Use product sampling        : %s",
			sampleProduct ? "yes" : "no");
		SLog(LOG_TYPE, "   Only sample learned BSDF    : %s",
			bsdfOnly ? "yes" : "no");
		SLog(LOG_TYPE, "   Decrease prior              : %s",
			decreasePrior ? "yes" : "no");
		SLog(LOG_TYPE, "   Correct state density       : %s",
			correctStateDensity ? "yes" : "no");
        SLog(LOG_TYPE, "   Saved samples per iterations: %i", savedSamplesPerPath);

		SLog(LOG_TYPE, "   Alpha                       : %f", alpha);
		SLog(LOG_TYPE, "   Batch iterations            : %i", batchIterations);
		SLog(LOG_TYPE, "   Init iterations             : %i", initIterations);
		SLog(LOG_TYPE, "   PER enabled                 : %s",
			enablePER ? "yes" : "no");
		SLog(LOG_TYPE, "   Replay buffer length        : %i", replayBufferLength);
		SLog(LOG_TYPE, "   Resample proportion         : %f", resampleProportion);
		#if SDMM_DEBUG == 1
		SLog(LOG_TYPE, "   Show weighted contributions : %s", showWeighted ? "yes" : "no");
		#endif
		SLog(LOG_TYPE, "   RNG seed                    : %i", rngSeed);
		SLog(LOG_TYPE, "   Sample mean for GMM initialization: %s", useInit ? "yes" : "no");
		SLog(LOG_TYPE, "   Sample covar for GMM initialization: %s", useInitCovar ? "yes" : "no");
		SLog(LOG_TYPE, "   Samples weights for GMM init means: %s", useInitWeightsForMeans ? "yes" : "no");
		SLog(LOG_TYPE, "   Samples weights for GMM init mixture weights: %s", useInitWeightsForMixture ? "yes" : "no");
		SLog(LOG_TYPE, "   Stop criterion in percent of swaps for GMM init: %f", initKMeansSwapTolerance);
#undef LOG_TYPE
	}
};

MTS_NAMESPACE_END

#endif /* __SDMM_CONFIG  */
