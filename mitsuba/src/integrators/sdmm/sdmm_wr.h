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

#if !defined(__SDMM_WR_H)
#define __SDMM_WR_H

#include <mitsuba/render/imageblock.h>
#include <mitsuba/core/fresolver.h>
#include "sdmm_config.h"

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                             Work result                              */
/* ==================================================================== */

/**
   Bidirectional path tracing needs its own WorkResult implementation,
   since each rendering thread simultaneously renders to a small 'camera
   image' block and potentially a full-resolution 'light image'.
*/
class SDMMWorkResult : public WorkResult {
public:

	SDMMWorkResult(const SDMMConfiguration &conf, const ReconstructionFilter *filter,
        Vector2i blockSize = Vector2i(-1, -1));

	// Clear the contents of the work result
	void clear();

	/// Fill the work result with content acquired from a binary data stream
	virtual void load(Stream *stream);

	/// Serialize a work result to a binary data stream
	virtual void save(Stream *stream) const;

	/// Aaccumulate another work result into this one
	void put(const SDMMWorkResult *workResult);

#if SDMM_DEBUG == 1
	/* In debug mode, this function allows to dump the contributions of
	   the individual sampling strategies to a series of images */
	void dump(const SDMMConfiguration &conf,
			const fs::path &prefix, const fs::path &stem) const;
#endif
	void dumpIndividual(
        int spp, int iteration, const fs::path &directory, Float time
    ) const;

	inline void putSample(const Point2 &sample, const Spectrum &spec) {
		m_block->put(sample, spec, 1.0f);
		m_blockSqr->put(sample, spec * spec, 1.0f);
	}

	inline const ImageBlock *getImageBlock() const {
		return m_block.get();
	}

	inline const ImageBlock *getImageBlockSqr() const {
		return m_blockSqr.get();
	}

	inline void setSize(const Vector2i &size) {
		m_block->setSize(size);
		m_blockSqr->setSize(size);
	}

    inline Vector2i getSize() {
        return m_block->getSize();
    }

    inline void setOffset(const Point2i &offset) {
        m_block->setOffset(offset);
        m_blockSqr->setOffset(offset);
    }

	/// Return a string representation
	std::string toString() const;

    Float averagePathLength = 0.f;
    unsigned long pathCount = 0;

	MTS_DECLARE_CLASS()
protected:
	/// Virtual destructor
	virtual ~SDMMWorkResult();

protected:
#if SDMM_DEBUG == 1
	ref_vector<ImageBlock> m_debugBlocks;
#endif
    ref<ImageBlock> m_block;
	ref<ImageBlock> m_blockSqr;
};

MTS_NAMESPACE_END

#endif /* __BDPT_WR_H */
