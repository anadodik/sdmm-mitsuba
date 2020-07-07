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

#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/fstream.h>
#include <iostream>
#include "sdmm_wr.h"

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                             Work result                              */
/* ==================================================================== */

SDMMWorkResult::SDMMWorkResult(const SDMMConfiguration &conf,
		const ReconstructionFilter *rfilter, Vector2i blockSize) {
    if (blockSize == Vector2i(-1, -1))
        blockSize = Vector2i(conf.blockSize, conf.blockSize);

	m_block = new ImageBlock(Bitmap::ESpectrumAlphaWeight, blockSize, rfilter);
	m_block->setOffset(Point2i(0, 0));
	m_block->setSize(blockSize);

	m_blockSqr = new ImageBlock(Bitmap::ESpectrumAlphaWeight, blockSize, rfilter);
	m_blockSqr->setOffset(Point2i(0, 0));
	m_blockSqr->setSize(blockSize);

#if SDMM_DEBUG == 1
	m_debugBlocks.resize(m_populations * m_iterations);
	for (size_t i = 0; i < m_debugBlocks.size(); ++i) {
		m_debugBlocks[i] = new ImageBlock(Bitmap::ESpectrum, blockSize, rfilter);
		m_debugBlocks[i]->setSize(blockSize);
		m_debugBlocks[i]->setOffset(Point2i(0,0));
	}

    m_manualImage = new ImageBlock(Bitmap::ESpectrum, blockSize, rfilter);
	m_manualImage->setSize(blockSize);
	m_manualImage->setOffset(Point2i(0, 0));
#endif

    m_outliers = new ImageBlock(Bitmap::ESpectrum, blockSize, rfilter);
    m_outliers->setSize(blockSize);
    m_outliers->setOffset(Point2i(0, 0));

    m_spatialDensity = new ImageBlock(Bitmap::ESpectrum, blockSize, rfilter);
    m_spatialDensity->setSize(blockSize);
    m_spatialDensity->setOffset(Point2i(0, 0));

    m_denoised = new ImageBlock(Bitmap::ESpectrum, blockSize, rfilter);
    m_denoised->setSize(blockSize);
    m_denoised->setOffset(Point2i(0, 0));
}

SDMMWorkResult::~SDMMWorkResult() { }

void SDMMWorkResult::put(const SDMMWorkResult *workResult) {
#if SDMM_DEBUG == 1
	for (size_t i = 0; i < m_debugBlocks.size(); ++i)
		m_debugBlocks[i]->put(workResult->m_debugBlocks[i].get());
    m_manualImage->put(workResult->m_manualImage.get());
#endif
	m_block->put(workResult->m_block.get());
	m_blockSqr->put(workResult->m_blockSqr.get());
}

void SDMMWorkResult::clear() {
#if SDMM_DEBUG == 1
	for (size_t i=0; i<m_debugBlocks.size(); ++i)
		m_debugBlocks[i]->clear();
    m_manualImage->clear();
#endif
	m_block->clear();
	m_blockSqr->clear();

    m_outliers->clear();
    m_spatialDensity->clear();
    m_denoised->clear();
}

#if SDMM_DEBUG == 1
/* In debug mode, this function allows to dump the contributions of
   the individual sampling strategies to a series of images */
void SDMMWorkResult::dump(
    const SDMMConfiguration &conf,
    const fs::path &prefix,
    const fs::path &stem
) const {
	Float weight = (Float) 1.0f / (Float) conf.samplesPerIteration;
    for(int populationId = 0; populationId < m_populations; ++populationId) {
        for(int iteration = 0; iteration < m_iterations; ++iteration) {
            int i = sub2ind(populationId, iteration);
            Bitmap *bitmap = const_cast<Bitmap *>(m_debugBlocks[i]->getBitmap());
            bitmap->convert(bitmap, weight);
            fs::path hdrFilename = prefix / fs::path(formatString("pop%02i_it%02i.exr", populationId, iteration));
            ref<FileStream> targetFileHDR = new FileStream(hdrFilename, FileStream::ETruncReadWrite);
            bitmap->write(Bitmap::EOpenEXR, targetFileHDR, 1);

            // fs::path ldrFilename = prefix / fs::path(formatString("pop%02i_it%02i.png", populationId, iteration));
            // ref<Bitmap> ldrBitmap = bitmap->convert(Bitmap::ERGB, Bitmap::EUInt8, -1, weight);
            // ref<FileStream> targetFileLDR = new FileStream(ldrFilename, FileStream::ETruncReadWrite);
            // ldrBitmap->write(Bitmap::EPNG, targetFileLDR, 1);
        }
    }
}

void SDMMWorkResult::dumpManual(const SDMMConfiguration &conf,
		const fs::path &prefix, const fs::path &stem) const {
    Bitmap *bitmap = const_cast<Bitmap *>(m_manualImage->getBitmap());
    fs::path hdrFilename = prefix / fs::path(formatString("manual_image.exr"));
    ref<FileStream> targetFileHDR = new FileStream(hdrFilename, FileStream::ETruncReadWrite);
    bitmap->write(Bitmap::EOpenEXR, targetFileHDR, 1);
}

#endif

void SDMMWorkResult::dumpOutliers(const SDMMConfiguration &conf,
		const fs::path &prefix, const fs::path &stem, int iteration) const {
	Float weight = (Float) 1.0f / (Float) (conf.samplesPerIteration);
    Bitmap *bitmap = const_cast<Bitmap *>(m_outliers->getBitmap());
    fs::path hdrFilename = prefix / "individual" / fs::path(formatString("outliers_it%05i.exr", iteration));;
    bitmap->convert(bitmap, weight);
    Vector2i borderSize{m_outliers->getBorderSize(), m_outliers->getBorderSize()};
    auto cropped = bitmap->crop(Point2i(borderSize), bitmap->getSize() - 2 * borderSize);
    ref<FileStream> targetFileHDR = new FileStream(hdrFilename, FileStream::ETruncReadWrite);
    cropped->write(Bitmap::EOpenEXR, targetFileHDR, 1);
}

void SDMMWorkResult::dumpSpatialDensity(
    int spp, const fs::path &prefix, const fs::path &stem, int iteration
) const {
	Float weight = (Float) 1.0f / (Float) spp;
    Bitmap *bitmap = const_cast<Bitmap *>(m_spatialDensity->getBitmap());
    fs::path hdrFilename = prefix / "individual" / fs::path(formatString("density_it%05i.exr", iteration));;
    bitmap->convert(bitmap, weight);
    Vector2i borderSize{m_spatialDensity->getBorderSize(), m_spatialDensity->getBorderSize()};
    auto cropped = bitmap->crop(Point2i(borderSize), bitmap->getSize() - 2 * borderSize);
    ref<FileStream> targetFileHDR = new FileStream(hdrFilename, FileStream::ETruncReadWrite);
    cropped->write(Bitmap::EOpenEXR, targetFileHDR, 1);
}

void SDMMWorkResult::dumpDenoised(
    int spp, const fs::path &prefix, const fs::path &stem, int iteration
) const {
	Float weight = (Float) 1.0f / (Float) spp;
    Bitmap *bitmap = const_cast<Bitmap *>(m_denoised->getBitmap());
    fs::path hdrFilename = prefix / "individual" / fs::path(formatString("denoised_it%05i.exr", iteration));;
    bitmap->convert(bitmap, weight);
    Vector2i borderSize{m_denoised->getBorderSize(), m_denoised->getBorderSize()};
    auto cropped = bitmap->crop(Point2i(borderSize), bitmap->getSize() - 2 * borderSize);
    ref<FileStream> targetFileHDR = new FileStream(hdrFilename, FileStream::ETruncReadWrite);
    cropped->write(Bitmap::EOpenEXR, targetFileHDR, 1);
}

void SDMMWorkResult::dumpIndividual(
    int spp,
    int iteration,
    const fs::path &directory,
    Float time
) const {
    if(!fs::is_directory(directory) || !fs::exists(directory)) {
        fs::create_directories(directory);
    }
    Properties imageProps;
    imageProps.setInteger("spp", spp);
    imageProps.setInteger("iteration", iteration);
    imageProps.setFloat("time", time);

    auto dumpImage = [directory, imageProps](
        const fs::path& filename, ref<ImageBlock> block
    ) {
        fs::path hdrFilename = directory / filename;
        ref<FileStream> targetFileHDR = new FileStream(
            hdrFilename, FileStream::ETruncReadWrite
        );
        Bitmap *bitmap = const_cast<Bitmap *>(block->getBitmap());
        ref<Bitmap> converted = bitmap->convert(
            Bitmap::ESpectrum, Bitmap::EFloat, 1.0f, 1.0f
        );
        converted->setMetadata(imageProps);
        converted->write(Bitmap::EOpenEXR, targetFileHDR, 1);
    };

    dumpImage(formatString("iteration%05i.exr", iteration), m_block);
    dumpImage(formatString("iteration_sqr%05i.exr", iteration), m_blockSqr);
}

void SDMMWorkResult::load(Stream *stream) {
#if SDMM_DEBUG == 1
	for (size_t i=0; i<m_debugBlocks.size(); ++i)
		m_debugBlocks[i]->load(stream);
#endif
	m_block->load(stream);
	m_blockSqr->load(stream);
    m_outliers->load(stream);
    m_spatialDensity->load(stream);
    m_denoised->load(stream);
}

void SDMMWorkResult::save(Stream *stream) const {
#if SDMM_DEBUG == 1
	for (size_t i=0; i<m_debugBlocks.size(); ++i)
		m_debugBlocks[i]->save(stream);
#endif
	m_block->save(stream);
	m_blockSqr->save(stream);
    m_outliers->save(stream);
    m_spatialDensity->save(stream);
    m_denoised->save(stream);
}

std::string SDMMWorkResult::toString() const {
	return m_block->toString();
}

MTS_IMPLEMENT_CLASS(SDMMWorkResult, false, WorkResult)
MTS_NAMESPACE_END
