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

#include "sdmm_wu.h"

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                          SDMMWorkUnit                                 */
/* ==================================================================== */

void SDMMWorkUnit::set(const WorkUnit *wu) {
    const SDMMWorkUnit *sdmmWu = static_cast<const SDMMWorkUnit *>(wu);
    m_populationId = sdmmWu->m_populationId;
    m_size = sdmmWu->m_size;
}

void SDMMWorkUnit::load(Stream *stream) {
    m_populationId = stream->readInt();
    int data[2];
    stream->readIntArray(data, 2);
    m_size.x   = data[0];
    m_size.y   = data[1];
}

void SDMMWorkUnit::save(Stream *stream) const {
    stream->writeInt(m_populationId);
    int data[2];
    data[0] = m_size.x;
    data[1] = m_size.y;
    stream->writeIntArray(data, 2);
}

std::string SDMMWorkUnit::toString() const {
    std::ostringstream oss;
    oss << "SDMMWorkUnit[id=" << m_populationId
        << ", size=" << m_size.toString() << "]";
    return oss.str();
}

MTS_IMPLEMENT_CLASS(SDMMWorkUnit, false, WorkUnit)
MTS_NAMESPACE_END
