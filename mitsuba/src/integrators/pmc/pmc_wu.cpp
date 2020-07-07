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

#include "pmc_wu.h"

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                          PMCWorkUnit                                 */
/* ==================================================================== */

void PMCWorkUnit::set(const WorkUnit *wu) {
    const PMCWorkUnit *pmcWu = static_cast<const PMCWorkUnit *>(wu);
    m_populationId = pmcWu->m_populationId;
    m_size = pmcWu->m_size;
}

void PMCWorkUnit::load(Stream *stream) {
    m_populationId = stream->readInt();
    int data[2];
    stream->readIntArray(data, 2);
    m_size.x   = data[0];
    m_size.y   = data[1];
}

void PMCWorkUnit::save(Stream *stream) const {
    stream->writeInt(m_populationId);
    int data[2];
    data[0] = m_size.x;
    data[1] = m_size.y;
    stream->writeIntArray(data, 2);
}

std::string PMCWorkUnit::toString() const {
    std::ostringstream oss;
    oss << "PMCWorkUnit[id=" << m_populationId
        << ", size=" << m_size.toString() << "]";
    return oss.str();
}

MTS_IMPLEMENT_CLASS(PMCWorkUnit, false, WorkUnit)
MTS_NAMESPACE_END
