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

#if !defined(__PMC_WU_H)
#define __PMC_WU_H

#include <mitsuba/core/sched.h>

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                             Work result                              */
/* ==================================================================== */

/**
   Bidirectional path tracing needs its own WorkResult implementation,
   since each rendering thread simultaneously renders to a small 'camera
   image' block and potentially a full-resolution 'light image'.
*/
class MTS_EXPORT_RENDER PMCWorkUnit : public WorkUnit {
public:
	PMCWorkUnit() : WorkUnit() { }

    /* WorkUnit implementation */
    void set(const WorkUnit *wu);
    void load(Stream *stream);
    void save(Stream *stream) const;

    inline int getId() const { return m_populationId; }
    inline const Vector2i &getSize() const { return m_size; }

    inline void setId(int id) { m_populationId = id; }
    inline void setSize(const Vector2i &size) { m_size = size; }

    std::string toString() const;

    MTS_DECLARE_CLASS()
protected:
    /// Virtual destructor
    virtual ~PMCWorkUnit() { }
private:
    int m_populationId;
    Vector2i m_size;
};

MTS_NAMESPACE_END

#endif /* __BDPT_WU_H */
