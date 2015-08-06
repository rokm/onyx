/* Linear LaRank: Pattern collection
 * Copyright (C) 2008- Antoine Bordes
 * Copyright (C) 2015 Rok Mandeljc
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 */

#ifndef LINEAR_LARANK__PATTERN_COLLECTION_H
#define LINEAR_LARANK__PATTERN_COLLECTION_H


#include "pattern.h"

#include <unordered_set>


namespace LinearLaRank {


class PatternCollection
{
public:
    PatternCollection ();

    virtual ~PatternCollection ();


    void insert (const Pattern &pattern);
    void remove (unsigned int i);

    bool empty () const;

    unsigned size () const;

    Pattern &sample ();


    unsigned maxcount () const;

    Pattern &operator [] (unsigned i);
    const Pattern &operator [] (unsigned i) const;

private:
    std::unordered_set<unsigned> freeidx;
    std::vector<Pattern> patterns;
};


}; // LinearLaRank


#endif
