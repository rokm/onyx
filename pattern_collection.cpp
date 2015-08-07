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

#include "pattern_collection.h"

namespace LinearLaRank {


PatternCollection::PatternCollection ()
{
}

PatternCollection::~PatternCollection ()
{
}


void PatternCollection::insert (const Pattern &pattern)
{
    if (freeidx.size()) {
        auto it = freeidx.begin();
        patterns[*it] = pattern;
        freeidx.erase(it);
    } else {
        patterns.push_back(pattern);
    }
}

void PatternCollection::remove (unsigned int i)
{
    patterns[i].clear();
    freeidx.insert(i);
}


unsigned int PatternCollection::numAllPatterns () const
{
    return patterns.size();
}

unsigned int PatternCollection::numValidPatterns () const
{
    return patterns.size() - freeidx.size();
}


const Pattern &PatternCollection::operator [] (unsigned int i) const
{
    return patterns[i];
}


const Pattern &PatternCollection::randomSample () const
{
    while (true) {
        unsigned r = rand() % patterns.size();
        if (patterns[r].isValid()) {
            return patterns[r];
        }
    }

    return patterns[0];
}


}; // LinearLaRank
