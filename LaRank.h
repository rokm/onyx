// -*- C++ -*-
// Copyright (C) 2008- Antoine Bordes

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA


#ifndef LARANK_H
#define LARANK_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>

#include <unordered_map>
#include <unordered_set>

#include <Eigen/Core>


// LARANKPATTERN: used to keep track of support patterns
class LaRankPattern
{
public:

    LaRankPattern (int x_id, const Eigen::VectorXd &x, int y, double w = 1.0)
        : x_id(x_id), x(x), y(y), w(w)
    {
    }

    LaRankPattern ()
        : x_id(0)
    {
    }

    bool exists () const
    {
        return x_id >= 0;
    }

    void clear ()
    {
        x_id = -1;
    }

public:
    int x_id;
    Eigen::VectorXd x;
    int y;
    double w; // Sample weight
};

// LARANKPATTERNS: collection of support patterns
class LaRankPatterns
{
public:
    LaRankPatterns ()
    {
    }

    virtual ~LaRankPatterns ()
    {
    }

    void insert (const LaRankPattern &pattern)
    {
        if (freeidx.size()) {
            std::unordered_set<unsigned>::iterator it = freeidx.begin();
            patterns[*it] = pattern;
            freeidx.erase(it);
        } else {
            patterns.push_back(pattern);
        }
    }

    void remove (unsigned i)
    {
        patterns[i].clear();
        freeidx.insert(i);
    }

    bool empty () const
    {
        return patterns.size() == freeidx.size();
    }

    unsigned size () const
    {
        return patterns.size() - freeidx.size();
    }

    LaRankPattern &sample ()
    {
        assert(!empty());
        while (true) {
            unsigned r = rand() % patterns.size();
            if (patterns[r].exists()) {
                return patterns[r];
            }
        }

        return patterns[0];
    }

    unsigned maxcount () const
    {
        return patterns.size();
    }

    LaRankPattern &operator [] (unsigned i)
    {
        return patterns[i];
    }

    const LaRankPattern &operator [] (unsigned i) const
    {
        return patterns[i];
    }

private:
    std::unordered_set<unsigned> freeidx;
    std::vector<LaRankPattern> patterns;
};


// MACHINE: the thing we learn
class Machine
{
public:
    virtual ~Machine() {};

    //MAIN functions for training and testing
    virtual int add (const Eigen::VectorXd &x, int classnumber, double weight = 1.0) = 0;
    virtual int predict (const Eigen::VectorXd &x) = 0;
    virtual int predict (const Eigen::VectorXd &x, Eigen::VectorXd &scores) = 0;

    // Information functions
    virtual void printStuff(double initime, bool dual) = 0;
    virtual double computeGap() = 0;

public:
    double C;
    double tau;
};

extern Machine *create_larank ();

#endif // LARANK_H
