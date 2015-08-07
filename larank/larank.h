/* Linear LaRank: Linear LaRank classifier
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

#ifndef LINEAR_LARANK__LARANK__LARANK_H
#define LINEAR_LARANK__LARANK__LARANK_H

#include <Eigen/Core>


namespace LinearLaRank {

class Classifier
{
public:
    virtual ~Classifier () {};

    virtual double getC () const = 0;
    virtual void setC (double C) = 0;

    virtual double getTau () const = 0;
    virtual void setTau (double) = 0;

    virtual int update (const Eigen::VectorXd &x, int label, double weight = 1.0) = 0;
    virtual int predict (const Eigen::VectorXd &x, Eigen::VectorXd &scores) const = 0;

    virtual double computeDualityGap () const = 0;

    virtual void seedRngEngine (unsigned int seed) = 0;

    virtual void saveToStream (std::ofstream &stream) const = 0;
    virtual void loadFromStream (std::ifstream &stream) = 0;
};

extern Classifier *create_linear_larank ();

}; // LinearLaRank

#endif
