/* Onyx: Linear LaRank: Linear LaRank classifier
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

#ifndef ONYX__LINEAR_LARANK__LARANK_H
#define ONYX__LINEAR_LARANK__LARANK_H

#include <cstdint>
#include <vector>
#include <Eigen/Core>

#include <onyx/export.h>


namespace Onyx {
namespace LinearLaRank {


class Classifier
{
public:
    virtual ~Classifier () {};

    virtual float getC () const = 0;
    virtual void setC (float C) = 0;

    virtual float getTau () const = 0;
    virtual void setTau (float) = 0;

    virtual unsigned int getNumFeatures () const = 0;

    virtual unsigned int getNumClasses () const = 0;
    virtual std::vector<int> getClassLabels () const = 0;

    virtual uint64_t getNumSeenSamples () const = 0;

    virtual int update (const Eigen::Ref<const Eigen::VectorXf> &features, int label, float weight = 1.0) = 0;
    virtual int update (const Eigen::Ref<const Eigen::VectorXd> &features, int label, float weight = 1.0) = 0;

    virtual int predict (const Eigen::Ref<const Eigen::VectorXf> &features) const = 0;
    virtual int predict (const Eigen::Ref<const Eigen::VectorXd> &features) const = 0;

    virtual int predict (const Eigen::Ref<const Eigen::VectorXf> &features, Eigen::Ref<Eigen::VectorXf> scores) const = 0;
    virtual int predict (const Eigen::Ref<const Eigen::VectorXd> &features, Eigen::Ref<Eigen::VectorXf> scores) const = 0;

    virtual float computeDualityGap () const = 0;

    virtual void seedRngEngine (unsigned int seed) = 0;

    virtual void saveToStream (std::ostream &stream) const = 0;
    virtual void loadFromStream (std::istream &stream) = 0;
};

extern ONYX_EXPORT Classifier *create_classifier ();


} // LinearLaRank
} // Onyx

#endif
