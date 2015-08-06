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

#include "larank.h"

#include "output.h"
#include "pattern.h"
#include "pattern_collection.h"

#include <chrono>
#include <limits>
#include <map>


namespace LinearLaRank {


// *********************************************************************
// *                   Linear LaRank implementation                    *
// *********************************************************************
class LaRank : public Classifier
{
public:
    LaRank ();
    virtual ~LaRank ();

    virtual double getC () const;
    virtual void setC (double C);

    virtual double getTau () const;
    virtual void setTau (double);

    virtual int update (const Eigen::VectorXd &x, int classnumber, double weight);
    virtual int predict (const Eigen::VectorXd &x, Eigen::VectorXd &scores) const;

    virtual double computeDualityGap () const;

private:
    // Output class gradient information
    struct outputgradient_t
    {
        int output;
        double gradient;

        outputgradient_t (int output = 0, double gradient = 0.0)
            : output(output), gradient(gradient)
        {
        }

        bool operator < (const outputgradient_t &og) const
        {
            return gradient > og.gradient;
        }
    };

    // Processing types in LaRank
    enum process_type_t
    {
        processNew,
        processOld,
        processOptimize
    };

    // Return type for ProcessNew
    struct process_return_t
    {
        double dual_increase;
        int ypred;

        process_return_t (double dual, int ypred)
            : dual_increase(dual) , ypred(ypred)
        {
        }
    };

private:
    Output *getOutput (int index);
    const Output *getOutput (int index) const;

    process_return_t process (const Pattern &pattern, process_type_t ptype);
    double reprocess ();
    double optimize ();
    unsigned int cleanup ();

    int getNSV () const;
    double getW2 () const;
    double getDualObjective () const;

private:
    // Parameters
    double C = 1.0;
    double tau = 0.0001;

    // Learnt output decision functions
    typedef std::map<int, Output> outputhash_t; // class index -> decision function
    outputhash_t outputs;

    // State
    PatternCollection patterns;

    uint64_t num_seen_samples = 0;
    uint64_t num_removed = 0;

    uint64_t num_pro = 0;
    uint64_t num_rep = 0;
    uint64_t num_opt = 0;

    double w_pro = 1.0;
    double w_rep = 1.0;
    double w_opt = 1.0;

    double dual = 0.0;
};


// *********************************************************************
// *                      Constructor/destructor                       *
// *********************************************************************
LaRank::LaRank ()
{
}

LaRank::~LaRank ()
{
}


// *********************************************************************
// *                         Public interface                          *
// *********************************************************************
double LaRank::getC () const
{
    return C;
}

void LaRank::setC (double C)
{
    this->C = C;
}


double LaRank::getTau () const
{
    return tau;
}

void LaRank::setTau (double tau)
{
    this->tau = tau;
}


// *********************************************************************
// *                              Update                               *
// *********************************************************************
// Adds new pattern and runs optimization steps chosen with adaptive schedule
int LaRank::update (const Eigen::VectorXd &features, int label, double weight)
{
    // Update counter of seen samples
    num_seen_samples++;

    // If we have not seen this class before, create a new output decision
    // function
    if (!getOutput(label)) {
        outputs.insert(std::make_pair(label, Output(features.size())));
    }

    // Fill pattern element
    Pattern pattern(num_seen_samples, features, label, weight);

    // *** ProcessNew with the "fresh" pattern ***
    auto timeStart = std::chrono::steady_clock::now();

    process_return_t pro_ret = process(pattern, processNew);
    double dual_increase = pro_ret.dual_increase;
    dual += dual_increase;

    double duration = std::chrono::duration<double>(std::chrono::steady_clock::now() - timeStart).count();
    double coeff = dual_increase / (10*(0.00001 + duration));
    num_pro++;
    w_pro = 0.05 * coeff + (1.0 - 0.05) * w_pro;

    // *** ProcessOld & Optimize until ready for a new ProcessNew ***
    // (Adaptive schedule here)
    for (;;) {
        double w_sum = w_pro + w_rep + w_opt;
        double prop_min = w_sum / 20;

        w_pro = std::max(w_pro, prop_min);
        w_rep = std::max(w_rep, prop_min);
        w_opt = std::max(w_opt, prop_min);

        w_sum = w_pro + w_rep + w_opt;

        double r = rand() / (double)RAND_MAX * w_sum;

        if (r <= w_pro) {
            break;
        } else if ((r > w_pro) && (r <= w_pro + w_rep)) {
            auto timeStart = std::chrono::steady_clock::now();

            double dual_increase = reprocess();

            dual += dual_increase;

            double duration = std::chrono::duration<double>(std::chrono::steady_clock::now() - timeStart).count();
            double coeff = dual_increase / (0.00001 + duration);

            num_rep++;
            w_rep = 0.05 * coeff + (1 - 0.05) * w_rep;
        } else {
            auto timeStart = std::chrono::steady_clock::now();

            double dual_increase = optimize();

            dual += dual_increase;

            double duration = std::chrono::duration<double>(std::chrono::steady_clock::now() - timeStart).count();
            double coeff = dual_increase / (0.00001 + duration);

            num_opt++;
            w_opt = 0.05 * coeff + (1 - 0.05) * w_opt;
        }
    }

    if (num_seen_samples % 100 == 0) {
        num_removed += cleanup();
    }

    return pro_ret.ypred;
}

// *********************************************************************
// *                              Predict                              *
// *********************************************************************
int LaRank::predict (const Eigen::VectorXd &features, Eigen::VectorXd &scores) const
{
    int res = -1;
    double score_max = -std::numeric_limits<double>::max();
    int nClass = 0;

    scores.resize(outputs.size());

    for (outputhash_t::const_iterator it = outputs.begin(); it != outputs.end(); it++) {
        double score = it->second.computeScore(features);

        if (score > score_max) {
            score_max = score;
            res = it->first;
        }

        scores[nClass++] = score; // Store output score
    }

    return res;
}


// *********************************************************************
// *                        Compute duality gap                        *
// *********************************************************************
double LaRank::computeDualityGap () const
{
    double sum_sl = 0.0;
    double sum_bi = 0.0;

    for (unsigned i = 0; i < patterns.maxcount(); i++) {
        const Pattern &p = patterns[i];
        if (!p.exists()) {
            continue;
        }

        const Output *out = getOutput(p.label);
        if (!out) {
            continue;
        }

        sum_bi += out->getBeta(p.id);
        double gi = out->computeGradient(p.features, p.label, p.label);
        double gmin = std::numeric_limits<double>::max();

        for (outputhash_t::const_iterator it = outputs.begin(); it != outputs.end(); it++) {
            if (it->first != p.label && it->second.isSupportVector(p.id)) {
                double g = it->second.computeGradient(p.features, p.label, it->first);
                if (g < gmin) {
                    gmin = g;
                }
            }
        }

        sum_sl += std::max(0.0, gi-gmin);
    }

    return std::max(0.0, getW2() + C * sum_sl - sum_bi);
}


// *********************************************************************
// *                            Processing                             *
// *********************************************************************
Output *LaRank::getOutput (int index)
{
    outputhash_t::iterator it = outputs.find(index);
    return it == outputs.end() ? NULL : &it->second;
}

const Output *LaRank::getOutput (int index) const
{
    outputhash_t::const_iterator it = outputs.find(index);
    return it == outputs.end() ? NULL : &it->second;
}

// Main optimization step
LaRank::process_return_t LaRank::process (const Pattern &pattern, process_type_t ptype)
{
    process_return_t pro_ret = process_return_t(0, 0);

    std::vector<outputgradient_t> outputgradients;
    outputgradients.reserve(/*getNumOutputs()*/outputs.size());

    std::vector<outputgradient_t> outputscores;
    outputscores.reserve(/*getNumOutputs()*/outputs.size());

    // Compute gradient and sort
    for (outputhash_t::iterator it = outputs.begin(); it != outputs.end(); it++) {
        if (ptype != processOptimize || it->second.isSupportVector(pattern.id)) {
            double g = it->second.computeGradient(pattern.features, pattern.label, it->first);

            outputgradients.push_back(outputgradient_t(it->first, g));

            if (it->first == pattern.label) {
                outputscores.push_back(outputgradient_t(it->first,(1.0 - g)));
            } else {
                outputscores.push_back(outputgradient_t(it->first, -g));
            }
        }
    }

    std::sort(outputgradients.begin(), outputgradients.end());

    // Determine the prediction and its confidence
    std::sort(outputscores.begin(), outputscores.end());
    pro_ret.ypred = outputscores[0].output;

    // Find yp
    outputgradient_t ygp;
    Output *outp = NULL;
    unsigned p;
    for (p = 0; p < outputgradients.size(); p++) {
        outputgradient_t &current = outputgradients[p];
        Output *output = getOutput(current.output);

        bool support = ptype == processOptimize || output->isSupportVector(pattern.id);
        bool goodclass = (current.output == pattern.label);

        if ((!support && goodclass) || (support && output->getBeta(pattern.id) < (goodclass ? C * pattern.weight : 0))) {
            ygp = current;
            outp = output;
            break;
        }
    }

    if (p == outputgradients.size()) {
        return pro_ret;
    }

    // Find ym
    outputgradient_t ygm;
    Output *outm = NULL;
    int m;
    for (m = outputgradients.size() - 1; m >= 0; m--) {
        outputgradient_t &current = outputgradients[m];
        Output *output = getOutput(current.output);

        bool support = ptype == processOptimize || output->isSupportVector(pattern.id);
        bool goodclass = (current.output == pattern.label);

        if (!goodclass || (support && output->getBeta(pattern.id) > 0)) {
            ygm = current;
            outm = output;
            break;
        }
    }

    if (m < 0) {
        return pro_ret;
    }

    // Throw or insert pattern
    if ((ygp.gradient - ygm.gradient) < tau) {
        return pro_ret;
    }

    if (ptype == processNew) {
        patterns.insert(pattern);
    }

    // Compute lambda and clip it
    double kii = pattern.features.squaredNorm();
    double lambda = (ygp.gradient - ygm.gradient) / (2 * kii);
    if (ptype == processOptimize || outp->isSupportVector(pattern.id)) {
        double beta = outp->getBeta(pattern.id);

        if (ygp.output == pattern.label) {
            lambda = std::min(lambda, C * pattern.weight - beta);
        } else {
            lambda = std::min(lambda, fabs(beta));
        }
    } else {
        lambda = std::min(lambda, C * pattern.weight);
    }

    // Update parameters
    outp->update(pattern.features, lambda, pattern.id);
    outm->update(pattern.features, -lambda, pattern.id);
    pro_ret.dual_increase = lambda * ((ygp.gradient - ygm.gradient) - lambda * kii);

    return pro_ret;
}

// 2nd optimization function (ProcessOld in the paper)
double LaRank::reprocess ()
{
    if (patterns.size()) {
        for (int n = 0; n < 10; n++) {
            process_return_t pro_ret = process(patterns.sample(), processOld);
            if (pro_ret.dual_increase) {
                return pro_ret.dual_increase;
            }
        }
    }
    return 0.0;
}

// 3rd optimization function
double LaRank::optimize ()
{
    double dual_increase = 0.0;

    if (patterns.size()) {
        for (int n = 0; n < 10; n++) {
            process_return_t pro_ret = process(patterns.sample(), processOptimize);
            dual_increase += pro_ret.dual_increase;
        }
    }

    return dual_increase;
}

// remove patterns and return the number of patterns that were removed
unsigned int LaRank::cleanup ()
{
    unsigned int res = 0;

    for (unsigned int i = 0; i < patterns.maxcount(); i++) {
        Pattern &p = patterns[i];
        if (p.exists() && !outputs[p.label].isSupportVector(p.id)) {
            patterns.remove(i);
            res++;
        }
    }

    return res;
}


// *********************************************************************
// *                       Information functions                       *
// *********************************************************************
// Number of Support Vectors
int LaRank::getNSV () const
{
    int res = 0;

    for (outputhash_t::const_iterator it = outputs.begin(); it != outputs.end(); it++) {
        res += it->second.getNSV();
    }

    return res;
}

// Square-norm of the weight vector w
double LaRank::getW2 () const
{
    double res = 0.0;

    for (outputhash_t::const_iterator it = outputs.begin(); it != outputs.end(); it++) {
        res += it->second.getW2();
    }

    return res;
}

// Dual objective value
double LaRank::getDualObjective () const
{
    double res = 0.0;

    for (unsigned int i = 0; i < patterns.maxcount(); ++i) {
        const Pattern &p = patterns[i];

        if (!p.exists()) {
            continue;
        }

        const Output *out = getOutput(p.label);
        if (!out) {
            continue;
        }

        res += out->getBeta(p.id);
    }

    return res - getW2() / 2;
}


// *********************************************************************
// *                     Exported create function                      *
// *********************************************************************
Classifier *create_linear_larank ()
{
  return new LaRank();
}

}; // LinearLaRank



#if 0
class LaRank : public Machine
{
public:


    LaRankOutput *getOutput (int index)
    {
        outputhash_t::iterator it = outputs.find(index);
        return it == outputs.end() ? NULL : &it->second;
    }

    const LaRankOutput *getOutput (int index) const
    {
        outputhash_t::const_iterator it = outputs.find(index);
        return it == outputs.end() ? NULL : &it->second;
    }

#endif
