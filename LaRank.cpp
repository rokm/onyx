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


#include <iostream>
#include <vector>
#include <algorithm>

#include <cstdlib>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cfloat>
#include <cassert>

#include <chrono>

#include "LaRank.h"


// OUTPUT: one per class of the training set.
// stores the value of the parameter vector w.
// (there is actually one w per class, wy)
class LaRankOutput
{
public:
    LaRankOutput (int numFeatures  = 0)
        : wy(Eigen::VectorXd::Zero(numFeatures))
    {
    }

    /*LaRankOutput (LaFVector &w)
        : wy(w)
    {
    }*/

    virtual ~LaRankOutput ()
    {
    }


    double computeGradient (const Eigen::VectorXd &features, int label, int this_label)
    {
        return (label == this_label ? 1.0 : 0.0) - computeScore(features);
    }

    double computeScore (const Eigen::VectorXd &features)
    {
        return wy.dot(features);
    }

    void update (const Eigen::VectorXd &features, double lambda, int pattern_id)
    {
        wy += lambda * features;

        // Update indicator value
        double beta_value = getBeta(pattern_id) + lambda;
        if (beta_value != 0.0)  {
            beta[pattern_id] = beta_value;
        } else {
            beta.erase(pattern_id);
        }
    }

    double getBeta (int pattern_id) const
    {
        std::unordered_map<int, double>::const_iterator it = beta.find(pattern_id);
        return (it == beta.end()) ? 0.0 : it->second;
    }

    bool isSupportVector (int pattern_id) const
    {
        return getBeta(pattern_id) != 0;
    }

    int getNSV () const
    {
        return beta.size();
    }

    double getW2 () const
    {
        return wy.dot(wy);
    }

private:
    // Beta (indicator) values of each support vector
    std::unordered_map<int, double> beta;

    // Hyperplane weights
    Eigen::VectorXd wy;
};


// LARANK: here is the big stuff
class LaRank : public Machine
{
public:
    LaRank ()
        : nb_seen_examples(0), nb_removed(0),
          n_pro(0), n_rep(0), n_opt(0),
          w_pro(1), w_rep(1), w_opt(1), dual(0)
    {
    }

    virtual ~LaRank ()
    {
    }

    // LEARNING FUNCTION: add new patterns and run optimization steps selected with dapatative schedule
    virtual int add (const Eigen::VectorXd &features, int label, double weight)
    {
        nb_seen_examples++;

        // create a new output object if never seen this one before
        if (!getOutput(label)) {
            outputs.insert(std::make_pair(label, LaRankOutput(features.size())));
        }

        LaRankPattern pattern(nb_seen_examples, features, label, weight);

        // ProcessNew with the "fresh" pattern
        auto timeStart = std::chrono::steady_clock::now();

        process_return_t pro_ret = process(pattern, processNew);
        double dual_increase = pro_ret.dual_increase;
        dual += dual_increase;

        double duration = std::chrono::duration<double>(std::chrono::steady_clock::now() - timeStart).count();
        double coeff = dual_increase / (10*(0.00001 + duration));
        n_pro++;
        w_pro = 0.05 * coeff + (1 - 0.05) * w_pro;

        // ProcessOld & Optimize until ready for a new processNew
        // (Adaptive schedule here)
        for (;;) {
            double w_sum = w_pro + w_rep + w_opt;
            double prop_min = w_sum / 20;

            if (w_pro < prop_min) {
                w_pro = prop_min;
            }
            if (w_rep < prop_min) {
                w_rep = prop_min;
            }
            if (w_opt < prop_min) {
                w_opt = prop_min;
            }

            w_sum = w_pro + w_rep + w_opt;

            double r = rand() / (double)RAND_MAX * w_sum;
            if (r <= w_pro) {
                break;
            } else if ( (r > w_pro) && (r <= w_pro + w_rep) ) {
                auto timeStart = std::chrono::steady_clock::now();

                double dual_increase = reprocess();

                dual += dual_increase;

                double duration = std::chrono::duration<double>(std::chrono::steady_clock::now() - timeStart).count();
                double coeff = dual_increase / (0.00001 + duration);

                n_rep++;
                w_rep = 0.05 * coeff + (1 - 0.05) * w_rep;
            } else {
                auto timeStart = std::chrono::steady_clock::now();

                double dual_increase = optimize();

                dual += dual_increase;

                double duration = std::chrono::duration<double>(std::chrono::steady_clock::now() - timeStart).count();
                double coeff = dual_increase / (0.00001 + duration);

                n_opt++;
                w_opt = 0.05 * coeff + (1 - 0.05) * w_opt;
            }
        }

        if (nb_seen_examples % 100 == 0) {
            nb_removed += cleanup();
        }

        return pro_ret.ypred;
    }

    // PREDICTION FUNCTION: main function in la_rank_classify
    virtual int predict (const Eigen::VectorXd &features)
    {
        int res = -1;
        double score_max = -DBL_MAX;

        for (outputhash_t::iterator it = outputs.begin(); it != outputs.end(); ++it) {
            double score = it->second.computeScore(features);

            if (score > score_max) {
                score_max = score;
                res = it->first;
            }
        }

        return res;
    }

    virtual int predict (const Eigen::VectorXd &features, Eigen::VectorXd &scores)
    {
        int res = -1;
        double score_max = -DBL_MAX;
        int nClass = 0;

        scores.resize(outputs.size());

        for (outputhash_t::iterator it = outputs.begin(); it != outputs.end(); it++) {
            double score = it->second.computeScore(features);

            if (score > score_max) {
                score_max = score;
                res = it->first;
            }

            scores[nClass++] = score; // Store output score
        }

        return res;
    }

    // Used for saving a model file
    /*virtual void save_outputs (std::ostream &ostr) {
        for (outputhash_t::const_iterator it = outputs.begin(); it != outputs.end(); ++it) {
            it->second.save_output(ostr, it->first);
        }
    }*/

    // Used for loading a model file
    /*virtual void add_output (int y, LaFVector wy) {
        outputs.insert(std::make_pair(y, LaRankOutput(wy)));
    }*/


    // Compute Duality gap (costly but used in stopping criteria in batch mode)
    virtual double computeGap ()
    {
        double sum_sl = 0.0;
        double sum_bi = 0.0;

        for (unsigned i = 0; i < patterns.maxcount(); i++) {
            const LaRankPattern &p = patterns[i];
            if (!p.exists()) {
                continue;
            }

            LaRankOutput *out = getOutput(p.y);
            if (!out) {
                continue;
            }

            sum_bi += out->getBeta(p.x_id);
            double gi = out->computeGradient(p.x, p.y, p.y);
            double gmin = DBL_MAX;

            for (outputhash_t::iterator it = outputs.begin(); it != outputs.end(); it++) {
                if (it->first != p.y && it->second.isSupportVector(p.x_id)) {
                    double g = it->second.computeGradient(p.x, p.y, it->first);
                    if (g < gmin) {
                        gmin = g;
                    }
                }
            }

            sum_sl += std::max(0.0, gi-gmin);
        }

        return std::max(0.0, getW2() + C * sum_sl - sum_bi);
    }

    // Display stuffs along learning
    virtual void printStuff (double initime, bool dual)
    {
        //std::cout << "Current duration (CPUs): " << getTime() - initime << std::endl;
        if (dual) {
            std::cout << "Dual: " << getDual() << " (w2: " << getW2() << ")" << std::endl;
        }
        std::cout << "Number of Support Patterns: " << patterns.size() << " / " << nb_seen_examples << " (removed:" << nb_removed << ")" << std::endl;
        double w_sum = w_pro + w_rep + w_opt;
        std::cout << "ProcessNew:" << n_pro << " ("<< w_pro/w_sum << ") ProcessOld:" << n_rep << " (" << w_rep/w_sum << ") Optimize:" << n_opt << " (" << w_opt/w_sum << ")" << std::endl;
        std::cout << "\t......"<< std::endl;
    }

    virtual unsigned getNumOutputs () const
    {
        return outputs.size();
    }

private:
    typedef std::unordered_map<int, LaRankOutput> outputhash_t; // class index -> LaRankOutput
    outputhash_t outputs;

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

    LaRankPatterns patterns;

    int nb_seen_examples;
    int nb_removed;

    int n_pro;
    int n_rep;
    int n_opt;

    double w_pro;
    double w_rep;
    double w_opt;

    double dual;

    struct outputgradient_t
    {
        outputgradient_t (int output, double gradient)
            : output(output), gradient(gradient)
        {
        }

        outputgradient_t ()
            : output(0), gradient(0)
        {
        }

        int output;
        double  gradient;

        bool operator < (const outputgradient_t &og) const
        {
            return gradient > og.gradient;
        }
    };

    // 3 types of operations in LaRank
    enum process_type
    {
        processNew,
        processOld,
        processOptimize
    };

    struct process_return_t
    {
        process_return_t (double dual, int ypred)
            : dual_increase(dual) , ypred(ypred)
        {
        }

        process_return_t ()
        {
        }

        double dual_increase;
        int ypred;
    };

    // Main optimization step
    process_return_t process (const LaRankPattern &pattern, process_type ptype)
    {
        process_return_t pro_ret = process_return_t(0, 0);
        std::vector<outputgradient_t> outputgradients;
        outputgradients.reserve(getNumOutputs());
        std::vector<outputgradient_t> outputscores;
        outputscores.reserve(getNumOutputs());

        // Compute gradient and sort
        for (outputhash_t::iterator it = outputs.begin(); it != outputs.end(); it++) {
            if (ptype != processOptimize || it->second.isSupportVector(pattern.x_id)) {
                double g = it->second.computeGradient(pattern.x, pattern.y, it->first);

                outputgradients.push_back(outputgradient_t(it->first, g));

                if (it->first == pattern.y) {
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
        LaRankOutput *outp = NULL;
        unsigned p;
        for (p = 0; p < outputgradients.size(); p++) {
            outputgradient_t &current = outputgradients[p];
            LaRankOutput *output = getOutput(current.output);

            bool support = ptype == processOptimize || output->isSupportVector(pattern.x_id);
            bool goodclass = current.output == pattern.y;

            if ((!support && goodclass) || (support && output->getBeta(pattern.x_id) < (goodclass ? C * pattern.w : 0))) {
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
        LaRankOutput *outm = NULL;
        int m;
        for (m = outputgradients.size() - 1; m >= 0; m--) {
            outputgradient_t &current = outputgradients[m];
            LaRankOutput *output = getOutput(current.output);

            bool support = ptype == processOptimize || output->isSupportVector(pattern.x_id);
            bool goodclass = current.output == pattern.y;

            if (!goodclass || (support && output->getBeta(pattern.x_id) > 0)) {
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
        double kii = /*dot(pattern.x,pattern.x);*/ pattern.x.dot(pattern.x);
        double lambda = (ygp.gradient - ygm.gradient) / (2 * kii);
        if (ptype == processOptimize || outp->isSupportVector(pattern.x_id)) {
            double beta = outp->getBeta(pattern.x_id);

            if (ygp.output == pattern.y) {
                lambda = std::min(lambda, C * pattern.w - beta);
            } else {
                lambda = std::min(lambda, fabs(beta));
            }
        } else {
            lambda = std::min(lambda, C * pattern.w);
        }

        // Update parameters
        outp->update(pattern.x, lambda, pattern.x_id);
        outm->update(pattern.x, -lambda, pattern.x_id);
        pro_ret.dual_increase = lambda * ((ygp.gradient - ygm.gradient) - lambda * kii);

        return pro_ret;
    }

    // 2nd optimization fiunction of LaRank (= ProcessOld in the paper)
    double reprocess ()
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

    // 3rd optimization function of LaRank
    double optimize ()
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
    unsigned cleanup ()
    {
        unsigned res = 0;

        for (unsigned i = 0; i < patterns.maxcount(); i++) {
            LaRankPattern &p = patterns[i];
            if (p.exists() && !outputs[p.y].isSupportVector(p.x_id)) {
                patterns.remove(i);
                res++;
            }
        }

        return res;
    }


    /*
      INFORMATION FUNCTIONS: display/compute parameter values of the algorithm
     */

    // Number of Support Vectors
    int getNSV () const
    {
        int res = 0;

        for (outputhash_t::const_iterator it = outputs.begin(); it != outputs.end(); it++) {
            res += it->second.getNSV();
        }

        return res;
    }

    // Square-norm of the weight vector w
    double getW2 () const
    {
        double res = 0.0;

        for (outputhash_t::const_iterator it = outputs.begin(); it != outputs.end(); it++) {
            res += it->second.getW2();
        }

        return res;
    }

    // Dual objective value
    double getDual()
    {
        double res = 0.0;

        for (unsigned i = 0; i < patterns.maxcount(); ++i) {
            const LaRankPattern &p = patterns[i];

            if (!p.exists()) {
                continue;
            }

            const LaRankOutput *out = getOutput(p.y);
            if (!out) {
                continue;
            }

            res += out->getBeta(p.x_id);
        }

        return res - getW2() / 2;
    }
};

/*
**  Create a LaRank object.
*/
Machine *create_larank ()
{
  return new LaRank();
}
