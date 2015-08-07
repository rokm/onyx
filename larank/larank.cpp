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

#include "decision_function.h"
#include "pattern.h"

#include <chrono>
#include <limits>
#include <map>
#include <unordered_set>
#include <random>


namespace LinearLaRank {


// *********************************************************************
// *                   Linear LaRank implementation                    *
// *********************************************************************
class LaRank : public Classifier
{
public:
    LaRank ();
    virtual ~LaRank ();

    virtual float getC () const;
    virtual void setC (float C);

    virtual float getTau () const;
    virtual void setTau (float);

    virtual int update (const Eigen::VectorXf &features, int label, float weight);

    virtual int predict (const Eigen::VectorXf &features) const;
    virtual int predict (const Eigen::VectorXf &features, Eigen::VectorXf &scores) const;

    virtual float computeDualityGap () const;

    virtual void seedRngEngine (unsigned int);

    virtual void saveToStream (std::ostream &stream) const;
    virtual void loadFromStream (std::istream &stream);

private:
    // Per-class gradient
    struct gradient_t
    {
        int label;
        float gradient;

        gradient_t (int label = 0, float gradient = 0.0)
            : label(label), gradient(gradient)
        {
        }

        bool operator < (const gradient_t &og) const
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
        float dual_increase;
        int predicted_label;

        process_return_t (float dual, int predicted_label)
            : dual_increase(dual), predicted_label(predicted_label)
        {
        }
    };

private:
    // Decision function management
    DecisionFunction *getDecisionFunction (int label);
    const DecisionFunction *getDecisionFunction (int label) const;

    // Pattern storage
    void storePattern (const Pattern &pattern);
    void removePattern (unsigned int i);
    const Pattern &getRandomPattern ();

    // Processing
    process_return_t process (const Pattern &pattern, process_type_t ptype);
    float reprocess ();
    float optimize ();
    unsigned int cleanup ();

    // Statistics
    int getNSV () const;
    float getW2 () const;
    float getDualObjective () const;

private:
    // Parameters
    float C = 1.0;
    float tau = 0.0001;

    // Learnt output decision functions
    std::map<int, DecisionFunction> decision_functions;

    // Pattern cache (used for storing support vectors)
    std::vector<Pattern> stored_patterns;
    std::unordered_set<unsigned int> free_pattern_idx;

    // Internal state
    uint32_t num_features = 0;

    uint64_t num_seen_samples = 0;
    uint64_t num_removed = 0;

    uint64_t num_pro = 0;
    uint64_t num_rep = 0;
    uint64_t num_opt = 0;

    float w_pro = 1.0;
    float w_rep = 1.0;
    float w_opt = 1.0;

    float dual = 0.0;

    // Random number generator
    std::mt19937 rng;
};


// *********************************************************************
// *                      Constructor/destructor                       *
// *********************************************************************
LaRank::LaRank ()
    : rng(std::random_device{}())
{
}

LaRank::~LaRank ()
{
}


// *********************************************************************
// *                         Public interface                          *
// *********************************************************************
float LaRank::getC () const
{
    return C;
}

void LaRank::setC (float C)
{
    this->C = C;
}


float LaRank::getTau () const
{
    return tau;
}

void LaRank::setTau (float tau)
{
    this->tau = tau;
}


void LaRank::seedRngEngine (unsigned int seed)
{
    rng.seed(seed);
}


// *********************************************************************
// *                   Serialization/deserialization                   *
// *********************************************************************
template<typename T>
std::ostream &binary_write (std::ostream &stream, const T &value)
{
    return stream.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

template<typename T>
std::istream &binary_read (std::istream &stream, T &value)
{
    return stream.read(reinterpret_cast<char *>(&value), sizeof(T));
}

void LaRank::saveToStream (std::ostream &stream) const
{
    // Signature
    binary_write(stream, 'O');
    binary_write(stream, 'L');
    binary_write(stream, 'L');
    binary_write(stream, 'R');

    // Format version
    binary_write(stream, static_cast<unsigned int>(1));

    // Parameters
    binary_write(stream, C);
    binary_write(stream, tau);

    // Internal state
    binary_write(stream, num_features);
    binary_write(stream, num_seen_samples);
    binary_write(stream, num_removed);

    binary_write(stream, num_pro);
    binary_write(stream, num_rep);
    binary_write(stream, num_opt);

    binary_write(stream, w_pro);
    binary_write(stream, w_rep);
    binary_write(stream, w_opt);

    binary_write(stream, dual);

    // Pattern cache (support vectors)
    // NOTE: we store only valid patterns, thereby effectively
    // pruning the stored_patterns vector and leaving the free_pattern_idx
    // set empty...
    unsigned int num_patterns = stored_patterns.size() - free_pattern_idx.size();
    binary_write(stream, num_patterns);

    for (auto const &pattern : stored_patterns) {
        // Skip invalid
        if (!pattern.isValid()) {
            continue;
        }

        // Sample ID
        binary_write(stream, pattern.id);

        // Sample features
        for (unsigned int f = 0; f < num_features; f++) {
            binary_write(stream, pattern.features[f]);
        }

        // Sample label
        binary_write(stream, pattern.label);

        // Sample weiight
        binary_write(stream, pattern.weight);
    }

    // Learnt decision functions
    binary_write(stream, static_cast<unsigned int>(decision_functions.size()));
    for (auto const &itd : decision_functions) {
        int label = itd.first;
        const DecisionFunction &out = itd.second;

        // Write decision function's label
        binary_write(stream, label);

        // Beta values
        binary_write(stream, static_cast<unsigned int>(out.beta.size()));
        for (auto const &itb : out.beta) {
            int64_t pattern_id = itb.first;
            float beta_value = itb.second;

            binary_write(stream, pattern_id);
            binary_write(stream, beta_value);
        }

        // Hyperplane weights
        for (unsigned int f = 0; f < num_features; f++) {
            binary_write(stream, out.w[f]);
        }
    }
}

void LaRank::loadFromStream (std::istream &stream)
{
    // Signature
    char sig[4];
    binary_read(stream, sig[0]);
    binary_read(stream, sig[1]);
    binary_read(stream, sig[2]);
    binary_read(stream, sig[3]);

    if (sig[0] != 'O' || sig[1] != 'L' || sig[2] != 'L' || sig[3] != 'R') {
        throw std::runtime_error("Invalid signature at beginning of the stream!");
    }

    // Format version
    unsigned int version;
    binary_read(stream, version);

    if (version != 1) {
        throw std::runtime_error("Invalid stream error!");
    }

    // Clear up the cache
    decision_functions.clear();
    stored_patterns.clear();
    free_pattern_idx.clear();

    // Parameters
    binary_read(stream, C);
    binary_read(stream, tau);

    // Internal state
    binary_read(stream, num_features);
    binary_read(stream, num_seen_samples);
    binary_read(stream, num_removed);

    binary_read(stream, num_pro);
    binary_read(stream, num_rep);
    binary_read(stream, num_opt);

    binary_read(stream, w_pro);
    binary_read(stream, w_rep);
    binary_read(stream, w_opt);

    binary_read(stream, dual);

    // Pattern cache (support vectors)
    // We stored only valid patterns, so we need to restore only the
    // stored_patterns vector, and leave free_pattern_idx empty
    unsigned int num_patterns;
    binary_read(stream, num_patterns);

    stored_patterns.resize(num_patterns);
    for (unsigned int p = 0; p < num_patterns; p++) {
        Pattern &pattern = stored_patterns[p];

        // Sample ID
        binary_read(stream, pattern.id);

        // Sample features
        pattern.features.resize(num_features);
        for (unsigned int f = 0; f < num_features; f++) {
            binary_read(stream, pattern.features[f]);
        }

        // Sample label
        binary_read(stream, pattern.label);

        // Sample weiight
        binary_read(stream, pattern.weight);
    }

    // Learnt decision functions
    unsigned int num_outputs;
    binary_read(stream, num_outputs);

    for (unsigned int o = 0; o < num_outputs; o++) {
        int label;

        // Label
        binary_read(stream, label);

        decision_functions.insert(std::make_pair(label, DecisionFunction(num_features)));
        DecisionFunction &out = decision_functions[label];

        // Beta values
        unsigned int num_betas;
        binary_read(stream, num_betas);

        for (unsigned int b = 0; b < num_betas; b++) {
            int64_t pattern_id;
            float beta_value;

            binary_read(stream, pattern_id);
            binary_read(stream, beta_value);

            out.beta[pattern_id] = beta_value;
        }

        // Hyperplane weights
        out.w.resize(num_features);
        for (unsigned int f = 0; f < num_features; f++) {
            binary_read(stream, out.w[f]);
        }
    }
}


// *********************************************************************
// *                              Update                               *
// *********************************************************************
// Adds new pattern and runs optimization steps chosen with adaptive schedule
int LaRank::update (const Eigen::VectorXf &features, int label, float weight)
{
    // If this is the first sample, deduce the number of features in the
    // feature vector
    if (!num_seen_samples) {
        num_features = features.size();
    }

    // Update counter of seen samples
    num_seen_samples++;

    // If we have not seen this class before, create a new decision
    // function
    if (!decision_functions.count(label)) {
        decision_functions.insert(std::make_pair(label, DecisionFunction(num_features)));
    }

    // Fill pattern element
    Pattern pattern(num_seen_samples, features, label, weight);

    // *** ProcessNew with the "fresh" pattern ***
    auto timeStart = std::chrono::steady_clock::now();

    process_return_t pro_ret = process(pattern, processNew);
    float dual_increase = pro_ret.dual_increase;
    dual += dual_increase;

    float duration = std::chrono::duration<float>(std::chrono::steady_clock::now() - timeStart).count();
    float coeff = dual_increase / (10*(0.00001 + duration));
    num_pro++;
    w_pro = 0.05 * coeff + (1.0 - 0.05) * w_pro;

    // *** ProcessOld & Optimize until ready for a new ProcessNew ***
    // (Adaptive schedule here)
    for (;;) {
        float w_sum = w_pro + w_rep + w_opt;
        float prop_min = w_sum / 20;

        w_pro = std::max(w_pro, prop_min);
        w_rep = std::max(w_rep, prop_min);
        w_opt = std::max(w_opt, prop_min);

        w_sum = w_pro + w_rep + w_opt;

        std::uniform_real_distribution<float> uniform_dist(0, w_sum);
        float r = uniform_dist(rng);

        if (r <= w_pro) {
            break;
        } else if ((r > w_pro) && (r <= w_pro + w_rep)) {
            auto timeStart = std::chrono::steady_clock::now();

            float dual_increase = reprocess();

            dual += dual_increase;

            float duration = std::chrono::duration<float>(std::chrono::steady_clock::now() - timeStart).count();
            float coeff = dual_increase / (0.00001 + duration);

            num_rep++;
            w_rep = 0.05 * coeff + (1 - 0.05) * w_rep;
        } else {
            auto timeStart = std::chrono::steady_clock::now();

            float dual_increase = optimize();

            dual += dual_increase;

            float duration = std::chrono::duration<float>(std::chrono::steady_clock::now() - timeStart).count();
            float coeff = dual_increase / (0.00001 + duration);

            num_opt++;
            w_opt = 0.05 * coeff + (1 - 0.05) * w_opt;
        }
    }

    if (num_seen_samples % 100 == 0) {
        num_removed += cleanup();
    }

    return pro_ret.predicted_label;
}

// *********************************************************************
// *                              Predict                              *
// *********************************************************************
int LaRank::predict (const Eigen::VectorXf &features) const
{
    int res = -1;
    float score_max = -std::numeric_limits<float>::max();

    for (auto it = decision_functions.begin(); it != decision_functions.end(); it++) {
        float score = it->second.computeScore(features);

        if (score > score_max) {
            score_max = score;
            res = it->first;
        }
    }

    return res;
}

int LaRank::predict (const Eigen::VectorXf &features, Eigen::VectorXf &scores) const
{
    int res = -1;
    float score_max = -std::numeric_limits<float>::max();
    int nClass = 0;

    for (auto it = decision_functions.begin(); it != decision_functions.end(); it++) {
        float score = it->second.computeScore(features);

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
float LaRank::computeDualityGap () const
{
    float sum_sl = 0.0;
    float sum_bi = 0.0;

    for (unsigned i = 0; i < stored_patterns.size(); i++) {
        const Pattern &pattern = stored_patterns[i];
        if (!pattern.isValid()) {
            continue;
        }

        const DecisionFunction *out = getDecisionFunction(pattern.label);
        if (!out) {
            continue;
        }

        sum_bi += out->getBeta(pattern.id);
        float gi = out->computeGradient(pattern.features, pattern.label, pattern.label);
        float gmin = std::numeric_limits<float>::max();

        for (auto it = decision_functions.begin(); it != decision_functions.end(); it++) {
            if (it->first != pattern.label && it->second.isSupportVector(pattern.id)) {
                float g = it->second.computeGradient(pattern.features, pattern.label, it->first);
                if (g < gmin) {
                    gmin = g;
                }
            }
        }

        sum_sl += std::max(0.0f, gi-gmin);
    }

    return std::max(0.0f, getW2() + C * sum_sl - sum_bi);
}


// *********************************************************************
// *                          Pattern storage                          *
// *********************************************************************
void LaRank::storePattern (const Pattern &pattern)
{
    // If we have a free pattern slot available, re-use it; otherwise,
    // enlarge the storage by appending the pattern
    if (free_pattern_idx.size()) {
        auto it = free_pattern_idx.begin();
        stored_patterns[*it] = pattern;
        free_pattern_idx.erase(it);
    } else {
        stored_patterns.push_back(pattern);
    }
}

void LaRank::removePattern (unsigned int i)
{
    stored_patterns[i].invalidate(); // Invalidate the pattern
    free_pattern_idx.insert(i); // Mark the slot as free
}

const Pattern &LaRank::getRandomPattern ()
{
    std::uniform_int_distribution<unsigned int> uniform_dist(0, stored_patterns.size() - 1);
    while (true) {
        unsigned int r = uniform_dist(rng);
        if (stored_patterns[r].isValid()) {
            return stored_patterns[r];
        }
    }

    return stored_patterns[0];
}


// *********************************************************************
// *                            Processing                             *
// *********************************************************************
DecisionFunction *LaRank::getDecisionFunction (int label)
{
    auto it = decision_functions.find(label);
    return it == decision_functions.end() ? NULL : &it->second;
}

const DecisionFunction *LaRank::getDecisionFunction (int label) const
{
    auto it = decision_functions.find(label);
    return it == decision_functions.end() ? NULL : &it->second;
}

// Main optimization step
LaRank::process_return_t LaRank::process (const Pattern &pattern, process_type_t ptype)
{
    process_return_t pro_ret = process_return_t(0, 0);

    std::vector<gradient_t> gradients;
    gradients.reserve(decision_functions.size());

    std::vector<gradient_t> scores;
    scores.reserve(decision_functions.size());

    // Compute gradient and sort
    for (auto it = decision_functions.begin(); it != decision_functions.end(); it++) {
        if (ptype != processOptimize || it->second.isSupportVector(pattern.id)) {
            float g = it->second.computeGradient(pattern.features, pattern.label, it->first);

            gradients.push_back(gradient_t(it->first, g));

            if (it->first == pattern.label) {
                scores.push_back(gradient_t(it->first,(1.0 - g)));
            } else {
                scores.push_back(gradient_t(it->first, -g));
            }
        }
    }

    std::sort(gradients.begin(), gradients.end());

    // Determine the prediction and its confidence
    std::sort(scores.begin(), scores.end());
    pro_ret.predicted_label = scores[0].label;

    // Find yp
    gradient_t ygp;
    DecisionFunction *outp = NULL;
    unsigned int p;
    for (p = 0; p < gradients.size(); p++) {
        gradient_t &current = gradients[p];
        DecisionFunction *output = getDecisionFunction(current.label);

        bool support = (ptype == processOptimize || output->isSupportVector(pattern.id));
        bool goodclass = (current.label == pattern.label);

        if ((!support && goodclass) || (support && output->getBeta(pattern.id) < (goodclass ? C * pattern.weight : 0))) {
            ygp = current;
            outp = output;
            break;
        }
    }

    if (p == gradients.size()) {
        return pro_ret;
    }

    // Find ym
    gradient_t ygm;
    DecisionFunction *outm = NULL;
    int m;
    for (m = gradients.size() - 1; m >= 0; m--) {
        gradient_t &current = gradients[m];
        DecisionFunction *output = getDecisionFunction(current.label);

        bool support = (ptype == processOptimize || output->isSupportVector(pattern.id));
        bool goodclass = (current.label == pattern.label);

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
        storePattern(pattern);
    }

    // Compute lambda and clip it
    float kii = pattern.features.squaredNorm();
    float lambda = (ygp.gradient - ygm.gradient) / (2 * kii);
    if (ptype == processOptimize || outp->isSupportVector(pattern.id)) {
        float beta = outp->getBeta(pattern.id);

        if (ygp.label == pattern.label) {
            lambda = std::min(lambda, C * pattern.weight - beta);
        } else {
            lambda = std::min(lambda, std::fabs(beta));
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
float LaRank::reprocess ()
{
    if (stored_patterns.size() == free_pattern_idx.size()) {
        // No valid patterns available...
        return 0.0;
    }

    for (int n = 0; n < 10; n++) {
        process_return_t pro_ret = process(getRandomPattern(), processOld);
        if (pro_ret.dual_increase) {
            return pro_ret.dual_increase;
        }
    }

    return 0.0;
}

// 3rd optimization function
float LaRank::optimize ()
{
    if (stored_patterns.size() == free_pattern_idx.size()) {
        // No valid patterns available...
        return 0.0;
    }

    float dual_increase = 0.0;
    for (int n = 0; n < 10; n++) {
        process_return_t pro_ret = process(getRandomPattern(), processOptimize);
        dual_increase += pro_ret.dual_increase;
    }

    return dual_increase;
}

// remove patterns and return the number of patterns that were removed
unsigned int LaRank::cleanup ()
{
    unsigned int count = 0;

    for (unsigned int i = 0; i < stored_patterns.size(); i++) {
        const Pattern &pattern = stored_patterns[i];
        if (pattern.isValid() && !decision_functions[pattern.label].isSupportVector(pattern.id)) {
            removePattern(i);
            count++;
        }
    }

    return count;
}


// *********************************************************************
// *                       Information functions                       *
// *********************************************************************
// Number of Support Vectors
int LaRank::getNSV () const
{
    int res = 0;

    for (auto it = decision_functions.begin(); it != decision_functions.end(); it++) {
        res += it->second.getNSV();
    }

    return res;
}

// Square-norm of the weight vector w
float LaRank::getW2 () const
{
    float res = 0.0;

    for (auto it = decision_functions.begin(); it != decision_functions.end(); it++) {
        res += it->second.getW2();
    }

    return res;
}

// Dual objective value
float LaRank::getDualObjective () const
{
    float res = 0.0;

    for (unsigned int i = 0; i < stored_patterns.size(); i++) {
        const Pattern &pattern = stored_patterns[i];

        if (!pattern.isValid()) {
            continue;
        }

        const DecisionFunction *out = getDecisionFunction(pattern.label);
        if (!out) {
            continue;
        }

        res += out->getBeta(pattern.id);
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
