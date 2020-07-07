#ifndef __SAMPLES_H
#define __SAMPLES_H

#include <iostream>
#include <fstream>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "distribution.h"
#include "mixture_model.h"

#define ENABLE_ALL 1

namespace jmm {

template<int t_dims, typename Scalar>
class Samples {
public:
    using Vectord = Eigen::Matrix<Scalar, t_dims, 1>;
    using Normal = Eigen::Matrix<Scalar, 3, 1>;
    using Color = Eigen::Matrix<Scalar, 3, 1>;

    Samples() : m_capacity(0), m_end(0) {}

    using SampleVector = Eigen::Matrix<Scalar, t_dims, Eigen::Dynamic>;
    using NormalsVector = Eigen::Matrix<Scalar, 3, Eigen::Dynamic>;
    using ColorVector = Eigen::Matrix<Scalar, 3, Eigen::Dynamic>;
    using ScalarVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using BooleanVector = Eigen::Matrix<bool, Eigen::Dynamic, 1>;
    
    SampleVector samples;
    ScalarVector samplingPdfs;
    ScalarVector learnedPdfs;
    ScalarVector heuristicPdfs;
    ScalarVector heuristicWeights;
    ScalarVector weights;
    ColorVector colorWeights;

    BooleanVector isDiffuse;
    NormalsVector normals;
    ScalarVector stateDensities;
    ScalarVector curvatures;

    ScalarVector rewards;
    ScalarVector discounts;

    // ColorVector functionValues;
    // ScalarVector bsdfWeights;
    // SampleVector nextSamples;
    // ScalarVector nextHeuristicPdfs;
    // ScalarVector nextHeuristicWeights;

    // From RenderingSamples:
    // Eigen::Matrix<Scalar, Eigen::Dynamic, 1> denoisedWeights;
    // std::vector<Point2> sensorPositions;
    // Eigen::Matrix<int, 1, Eigen::Dynamic> depths;
    // std::vector<Spectrum> throughputs;
    
    Vectord meanPosition = Vectord::Zero();
    Vectord meanSquarePosition = Vectord::Zero();
    Normal meanNormal = Normal::Zero();
    Normal meanSquareNormal = Normal::Zero();
    Scalar nSamples = 0.f;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Samples(const Samples& other) {
        meanPosition = other.meanPosition;
        meanSquarePosition = other.meanSquarePosition;
        meanNormal = other.meanNormal;
        meanSquareNormal = other.meanSquareNormal;
        nSamples = other.nSamples;

        m_end = other.m_end;
        m_capacity = other.m_capacity;
        m_totalSamplesCount = other.m_totalSamplesCount;

        samples = other.samples;
        samplingPdfs = other.samplingPdfs;
        learnedPdfs = other.learnedPdfs;
        heuristicPdfs = other.heuristicPdfs;
        heuristicWeights = other.heuristicWeights;
        weights = other.weights;

        isDiffuse = other.isDiffuse;
        normals = other.normals;
        curvatures = other.curvatures;

        #if ENABLE_ALL == 1
        colorWeights = other.colorWeights;
        rewards = other.rewards;
        discounts = other.discounts;
        #endif // ENABLE_ALL == 1

        // functionValues = std::move(other.functionValues);
        // bsdfWeights = std::move(other.bsdfWeights);
        // nextSamples = std::move(other.nextSamples);
        // nextHeuristicPdfs = std::move(other.nextHeuristicPdfs);
        // nextHeuristicWeights = std::move(other.nextHeuristicWeights);
    }

    Samples(Samples&& other) {
        meanPosition = other.meanPosition;
        meanSquarePosition = other.meanSquarePosition;
        meanNormal = other.meanNormal;
        meanSquareNormal = other.meanSquareNormal;
        nSamples = other.nSamples;

        m_end = other.m_end;
        m_capacity = other.m_capacity;
        m_totalSamplesCount = other.m_totalSamplesCount;

        samples = std::move(other.samples);
        samplingPdfs = std::move(other.samplingPdfs);
        learnedPdfs = std::move(other.learnedPdfs);
        heuristicPdfs = std::move(other.heuristicPdfs);
        heuristicWeights = std::move(other.heuristicWeights);
        weights = std::move(other.weights);
        colorWeights = std::move(other.colorWeights);

        isDiffuse = std::move(other.isDiffuse);
        normals = std::move(other.normals);
        curvatures = std::move(other.curvatures);

        rewards = std::move(other.rewards);
        discounts = std::move(other.discounts);

        // functionValues = std::move(other.functionValues);
        // bsdfWeights = std::move(other.bsdfWeights);
        // nextSamples = std::move(other.nextSamples);
        // nextHeuristicPdfs = std::move(other.nextHeuristicPdfs);
        // nextHeuristicWeights = std::move(other.nextHeuristicWeights);
    }

    Samples& operator=(const Samples& other) {
        meanPosition = other.meanPosition;
        meanSquarePosition = other.meanSquarePosition;
        meanNormal = other.meanNormal;
        meanSquareNormal = other.meanSquareNormal;
        nSamples = other.nSamples;

        m_end = other.m_end;
        m_capacity = other.m_capacity;
        m_totalSamplesCount = other.m_totalSamplesCount;

        samples = other.samples;
        samplingPdfs = other.samplingPdfs;
        learnedPdfs = other.learnedPdfs;
        heuristicPdfs = other.heuristicPdfs;
        heuristicWeights = other.heuristicWeights;
        weights = other.weights;
        colorWeights = other.colorWeights;

        isDiffuse = other.isDiffuse;
        normals = other.normals;
        curvatures = other.curvatures;

        rewards = other.rewards;
        discounts = other.discounts;

        // functionValues = other.functionValues;
        // bsdfWeights = other.bsdfWeights;
        // nextSamples = other.nextSamples;
        // nextHeuristicPdfs = other.nextHeuristicPdfs;
        // nextHeuristicWeights = other.nextHeuristicWeights;
        return *this;
    }

    Samples& operator=(Samples&& other) {
        meanPosition = other.meanPosition;
        meanSquarePosition = other.meanSquarePosition;
        meanNormal = other.meanNormal;
        meanSquareNormal = other.meanSquareNormal;
        nSamples = other.nSamples;

        m_end = other.m_end;
        m_capacity = other.m_capacity;
        m_totalSamplesCount = other.m_totalSamplesCount;

        samples = std::move(other.samples);
        samplingPdfs = std::move(other.samplingPdfs);
        learnedPdfs = std::move(other.learnedPdfs);
        heuristicPdfs = std::move(other.heuristicPdfs);
        heuristicWeights = std::move(other.heuristicWeights);
        weights = std::move(other.weights);
        colorWeights = std::move(other.colorWeights);

        isDiffuse = std::move(other.isDiffuse);
        normals = std::move(other.normals);
        curvatures = std::move(other.curvatures);

        rewards = std::move(other.rewards);
        discounts = std::move(other.discounts);

        // functionValues = std::move(other.functionValues);
        // bsdfWeights = std::move(other.bsdfWeights);
        // nextSamples = std::move(other.nextSamples);
        // nextHeuristicPdfs = std::move(other.nextHeuristicPdfs);
        // nextHeuristicWeights = std::move(other.nextHeuristicWeights);
        return *this;
    }

    virtual void reserve(int size) {
        samples.conservativeResize(Eigen::NoChange, size);
        samplingPdfs.conservativeResize(size);
        learnedPdfs.conservativeResize(size);
        heuristicPdfs.conservativeResize(size);
        heuristicWeights.conservativeResize(size);
        weights.conservativeResize(size);
        colorWeights.conservativeResize(Eigen::NoChange, size);

        isDiffuse.conservativeResize(size);
        normals.conservativeResize(Eigen::NoChange, size);
        stateDensities.conservativeResize(size);
        curvatures.conservativeResize(size);

        rewards.conservativeResize(size);
        discounts.conservativeResize(size);

        // functionValues.conservativeResize(Eigen::NoChange, size);
        // bsdfWeights.conservativeResize(size);
        // nextSamples.conservativeResize(Eigen::NoChange, size);
        // nextHeuristicPdfs.conservativeResize(size);
        // nextHeuristicWeights.conservativeResize(size);

        m_capacity = size;
        if(size < m_end) {
            m_end = size;
        }
    }

    void normalizeWeights() {
        weights.topRows(m_end) /= weights.topRows(m_end).sum();
    }

    void clampWeights(Scalar clamping) {
        weights.topRows(m_end) = weights.topRows(m_end).cwiseMin(clamping);
    }

    void clear() {
        m_end = 0;
        m_totalSamplesCount = 0;

        // meanPosition = Vectord::Zero();
        // meanSquarePosition = Vectord::Zero();
        // nSamples = 0;
    }

    template<typename ...FwdArgs>
    void push_back_synchronized(
        FwdArgs&& ...fwdArgs
    ) {
        boost::unique_lock<boost::mutex> lock(mutex);
        push_back(std::forward<FwdArgs>(fwdArgs) ...);
    }

    virtual bool push_back(
        const Vectord& sample,
        // Color functionValue,
        Scalar samplingPdf,
        Scalar learnedPdf,
        Scalar heuristicPdf,
        Scalar heuristicWeight,
        Scalar weight,
        Color colorWeight,

        bool _isDiffuse = 0,
        const Normal& normal = Normal::Zero(),
        Scalar curvature = 0,

        Scalar reward = 0,
        Scalar discount = 0
        // Scalar bsdfWeight = 0,
        // const Vectord& nextSample = Vectord::Zero(),
        // Scalar nextHeuristicPdf = 0,
        // Scalar nextHeuristicWeight = 0
    ) {
        ++m_totalSamplesCount;
        if(weight == 0) {
            return false;
        }

        nSamples += 1;
        meanPosition += sample;
        meanSquarePosition.array() += sample.array().square();
        meanNormal += normal;
        meanSquareNormal.array() += normal.array().square();

        if(m_end == m_capacity) {
            reserve(2 * m_end);
            std::cerr << "WARNING: Resizing sample array.\n";
            // std::cerr << "m_end=" << m_end << " ... " << "samples.cols()=" << samples.cols() << "\n";
        }
        samples.col(m_end) = sample;
        samplingPdfs(m_end) = samplingPdf;
        learnedPdfs(m_end) = learnedPdf;
        heuristicPdfs(m_end) = heuristicPdf;
        heuristicWeights(m_end) = heuristicWeight;
        weights(m_end) = weight;
        colorWeights.col(m_end) = colorWeight;

        isDiffuse(m_end) = _isDiffuse;
        normals.col(m_end) = normal;
        curvatures(m_end) = curvature;

        rewards(m_end) = reward;
        discounts(m_end) = discount;

        // functionValues(m_end) = functionValue;
        // bsdfWeights(m_end) = bsdfWeight;
        // nextSamples.col(m_end) = nextSample;
        // nextHeuristicPdfs(m_end) = nextHeuristicPdf;
        // nextHeuristicWeights(m_end) = nextHeuristicWeight;
        
        // averageWeight = (averageWeight * (Scalar) m_end + weight) / ((Scalar) m_end + 1.f);

        ++m_end;
        return true;
    }

    void push_back(const Samples<t_dims, Scalar>& other, int sample_i) {
        push_back(
            other.samples.col(sample_i),
            other.samplingPdfs(sample_i),
            other.learnedPdfs(sample_i),
            other.heuristicPdfs(sample_i),
            other.heuristicWeights(sample_i),
            other.weights(sample_i),
            other.colorWeights.col(sample_i),

            other.isDiffuse(sample_i),
            other.normals.col(sample_i),
            other.curvatures(sample_i),

            other.rewards(sample_i),
            other.discounts(sample_i)
            // other.functionValues(sample_i),
            // other.bsdfWeights(sample_i),
            // other.nextSamples.col(sample_i),
            // other.nextHeuristicPdfs(sample_i),
            // other.nextHeuristicWeights(sample_i),
        );
    }

    void push_back(const Samples<t_dims, Scalar>& other) {
        // if(m_capacity < m_end + other.m_end) {
        //     reserve(2 * (m_end + other.m_end));
        // }
        for(int sample_i = 0; sample_i < other.m_end; ++sample_i) {
            push_back(other, sample_i);
        }
    }

    Samples prioritizedSample(
        const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& error,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& randomUniform,
        Scalar proportionDone
    ) {
        Scalar priority = 0.7;
        Scalar isWeightDecay = 0.3;
        // isWeightDecay += (1 - isWeightDecay) * proportionDone;

        int bufferSize = error.rows();
        int nSamples = randomUniform.rows();
        assert(bufferSize == m_end);
        std::cerr << "Sampling nSamples: " << nSamples << "\n";
        Samples<t_dims, Scalar> drawn;
        drawn.reserve(nSamples);

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> pdf = error.array().max(0);
        pdf.array() = pdf.array().pow(priority);

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> cdf(bufferSize, 1);
        createCdfEigen(pdf, cdf, true);

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> isWeights = 
            ((Scalar) nSamples / ((Scalar) bufferSize * pdf.array())).pow(isWeightDecay);
        isWeights.array() /= isWeights.maxCoeff();
        std::cerr << "Nan isWeights finite: " << isWeights.array().isFinite().all() << '\n';

        #pragma omp parallel for
        for(int sample_i = 0; sample_i < nSamples; ++sample_i) {
            int sample_j = sampleDiscreteCdf(cdf, randomUniform(sample_i));
            drawn.push_back(*this, sample_j);
            drawn.weights(sample_i) *= isWeights(sample_j);
        }
        return std::move(drawn);
    }

    void russianRoulette(
        const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& randomUniform,
        int nRemainingSamples,
        bool correctWeights
    ) {
        // Scalar sum = weights.topRows(m_end).sum();
        Scalar mean = weights.topRows(m_end).mean();
        // Scalar std = std::sqrt(
        //     (weights.topRows(m_end).array() - mean).square().sum() / ((Scalar) m_end - 1)
        // );
        
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> priorities(size(), 1);
        // priorities.setConstant((Scalar) nRemainingSamples / (Scalar) size());
        priorities = weights.topRows(m_end);
        // priorities.array() *= nRemainingSamples / mean;

        // std::cerr << randomUniform.rows() << "vs." << m_end << std::endl;

        int oldEnd = m_end;
        m_end = 0;
        for(int sample_i = 0; sample_i < oldEnd; ++sample_i) {
            // if(weights(sample_i) / mean < 1.f) {
            //     continue;
            // }
            Scalar keepProbability = std::min(Scalar(1), std::max(Scalar(0.1), priorities(sample_i)));
            // std::cerr << keepProbability << "\n";

            // TERMINATE 
            if(randomUniform(sample_i) > keepProbability) {
                continue;
            }

            if(correctWeights) {
                weights(sample_i) /= keepProbability;
            }
            push_back(*this, sample_i);
        }
        // std::cerr << "RR removed " << oldEnd - m_end << "/" << oldEnd << " samples (remaining=" << m_end << ").\n";
    }


    void save(const std::string& filename) const {
        // make an archive
        std::ofstream ofs(filename.c_str());
        boost::archive::binary_oarchive oa(ofs);
        oa << BOOST_SERIALIZATION_NVP(*this);
    }

    void load(const std::string& filename) {
        // open the archive
        std::ifstream ifs(filename.c_str());
        boost::archive::binary_iarchive ia(ifs);
        ia >> BOOST_SERIALIZATION_NVP(*this);
    }

    int size() const {
        return m_end;
    }

    int capacity() const {
        return m_capacity;
    }

    int totalSamplesCount() const {
        return m_totalSamplesCount;
    }

    void setSize(int size) {
        m_end = size;
    }

protected:
    int m_capacity = 0;
    int m_end = 0;
    int m_totalSamplesCount = 0;

    boost::mutex mutex;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        reserve(m_end);
        ar  & m_capacity;
        ar  & m_end;
        ar  & samples;
        ar  & samplingPdfs;
        ar  & learnedPdfs;
        ar  & heuristicPdfs;
        ar  & heuristicWeights;
        ar  & weights;
        ar  & colorWeights;

        ar  & isDiffuse;
        ar  & stateDensities;
        ar  & curvatures;

        // ar  & functionValues;
        // ar  & bsdfWeights;
        // ar  & nextSamples;
        // ar  & rewards;
        // ar  & discounts;
        // ar  & nextHeuristicPdfs;
        // ar  & nextHeuristicWeights;
    }
};

}

#endif /* __SAMPLES_H */
