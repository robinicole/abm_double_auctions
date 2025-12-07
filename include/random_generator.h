#ifndef ABM_RANDOM_GENERATOR_H
#define ABM_RANDOM_GENERATOR_H

#include <random>
#include <cstdint>

namespace abm {

/**
 * @brief Interface for random number generation
 *
 * This allows dependency injection for testing purposes.
 */
class IRandomGenerator {
public:
    virtual ~IRandomGenerator() = default;

    /**
     * @brief Generate a random double in [0, 1)
     */
    virtual double uniform() = 0;

    /**
     * @brief Generate a random 32-bit integer
     */
    virtual uint32_t int32() = 0;

    /**
     * @brief Generate a Gaussian random number
     * @param mean The mean of the distribution
     * @param stddev The standard deviation
     */
    virtual double gaussian(double mean, double stddev) = 0;
};

/**
 * @brief Default random number generator using the Mersenne Twister algorithm
 */
class RandomGenerator : public IRandomGenerator {
public:
    /**
     * @brief Construct with a seed
     * @param seed The random seed (default: random_device)
     */
    explicit RandomGenerator(uint64_t seed = std::random_device{}());

    double uniform() override;
    uint32_t int32() override;
    double gaussian(double mean, double stddev) override;

    /**
     * @brief Reseed the generator
     */
    void seed(uint64_t seed);

private:
    std::mt19937_64 engine_;
    std::uniform_real_distribution<double> uniform_dist_{0.0, 1.0};
    std::normal_distribution<double> normal_dist_{0.0, 1.0};
};

/**
 * @brief Mock random generator for deterministic testing
 */
class MockRandomGenerator : public IRandomGenerator {
public:
    explicit MockRandomGenerator(double fixed_uniform = 0.5, double fixed_gaussian = 0.0);

    double uniform() override;
    uint32_t int32() override;
    double gaussian(double mean, double stddev) override;

    void setUniformValue(double value);
    void setGaussianValue(double value);
    void setUniformSequence(const std::vector<double>& values);

private:
    double fixed_uniform_;
    double fixed_gaussian_;
    std::vector<double> uniform_sequence_;
    size_t sequence_index_ = 0;
    bool use_sequence_ = false;
};

/**
 * @brief Global random generator singleton (for backward compatibility)
 *
 * @note Prefer dependency injection over using this global instance
 */
IRandomGenerator& getGlobalRandom();
void setGlobalRandom(std::unique_ptr<IRandomGenerator> rng);

} // namespace abm

#endif // ABM_RANDOM_GENERATOR_H
