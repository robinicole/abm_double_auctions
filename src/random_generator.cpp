#include "random_generator.h"

#include <memory>

namespace abm {

// RandomGenerator implementation
RandomGenerator::RandomGenerator(uint64_t seed) : engine_(seed) {}

double RandomGenerator::uniform() { return uniform_dist_(engine_); }

uint32_t RandomGenerator::int32() { return static_cast<uint32_t>(engine_()); }

double RandomGenerator::gaussian(double mean, double stddev) {
  return mean + stddev * normal_dist_(engine_);
}

void RandomGenerator::seed(uint64_t seed) { engine_.seed(seed); }

// MockRandomGenerator implementation
MockRandomGenerator::MockRandomGenerator(double fixed_uniform,
                                         double fixed_gaussian)
    : fixed_uniform_(fixed_uniform), fixed_gaussian_(fixed_gaussian) {}

double MockRandomGenerator::uniform() {
  if (use_sequence_ && sequence_index_ < uniform_sequence_.size()) {
    return uniform_sequence_[sequence_index_++];
  }
  return fixed_uniform_;
}

uint32_t MockRandomGenerator::int32() {
  return static_cast<uint32_t>(fixed_uniform_ * UINT32_MAX);
}

double MockRandomGenerator::gaussian(double mean, double stddev) {
  return mean + stddev * fixed_gaussian_;
}

void MockRandomGenerator::setUniformValue(double value) {
  fixed_uniform_ = value;
  use_sequence_ = false;
}

void MockRandomGenerator::setGaussianValue(double value) {
  fixed_gaussian_ = value;
}

void MockRandomGenerator::setUniformSequence(
    const std::vector<double>& values) {
  uniform_sequence_ = values;
  sequence_index_ = 0;
  use_sequence_ = true;
}

// Global random generator
namespace {
std::unique_ptr<IRandomGenerator> global_rng;
}

IRandomGenerator& getGlobalRandom() {
  if (!global_rng) {
    global_rng = std::make_unique<RandomGenerator>();
  }
  return *global_rng;
}

void setGlobalRandom(std::unique_ptr<IRandomGenerator> rng) {
  global_rng = std::move(rng);
}

}  // namespace abm
