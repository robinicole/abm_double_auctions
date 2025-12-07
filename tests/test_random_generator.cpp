#include "random_generator.h"

#include <gtest/gtest.h>

#include <cmath>
#include <set>

namespace abm {
namespace {

class RandomGeneratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    rng_ = std::make_unique<RandomGenerator>(42);  // Fixed seed for reproducibility
  }

  std::unique_ptr<RandomGenerator> rng_;
};

TEST_F(RandomGeneratorTest, UniformReturnsValueInRange) {
  for (int i = 0; i < 1000; i++) {
    double val = rng_->uniform();
    EXPECT_GE(val, 0.0);
    EXPECT_LT(val, 1.0);
  }
}

TEST_F(RandomGeneratorTest, UniformDistributionApproximatelyUniform) {
  const int kNumSamples = 10000;
  const int kNumBuckets = 10;
  std::vector<int> buckets(kNumBuckets, 0);

  for (int i = 0; i < kNumSamples; i++) {
    double val = rng_->uniform();
    int bucket = static_cast<int>(val * kNumBuckets);
    if (bucket >= kNumBuckets) bucket = kNumBuckets - 1;
    buckets[bucket]++;
  }

  // Each bucket should have approximately kNumSamples/kNumBuckets samples
  double expected = static_cast<double>(kNumSamples) / kNumBuckets;
  for (int i = 0; i < kNumBuckets; i++) {
    EXPECT_NEAR(buckets[i], expected, expected * 0.2);  // Within 20%
  }
}

TEST_F(RandomGeneratorTest, Int32ProducesVariety) {
  std::set<uint32_t> values;
  for (int i = 0; i < 100; i++) {
    values.insert(rng_->int32());
  }
  // Should produce at least 90 unique values out of 100
  EXPECT_GE(values.size(), 90u);
}

TEST_F(RandomGeneratorTest, GaussianMeanIsCorrect) {
  const int kNumSamples = 10000;
  const double kMean = 5.0;
  const double kStddev = 2.0;

  double sum = 0.0;
  for (int i = 0; i < kNumSamples; i++) {
    sum += rng_->gaussian(kMean, kStddev);
  }

  double sample_mean = sum / kNumSamples;
  EXPECT_NEAR(sample_mean, kMean, 0.1);  // Within 0.1 of expected mean
}

TEST_F(RandomGeneratorTest, GaussianStddevIsCorrect) {
  const int kNumSamples = 10000;
  const double kMean = 0.0;
  const double kStddev = 3.0;

  double sum = 0.0;
  double sum_sq = 0.0;
  for (int i = 0; i < kNumSamples; i++) {
    double val = rng_->gaussian(kMean, kStddev);
    sum += val;
    sum_sq += val * val;
  }

  double sample_mean = sum / kNumSamples;
  double sample_variance = (sum_sq / kNumSamples) - (sample_mean * sample_mean);
  double sample_stddev = std::sqrt(sample_variance);

  EXPECT_NEAR(sample_stddev, kStddev, 0.2);  // Within 0.2 of expected stddev
}

TEST_F(RandomGeneratorTest, SeedProducesReproducibleResults) {
  RandomGenerator rng1(12345);
  RandomGenerator rng2(12345);

  for (int i = 0; i < 100; i++) {
    EXPECT_EQ(rng1.uniform(), rng2.uniform());
    EXPECT_EQ(rng1.int32(), rng2.int32());
  }
}

TEST_F(RandomGeneratorTest, DifferentSeedsProduceDifferentResults) {
  RandomGenerator rng1(12345);
  RandomGenerator rng2(54321);

  bool all_same = true;
  for (int i = 0; i < 100; i++) {
    if (rng1.uniform() != rng2.uniform()) {
      all_same = false;
      break;
    }
  }
  EXPECT_FALSE(all_same);
}

// MockRandomGenerator tests
class MockRandomGeneratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_rng_ = std::make_unique<MockRandomGenerator>(0.5, 0.0);
  }

  std::unique_ptr<MockRandomGenerator> mock_rng_;
};

TEST_F(MockRandomGeneratorTest, ReturnsFixedUniformValue) {
  mock_rng_->setUniformValue(0.75);
  for (int i = 0; i < 10; i++) {
    EXPECT_DOUBLE_EQ(mock_rng_->uniform(), 0.75);
  }
}

TEST_F(MockRandomGeneratorTest, ReturnsGaussianWithOffset) {
  mock_rng_->setGaussianValue(1.5);
  double result = mock_rng_->gaussian(10.0, 2.0);
  EXPECT_DOUBLE_EQ(result, 10.0 + 2.0 * 1.5);
}

TEST_F(MockRandomGeneratorTest, SequenceReturnsValuesInOrder) {
  mock_rng_->setUniformSequence({0.1, 0.2, 0.3, 0.4});

  EXPECT_DOUBLE_EQ(mock_rng_->uniform(), 0.1);
  EXPECT_DOUBLE_EQ(mock_rng_->uniform(), 0.2);
  EXPECT_DOUBLE_EQ(mock_rng_->uniform(), 0.3);
  EXPECT_DOUBLE_EQ(mock_rng_->uniform(), 0.4);
}

// Global random generator tests
TEST(GlobalRandomTest, GetGlobalRandomReturnsValidGenerator) {
  IRandomGenerator& rng = getGlobalRandom();
  double val = rng.uniform();
  EXPECT_GE(val, 0.0);
  EXPECT_LT(val, 1.0);
}

TEST(GlobalRandomTest, SetGlobalRandomChangesGenerator) {
  auto mock = std::make_unique<MockRandomGenerator>(0.99);
  setGlobalRandom(std::move(mock));

  IRandomGenerator& rng = getGlobalRandom();
  EXPECT_DOUBLE_EQ(rng.uniform(), 0.99);

  // Reset to default
  setGlobalRandom(std::make_unique<RandomGenerator>());
}

}  // namespace
}  // namespace abm
