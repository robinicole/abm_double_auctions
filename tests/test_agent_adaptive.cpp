#include "agent_adaptive.h"

#include <gtest/gtest.h>

#include <cmath>

namespace abm {
namespace {

class AgentAdaptiveTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_rng_ = std::make_unique<MockRandomGenerator>(0.5, 0.0);
    agent_ = std::make_unique<AgentAdaptive>(*mock_rng_);

    AgentParams params;
    params.id = 99;
    params.sigma = 1.0;
    params.forget_param = 0.1;
    params.mean_buy = 11.0;
    params.mean_sell = 10.0;
    params.temperature = 0.2;

    agent_->Initialize(params);
  }

  std::unique_ptr<MockRandomGenerator> mock_rng_;
  std::unique_ptr<AgentAdaptive> agent_;
};

TEST_F(AgentAdaptiveTest, InitializeSetsCorrectValues) {
  EXPECT_EQ(agent_->GetId(), 99);
  EXPECT_DOUBLE_EQ(agent_->GetForgetParam(), 0.1);
  EXPECT_DOUBLE_EQ(agent_->GetTemperature(), 0.2);
  EXPECT_DOUBLE_EQ(agent_->GetBuyProbability(), 0.0);  // Not applicable
  EXPECT_EQ(agent_->TypeAgent(), "adaptive");
}

TEST_F(AgentAdaptiveTest, InitializeScoreSetsRandomValues) {
  mock_rng_->setGaussianValue(2.0);
  agent_->InitializeScore(0.5);

  // With gaussian returning mean + stddev * 2.0 = 0 + 0.5 * 2.0 = 1.0
  EXPECT_DOUBLE_EQ(agent_->GetAB1(), 1.0);
  EXPECT_DOUBLE_EQ(agent_->GetAS1(), 1.0);
  EXPECT_DOUBLE_EQ(agent_->GetAB2(), 1.0);
  EXPECT_DOUBLE_EQ(agent_->GetAS2(), 1.0);
}

TEST_F(AgentAdaptiveTest, ChoiceMarketSelectsSellMarket1WhenRandomVeryLow) {
  // With all scores at 0 and T=0.2:
  // exp(0/0.2) = 1 for all four options
  // Each option has probability 0.25
  // Cumulative: [0, 0.25) = sell@1, [0.25, 0.5) = buy@1,
  //            [0.5, 0.75) = sell@2, [0.75, 1.0) = buy@2

  mock_rng_->setUniformValue(0.1);  // Falls in [0, 0.25)
  agent_->ChoiceMarket();

  EXPECT_EQ(agent_->GetChoice(), 1);
  EXPECT_EQ(agent_->GetAction(), Action::kSell);
}

TEST_F(AgentAdaptiveTest, ChoiceMarketSelectsBuyMarket1WhenRandomLow) {
  mock_rng_->setUniformValue(0.3);  // Falls in [0.25, 0.5)
  agent_->ChoiceMarket();

  EXPECT_EQ(agent_->GetChoice(), 1);
  EXPECT_EQ(agent_->GetAction(), Action::kBuy);
}

TEST_F(AgentAdaptiveTest, ChoiceMarketSelectsSellMarket2WhenRandomMedium) {
  mock_rng_->setUniformValue(0.6);  // Falls in [0.5, 0.75)
  agent_->ChoiceMarket();

  EXPECT_EQ(agent_->GetChoice(), 2);
  EXPECT_EQ(agent_->GetAction(), Action::kSell);
}

TEST_F(AgentAdaptiveTest, ChoiceMarketSelectsBuyMarket2WhenRandomHigh) {
  mock_rng_->setUniformValue(0.9);  // Falls in [0.75, 1.0)
  agent_->ChoiceMarket();

  EXPECT_EQ(agent_->GetChoice(), 2);
  EXPECT_EQ(agent_->GetAction(), Action::kBuy);
}

TEST_F(AgentAdaptiveTest, UpdateScoreUpdatesAB1WhenBuyingAtMarket1) {
  mock_rng_->setUniformValue(0.3);  // Buy at market 1
  agent_->ChoiceMarket();

  agent_->UpdateScore(1.0);

  // AB1 = (1-0.1)*0 + 1.0*0.1 = 0.1
  // Others decay by (1 - 1.0 * 0.1) = 0.9, but they're already 0
  EXPECT_NEAR(agent_->GetAB1(), 0.1, 1e-10);
  EXPECT_DOUBLE_EQ(agent_->GetAS1(), 0.0);
  EXPECT_DOUBLE_EQ(agent_->GetAB2(), 0.0);
  EXPECT_DOUBLE_EQ(agent_->GetAS2(), 0.0);
  EXPECT_DOUBLE_EQ(agent_->GetLastScore(), 1.0);
}

TEST_F(AgentAdaptiveTest, UpdateScoreUpdatesAS1WhenSellingAtMarket1) {
  mock_rng_->setUniformValue(0.1);  // Sell at market 1
  agent_->ChoiceMarket();

  agent_->UpdateScore(2.0);

  // AS1 = (1-0.1)*0 + 2.0*0.1 = 0.2
  EXPECT_DOUBLE_EQ(agent_->GetAB1(), 0.0);
  EXPECT_NEAR(agent_->GetAS1(), 0.2, 1e-10);
  EXPECT_DOUBLE_EQ(agent_->GetAB2(), 0.0);
  EXPECT_DOUBLE_EQ(agent_->GetAS2(), 0.0);
}

TEST_F(AgentAdaptiveTest, UpdateScoreUpdatesAB2WhenBuyingAtMarket2) {
  mock_rng_->setUniformValue(0.9);  // Buy at market 2
  agent_->ChoiceMarket();

  agent_->UpdateScore(3.0);

  // AB2 = (1-0.1)*0 + 3.0*0.1 = 0.3
  EXPECT_DOUBLE_EQ(agent_->GetAB1(), 0.0);
  EXPECT_DOUBLE_EQ(agent_->GetAS1(), 0.0);
  EXPECT_NEAR(agent_->GetAB2(), 0.3, 1e-10);
  EXPECT_DOUBLE_EQ(agent_->GetAS2(), 0.0);
}

TEST_F(AgentAdaptiveTest, UpdateScoreUpdatesAS2WhenSellingAtMarket2) {
  mock_rng_->setUniformValue(0.6);  // Sell at market 2
  agent_->ChoiceMarket();

  agent_->UpdateScore(4.0);

  // AS2 = (1-0.1)*0 + 4.0*0.1 = 0.4
  EXPECT_DOUBLE_EQ(agent_->GetAB1(), 0.0);
  EXPECT_DOUBLE_EQ(agent_->GetAS1(), 0.0);
  EXPECT_DOUBLE_EQ(agent_->GetAB2(), 0.0);
  EXPECT_NEAR(agent_->GetAS2(), 0.4, 1e-10);
}

TEST_F(AgentAdaptiveTest, UpdateScoreDecaysOtherScores) {
  // First, set some initial scores
  mock_rng_->setGaussianValue(1.0);
  agent_->InitializeScore(1.0);  // All scores = 1.0

  // Choose to buy at market 1
  mock_rng_->setUniformValue(0.3);
  agent_->ChoiceMarket();

  // Update with score = 0 (just to see decay)
  agent_->UpdateScore(0.0);

  // AB1 = (1-0.1)*1.0 + 0*0.1 = 0.9
  // Others decay by (1 - 1.0 * 0.1) = 0.9
  EXPECT_NEAR(agent_->GetAB1(), 0.9, 1e-10);
  EXPECT_NEAR(agent_->GetAS1(), 0.9, 1e-10);
  EXPECT_NEAR(agent_->GetAB2(), 0.9, 1e-10);
  EXPECT_NEAR(agent_->GetAS2(), 0.9, 1e-10);
}

TEST_F(AgentAdaptiveTest, MakeOfferWorks) {
  mock_rng_->setUniformValue(0.3);  // Buy at market 1
  agent_->ChoiceMarket();

  mock_rng_->setGaussianValue(0.0);
  double offer = agent_->MakeOffer();

  // For buying: mean_buy=11.0, sigma=1.0, gaussian=0 -> 11.0
  EXPECT_DOUBLE_EQ(offer, 11.0);
}

TEST_F(AgentAdaptiveTest, GetScoreReturnsFourValues) {
  mock_rng_->setGaussianValue(0.5);
  agent_->InitializeScore(2.0);

  auto scores = agent_->GetScore();
  EXPECT_EQ(scores.size(), 4u);
  EXPECT_DOUBLE_EQ(scores[0], 1.0);  // AB1
  EXPECT_DOUBLE_EQ(scores[1], 1.0);  // AS1
  EXPECT_DOUBLE_EQ(scores[2], 1.0);  // AB2
  EXPECT_DOUBLE_EQ(scores[3], 1.0);  // AS2
}

TEST_F(AgentAdaptiveTest, ChangeParameterAlpha) {
  EXPECT_TRUE(agent_->ChangeParameter("alpha", 0.5));
  EXPECT_DOUBLE_EQ(agent_->GetAlpha(), 0.5);

  EXPECT_FALSE(agent_->ChangeParameter("alpha", 1.5));  // Out of range
}

TEST_F(AgentAdaptiveTest, ChangeParameterUnknownReturnsFalse) {
  EXPECT_FALSE(agent_->ChangeParameter("unknown", 1.0));
}

TEST_F(AgentAdaptiveTest, NewTurnResetsBid) {
  mock_rng_->setUniformValue(0.3);
  agent_->ChoiceMarket();
  mock_rng_->setGaussianValue(0.0);
  agent_->MakeOffer();

  agent_->NewTurn();
  EXPECT_DOUBLE_EQ(agent_->GetBid(), 0.0);
}

TEST_F(AgentAdaptiveTest, SetSuccessStoresSuccess) {
  agent_->SetSuccess(true);
  // No getter for success, but at least verify it doesn't crash
  agent_->SetSuccess(false);
}

TEST_F(AgentAdaptiveTest, CumulativeScoreUpdates) {
  mock_rng_->setUniformValue(0.3);
  agent_->ChoiceMarket();

  agent_->UpdateScore(10.0);
  double cs1 = agent_->GetCumulativeScore();
  // cgscore = 0 * (1-0.1) + 10 * 0.1 = 1.0
  EXPECT_NEAR(cs1, 1.0, 1e-10);

  mock_rng_->setUniformValue(0.3);
  agent_->ChoiceMarket();
  agent_->UpdateScore(10.0);
  // cgscore = 1.0 * 0.9 + 10 * 0.1 = 1.9
  EXPECT_NEAR(agent_->GetCumulativeScore(), 1.9, 1e-10);
}

}  // namespace
}  // namespace abm
