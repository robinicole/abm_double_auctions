#include <gtest/gtest.h>

#include <cmath>

#include "agent_fbfs.h"

namespace abm {
namespace {

class AgentFBFSTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_rng_ = std::make_unique<MockRandomGenerator>(0.5, 0.0);
    agent_ = std::make_unique<AgentFBFS>(*mock_rng_);

    AgentParams params;
    params.id = 42;
    params.sigma = 1.0;
    params.forget_param = 0.1;
    params.mean_buy = 11.0;
    params.mean_sell = 10.0;
    params.temperature = 0.2;
    params.buy_probability = 0.3;

    agent_->Initialize(params);
  }

  std::unique_ptr<MockRandomGenerator> mock_rng_;
  std::unique_ptr<AgentFBFS> agent_;
};

TEST_F(AgentFBFSTest, InitializeSetsCorrectValues) {
  EXPECT_EQ(agent_->GetId(), 42);
  EXPECT_DOUBLE_EQ(agent_->GetForgetParam(), 0.1);
  EXPECT_DOUBLE_EQ(agent_->GetTemperature(), 0.2);
  EXPECT_DOUBLE_EQ(agent_->GetBuyProbability(), 0.3);
  EXPECT_EQ(agent_->TypeAgent(), "fbfs");
}

TEST_F(AgentFBFSTest, InitializeScoreSetsRandomValues) {
  mock_rng_->setGaussianValue(1.5);
  agent_->InitializeScore(0.5);

  // With gaussian returning mean + stddev * 1.5 = 0 + 0.5 * 1.5 = 0.75
  EXPECT_DOUBLE_EQ(agent_->GetA1(), 0.75);
  EXPECT_DOUBLE_EQ(agent_->GetA2(), 0.75);
  EXPECT_DOUBLE_EQ(agent_->GetDelta(), 0.75);
}

TEST_F(AgentFBFSTest, ChoiceMarketSelectsMarket1WhenRandomLow) {
  mock_rng_->setUniformSequence(
      {0.1, 0.1});  // Low random for market choice and action

  int choice = agent_->ChoiceMarket();

  // With delta=0, sigmoid(0)=0.5, and random=0.1 < 0.5, should choose market 1
  EXPECT_EQ(choice, 1);
  EXPECT_EQ(agent_->GetChoice(), 1);
}

TEST_F(AgentFBFSTest, ChoiceMarketSelectsMarket2WhenRandomHigh) {
  mock_rng_->setUniformSequence({0.9, 0.1});  // High random for market choice

  int choice = agent_->ChoiceMarket();

  // With delta=0, sigmoid(0)=0.5, and random=0.9 > 0.5, should choose market 2
  EXPECT_EQ(choice, 2);
  EXPECT_EQ(agent_->GetChoice(), 2);
}

TEST_F(AgentFBFSTest, ChooseActionBuysWhenRandomLow) {
  mock_rng_->setUniformSequence({0.5, 0.1});  // Market choice, then action

  agent_->ChoiceMarket();

  // buy_probability = 0.3, random = 0.1 < 0.3, should buy
  EXPECT_EQ(agent_->GetAction(), Action::kBuy);
}

TEST_F(AgentFBFSTest, ChooseActionSellsWhenRandomHigh) {
  mock_rng_->setUniformSequence({0.5, 0.5});  // Market choice, then action

  agent_->ChoiceMarket();

  // buy_probability = 0.3, random = 0.5 > 0.3, should sell
  EXPECT_EQ(agent_->GetAction(), Action::kSell);
}

TEST_F(AgentFBFSTest, MakeOfferGeneratesBidForBuyer) {
  mock_rng_->setUniformSequence({0.5, 0.1});  // Force buying
  agent_->ChoiceMarket();

  mock_rng_->setGaussianValue(0.5);
  double offer = agent_->MakeOffer();

  // mean_buy=11.0, sigma=1.0, gaussian returns mean + stddev * 0.5 = 11.5
  EXPECT_DOUBLE_EQ(offer, 11.5);
  EXPECT_DOUBLE_EQ(agent_->GetBid(), 11.5);
}

TEST_F(AgentFBFSTest, MakeOfferGeneratesAskForSeller) {
  mock_rng_->setUniformSequence({0.5, 0.5});  // Force selling
  agent_->ChoiceMarket();

  mock_rng_->setGaussianValue(-0.5);
  double offer = agent_->MakeOffer();

  // mean_sell=10.0, sigma=1.0, gaussian returns mean + stddev * (-0.5) = 9.5
  EXPECT_DOUBLE_EQ(offer, 9.5);
}

TEST_F(AgentFBFSTest, UpdateScoreUpdatesA1WhenMarket1Chosen) {
  mock_rng_->setUniformSequence({0.1, 0.1});  // Choose market 1
  agent_->ChoiceMarket();

  // Set initial values
  agent_->ChangeParameter("score", 0.0);
  EXPECT_DOUBLE_EQ(agent_->GetA1(), 0.0);
  EXPECT_DOUBLE_EQ(agent_->GetA2(), 0.0);

  agent_->UpdateScore(1.0);

  // A1 = (1-0.1)*0 + 1.0*0.1 = 0.1
  // A2 = 0 * (1-1.0*0.1) = 0
  EXPECT_NEAR(agent_->GetA1(), 0.1, 1e-10);
  EXPECT_DOUBLE_EQ(agent_->GetA2(), 0.0);
  EXPECT_NEAR(agent_->GetDelta(), 0.1, 1e-10);
  EXPECT_DOUBLE_EQ(agent_->GetLastScore(), 1.0);
}

TEST_F(AgentFBFSTest, UpdateScoreUpdatesA2WhenMarket2Chosen) {
  mock_rng_->setUniformSequence({0.9, 0.1});  // Choose market 2
  agent_->ChoiceMarket();

  agent_->ChangeParameter("score", 0.0);
  agent_->UpdateScore(2.0);

  // A1 = 0 * (1-1.0*0.1) = 0
  // A2 = (1-0.1)*0 + 2.0*0.1 = 0.2
  EXPECT_DOUBLE_EQ(agent_->GetA1(), 0.0);
  EXPECT_NEAR(agent_->GetA2(), 0.2, 1e-10);
  EXPECT_NEAR(agent_->GetDelta(), -0.2, 1e-10);
}

TEST_F(AgentFBFSTest, NewTurnResetsBid) {
  mock_rng_->setUniformSequence({0.5, 0.1});
  agent_->ChoiceMarket();
  mock_rng_->setGaussianValue(0.0);
  agent_->MakeOffer();
  EXPECT_NE(agent_->GetBid(), 0.0);

  agent_->NewTurn();
  EXPECT_DOUBLE_EQ(agent_->GetBid(), 0.0);
}

TEST_F(AgentFBFSTest, SetTemperatureChangesTemperature) {
  agent_->SetTemperature(0.5);
  EXPECT_DOUBLE_EQ(agent_->GetTemperature(), 0.5);
}

TEST_F(AgentFBFSTest, SetForgetParamChangesForgetParam) {
  agent_->SetForgetParam(0.05);
  EXPECT_DOUBLE_EQ(agent_->GetForgetParam(), 0.05);
}

TEST_F(AgentFBFSTest, ChangeParameterAlpha) {
  EXPECT_TRUE(agent_->ChangeParameter("alpha", 0.5));
  EXPECT_DOUBLE_EQ(agent_->GetAlpha(), 0.5);

  EXPECT_FALSE(agent_->ChangeParameter("alpha", 1.5));  // Out of range
}

TEST_F(AgentFBFSTest, ChangeParameterScore) {
  EXPECT_TRUE(agent_->ChangeParameter("score", 2.0));
  EXPECT_DOUBLE_EQ(agent_->GetDelta(), 2.0);
  EXPECT_DOUBLE_EQ(agent_->GetA1(), 1.0);
  EXPECT_DOUBLE_EQ(agent_->GetA2(), -1.0);
}

TEST_F(AgentFBFSTest, SigmoidReturnsCorrectValues) {
  // Test sigmoid function
  EXPECT_DOUBLE_EQ(AgentFBFS::Sigmoid(0.0, 0.2), 0.5);

  // Large positive x should approach 1
  EXPECT_GT(AgentFBFS::Sigmoid(10.0, 0.2), 0.99);

  // Large negative x should approach 0
  EXPECT_LT(AgentFBFS::Sigmoid(-10.0, 0.2), 0.01);

  // Temperature 0 should return step function
  EXPECT_DOUBLE_EQ(AgentFBFS::Sigmoid(1.0, 0.0), 1.0);
  EXPECT_DOUBLE_EQ(AgentFBFS::Sigmoid(-1.0, 0.0), 0.0);
}

TEST_F(AgentFBFSTest, GetScoreReturnsDelta) {
  agent_->ChangeParameter("score", 1.5);
  auto scores = agent_->GetScore();
  EXPECT_EQ(scores.size(), 1u);
  EXPECT_DOUBLE_EQ(scores[0], 1.5);
}

}  // namespace
}  // namespace abm
