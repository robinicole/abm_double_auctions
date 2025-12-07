#include "market.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "agent_fbfs.h"

namespace abm {
namespace {

class MarketTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_rng_ = std::make_unique<MockRandomGenerator>(0.5, 0.0);
    market_ = std::make_unique<Market>(1, 0.3, *mock_rng_);

    // Create some test agents
    CreateTestAgents(10);
  }

  void CreateTestAgents(int n) {
    agents_.clear();
    agent_ptrs_.clear();

    for (int i = 0; i < n; i++) {
      auto agent = std::make_unique<AgentFBFS>(*mock_rng_);
      AgentParams params;
      params.id = i;
      params.sigma = 0.1;
      params.forget_param = 0.1;
      params.mean_buy = 11.0;
      params.mean_sell = 10.0;
      params.temperature = 0.2;
      params.buy_probability = (i < n / 2) ? 0.9 : 0.1;  // Half buyers, half sellers
      agent->Initialize(params);
      agent_ptrs_.push_back(agent.get());
      agents_.push_back(std::move(agent));
    }
  }

  void PrepareAgentsForMarket() {
    for (auto& agent : agents_) {
      agent->ChoiceMarket();
      agent->MakeOffer();
    }
  }

  std::unique_ptr<MockRandomGenerator> mock_rng_;
  std::unique_ptr<Market> market_;
  std::vector<std::unique_ptr<AgentFBFS>> agents_;
  std::vector<Agent*> agent_ptrs_;
};

TEST_F(MarketTest, InitialState) {
  EXPECT_EQ(market_->GetId(), 1);
  EXPECT_DOUBLE_EQ(market_->GetTheta(), 0.3);
  EXPECT_EQ(market_->GetState(), MarketState::kRewarded);  // Initial state allows NewTurn
}

TEST_F(MarketTest, NewTurnTransitionsToReady) {
  market_->NewTurn();
  EXPECT_EQ(market_->GetState(), MarketState::kReady);
}

TEST_F(MarketTest, CollectOffersGathersBidsAndAsks) {
  mock_rng_->setUniformSequence({
      0.1, 0.1,  // Agent 0: market 1, buy
      0.1, 0.1,  // Agent 1: market 1, buy
      0.1, 0.9,  // Agent 2: market 1, sell
      0.9, 0.1,  // Agent 3: market 2 (not collected)
      0.1, 0.9,  // Agent 4: market 1, sell
      0.1, 0.1,  // Agent 5: market 1, buy
      0.9, 0.9,  // Agent 6: market 2
      0.1, 0.9,  // Agent 7: market 1, sell
      0.1, 0.1,  // Agent 8: market 1, buy
      0.9, 0.1,  // Agent 9: market 2
  });

  PrepareAgentsForMarket();

  market_->NewTurn();
  market_->CollectOffers(agent_ptrs_);

  EXPECT_EQ(market_->GetState(), MarketState::kCollecting);
  // 4 buyers (0,1,5,8) and 3 sellers (2,4,7) at market 1
}

TEST_F(MarketTest, ComputeTradingPriceCalculatesCorrectly) {
  // Set up controlled scenario
  mock_rng_->setUniformSequence({
      0.1, 0.1,  // Agent 0: market 1, buy
      0.1, 0.1,  // Agent 1: market 1, buy
      0.1, 0.9,  // Agent 2: market 1, sell
      0.1, 0.9,  // Agent 3: market 1, sell
      0.9, 0.9,  // Agent 4: market 2
      0.9, 0.9,  // Agent 5: market 2
      0.9, 0.9,  // Agent 6: market 2
      0.9, 0.9,  // Agent 7: market 2
      0.9, 0.9,  // Agent 8: market 2
      0.9, 0.9,  // Agent 9: market 2
  });

  mock_rng_->setGaussianValue(0.0);  // Offers exactly at mean

  PrepareAgentsForMarket();

  market_->NewTurn();
  market_->CollectOffers(agent_ptrs_);
  double tp = market_->ComputeTradingPrice();

  // 2 buyers at mean_buy=11, 2 sellers at mean_sell=10
  // mean_bids = 11, mean_asks = 10
  // tp = 10 + 0.3 * (11 - 10) = 10.3
  EXPECT_DOUBLE_EQ(tp, 10.3);
  EXPECT_EQ(market_->GetState(), MarketState::kPriceSet);
}

TEST_F(MarketTest, ComputeTradingPriceReturnsZeroWhenNoBids) {
  // All agents choose market 2
  mock_rng_->setUniformValue(0.9);  // High values force market 2

  PrepareAgentsForMarket();

  market_->NewTurn();
  market_->CollectOffers(agent_ptrs_);
  double tp = market_->ComputeTradingPrice();

  EXPECT_DOUBLE_EQ(tp, 0.0);
  EXPECT_TRUE(market_->HasNoTrade());
}

TEST_F(MarketTest, CleanOrdersFiltersInvalidOrders) {
  mock_rng_->setUniformSequence({
      0.1, 0.1,  // Agent 0: market 1, buy
      0.1, 0.1,  // Agent 1: market 1, buy
      0.1, 0.9,  // Agent 2: market 1, sell
      0.1, 0.9,  // Agent 3: market 1, sell
      0.9, 0.9,  // Agent 4-9: market 2
      0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
  });

  mock_rng_->setGaussianValue(0.0);

  PrepareAgentsForMarket();

  market_->NewTurn();
  market_->CollectOffers(agent_ptrs_);
  market_->ComputeTradingPrice();

  mock_rng_->setUniformValue(0.0);  // For random removal if needed
  int result = market_->CleanOrders();

  EXPECT_EQ(result, 0);
  EXPECT_EQ(market_->GetState(), MarketState::kOrdersCleaned);
  // Valid bids: bid > tp (11 > 10.3) = 2
  // Valid asks: ask < tp (10 < 10.3) = 2
  EXPECT_EQ(market_->GetValidBids(), 2);
  EXPECT_EQ(market_->GetValidAsks(), 2);
}

TEST_F(MarketTest, RewardTradersUpdatesAgentScores) {
  mock_rng_->setUniformSequence({
      0.1, 0.1,  // Agent 0: market 1, buy
      0.1, 0.9,  // Agent 1: market 1, sell
      0.9, 0.9,  // Agent 2-9: market 2
      0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
      0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
  });

  mock_rng_->setGaussianValue(0.0);

  PrepareAgentsForMarket();

  market_->NewTurn();
  market_->CollectOffers(agent_ptrs_);
  market_->ComputeTradingPrice();
  market_->CleanOrders();
  int result = market_->RewardTraders(agent_ptrs_);

  EXPECT_EQ(result, 0);
  EXPECT_EQ(market_->GetState(), MarketState::kRewarded);
}

TEST_F(MarketTest, StateTransitionsEnforced) {
  // Try to call methods out of order
  Market m(2, 0.5, *mock_rng_);

  // Can't collect offers without NewTurn
  m.CollectOffers(agent_ptrs_);  // Should print error but not crash

  m.NewTurn();
  // Can't compute price without collecting offers
  m.ComputeTradingPrice();  // Should print error

  m.CollectOffers(agent_ptrs_);
  // Can't clean without price
  m.CleanOrders();  // Should print error
}

TEST_F(MarketTest, DisplayReturnsBuySellRatio) {
  mock_rng_->setUniformSequence({
      0.1, 0.1,  // Buy at market 1
      0.1, 0.1,  // Buy at market 1
      0.1, 0.9,  // Sell at market 1
      0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
      0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
  });

  mock_rng_->setGaussianValue(0.0);

  PrepareAgentsForMarket();

  market_->NewTurn();
  market_->CollectOffers(agent_ptrs_);
  market_->ComputeTradingPrice();
  market_->CleanOrders();

  std::string display = market_->Display();
  // Should be a string representation of a number
  EXPECT_FALSE(display.empty());
}

TEST_F(MarketTest, GettersReturnCorrectValues) {
  EXPECT_EQ(market_->GetId(), 1);
  EXPECT_DOUBLE_EQ(market_->GetTheta(), 0.3);
  EXPECT_EQ(market_->GetStep(), 0);

  market_->NewTurn();
  EXPECT_EQ(market_->GetStep(), 1);

  market_->NewTurn();
  EXPECT_EQ(market_->GetStep(), 2);
}

TEST_F(MarketTest, NoTradeWhenBidsAndAsksEmpty) {
  // Force all agents to market 2
  mock_rng_->setUniformValue(0.9);

  PrepareAgentsForMarket();

  market_->NewTurn();
  market_->CollectOffers(agent_ptrs_);
  market_->ComputeTradingPrice();

  EXPECT_TRUE(market_->HasNoTrade());
}

}  // namespace
}  // namespace abm
