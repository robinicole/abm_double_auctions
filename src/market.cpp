#include "market.h"

#include <algorithm>
#include <iostream>

namespace abm {

Market::Market(int id, double theta, IRandomGenerator& rng,
               const std::string& output_filename)
    : id_(id), theta_(theta), rng_(&rng) {
  if (!output_filename.empty() && output_filename != "no") {
    write_time_series_ = true;
    time_series_file_.open(output_filename);
    time_series_file_ << "id\tnstep\ttp\ttheta\tnotrade\tvbids\tvasks\n";
  }
}

void Market::NewTurn() {
  if (state_ != MarketState::kRewarded) {
    std::cerr << "Warning: reinitializing market without completing previous "
                 "turn\n";
  }

  bids_.clear();
  asks_.clear();
  failed_traders_.clear();
  trading_price_ = -1.0;
  no_trade_ = false;
  state_ = MarketState::kReady;
  step_++;
}

void Market::CollectOffers(const std::vector<Agent*>& agents) {
  if (state_ != MarketState::kReady) {
    std::cerr << "Error: must call NewTurn() before CollectOffers()\n";
    return;
  }

  for (const auto* agent : agents) {
    if (agent->GetChoice() == id_) {
      Proposal prop;
      prop.trader_id = agent->GetId();
      prop.value = agent->GetBid();

      if (agent->GetAction() == Action::kBuy) {
        bids_.push_back(prop);
      } else {
        asks_.push_back(prop);
      }
    }
  }

  total_bids_ = static_cast<int>(bids_.size());
  total_asks_ = static_cast<int>(asks_.size());
  num_buyers_ = total_bids_;
  num_sellers_ = total_asks_;
  state_ = MarketState::kCollecting;
}

double Market::ComputeTradingPrice() {
  if (state_ != MarketState::kCollecting) {
    std::cerr << "Error: must call CollectOffers() before "
                 "ComputeTradingPrice()\n";
    return 0.0;
  }

  if (bids_.empty() || asks_.empty()) {
    no_trade_ = true;
    state_ = MarketState::kPriceSet;
    return 0.0;
  }

  // Calculate average bids and asks
  double sum_bids = 0.0;
  for (const auto& bid : bids_) {
    sum_bids += bid.value;
  }
  double mean_bids = sum_bids / static_cast<double>(bids_.size());

  double sum_asks = 0.0;
  for (const auto& ask : asks_) {
    sum_asks += ask.value;
  }
  double mean_asks = sum_asks / static_cast<double>(asks_.size());

  // Trading price formula: tp = <ask> + theta * (<bid> - <ask>)
  trading_price_ = mean_asks + theta_ * (mean_bids - mean_asks);
  state_ = MarketState::kPriceSet;
  return trading_price_;
}

int Market::CleanOrders() {
  if (state_ != MarketState::kPriceSet) {
    std::cerr << "Error: must call ComputeTradingPrice() before "
                 "CleanOrders()\n";
    return -1;
  }

  // Filter valid bids (bid > trading_price) and asks (ask < trading_price)
  std::vector<Proposal> valid_bids;
  std::vector<Proposal> valid_asks;

  for (const auto& bid : bids_) {
    if (bid.value > trading_price_) {
      valid_bids.push_back(bid);
    } else {
      failed_traders_.push_back(bid.trader_id);
    }
  }

  for (const auto& ask : asks_) {
    if (ask.value < trading_price_) {
      valid_asks.push_back(ask);
    } else {
      failed_traders_.push_back(ask.trader_id);
    }
  }

  valid_bids_ = static_cast<int>(valid_bids.size());
  valid_asks_ = static_cast<int>(valid_asks.size());

  bids_ = std::move(valid_bids);
  asks_ = std::move(valid_asks);

  if (bids_.empty() || asks_.empty()) {
    no_trade_ = true;
    state_ = MarketState::kOrdersCleaned;
    return 0;
  }

  // Compute buy/sell ratio
  buy_sell_ratio_ =
      static_cast<double>(valid_bids_) / static_cast<double>(valid_asks_);

  // Match the number of buyers and sellers by randomly removing excess
  while (asks_.size() > bids_.size()) {
    size_t idx = rng_->int32() % asks_.size();
    failed_traders_.push_back(asks_[idx].trader_id);
    asks_[idx] = asks_.back();
    asks_.pop_back();
  }

  while (bids_.size() > asks_.size()) {
    size_t idx = rng_->int32() % bids_.size();
    failed_traders_.push_back(bids_[idx].trader_id);
    bids_[idx] = bids_.back();
    bids_.pop_back();
  }

  state_ = MarketState::kOrdersCleaned;
  return 0;
}

int Market::RewardTraders(std::vector<Agent*>& agents) {
  if (state_ != MarketState::kOrdersCleaned) {
    std::cerr << "Error: must call CleanOrders() before RewardTraders()\n";
    return -1;
  }

  // Handle case when no trades occur
  if (no_trade_) {
    for (const auto& bid : bids_) {
      agents[bid.trader_id]->UpdateScore(0.0);
      agents[bid.trader_id]->SetSuccess(false);
    }
    for (const auto& ask : asks_) {
      agents[ask.trader_id]->UpdateScore(0.0);
      agents[ask.trader_id]->SetSuccess(false);
    }
  }

  // Update failed traders
  for (int trader_id : failed_traders_) {
    agents[trader_id]->UpdateScore(0.0);
    agents[trader_id]->SetSuccess(false);
  }

  // Reward successful buyers (profit = bid - trading_price)
  for (const auto& bid : bids_) {
    double profit = bid.value - trading_price_;
    if (profit < 0) {
      std::cerr << "Warning: negative buyer profit: " << profit << "\n";
    }
    agents[bid.trader_id]->UpdateScore(profit);
    agents[bid.trader_id]->SetSuccess(true);
  }

  // Reward successful sellers (profit = trading_price - ask)
  for (const auto& ask : asks_) {
    double profit = trading_price_ - ask.value;
    if (profit < 0) {
      std::cerr << "Warning: negative seller profit: " << profit << "\n";
    }
    agents[ask.trader_id]->UpdateScore(profit);
    agents[ask.trader_id]->SetSuccess(true);
  }

  // Write time series if enabled
  if (write_time_series_) {
    WriteTimeSeries();
  }

  state_ = MarketState::kRewarded;
  return 0;
}

void Market::WriteTimeSeries() {
  time_series_file_ << id_ << "\t" << step_ << "\t" << trading_price_ << "\t"
                    << theta_ << "\t" << (no_trade_ ? 1 : 0) << "\t"
                    << valid_bids_ << "\t" << valid_asks_ << "\n";
}

std::string Market::Display() const {
  return std::to_string(buy_sell_ratio_);
}

}  // namespace abm
