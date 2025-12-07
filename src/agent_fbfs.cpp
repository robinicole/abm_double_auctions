#include "agent_fbfs.h"

#include <cmath>
#include <iostream>

namespace abm {

AgentFBFS::AgentFBFS(IRandomGenerator& rng) : AgentBase(rng) {}

void AgentFBFS::Initialize(const AgentParams& params) {
  InitializeCommon(params);
  buy_probability_ = params.buy_probability;

  // Reset FBFS-specific state
  a1_ = 0.0;
  a2_ = 0.0;
  delta_ = 0.0;
}

void AgentFBFS::InitializeScore(double variance) {
  if (variance <= 0) {
    std::cerr << "Warning: variance must be positive, using 0 for initial "
                 "scores\n";
    delta_ = 0.0;
    return;
  }

  a1_ = rng_.gaussian(0.0, variance);
  a2_ = rng_.gaussian(0.0, variance);
  delta_ = rng_.gaussian(0.0, variance);
}

int AgentFBFS::ChoiceMarket() {
  double random = rng_.uniform();
  if (random < Sigmoid(delta_, temperature_)) {
    choice_ = 1;
  } else {
    choice_ = 2;
  }
  action_ = ChooseAction();
  return choice_;
}

Action AgentFBFS::ChooseAction() {
  double random = rng_.uniform();
  if (random < buy_probability_) {
    action_ = Action::kBuy;
  } else {
    action_ = Action::kSell;
  }
  return action_;
}

void AgentFBFS::UpdateScore(double score) {
  if (choice_ == 1) {
    a1_ = (1.0 - forget_param_) * a1_ + score * forget_param_;
    a2_ = a2_ * (1.0 - alpha_ * forget_param_);
  } else {
    a2_ = (1.0 - forget_param_) * a2_ + score * forget_param_;
    a1_ = a1_ * (1.0 - alpha_ * forget_param_);
  }
  delta_ = a1_ - a2_;
  last_score_ = score;
  UpdateCumulativeScore(score);
}

bool AgentFBFS::ChangeParameter(const std::string& name, double value) {
  if (name == "alpha") {
    if (value < 0.0 || value > 1.0) {
      std::cerr << "Error: alpha must be between 0 and 1\n";
      return false;
    }
    alpha_ = value;
    return true;
  }
  if (name == "score") {
    delta_ = value;
    a1_ = value / 2.0;
    a2_ = -value / 2.0;
    return true;
  }
  if (name == "pm1") {
    double val_tmp = -temperature_ * std::log(1.0 / value - 1.0);
    delta_ = val_tmp;
    a1_ = val_tmp / 2.0;
    a2_ = -val_tmp / 2.0;
    return true;
  }

  std::cerr << "Error: unknown parameter name '" << name << "'\n";
  return false;
}

void AgentFBFS::Print() const {
  std::cout << "AgentFBFS:\n"
            << "  id: " << id_ << "\n"
            << "  alpha: " << alpha_ << "\n"
            << "  choice: " << choice_ << "\n"
            << "  action: " << (action_ == Action::kBuy ? "Buy" : "Sell")
            << "\n"
            << "  sigma: " << sigma_ << "\n"
            << "  delta: " << delta_ << "\n"
            << "  forget_param: " << forget_param_ << "\n"
            << "  bid: " << bid_ << "\n"
            << "  mean_buy: " << mean_buy_ << "\n"
            << "  mean_sell: " << mean_sell_ << "\n"
            << "  success: " << (success_ ? "true" : "false") << "\n"
            << "  last_score: " << last_score_ << "\n"
            << "  buy_probability: " << buy_probability_ << "\n"
            << "  temperature: " << temperature_ << "\n"
            << "  A1: " << a1_ << "\n"
            << "  A2: " << a2_ << "\n";
}

}  // namespace abm
