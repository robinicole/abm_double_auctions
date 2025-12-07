#include "agent_adaptive.h"

#include <cmath>
#include <iostream>

namespace abm {

AgentAdaptive::AgentAdaptive(IRandomGenerator& rng) : AgentBase(rng) {}

void AgentAdaptive::Initialize(const AgentParams& params) {
  InitializeCommon(params);

  // Reset Adaptive-specific state
  ab1_ = 0.0;
  as1_ = 0.0;
  ab2_ = 0.0;
  as2_ = 0.0;
}

void AgentAdaptive::InitializeScore(double variance) {
  if (variance <= 0) {
    std::cerr << "Warning: variance must be positive, using 0 for initial "
                 "scores\n";
    ab1_ = as1_ = ab2_ = as2_ = 0.0;
    return;
  }

  ab1_ = rng_.gaussian(0.0, variance);
  ab2_ = rng_.gaussian(0.0, variance);
  as1_ = rng_.gaussian(0.0, variance);
  as2_ = rng_.gaussian(0.0, variance);
}

std::vector<double> AgentAdaptive::GetScore() const {
  return {ab1_, as1_, ab2_, as2_};
}

int AgentAdaptive::ChoiceMarket() {
  double random = rng_.uniform();

  // Compute softmax probabilities for all 4 action-market combinations
  double exp_as1 = std::exp(as1_ / temperature_);
  double exp_ab1 = std::exp(ab1_ / temperature_);
  double exp_as2 = std::exp(as2_ / temperature_);
  double exp_ab2 = std::exp(ab2_ / temperature_);

  double norm = exp_as1 + exp_ab1 + exp_as2 + exp_ab2;

  // Cumulative probabilities
  double p_as1 = exp_as1 / norm;
  double p_ab1 = p_as1 + exp_ab1 / norm;
  double p_as2 = p_ab1 + exp_as2 / norm;

  if (random < p_as1) {
    action_ = Action::kSell;
    choice_ = 1;
  } else if (random < p_ab1) {
    action_ = Action::kBuy;
    choice_ = 1;
  } else if (random < p_as2) {
    action_ = Action::kSell;
    choice_ = 2;
  } else {
    action_ = Action::kBuy;
    choice_ = 2;
  }

  return choice_;
}

void AgentAdaptive::UpdateScore(double score) {
  double decay = 1.0 - alpha_ * forget_param_;
  double update_factor = forget_param_;

  if (choice_ == 1 && action_ == Action::kBuy) {
    ab1_ = (1.0 - forget_param_) * ab1_ + score * update_factor;
    ab2_ *= decay;
    as1_ *= decay;
    as2_ *= decay;
  } else if (choice_ == 1 && action_ == Action::kSell) {
    as1_ = (1.0 - forget_param_) * as1_ + score * update_factor;
    ab2_ *= decay;
    ab1_ *= decay;
    as2_ *= decay;
  } else if (choice_ == 2 && action_ == Action::kSell) {
    as2_ = (1.0 - forget_param_) * as2_ + score * update_factor;
    ab2_ *= decay;
    ab1_ *= decay;
    as1_ *= decay;
  } else {  // choice_ == 2 && action_ == Action::kBuy
    ab2_ = (1.0 - forget_param_) * ab2_ + score * update_factor;
    as2_ *= decay;
    as1_ *= decay;
    ab1_ *= decay;
  }

  last_score_ = score;
  UpdateCumulativeScore(score);
}

bool AgentAdaptive::ChangeParameter(const std::string& name, double value) {
  if (name == "alpha") {
    if (value < 0.0 || value > 1.0) {
      std::cerr << "Error: alpha must be between 0 and 1\n";
      return false;
    }
    alpha_ = value;
    return true;
  }

  std::cerr << "Error: unknown parameter name '" << name << "'\n";
  return false;
}

void AgentAdaptive::Print() const {
  std::cout << "AgentAdaptive:\n"
            << "  id: " << id_ << "\n"
            << "  alpha: " << alpha_ << "\n"
            << "  choice: " << choice_ << "\n"
            << "  action: " << (action_ == Action::kBuy ? "Buy" : "Sell")
            << "\n"
            << "  sigma: " << sigma_ << "\n"
            << "  forget_param: " << forget_param_ << "\n"
            << "  bid: " << bid_ << "\n"
            << "  mean_buy: " << mean_buy_ << "\n"
            << "  mean_sell: " << mean_sell_ << "\n"
            << "  success: " << (success_ ? "true" : "false") << "\n"
            << "  last_score: " << last_score_ << "\n"
            << "  temperature: " << temperature_ << "\n"
            << "  AB1: " << ab1_ << "\n"
            << "  AS1: " << as1_ << "\n"
            << "  AB2: " << ab2_ << "\n"
            << "  AS2: " << as2_ << "\n";
}

}  // namespace abm
