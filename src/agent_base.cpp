#include <cmath>
#include <iostream>
#include <stdexcept>

#include "agent.h"

namespace abm {

AgentBase::AgentBase(IRandomGenerator& rng) : rng_(rng) {}

void AgentBase::SetAlpha(double alpha) {
  if (alpha < 0.0 || alpha > 1.0) {
    throw std::invalid_argument("alpha must be between 0 and 1");
  }
  alpha_ = alpha;
}

double AgentBase::MakeOffer() {
  if (action_ == Action::kBuy) {
    bid_ = rng_.gaussian(mean_buy_, sigma_);
  } else {
    bid_ = rng_.gaussian(mean_sell_, sigma_);
  }
  return bid_;
}

double AgentBase::Sigmoid(double x, double temperature) {
  if (temperature > 0) {
    return 1.0 / (1.0 + std::exp(-x / temperature));
  }
  return x > 0 ? 1.0 : 0.0;
}

void AgentBase::InitializeCommon(const AgentParams& params) {
  sigma_ = params.sigma;
  forget_param_ = params.forget_param;
  mean_buy_ = params.mean_buy;
  mean_sell_ = params.mean_sell;
  id_ = params.id;
  temperature_ = params.temperature;

  // Reset state
  choice_ = 1;
  action_ = Action::kBuy;
  success_ = false;
  last_score_ = 0.0;
  cumulative_score_ = 0.0;
}

void AgentBase::UpdateCumulativeScore(double score) {
  cumulative_score_ =
      cumulative_score_ * (1.0 - forget_param_) + score * forget_param_;
}

std::string MeanScore(const std::vector<Agent*>& agents) {
  std::stringstream ss;
  double sum_p1 = 0.0, sum_p2 = 0.0;
  int count_p1 = 0, count_p2 = 0;

  for (const auto* agent : agents) {
    double prob = agent->GetBuyProbability();
    if (std::abs(prob - 0.2) < 0.001) {
      sum_p1 += agent->GetLastScore();
      count_p1++;
    } else if (std::abs(prob - 0.8) < 0.001) {
      sum_p2 += agent->GetLastScore();
      count_p2++;
    }
  }

  if (count_p1 > 0 && count_p2 > 0) {
    ss << sum_p1 / count_p1 << "\t" << sum_p2 / count_p2 << "\n";
  }
  return ss.str();
}

std::string DisplayScore(const std::vector<Agent*>& agents, int index) {
  if (index < 0 || index >= static_cast<int>(agents.size())) {
    throw std::out_of_range("Agent index out of range");
  }
  const auto* agent = agents[index];
  std::stringstream ss;
  ss << agent->GetId() << "\t" << agent->GetBuyProbability() << "\t"
     << agent->GetForgetParam() << "\t" << agent->GetLastScore();
  return ss.str();
}

}  // namespace abm
