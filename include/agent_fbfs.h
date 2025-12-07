#ifndef ABM_AGENT_FBFS_H
#define ABM_AGENT_FBFS_H

#include "agent.h"

namespace abm {

/**
 * @brief Fixed Buy/Fixed Sell Agent
 *
 * This agent type has fixed probabilities for buying vs selling,
 * but learns which market to trade in using Experience Weighted
 * Attraction (EWA) learning.
 */
class AgentFBFS : public AgentBase {
 public:
  /**
   * @brief Construct an agent with a random generator
   * @param rng Reference to a random number generator
   */
  explicit AgentFBFS(IRandomGenerator& rng);

  // Type identification
  std::string TypeAgent() const override { return "fbfs"; }

  // Getters
  std::vector<double> GetScore() const override { return {a1_ - a2_}; }
  double GetBuyProbability() const override { return buy_probability_; }

  // Additional getters for FBFS-specific state
  double GetA1() const { return a1_; }
  double GetA2() const { return a2_; }
  double GetDelta() const { return delta_; }

  // Parameter modification
  bool ChangeParameter(const std::string& name, double value) override;

  // Core behavior
  void Initialize(const AgentParams& params) override;
  void InitializeScore(double variance) override;
  int ChoiceMarket() override;
  void UpdateScore(double score) override;

  // Debug
  void Print() const override;

 private:
  Action ChooseAction();

  // FBFS-specific state
  double buy_probability_ = 0.5;
  double a1_ = 0.0;         // Attraction to market 1
  double a2_ = 0.0;         // Attraction to market 2
  double delta_ = 0.0;      // a1_ - a2_
};

}  // namespace abm

#endif  // ABM_AGENT_FBFS_H
