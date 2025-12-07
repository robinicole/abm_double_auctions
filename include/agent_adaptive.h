#ifndef ABM_AGENT_ADAPTIVE_H
#define ABM_AGENT_ADAPTIVE_H

#include "agent.h"

namespace abm {

/**
 * @brief Adaptive Agent with four-dimensional learning
 *
 * This agent type learns both which market to trade in AND whether
 * to buy or sell, using Experience Weighted Attraction (EWA) learning
 * across four action-market combinations.
 */
class AgentAdaptive : public AgentBase {
 public:
  /**
   * @brief Construct an agent with a random generator
   * @param rng Reference to a random number generator
   */
  explicit AgentAdaptive(IRandomGenerator& rng);

  // Type identification
  std::string TypeAgent() const override { return "adaptive"; }

  // Getters
  std::vector<double> GetScore() const override;
  double GetBuyProbability() const override { return 0.0; }  // Not applicable

  // Additional getters for Adaptive-specific state
  double GetAB1() const { return ab1_; }
  double GetAS1() const { return as1_; }
  double GetAB2() const { return ab2_; }
  double GetAS2() const { return as2_; }

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
  // Adaptive-specific state (4 dimensions)
  double ab1_ = 0.0;  // Attraction to buying at market 1
  double as1_ = 0.0;  // Attraction to selling at market 1
  double ab2_ = 0.0;  // Attraction to buying at market 2
  double as2_ = 0.0;  // Attraction to selling at market 2
};

}  // namespace abm

#endif  // ABM_AGENT_ADAPTIVE_H
