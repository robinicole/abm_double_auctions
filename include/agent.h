#ifndef ABM_AGENT_H
#define ABM_AGENT_H

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "random_generator.h"

namespace abm {

/**
 * @brief Enumeration for agent actions
 */
enum class Action {
  kBuy = 0,
  kSell = 1
};

/**
 * @brief Parameters for initializing an agent
 */
struct AgentParams {
  double sigma = 1.0;           ///< Variance of agent's bid
  double forget_param = 0.1;    ///< Forgetting parameter
  double mean_buy = 11.0;       ///< Average bid price
  double mean_sell = 10.0;      ///< Average ask price
  int id = 0;                   ///< Agent identifier
  double temperature = 0.2;     ///< Temperature for softmax decisions
  double buy_probability = 0.5; ///< Probability of buying (for FBFS agents)
};

/**
 * @brief Abstract base class for all agent types
 *
 * This class defines the interface for agents participating in the
 * double auction market simulation.
 */
class Agent {
 public:
  virtual ~Agent() = default;

  // Type identification
  virtual std::string TypeAgent() const = 0;

  // Getters
  virtual int GetId() const = 0;
  virtual double GetForgetParam() const = 0;
  virtual double GetTemperature() const = 0;
  virtual int GetChoice() const = 0;
  virtual Action GetAction() const = 0;
  virtual double GetBid() const = 0;
  virtual std::vector<double> GetScore() const = 0;
  virtual double GetLastScore() const = 0;
  virtual double GetBuyProbability() const = 0;
  virtual double GetCumulativeScore() const = 0;
  virtual double GetAlpha() const = 0;

  // Setters
  virtual void SetTemperature(double temp) = 0;
  virtual void SetForgetParam(double fp) = 0;
  virtual void SetAlpha(double alpha) = 0;

  // Parameter modification
  virtual bool ChangeParameter(const std::string& name, double value) = 0;

  // Core behavior
  virtual void Initialize(const AgentParams& params) = 0;
  virtual void InitializeScore(double variance) = 0;
  virtual void NewTurn() = 0;
  virtual int ChoiceMarket() = 0;
  virtual double MakeOffer() = 0;
  virtual void UpdateScore(double score) = 0;
  virtual void SetSuccess(bool success) = 0;

  // Debug
  virtual void Print() const = 0;
};

/**
 * @brief Base implementation class for common agent functionality
 *
 * This class provides shared implementation for AgentFBFS and AgentAdaptive.
 */
class AgentBase : public Agent {
 public:
  explicit AgentBase(IRandomGenerator& rng);
  ~AgentBase() override = default;

  // Getters (common implementation)
  int GetId() const override { return id_; }
  double GetForgetParam() const override { return forget_param_; }
  double GetTemperature() const override { return temperature_; }
  int GetChoice() const override { return choice_; }
  Action GetAction() const override { return action_; }
  double GetBid() const override { return bid_; }
  double GetLastScore() const override { return last_score_; }
  double GetCumulativeScore() const override { return cumulative_score_; }
  double GetAlpha() const override { return alpha_; }

  // Setters (common implementation)
  void SetTemperature(double temp) override { temperature_ = temp; }
  void SetForgetParam(double fp) override { forget_param_ = fp; }
  void SetAlpha(double alpha) override;

  // Common behavior
  void NewTurn() override { bid_ = 0.0; }
  double MakeOffer() override;
  void SetSuccess(bool success) override { success_ = success; }

  // Utility functions
  static double Sigmoid(double x, double temperature);

 protected:
  // Initialize common parameters
  void InitializeCommon(const AgentParams& params);

  // Update cumulative score
  void UpdateCumulativeScore(double score);

  // Reference to random generator (injected dependency)
  IRandomGenerator& rng_;

  // Agent state
  int id_ = 0;
  int choice_ = 1;          // Market choice (1 or 2)
  Action action_ = Action::kBuy;
  double sigma_ = 1.0;      // Variance of bids
  double forget_param_ = 0.1;
  double bid_ = 0.0;
  double mean_buy_ = 11.0;
  double mean_sell_ = 10.0;
  bool success_ = false;
  double last_score_ = 0.0;
  double temperature_ = 0.2;
  double alpha_ = 1.0;      // Fictitious play coefficient
  double cumulative_score_ = 0.0;
};

/**
 * @brief Calculate mean score for agents with a specific buy probability
 */
std::string MeanScore(const std::vector<Agent*>& agents);

/**
 * @brief Display agent information
 */
std::string DisplayScore(const std::vector<Agent*>& agents, int index);

}  // namespace abm

#endif  // ABM_AGENT_H
