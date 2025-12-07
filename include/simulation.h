#ifndef ABM_SIMULATION_H
#define ABM_SIMULATION_H

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "agent.h"
#include "market.h"

namespace abm {

/**
 * @brief Simulation mode enumeration
 */
enum class SimulationMode {
  kFBFS,     // Fixed Buy/Fixed Sell agents
  kAdaptive  // Adaptive agents
};

/**
 * @brief Multi-agent simulation controller
 *
 * Orchestrates the interaction between agents and markets over
 * multiple trading rounds.
 */
class Simulation {
 public:
  /**
   * @brief Construct a simulation
   * @param market1 Reference to the first market
   * @param market2 Reference to the second market
   * @param agents Vector of agent unique_ptrs (takes ownership)
   * @param mode Simulation mode
   * @param prefix Filename prefix for output files
   */
  Simulation(Market& market1, Market& market2,
             std::vector<std::unique_ptr<Agent>> agents, SimulationMode mode,
             const std::string& prefix = "");

  /**
   * @brief Execute a single trading iteration
   * @return 0 on success
   */
  int OneIteration();

  /**
   * @brief Run multiple iterations
   * @param n Number of iterations
   * @param snapshot_interval Interval for saving snapshots (-1 = no snapshots)
   * @return 0 on success
   */
  int RunIterations(int n, int snapshot_interval = -1);

  /**
   * @brief Calculate average score for agents with given buy probability
   */
  double AverageScore(double buy_probability) const;

  /**
   * @brief Set temperature for all agents
   */
  void SetTemperature(double temp);

  /**
   * @brief Set forgetting parameter for all agents
   */
  void SetForgetParam(double fp);

  // Getters
  int GetNumAgents() const { return static_cast<int>(agents_.size()); }
  SimulationMode GetMode() const { return mode_; }
  const std::vector<std::unique_ptr<Agent>>& GetAgents() const {
    return agents_;
  }

  /**
   * @brief Get raw pointers to agents (for compatibility with Market)
   */
  std::vector<Agent*> GetAgentPointers();

 private:
  void SaveSnapshot(int snapshot_number);

  std::vector<std::unique_ptr<Agent>> agents_;
  Market* market1_;
  Market* market2_;
  SimulationMode mode_;
  std::string prefix_;

  // Snapshot tracking
  int current_snapshot_ = 0;
  int iterations_since_snapshot_ = 0;

  // Output file for fractions
  std::ofstream fractions_file_;
};

/**
 * @brief Compute sigmoid function
 */
inline double Sigmoid(double x, double temperature) {
  if (temperature > 0) {
    return 1.0 / (1.0 + std::exp(-x / temperature));
  }
  return x > 0 ? 1.0 : 0.0;
}

}  // namespace abm

#endif  // ABM_SIMULATION_H
