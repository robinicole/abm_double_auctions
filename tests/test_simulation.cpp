#include "simulation.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "agent_adaptive.h"
#include "agent_fbfs.h"
#include "market.h"

namespace abm {
namespace {

class SimulationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_rng_ = std::make_unique<MockRandomGenerator>(0.5, 0.0);
  }

  std::vector<std::unique_ptr<Agent>> CreateFBFSAgents(int n) {
    std::vector<std::unique_ptr<Agent>> agents;
    for (int i = 0; i < n; i++) {
      auto agent = std::make_unique<AgentFBFS>(*mock_rng_);
      AgentParams params;
      params.id = i;
      params.sigma = 1.0;
      params.forget_param = 0.1;
      params.mean_buy = 11.0;
      params.mean_sell = 10.0;
      params.temperature = 0.2;
      params.buy_probability = (i < n / 2) ? 0.2 : 0.8;
      agent->Initialize(params);
      agent->InitializeScore(0.1);
      agents.push_back(std::move(agent));
    }
    return agents;
  }

  std::vector<std::unique_ptr<Agent>> CreateAdaptiveAgents(int n) {
    std::vector<std::unique_ptr<Agent>> agents;
    for (int i = 0; i < n; i++) {
      auto agent = std::make_unique<AgentAdaptive>(*mock_rng_);
      AgentParams params;
      params.id = i;
      params.sigma = 1.0;
      params.forget_param = 0.1;
      params.mean_buy = 11.0;
      params.mean_sell = 10.0;
      params.temperature = 0.2;
      agent->Initialize(params);
      agent->InitializeScore(0.1);
      agents.push_back(std::move(agent));
    }
    return agents;
  }

  std::unique_ptr<MockRandomGenerator> mock_rng_;
};

TEST_F(SimulationTest, ConstructWithFBFSAgents) {
  auto agents = CreateFBFSAgents(100);
  Market m1(1, 0.3, *mock_rng_);
  Market m2(2, 0.7, *mock_rng_);

  Simulation sim(m1, m2, std::move(agents), SimulationMode::kFBFS, "test_");

  EXPECT_EQ(sim.GetNumAgents(), 100);
  EXPECT_EQ(sim.GetMode(), SimulationMode::kFBFS);
}

TEST_F(SimulationTest, ConstructWithAdaptiveAgents) {
  auto agents = CreateAdaptiveAgents(50);
  Market m1(1, 0.3, *mock_rng_);
  Market m2(2, 0.7, *mock_rng_);

  Simulation sim(m1, m2, std::move(agents), SimulationMode::kAdaptive, "test_");

  EXPECT_EQ(sim.GetNumAgents(), 50);
  EXPECT_EQ(sim.GetMode(), SimulationMode::kAdaptive);
}

TEST_F(SimulationTest, OneIterationCompletesSuccessfully) {
  auto agents = CreateFBFSAgents(20);
  Market m1(1, 0.3, *mock_rng_);
  Market m2(2, 0.7, *mock_rng_);

  Simulation sim(m1, m2, std::move(agents), SimulationMode::kFBFS);

  int result = sim.OneIteration();
  EXPECT_EQ(result, 0);
}

TEST_F(SimulationTest, RunIterationsCompletesSuccessfully) {
  auto agents = CreateFBFSAgents(20);
  Market m1(1, 0.3, *mock_rng_);
  Market m2(2, 0.7, *mock_rng_);

  Simulation sim(m1, m2, std::move(agents), SimulationMode::kFBFS);

  int result = sim.RunIterations(10);
  EXPECT_EQ(result, 0);
}

TEST_F(SimulationTest, AverageScoreCalculatesCorrectly) {
  auto agents = CreateFBFSAgents(20);
  Market m1(1, 0.3, *mock_rng_);
  Market m2(2, 0.7, *mock_rng_);

  Simulation sim(m1, m2, std::move(agents), SimulationMode::kFBFS);

  // Run a few iterations to generate some scores
  sim.RunIterations(5);

  // Average score should be between 0 and 1 (it's a sigmoid)
  double avg1 = sim.AverageScore(0.2);
  double avg2 = sim.AverageScore(0.8);

  EXPECT_GE(avg1, 0.0);
  EXPECT_LE(avg1, 1.0);
  EXPECT_GE(avg2, 0.0);
  EXPECT_LE(avg2, 1.0);
}

TEST_F(SimulationTest, SetTemperatureChangesAllAgents) {
  auto agents = CreateFBFSAgents(10);
  Market m1(1, 0.3, *mock_rng_);
  Market m2(2, 0.7, *mock_rng_);

  Simulation sim(m1, m2, std::move(agents), SimulationMode::kFBFS);

  sim.SetTemperature(0.5);

  const auto& sim_agents = sim.GetAgents();
  for (const auto& agent : sim_agents) {
    EXPECT_DOUBLE_EQ(agent->GetTemperature(), 0.5);
  }
}

TEST_F(SimulationTest, SetForgetParamChangesAllAgents) {
  auto agents = CreateFBFSAgents(10);
  Market m1(1, 0.3, *mock_rng_);
  Market m2(2, 0.7, *mock_rng_);

  Simulation sim(m1, m2, std::move(agents), SimulationMode::kFBFS);

  sim.SetForgetParam(0.05);

  const auto& sim_agents = sim.GetAgents();
  for (const auto& agent : sim_agents) {
    EXPECT_DOUBLE_EQ(agent->GetForgetParam(), 0.05);
  }
}

TEST_F(SimulationTest, GetAgentPointersReturnsCorrectPointers) {
  auto agents = CreateFBFSAgents(10);
  Market m1(1, 0.3, *mock_rng_);
  Market m2(2, 0.7, *mock_rng_);

  Simulation sim(m1, m2, std::move(agents), SimulationMode::kFBFS);

  auto ptrs = sim.GetAgentPointers();
  EXPECT_EQ(ptrs.size(), 10u);

  // Verify pointers match the agents in the simulation
  const auto& sim_agents = sim.GetAgents();
  for (size_t i = 0; i < ptrs.size(); i++) {
    EXPECT_EQ(ptrs[i], sim_agents[i].get());
  }
}

TEST_F(SimulationTest, AgentsLearnOverIterations) {
  auto agents = CreateFBFSAgents(100);
  Market m1(1, 0.3, *mock_rng_);
  Market m2(2, 0.7, *mock_rng_);

  Simulation sim(m1, m2, std::move(agents), SimulationMode::kFBFS);

  // Capture initial scores
  const auto& sim_agents = sim.GetAgents();
  std::vector<double> initial_scores;
  for (const auto& agent : sim_agents) {
    initial_scores.push_back(agent->GetLastScore());
  }

  // Run several iterations
  sim.RunIterations(50);

  // Verify that scores have changed
  int changed_count = 0;
  for (size_t i = 0; i < sim_agents.size(); i++) {
    if (sim_agents[i]->GetLastScore() != initial_scores[i]) {
      changed_count++;
    }
  }

  // Most agents should have different scores after trading
  EXPECT_GT(changed_count, static_cast<int>(sim_agents.size()) / 2);
}

TEST_F(SimulationTest, AdaptiveSimulationWorks) {
  auto agents = CreateAdaptiveAgents(50);
  Market m1(1, 0.3, *mock_rng_);
  Market m2(2, 0.7, *mock_rng_);

  Simulation sim(m1, m2, std::move(agents), SimulationMode::kAdaptive);

  int result = sim.RunIterations(20);
  EXPECT_EQ(result, 0);

  // Verify agents have 4 scores
  const auto& sim_agents = sim.GetAgents();
  for (const auto& agent : sim_agents) {
    auto scores = agent->GetScore();
    EXPECT_EQ(scores.size(), 4u);
  }
}

// Test the global Sigmoid function
TEST(SigmoidTest, ReturnsCorrectValues) {
  EXPECT_DOUBLE_EQ(Sigmoid(0.0, 0.2), 0.5);
  EXPECT_GT(Sigmoid(10.0, 0.2), 0.99);
  EXPECT_LT(Sigmoid(-10.0, 0.2), 0.01);
  EXPECT_DOUBLE_EQ(Sigmoid(1.0, 0.0), 1.0);
  EXPECT_DOUBLE_EQ(Sigmoid(-1.0, 0.0), 0.0);
}

}  // namespace
}  // namespace abm
