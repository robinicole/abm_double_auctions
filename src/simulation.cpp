#include "simulation.h"

#include <cmath>
#include <iostream>

namespace abm {

Simulation::Simulation(Market& market1, Market& market2,
                       std::vector<std::unique_ptr<Agent>> agents,
                       SimulationMode mode, const std::string& prefix)
    : agents_(std::move(agents)),
      market1_(&market1),
      market2_(&market2),
      mode_(mode),
      prefix_(prefix) {
  fractions_file_.open(prefix + "fractions.dat");
  fractions_file_ << "f1\tf2\n";
}

std::vector<Agent*> Simulation::GetAgentPointers() {
  std::vector<Agent*> ptrs;
  ptrs.reserve(agents_.size());
  for (const auto& agent : agents_) {
    ptrs.push_back(agent.get());
  }
  return ptrs;
}

double Simulation::AverageScore(double buy_probability) const {
  double cumul = 0.0;
  int size = 0;

  for (const auto& agent : agents_) {
    if (std::abs(agent->GetBuyProbability() - buy_probability) < 0.001) {
      double score = agent->GetScore()[0];
      double temp = agent->GetTemperature();
      cumul += 1.0 / (1.0 + std::exp(-score / temp));
      size++;
    }
  }

  return size > 0 ? cumul / static_cast<double>(size) : 0.0;
}

int Simulation::OneIteration() {
  // Get raw pointers for market operations
  auto agent_ptrs = GetAgentPointers();

  // Phase 1: Each agent chooses market and makes offer
  for (auto& agent : agents_) {
    agent->ChoiceMarket();
    agent->MakeOffer();
  }

  // Phase 2: Market 1 processes trades
  market1_->NewTurn();
  market1_->CollectOffers(agent_ptrs);
  market1_->ComputeTradingPrice();
  market1_->CleanOrders();
  market1_->RewardTraders(agent_ptrs);

  // Phase 3: Market 2 processes trades
  market2_->NewTurn();
  market2_->CollectOffers(agent_ptrs);
  market2_->ComputeTradingPrice();
  market2_->CleanOrders();
  market2_->RewardTraders(agent_ptrs);

  return 0;
}

int Simulation::RunIterations(int n, int snapshot_interval) {
  for (int i = 0; i < n; i++) {
    // Progress output
    if (mode_ == SimulationMode::kAdaptive) {
      std::cout << "\r" << static_cast<double>(i) * 100.0 / static_cast<double>(n)
                << "%" << std::flush;
    } else {
      double f1 = AverageScore(0.2);
      double f2 = AverageScore(0.8);
      std::cout << "\r" << static_cast<double>(i) * 100.0 / static_cast<double>(n)
                << "% f1 = " << f1 << "\t f2 = " << f2 << std::flush;
      fractions_file_ << f1 << "\t" << f2 << "\n";
    }

    // Run one iteration
    OneIteration();

    // Handle snapshots
    if (snapshot_interval > 0) {
      iterations_since_snapshot_++;
      if (iterations_since_snapshot_ >= snapshot_interval) {
        SaveSnapshot(current_snapshot_);
        current_snapshot_++;
        iterations_since_snapshot_ = 0;
      }
    }
  }

  std::cout << "\n";
  return 0;
}

void Simulation::SaveSnapshot(int snapshot_number) {
  std::ofstream file(prefix_ + "dist_snapshot" + std::to_string(snapshot_number) +
                     ".dat");

  if (mode_ == SimulationMode::kFBFS) {
    file << "score\tpb\tfparam\tlast_score\tcgscore\tlast_choice\n";
    for (const auto& agent : agents_) {
      file << agent->GetScore()[0] << "\t"
           << agent->GetBuyProbability() << "\t"
           << agent->GetForgetParam() << "\t"
           << agent->GetLastScore() << "\t"
           << agent->GetCumulativeScore() << "\t"
           << agent->GetChoice() << "\n";
    }
  } else {
    file << "AB1\tAS1\tAB2\tAS2\tfparam\tlast_score\tcgscore\n";
    for (const auto& agent : agents_) {
      auto scores = agent->GetScore();
      file << scores[0] << "\t" << scores[1] << "\t"
           << scores[2] << "\t" << scores[3] << "\t"
           << agent->GetForgetParam() << "\t"
           << agent->GetLastScore() << "\t"
           << agent->GetCumulativeScore() << "\n";
    }
  }
}

void Simulation::SetTemperature(double temp) {
  for (auto& agent : agents_) {
    agent->SetTemperature(temp);
  }
}

void Simulation::SetForgetParam(double fp) {
  for (auto& agent : agents_) {
    agent->SetForgetParam(fp);
  }
}

}  // namespace abm
