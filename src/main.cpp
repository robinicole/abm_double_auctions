#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "agent_adaptive.h"
#include "agent_fbfs.h"
#include "market.h"
#include "random_generator.h"
#include "simulation.h"

namespace po = boost::program_options;

int main(int argc, char** argv) {
  // Default parameters
  int num_agents = 20000;
  double sigma = 1.0;
  double mean_buy = 11.0;
  double mean_sell = 10.0;
  double temperature = 0.2;
  double forget_param = 0.01;
  double p1 = 0.2;
  double p2 = 0.8;
  double theta1 = 0.3;
  double theta2 = 0.7;
  int iterations = 5000;
  double alpha = 1.0;
  double fast_fraction = 0.0;
  std::string output_dist = "findist.dat";
  std::string market_file = "marketts.dat";
  std::string mode = "fbfs";
  std::string prefix = "";
  int snapshot_interval = -1;

  // Command line argument parsing
  po::options_description desc(
      "Multi-agent simulation of double auction model with 2 markets");
  desc.add_options()
      ("help,h", "Print help message")
      ("nagents", po::value<int>(&num_agents)->default_value(20000),
          "Number of agents in the simulation")
      ("NSnapshot", po::value<int>(&snapshot_interval)->default_value(-1),
          "Number of steps between snapshots (-1 = no snapshots)")
      ("th1", po::value<double>(&theta1)->default_value(0.3),
          "Value of theta1 (market 1 preference)")
      ("th2", po::value<double>(&theta2)->default_value(0.7),
          "Value of theta2 (market 2 preference)")
      ("fastfrac", po::value<double>(&fast_fraction)->default_value(0.0),
          "Fraction of agents with fparam = 1")
      ("nsteps", po::value<int>(&iterations)->default_value(5000),
          "Number of simulation steps")
      ("fparam", po::value<double>(&forget_param)->default_value(0.01),
          "Forgetting parameter")
      ("pb1", po::value<double>(&p1)->default_value(0.2),
          "Buying probability of population 1")
      ("pb2", po::value<double>(&p2)->default_value(0.8),
          "Buying probability of population 2")
      ("T", po::value<double>(&temperature)->default_value(0.2),
          "Temperature")
      ("outdist", po::value<std::string>(&output_dist)->default_value("findist.dat"),
          "Output file for final distribution (use 'no' to disable)")
      ("mbuy", po::value<double>(&mean_buy)->default_value(11.0),
          "Average bid price")
      ("msell", po::value<double>(&mean_sell)->default_value(10.0),
          "Average ask price")
      ("sigma", po::value<double>(&sigma)->default_value(1.0),
          "Variance of bids/asks")
      ("alpha", po::value<double>(&alpha)->default_value(1.0),
          "Fictitious play coefficient")
      ("mar_ts", po::value<std::string>(&market_file)->default_value("marketts.dat"),
          "Market time series file prefix (use 'no' to disable)")
      ("mode", po::value<std::string>(&mode)->default_value("fbfs"),
          "Agent type: 'fbfs' or 'adaptive'")
      ("prefix", po::value<std::string>(&prefix)->default_value(""),
          "Prefix for output files");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 0;
    }
    po::notify(vm);
  } catch (const po::error& e) {
    std::cerr << "ERROR: " << e.what() << "\n\n" << desc << "\n";
    return 1;
  }

  // Print simulation parameters
  std::cout << "============================\n"
            << "Multi-agent simulation of the double auction model\n"
            << "with 2 markets\n"
            << "============================\n\n"
            << "p1 = " << p1 << "\n"
            << "p2 = " << p2 << "\n"
            << "th1 = " << theta1 << "\n"
            << "th2 = " << theta2 << "\n"
            << "r = " << forget_param << "\n"
            << "Temperature = " << temperature << "\n"
            << "Fraction of agents with r = 1: " << fast_fraction << "\n"
            << "Fictitious play coefficient (alpha) = " << alpha << "\n"
            << "<b> = " << mean_buy << "\n"
            << "<a> = " << mean_sell << "\n"
            << "Offer variance = " << sigma << "\n"
            << "Type of traders = " << mode << "\n"
            << "=======================================\n"
            << "Iterations between snapshots = " << snapshot_interval << "\n"
            << "Number of iterations = " << iterations << "\n"
            << "Number of agents: " << num_agents << "\n"
            << "=======================================\n"
            << "Simulation progress:\n";

  // Initialize random generator
  abm::RandomGenerator rng;

  // Create market output filenames
  std::string ts_m1 = (market_file == "no") ? "no" : prefix + "id_1_" + market_file;
  std::string ts_m2 = (market_file == "no") ? "no" : prefix + "id_2_" + market_file;

  // Create markets
  abm::Market market1(1, theta1, rng, ts_m1);
  abm::Market market2(2, theta2, rng, ts_m2);

  // Create agents
  std::vector<std::unique_ptr<abm::Agent>> agents;
  agents.reserve(num_agents);

  abm::SimulationMode sim_mode;
  if (mode == "fbfs") {
    sim_mode = abm::SimulationMode::kFBFS;

    int x1 = static_cast<int>((num_agents * fast_fraction) / 2.0);
    int x2 = static_cast<int>(num_agents * fast_fraction);
    int x3 = static_cast<int>((num_agents * (1 + fast_fraction)) / 2.0);

    std::cout << "Number of fast agents: " << x1 << "; " << x2 << "; " << x3 << "\n";

    for (int i = 0; i < num_agents; i++) {
      auto agent = std::make_unique<abm::AgentFBFS>(rng);

      abm::AgentParams params;
      params.sigma = sigma;
      params.mean_buy = mean_buy;
      params.mean_sell = mean_sell;
      params.id = i;
      params.temperature = temperature;

      // Assign buy probability and forget param based on population
      if (i < x1) {
        params.forget_param = 1.0;
        params.buy_probability = p1;
      } else if (i < x2) {
        params.forget_param = 1.0;
        params.buy_probability = p2;
      } else if (i < x3) {
        params.forget_param = forget_param;
        params.buy_probability = p1;
      } else {
        params.forget_param = forget_param;
        params.buy_probability = p2;
      }

      agent->Initialize(params);
      agent->ChangeParameter("alpha", alpha);
      agent->InitializeScore(forget_param);
      agents.push_back(std::move(agent));
    }
  } else if (mode == "adaptive") {
    sim_mode = abm::SimulationMode::kAdaptive;

    for (int i = 0; i < num_agents; i++) {
      auto agent = std::make_unique<abm::AgentAdaptive>(rng);

      abm::AgentParams params;
      params.sigma = sigma;
      params.mean_buy = mean_buy;
      params.mean_sell = mean_sell;
      params.id = i;
      params.temperature = temperature;
      params.forget_param = (i < static_cast<int>(num_agents * fast_fraction))
                                ? 1.0
                                : forget_param;

      agent->Initialize(params);
      agent->ChangeParameter("alpha", alpha);
      agent->InitializeScore(forget_param);
      agents.push_back(std::move(agent));
    }
  } else {
    std::cerr << "Error: unknown mode '" << mode << "'\n";
    return 1;
  }

  // Run simulation
  abm::Simulation sim(market1, market2, std::move(agents), sim_mode, prefix);
  sim.RunIterations(iterations, snapshot_interval);

  // Write final distribution
  if (output_dist != "no") {
    std::ofstream out_file(prefix + output_dist);
    const auto& final_agents = sim.GetAgents();

    if (mode == "adaptive") {
      out_file << "AB1\tAS1\tAB2\tAS2\tfparam\tT\tth1\tth2\tfastfrac\t"
               << "last_score\tcgscore\n";
      for (const auto& agent : final_agents) {
        auto scores = agent->GetScore();
        out_file << scores[0] << "\t" << scores[1] << "\t" << scores[2] << "\t"
                 << scores[3] << "\t" << agent->GetForgetParam() << "\t"
                 << temperature << "\t" << theta1 << "\t" << theta2 << "\t"
                 << fast_fraction << "\t" << agent->GetLastScore() << "\t"
                 << agent->GetCumulativeScore() << "\n";
      }
    } else {
      out_file << "score\tpb\tfparam\tT\tth1\tth2\tfastfrac\tlast_score\t"
               << "cgscore\tlast_choice\n";
      for (const auto& agent : final_agents) {
        out_file << agent->GetScore()[0] << "\t" << agent->GetBuyProbability()
                 << "\t" << agent->GetForgetParam() << "\t" << temperature
                 << "\t" << theta1 << "\t" << theta2 << "\t" << fast_fraction
                 << "\t" << agent->GetLastScore() << "\t"
                 << agent->GetCumulativeScore() << "\t" << agent->GetChoice()
                 << "\n";
      }
    }
  }

  // Write market saturation statistics
  std::ofstream mfinstat(prefix + "mfinstat.dat", std::ios::app);
  mfinstat << "p1\tp2\tth1\tth2\tfparam\tT\tfastfrac\tm1bos\tm2bos\n";
  mfinstat << p1 << "\t" << p2 << "\t" << theta1 << "\t" << theta2 << "\t"
           << forget_param << "\t" << temperature << "\t" << fast_fraction
           << "\t" << market1.Display() << "\t" << market2.Display() << "\n";

  return 0;
}
