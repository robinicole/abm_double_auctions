#ifndef ABM_MARKET_H
#define ABM_MARKET_H

#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "agent.h"
#include "random_generator.h"

namespace abm {

/**
 * @brief Represents a proposal (bid or ask) from a trader
 */
struct Proposal {
  int trader_id;
  double value;
};

/**
 * @brief Market state enumeration for state machine validation
 */
enum class MarketState {
  kReady,          // Ready for new turn
  kCollecting,     // Collecting offers
  kPriceSet,       // Trading price computed
  kOrdersCleaned,  // Orders have been filtered
  kRewarded        // Traders have been rewarded
};

/**
 * @brief Double auction market implementation
 *
 * The market operates in a 5-step cycle:
 * 1. NewTurn() - Initialize for a new trading session
 * 2. CollectOffers() - Gather bids/asks from agents
 * 3. ComputeTradingPrice() - Calculate the clearing price
 * 4. CleanOrders() - Filter invalid orders and match buyers/sellers
 * 5. RewardTraders() - Distribute payoffs to successful traders
 */
class Market {
 public:
  /**
   * @brief Default constructor (for testing)
   */
  Market() = default;

  /**
   * @brief Construct a market
   * @param id Market identifier (1 or 2)
   * @param theta Market preference parameter (0=seller, 1=buyer)
   * @param rng Reference to random number generator
   * @param output_filename Optional filename for time series output
   */
  Market(int id, double theta, IRandomGenerator& rng,
         const std::string& output_filename = "");

  /**
   * @brief Initialize the market for a new trading turn
   */
  void NewTurn();

  /**
   * @brief Collect offers from all agents targeting this market
   * @param agents Vector of agent pointers
   */
  void CollectOffers(const std::vector<Agent*>& agents);

  /**
   * @brief Compute the trading price based on collected bids/asks
   * @return The trading price, or 0 if no trades possible
   */
  double ComputeTradingPrice();

  /**
   * @brief Filter invalid orders and randomly match excess buyers/sellers
   * @return 0 on success, -1 on error
   */
  int CleanOrders();

  /**
   * @brief Distribute rewards to successful traders
   * @param agents Vector of agent pointers
   * @return 0 on success, -1 on error
   */
  int RewardTraders(std::vector<Agent*>& agents);

  /**
   * @brief Get the current buy/sell ratio
   */
  std::string Display() const;

  // Getters
  int GetId() const { return id_; }
  double GetTheta() const { return theta_; }
  double GetTradingPrice() const { return trading_price_; }
  bool HasNoTrade() const { return no_trade_; }
  int GetValidBids() const { return valid_bids_; }
  int GetValidAsks() const { return valid_asks_; }
  int GetNumBuyers() const { return num_buyers_; }
  int GetNumSellers() const { return num_sellers_; }
  double GetBuySellRatio() const { return buy_sell_ratio_; }
  MarketState GetState() const { return state_; }
  int GetStep() const { return step_; }

 private:
  void WriteTimeSeries();

  // Market configuration
  int id_ = 0;
  double theta_ = 0.5;  // Price preference (0=seller, 1=buyer)

  // Random number generator
  IRandomGenerator* rng_ = nullptr;

  // State machine
  MarketState state_ = MarketState::kRewarded;
  int step_ = 0;

  // Trading data
  double trading_price_ = -1.0;
  std::vector<Proposal> bids_;
  std::vector<Proposal> asks_;
  std::vector<int> failed_traders_;

  // Trading statistics
  bool no_trade_ = false;
  int total_bids_ = 0;
  int total_asks_ = 0;
  int valid_bids_ = 0;
  int valid_asks_ = 0;
  int num_buyers_ = 0;
  int num_sellers_ = 0;
  double buy_sell_ratio_ = 0.0;

  // Time series output
  std::ofstream time_series_file_;
  bool write_time_series_ = false;
};

}  // namespace abm

#endif  // ABM_MARKET_H
