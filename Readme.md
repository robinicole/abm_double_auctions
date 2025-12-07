# ABM Double Auctions

Agent-Based Model simulation of double auction markets with Experience Weighted Attraction (EWA) Learning.

[![CI](https://github.com/yourusername/abm_double_auctions/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/abm_double_auctions/actions/workflows/ci.yml)

## Overview

This code is supplementary material to the article "Dynamical selection of Nash equilibria using Experience Weighted Attraction Learning" (see the preprint [here](https://arxiv.org/abs/1706.09763)).

The simulation models:
- **Two markets** with different clearing price mechanisms (theta parameter)
- **Heterogeneous agents** with either Fixed Buy/Sell (FBFS) or Adaptive preferences
- **Experience Weighted Attraction (EWA) learning** for market and action selection

## Project Structure

```
abm_double_auctions/
├── include/           # Header files
│   ├── agent.h        # Agent base class and interface
│   ├── agent_fbfs.h   # Fixed Buy/Fixed Sell agent
│   ├── agent_adaptive.h # Adaptive agent
│   ├── market.h       # Market class
│   ├── simulation.h   # Simulation orchestrator
│   └── random_generator.h # Random number generation
├── src/               # Implementation files
│   ├── main.cpp       # Main entry point
│   ├── agent_base.cpp
│   ├── agent_fbfs.cpp
│   ├── agent_adaptive.cpp
│   ├── market.cpp
│   ├── simulation.cpp
│   └── random_generator.cpp
├── tests/             # Unit tests
│   ├── test_random_generator.cpp
│   ├── test_agent_fbfs.cpp
│   ├── test_agent_adaptive.cpp
│   ├── test_market.cpp
│   └── test_simulation.cpp
├── .github/workflows/ # CI/CD configuration
│   └── ci.yml
├── CMakeLists.txt     # CMake build configuration
├── .clang-format      # Code formatting (Google style)
└── .clang-tidy        # Static analysis configuration
```

## Requirements

- C++17 compatible compiler (GCC 9+, Clang 10+)
- CMake 3.14+
- Boost library with `program_options`

### Installing Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get install cmake g++ libboost-all-dev
```

**macOS (Homebrew):**
```bash
brew install cmake boost
```

## Building

### Using CMake (Recommended)

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . -j$(nproc)
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TESTS` | ON | Build unit tests |
| `ENABLE_CLANG_TIDY` | OFF | Enable clang-tidy static analysis |
| `CMAKE_BUILD_TYPE` | Release | Build type (Debug/Release) |

## Running Tests

```bash
cd build
ctest --output-on-failure
```

Or run the test executable directly:
```bash
./build/abm_tests
```

## Usage

```bash
./build/mafbfs [options]
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `-h, --help` | | Print help message |
| `--nagents` | 20000 | Number of agents in the simulation |
| `--nsteps` | 5000 | Number of simulation steps |
| `--th1` | 0.3 | Theta parameter for market 1 |
| `--th2` | 0.7 | Theta parameter for market 2 |
| `--fparam` | 0.01 | Forgetting parameter (r in the paper) |
| `--T` | 0.2 | Temperature (1/β in the paper) |
| `--pb1` | 0.2 | Buying probability of population 1 |
| `--pb2` | 0.8 | Buying probability of population 2 |
| `--mbuy` | 11 | Average bid price |
| `--msell` | 10 | Average ask price |
| `--sigma` | 1 | Variance of bids/asks |
| `--alpha` | 1 | Fictitious play coefficient |
| `--fastfrac` | 0 | Fraction of fast learners (fparam=1) |
| `--mode` | fbfs | Agent type: `fbfs` or `adaptive` |
| `--outdist` | findist.dat | Output file for final distribution |
| `--mar_ts` | marketts.dat | Market time series file prefix |
| `--NSnapshot` | -1 | Steps between snapshots (-1 = none) |
| `--prefix` | | Prefix for output files |

### Example

```bash
# Run with 10000 agents for 1000 steps
./build/mafbfs --nagents 10000 --nsteps 1000 --T 0.1

# Run adaptive agents
./build/mafbfs --mode adaptive --nagents 5000 --nsteps 2000
```

## Code Quality

This project follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) and includes:

- **Unit tests** using Google Test framework
- **Static analysis** with clang-tidy
- **Code formatting** with clang-format
- **CI/CD** with GitHub Actions (build, test, lint, sanitizers, code coverage)

### Running Linting Locally

```bash
# Format code
find src include tests -name '*.cpp' -o -name '*.h' | xargs clang-format -i

# Run clang-tidy
cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
clang-tidy -p build src/*.cpp include/*.h
```

## Architecture

### Class Hierarchy

```
Agent (abstract interface)
└── AgentBase (common implementation)
    ├── AgentFBFS (Fixed Buy/Fixed Sell)
    └── AgentAdaptive (Adaptive preferences)
```

### Key Components

- **Agent**: Implements EWA learning and trading behavior
- **Market**: Double auction clearing mechanism with state machine
- **Simulation**: Orchestrates agents and markets over multiple rounds
- **RandomGenerator**: Dependency-injectable RNG for deterministic testing

## Reference

If you use this code, please cite:

```
Dynamical selection of Nash equilibria using Experience Weighted Attraction Learning
arXiv:1706.09763
```

## License

See LICENSE file for details.
