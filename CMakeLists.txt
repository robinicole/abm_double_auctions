cmake_minimum_required(VERSION 3.14)
project(abm_double_auctions VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add compile flags for better warnings
add_compile_options(-Wall -Wextra -Wpedantic)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Find Boost
find_package(Boost REQUIRED COMPONENTS program_options)

# Main library (shared between main and tests)
add_library(abm_lib STATIC
    src/random_generator.cpp
    src/agent_base.cpp
    src/agent_fbfs.cpp
    src/agent_adaptive.cpp
    src/market.cpp
    src/simulation.cpp
)
target_include_directories(abm_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(abm_lib PUBLIC Boost::program_options)

# Main executable
add_executable(mafbfs src/main.cpp)
target_link_libraries(mafbfs PRIVATE abm_lib)

# Testing
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    enable_testing()
    include(FetchContent)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    # Prevent overriding the parent project's compiler/linker settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # Test executable
    add_executable(abm_tests
        tests/test_random_generator.cpp
        tests/test_agent_fbfs.cpp
        tests/test_agent_adaptive.cpp
        tests/test_market.cpp
        tests/test_simulation.cpp
    )
    target_link_libraries(abm_tests PRIVATE abm_lib gtest gtest_main)
    target_include_directories(abm_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)

    include(GoogleTest)
    gtest_discover_tests(abm_tests)
endif()

# Clang-tidy support
option(ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
if(ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
    if(CLANG_TIDY_EXE)
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
    endif()
endif()
