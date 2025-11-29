cmake_minimum_required(VERSION 3.16)
project(spatialcmc LANGUAGES CXX)

# --- Standard C++ ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Include directories ---
include_directories(
        Step1_Sharding
        Step2_MCMC
        Step3_Consensus
        SpatialCMC
        Distance
        Utils
)

# --- Source files ---
set(SOURCES
        main.cpp
        SpatialCMC/SpatialCMC.cpp
        Step1_Sharding/Sharding.cpp
        Step2_MCMC/MCMC.cpp
        Step3_Consensus/Consensus.cpp
        Distance/Distance.cpp
        Utils/Utils.cpp
)

# --- Executable target ---
add_executable(spatialcmc ${SOURCES})

# --- Optional: Build type ---
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()


