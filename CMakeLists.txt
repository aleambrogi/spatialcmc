cmake_minimum_required(VERSION 3.20)
project(spatialcmc LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Threads REQUIRED)
add_library(spatialcmc_core
  MCMC/DP_MixtureModel.cpp MCMC/IModelPrior.cpp
  MCMC/aPPM_MixtureModel.cpp MCMC/ParallelDPMixture.cpp
  Consensus/CohesionRepulsionDP.cpp Step0/CSVLoader.cpp
  Step1_Sharding/Sharding.cpp)
target_include_directories(spatialcmc_core PUBLIC
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/MCMC
  ${PROJECT_SOURCE_DIR}/Consensus ${PROJECT_SOURCE_DIR}/Step0
  ${PROJECT_SOURCE_DIR}/Step1_Sharding)
target_link_libraries(spatialcmc_core PUBLIC Threads::Threads)
add_executable(spatialcmc main.cpp)
target_link_libraries(spatialcmc PRIVATE spatialcmc_core)
include(CTest)
if(BUILD_TESTING)
  add_executable(sampler_smoke tests/sampler_smoke.cpp)
  target_link_libraries(sampler_smoke PRIVATE spatialcmc_core)
  add_test(NAME sampler_smoke COMMAND sampler_smoke)
endif()
