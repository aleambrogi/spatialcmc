cmake_minimum_required(VERSION 3.29)
project(NEAL_ALGORITHM CXX)

set(CMAKE_CXX_STANDARD 20)

# ----------------------------------------------------------------------
# BOOST
# ----------------------------------------------------------------------
set(BOOST_ROOT "C:/boost_1_89_0")

# ----------------------------------------------------------------------
# JSON
# ----------------------------------------------------------------------
include_directories(${PROJECT_SOURCE_DIR}/external)
include_directories(${PROJECT_SOURCE_DIR}/external/nlohmann)

# ----------------------------------------------------------------------
# SOURCES
# ----------------------------------------------------------------------
set(SOURCES
        main.cpp

        # ---- MCMC ----
        MCMC/DP_MixtureModel.cpp
        MCMC/IModelPrior.cpp
        MCMC/aPPM_MixtureModel.cpp
        MCMC/ParallelDPMixture.cpp

        # ---- Consensus ----
        Consensus/CohesionRepulsionDP.cpp

        # ---- Step0 ----
        Step0/CSVLoader.cpp

        # ---- Step1 ----
        Step1_Sharding/Sharding.cpp
        Step1_Sharding/test_sharding.cpp

        dataset_median.csv
        adjacency_list.json
        MCMC/aPPM_MixtureModel_Neal8.cpp
        MCMC/aPPM_MixtureModel_Neal8.h
        MCMC/BetaNonConjugateModel.cpp
        MCMC/BetaNonConjugateModel.h
)

add_executable(NEAL_ALGORITHM ${SOURCES})

# ----------------------------------------------------------------------
# INCLUDE DIRECTORIES
# ----------------------------------------------------------------------
target_include_directories(NEAL_ALGORITHM PRIVATE
        "${PROJECT_SOURCE_DIR}"
        "${PROJECT_SOURCE_DIR}/MCMC"
        "${PROJECT_SOURCE_DIR}/Consensus"
        "${PROJECT_SOURCE_DIR}/Step0"
        "${PROJECT_SOURCE_DIR}/Step1_Sharding"
        "${PROJECT_SOURCE_DIR}/external/nlohmann"
        "C:/boost_1_89_0"
)

# ----------------------------------------------------------------------
# LINK LIBRARIES
# ----------------------------------------------------------------------
target_link_libraries(NEAL_ALGORITHM PRIVATE ${BOOST_LIBS})
