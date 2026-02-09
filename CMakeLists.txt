cmake_minimum_required(VERSION 3.29)
project(NEAL_ALGORITHM CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------------------------------------
# BOOST
# ----------------------------------------------------------------------
set(BOOST_ROOT "C:/boost_1_89_0")

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
        MCMC/aPPM_MixtureModel_Neal8.cpp
        MCMC/BetaNonConjugateModel.cpp
        MCMC/PoissonGammaModel.cpp

        # ---- Consensus ----
        Consensus/CohesionRepulsionDP.cpp

        # ---- Step0 ----
        Step0/CSVLoader.cpp
        Step0/DataLoader.cpp
        Step0/GeometryAdjacency.cpp

        # ---- Step1 ----
        Step1_Sharding/Sharding.cpp
        Step1_Sharding/test_sharding.cpp

        # ---- utils ----
        utils/adjacency.cpp
        utils/distances.cpp

        # ---- output ----
        output/BayesianPipelineBaseline.cpp

        # ---- interface ----
        SpatialCMCInterface.cpp
)

add_executable(NEAL_ALGORITHM ${SOURCES})

# ----------------------------------------------------------------------
# GDAL LINKING FIX
# ----------------------------------------------------------------------
# 1. Use standard module
find_package(GDAL REQUIRED)

# 2. Debugging: Print exactly what was found (Check CMake tab in CLion)
message(STATUS "GDAL_LIBRARY: ${GDAL_LIBRARY}")
message(STATUS "GDAL_LIBRARIES: ${GDAL_LIBRARIES}")

# 3. Link explicitly using the variable that MinGW actually sets
target_link_libraries(NEAL_ALGORITHM PRIVATE
        ${GDAL_LIBRARY}    # Legacy variable (common on MinGW)
        ${GDAL_LIBRARIES}  # Modern variable
)

target_include_directories(NEAL_ALGORITHM PRIVATE ${GDAL_INCLUDE_DIR})
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
        "${PROJECT_SOURCE_DIR}/output"
        "${PROJECT_SOURCE_DIR}/utils"
        "${BOOST_ROOT}"
        "${GDAL_INCLUDE_DIRS}" # Explicitly add GDAL includes
)

# ----------------------------------------------------------------------
# LINK LIBRARIES
# ----------------------------------------------------------------------
# Link ${GDAL_LIBRARIES} explicitly to handle MinGW quirks
target_link_libraries(NEAL_ALGORITHM PRIVATE ${GDAL_LIBRARIES})

# Only link BOOST_LIBS if you are using compiled boost components (Filesystem, System, etc.)
# target_link_libraries(NEAL_ALGORITHM PRIVATE ${BOOST_LIBS})