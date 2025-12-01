cmake_minimum_required(VERSION 3.29)
project(NEAL_ALGORITHM CXX)

set(CMAKE_CXX_STANDARD 20)

# ----------------------------------------------------------------------
# BOOST SETUP
# ----------------------------------------------------------------------
set(BOOST_ROOT "C:/boost_1_89_0")
set(BOOST_LIBRARYDIR "C:/boost_1_89_0/stage/lib")

set(BOOST_LIBS
        "${BOOST_LIBRARYDIR}/libboost_math_c99-mgw15-mt-s-x64-1_89.a"
)

# ----------------------------------------------------------------------
# SOURCES
# ----------------------------------------------------------------------
set(SOURCES
        MCM/main.cpp
        MCM/DP_MixtureModel.cpp
        MCM/IModelPrior.cpp
        MCM/aPPM_MixtureModel.cpp
        MCM/CohesionRepulsionDP.cpp
        Step1_Sharding/test_sharding.cpp
)

add_executable(NEAL_ALGORITHM ${SOURCES})

# ----------------------------------------------------------------------
# INCLUDE DIRECTORIES (THIS IS THE IMPORTANT PART)
# ----------------------------------------------------------------------
target_include_directories(NEAL_ALGORITHM PRIVATE
        "C:/boost_1_89_0"
        "${PROJECT_SOURCE_DIR}/MCM"
        "${PROJECT_SOURCE_DIR}/Step0"
        "${PROJECT_SOURCE_DIR}/Step1_Sharding"
)

# ----------------------------------------------------------------------
# LINK LIBRARIES
# ----------------------------------------------------------------------
target_link_libraries(NEAL_ALGORITHM PRIVATE ${BOOST_LIBS})
