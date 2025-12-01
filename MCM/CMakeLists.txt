cmake_minimum_required(VERSION 3.29)
project(NEAL_ALGORITHM CXX)

set(CMAKE_CXX_STANDARD 20)

# ----------------------------------------------------------------------
# BOOST SETUP (MinGW workaround)
# ----------------------------------------------------------------------
set(BOOST_ROOT "C:/boost_1_89_0")
set(BOOST_LIBRARYDIR "C:/boost_1_89_0/stage/lib")

# Include directory
include_directories("${BOOST_ROOT}")

# Libraries with exact names (MinGW static .a files)
set(BOOST_LIBS
        "${BOOST_LIBRARYDIR}/libboost_math_c99-mgw15-mt-s-x64-1_89.a"
)


# ----------------------------------------------------------------------
# TARGET DEFINITION
# ----------------------------------------------------------------------
add_executable(NEAL_ALGORITHM
        main.cpp
        DP_MixtureModel.cpp
        DP_MixtureModel.h
        IModelPrior.cpp
        IModelPrior.h
        aPPM_MixtureModel.cpp
        aPPM_MixtureModel.h
        CohesionRepulsionDP.cpp
        CohesionRepulsionDP.h
)

# Link Boost libraries manually
target_link_libraries(NEAL_ALGORITHM PRIVATE ${BOOST_LIBS})
