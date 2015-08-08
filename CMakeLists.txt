cmake_minimum_required(VERSION 2.8)
project(onyx CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

include(GNUInstallDirs)

# Common settings for gcc compiler
if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++11)
    add_definitions(-Wall -Wextra)
    add_definitions(-Wno-unused-parameter)
    add_definitions(-Wno-deprecated-declarations)
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR})

# Eigen is a global dependency
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Library
add_subdirectory(onyx)

# Applications
add_subdirectory(apps)

# MATLAB
add_subdirectory(matlab)
