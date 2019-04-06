cmake_minimum_required(VERSION 3.2)
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

# Library
add_subdirectory(onyx)

# Applications
add_subdirectory(apps)

# MATLAB
add_subdirectory(matlab)
