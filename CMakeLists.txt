cmake_minimum_required(VERSION 3.6)
project(onyx VERSION 1.0 LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

include(GNUInstallDirs)

# Enable warnings for gcc
if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-Wall -Wextra)
endif()

set(CMAKE_CXX_STANDARD 11)

# Library
add_subdirectory(onyx)

# Applications
add_subdirectory(apps)

# MATLAB
add_subdirectory(matlab)
