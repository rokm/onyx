cmake_minimum_required(VERSION 2.8)
project(apps CXX)

find_package(Boost 1.42 COMPONENTS program_options REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
add_definitions(-DBOOST_ALL_DYN_LINK) # Disable autolinking on MSVC

# *** Test application ***
add_executable(larank_app dataset.cpp main.cpp)
target_link_libraries(larank_app larank)
target_link_libraries(larank_app ${Boost_PROGRAM_OPTIONS_LIBRARY})

install(TARGETS larank_app DESTINATION ${CMAKE_INSTALL_BINDIR})
