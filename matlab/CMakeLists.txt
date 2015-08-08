cmake_minimum_required(VERSION 2.8)
project(onyx_matlab CXX)

find_package(Matlab)

if(MATLAB_FOUND)
    include_directories(${MATLAB_INCLUDE_DIRS})
    add_definitions(-DMATLAB_MEX_FILE)

    # onyx.LinearLaRank: build
    add_library(larank_mex SHARED +onyx/@LinearLaRank/private/larank_mex.cpp)
    set_target_properties(larank_mex PROPERTIES PREFIX "" SUFFIX ".${MATLAB_MEXEXT}")
    if(MSVC)
        set_target_properties(larank_mex PROPERTIES LINK_FLAGS "/export:mexFunction")
    endif()
    target_link_libraries(larank_mex ${MATLAB_LIBRARIES})
    target_link_libraries(larank_mex onyx_static)

    # onyx.LinearLaRank: install
    install(TARGETS larank_mex DESTINATION ${CMAKE_INSTALL_PREFIX}/matlab/+onyx/@LinearLaRank/private/)
    install(FILES +onyx/@LinearLaRank/LinearLaRank.m DESTINATION ${CMAKE_INSTALL_PREFIX}/matlab/+onyx/@LinearLaRank)
    install(FILES onyx_app.m DESTINATION ${CMAKE_INSTALL_PREFIX}/matlab/)
else()
    message(STATUS "MATLAB bindings disabled")
endif()
