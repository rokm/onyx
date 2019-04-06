find_package(Boost 1.42 COMPONENTS program_options)

if(Boost_FOUND)
    # *** Test application ***
    add_executable(onyx_app dataset.cpp main.cpp)
    target_link_libraries(onyx_app onyx)
    target_link_libraries(onyx_app Boost::program_options Boost::dynamic_linking)

    install(TARGETS onyx_app DESTINATION ${CMAKE_INSTALL_BINDIR})
else()
    message(STATUS "Boost not found; disabling demo application")
endif()
