cmake_minimum_required(VERSION 2.8)
project(libonyx C CXX)

include(GenerateExportHeader)

find_package(Matlab)

# Sublibraries
add_subdirectory(larank)


set(onyx_HEADERS
    ${PROJECT_BINARY_DIR}/export.h
)

# Shared library
add_library(onyx SHARED ${onyx_SOURCES} ${onyx_HEADERS} $<TARGET_OBJECTS:larank>)

# Static library (for Matlab bindings)
if(MATLAB_FOUND)
    add_library(onyx_static STATIC ${onyx_SOURCES} ${onyx_HEADERS} $<TARGET_OBJECTS:larank>)
    target_compile_definitions(onyx_static PUBLIC ONYX_STATIC_DEFINE)
    set_target_properties(onyx_static PROPERTIES POSITION_INDEPENDENT_CODE 1) # need -fPIC for linking with MEX files
endif()


# Export headers
generate_export_header(onyx EXPORT_FILE_NAME export.h)

install(TARGETS onyx DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${onyx_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onyx)
