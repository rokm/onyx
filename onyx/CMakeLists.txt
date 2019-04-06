include(GenerateExportHeader)

# NOTE: versions prior to 3.2.5 have a bug with Eigen::Ref ambiguity...
find_package(Eigen3 3.2.5 REQUIRED)


set(onyx_HEADERS
    ${CMAKE_CURRENT_BINARY_DIR}/export.h
)

# Linear LaRank
set(linear_larank_SOURCES
    linear_larank/decision_function.cpp
    linear_larank/larank.cpp
    linear_larank/pattern.cpp
)

set(linear_larank_HEADERS
    linear_larank/larank.h
)

# Shared library
add_library(onyx SHARED
    ${onyx_SOURCES} ${onyx_HEADERS}
    ${linear_larank_SOURCES} ${linear_larank_HEADERS}
)

target_include_directories(onyx PUBLIC ${EIGEN3_INCLUDE_DIR})
target_include_directories(onyx PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(onyx PUBLIC ${PROJECT_BINARY_DIR})

# Export headers
generate_export_header(onyx EXPORT_FILE_NAME export.h)

# Install library
install(TARGETS onyx DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Install headers
install(FILES ${onyx_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onyx)
install(FILES ${linear_larank_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onyx/linear_larank)

