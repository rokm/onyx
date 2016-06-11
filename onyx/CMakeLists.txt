cmake_minimum_required(VERSION 2.8)
project(libonyx CXX)

include(GenerateExportHeader)

set(onyx_HEADERS
    ${PROJECT_BINARY_DIR}/export.h
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

# Export headers
generate_export_header(onyx EXPORT_FILE_NAME export.h)

# Install library
install(TARGETS onyx DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Install headers
install(FILES ${onyx_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onyx)
install(FILES ${linear_larank_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onyx/linear_larank)

