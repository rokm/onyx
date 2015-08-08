cmake_minimum_required(VERSION 2.8)
project(larank C CXX)

include(GenerateExportHeader)

set(larank_SOURCES
    decision_function.cpp
    larank.cpp
    pattern.cpp
)

set(larank_HEADERS
    ${PROJECT_BINARY_DIR}/export.h
    larank.h
)

# Shared library
add_library(larank SHARED ${larank_SOURCES} ${larank_HEADERS})

# Export headers
generate_export_header(larank EXPORT_FILE_NAME export.h)

install(TARGETS larank DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${larank_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/larank)
