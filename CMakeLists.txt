cmake_minimum_required(VERSION 3.28)

set(INFINITRAIN_BACKEND "maca" CACHE STRING
  "Accelerator provider to build from backends/<provider>")
string(TOLOWER "${INFINITRAIN_BACKEND}" INFINITRAIN_BACKEND)
if(NOT INFINITRAIN_BACKEND MATCHES "^[a-z0-9_]+$")
  message(FATAL_ERROR
    "INFINITRAIN_BACKEND must name a directory under backends/: "
    "${INFINITRAIN_BACKEND}")
endif()

set(INFINITRAIN_BACKEND_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/backends/${INFINITRAIN_BACKEND}")
set(INFINITRAIN_BACKEND_PRE_PROJECT
  "${INFINITRAIN_BACKEND_DIR}/cmake/pre_project.cmake")
if(NOT EXISTS "${INFINITRAIN_BACKEND_PRE_PROJECT}")
  message(FATAL_ERROR
    "Unknown or incomplete InfiniTrain backend '${INFINITRAIN_BACKEND}': "
    "${INFINITRAIN_BACKEND_PRE_PROJECT} was not found.")
endif()

# A provider owns compiler and dependency-probe setup that must happen before
# project(). This keeps vendor SDK assumptions out of the repository root.
include("${INFINITRAIN_BACKEND_PRE_PROJECT}")

project(InfiniTrainBackends VERSION 0.1.0 LANGUAGES CXX)

option(INFINITRAIN_BACKENDS_BUILD_EXAMPLES
  "Build backend-enabled InfiniTrain examples" ${PROJECT_IS_TOP_LEVEL})
option(INFINITRAIN_BACKENDS_BUILD_TESTS
  "Build backend hardware tests" ${PROJECT_IS_TOP_LEVEL})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(INFINITRAIN_BACKENDS_BUILD_TESTS)
  enable_testing()
endif()

set(INFINITRAIN_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/InfiniTrain"
    CACHE PATH "InfiniTrain source tree (normally the pinned submodule)")
if(NOT EXISTS "${INFINITRAIN_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "InfiniTrain was not found at ${INFINITRAIN_SOURCE_DIR}. "
    "Run git submodule update --init --recursive or set INFINITRAIN_SOURCE_DIR.")
endif()

add_subdirectory("${INFINITRAIN_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/InfiniTrain")
add_subdirectory("backends/${INFINITRAIN_BACKEND}")
