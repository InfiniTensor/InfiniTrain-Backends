cmake_minimum_required(VERSION 3.28)

set(INFINITRAIN_BACKEND "" CACHE STRING
  "Accelerator provider to build from backends/<provider>")
if(INFINITRAIN_BACKEND STREQUAL "")
  message(FATAL_ERROR
    "INFINITRAIN_BACKEND is required. Configure with "
    "-DINFINITRAIN_BACKEND=<provider>.")
endif()
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

if(USE_CUDA)
  message(FATAL_ERROR
    "USE_CUDA=ON is incompatible with the '${INFINITRAIN_BACKEND}' "
    "PrivateUse1 provider. Configure with -DUSE_CUDA=OFF or use a separate "
    "build directory.")
endif()

# A provider owns compiler and dependency-probe setup that must happen before
# project(). This keeps vendor SDK assumptions out of the repository root.
include("${INFINITRAIN_BACKEND_PRE_PROJECT}")

project(InfiniTrainBackends VERSION 0.1.0 LANGUAGES CXX)

option(INFINITRAIN_BACKENDS_BUILD_EXAMPLES
  "Build backend-enabled InfiniTrain examples" ${PROJECT_IS_TOP_LEVEL})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(INFINITRAIN_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/InfiniTrain"
    CACHE PATH "InfiniTrain source tree (normally the pinned submodule)")
if(NOT EXISTS "${INFINITRAIN_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "InfiniTrain was not found at ${INFINITRAIN_SOURCE_DIR}. "
    "Run git submodule update --init --recursive or set INFINITRAIN_SOURCE_DIR.")
endif()

add_subdirectory("${INFINITRAIN_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/InfiniTrain")
if(INFINITRAIN_BACKENDS_BUILD_EXAMPLES AND NOT TARGET infini_run)
  # InfiniTrain omits top-level tools when embedded, but multi-process model
  # runs still require its launcher next to the provider-enabled examples.
  add_subdirectory(
    "${INFINITRAIN_SOURCE_DIR}/tools/infini_run"
    "${CMAKE_CURRENT_BINARY_DIR}/InfiniTrain/tools/infini_run")
  set_target_properties(infini_run PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
if(BUILD_TEST)
  # InfiniTrain registers the shared suites in its subdirectory. Enable CTest
  # at this repository root before the provider instantiates PrivateUse1.
  enable_testing()
endif()
add_subdirectory("backends/${INFINITRAIN_BACKEND}")
