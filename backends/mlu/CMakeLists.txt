# TODO(mlu): CNCL distributed collectives land with the kernels milestone.
find_library(MLU_RUNTIME_LIBRARY NAMES cnrt HINTS "${NEUWARE_HOME}/lib64" REQUIRED)
find_library(MLU_DNN_LIBRARY NAMES cnnl HINTS "${NEUWARE_HOME}/lib64" REQUIRED)

set(MLU_BACKEND_SOURCES
  src/backend.cc
  src/runtime/mlu_guard_impl.cc
  src/runtime/mlu_runtime_common.cc
)

add_library(infini_train_backend_mlu STATIC ${MLU_BACKEND_SOURCES})
add_library(InfiniTrain::Backend::MLU ALIAS infini_train_backend_mlu)

target_include_directories(infini_train_backend_mlu
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${INFINITRAIN_SOURCE_DIR}"
    "${NEUWARE_HOME}/include"
)

target_link_libraries(infini_train_backend_mlu PUBLIC
  InfiniTrain::infini_train
  ${MLU_RUNTIME_LIBRARY}
  ${MLU_DNN_LIBRARY}
)

# Static registries in InfiniTrain and its CPU kernel library are discovered by
# global initialization. Executables should link this target so those objects
# are retained even when no ordinary symbol references them directly.
add_library(infini_train_backend_mlu_executable INTERFACE)
add_library(InfiniTrain::Backend::MLUExecutable ALIAS
  infini_train_backend_mlu_executable)
target_link_libraries(infini_train_backend_mlu_executable INTERFACE
  "-Wl,--start-group"
  "-Wl,--whole-archive"
  infini_train
  infini_train_cpu_kernels
  infini_train_backend_mlu
  "-Wl,--no-whole-archive"
  "-Wl,--end-group"
)

if(BUILD_TEST)
  if(NOT COMMAND infini_train_add_privateuse1_test_suites)
    message(FATAL_ERROR
      "The selected InfiniTrain revision does not provide PrivateUse1 test suites. "
      "Update INFINITRAIN_SOURCE_DIR or the pinned submodule revision.")
  endif()

  infini_train_add_privateuse1_test_suites(
    BACKEND_NAME ${INFINITRAIN_BACKEND}
    DEVICE_INDEX 0
    LINK_LIBRARIES InfiniTrain::Backend::MLUExecutable
    BACKEND_HEADER "infini_train_mlu/backend.h"
    BACKEND_REGISTRAR infini_train::mlu::RegisterBackend
    TEST_TIMEOUT 300
    RUN_SERIAL
  )
endif()

if(INFINITRAIN_BACKENDS_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
