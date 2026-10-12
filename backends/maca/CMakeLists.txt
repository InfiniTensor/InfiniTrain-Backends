option(INFINITRAIN_MACA_WITH_MCCL "Enable MCCL distributed collectives" ON)

find_library(MACA_RUNTIME_LIBRARY NAMES mcruntime HINTS "${MACA_PATH}/lib" REQUIRED)
find_library(MACA_DNN_LIBRARY NAMES mcdnn HINTS "${MACA_PATH}/lib" REQUIRED)
find_library(MACA_BLAS_LIBRARY NAMES mcblas HINTS "${MACA_PATH}/lib" REQUIRED)

file(GLOB MACA_DEVICE_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.maca")
set_source_files_properties(${MACA_DEVICE_SOURCES} PROPERTIES
  LANGUAGE CXX
  COMPILE_OPTIONS "-x;maca"
)

set(MACA_BACKEND_SOURCES
  src/backend.cc
  src/runtime/maca_guard_impl.cc
  src/runtime/maca_runtime_common.cc
  src/kernels/common/gemm.cc
  ${MACA_DEVICE_SOURCES}
)

if(INFINITRAIN_MACA_WITH_MCCL)
  find_library(MACA_COMM_LIBRARY NAMES mccl HINTS "${MACA_PATH}/lib" REQUIRED)
  list(APPEND MACA_BACKEND_SOURCES
    src/ccl/mccl_common.cc
    src/ccl/mccl_impl.cc
  )
endif()

add_library(infini_train_backend_maca STATIC ${MACA_BACKEND_SOURCES})
add_library(InfiniTrain::Backend::MACA ALIAS infini_train_backend_maca)

target_include_directories(infini_train_backend_maca
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${INFINITRAIN_SOURCE_DIR}"
    "${MACA_PATH}/include"
)

target_compile_definitions(infini_train_backend_maca PRIVATE
  $<$<BOOL:${INFINITRAIN_MACA_WITH_MCCL}>:USE_MCCL=1>
)

target_link_libraries(infini_train_backend_maca PUBLIC
  InfiniTrain::infini_train
  ${MACA_RUNTIME_LIBRARY}
  ${MACA_DNN_LIBRARY}
  ${MACA_BLAS_LIBRARY}
)
if(INFINITRAIN_MACA_WITH_MCCL)
  target_link_libraries(infini_train_backend_maca PUBLIC ${MACA_COMM_LIBRARY})
endif()

# Static registries in InfiniTrain and its CPU kernel library are discovered by
# global initialization. Executables should link this target so those objects
# are retained even when no ordinary symbol references them directly.
add_library(infini_train_backend_maca_executable INTERFACE)
add_library(InfiniTrain::Backend::MACAExecutable ALIAS
  infini_train_backend_maca_executable)
# Keep target names in the link interface so their usage requirements and build
# dependencies remain attached. This matches InfiniTrain's validated static
# executable link order for MACA.
target_link_libraries(infini_train_backend_maca_executable INTERFACE
  "-Wl,--start-group"
  "-Wl,--whole-archive"
  infini_train
  infini_train_cpu_kernels
  infini_train_backend_maca
  "-Wl,--no-whole-archive"
  "-Wl,--end-group"
)

if(BUILD_TEST)
  if(NOT COMMAND infini_train_add_privateuse1_test_suites)
    message(FATAL_ERROR
      "The selected InfiniTrain revision does not provide PrivateUse1 test suites. "
      "Update INFINITRAIN_SOURCE_DIR or the pinned submodule revision.")
  endif()

  # The target suffix and CTest label identify this provider; execution still
  # uses the backend-neutral PrivateUse1 device and GTest prefix.
  infini_train_add_privateuse1_test_suites(
    BACKEND_NAME ${INFINITRAIN_BACKEND}
    DEVICE_INDEX 0
    LINK_LIBRARIES InfiniTrain::Backend::MACAExecutable
    BACKEND_HEADER "infini_train_maca/backend.h"
    BACKEND_REGISTRAR infini_train::maca::RegisterBackend
    TEST_TIMEOUT 300
    RUN_SERIAL
  )
endif()

if(INFINITRAIN_BACKENDS_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
