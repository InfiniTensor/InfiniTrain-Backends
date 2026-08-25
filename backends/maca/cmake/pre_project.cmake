set(MACA_PATH "$ENV{MACA_PATH}" CACHE PATH "Path to the MACA SDK")
if(NOT MACA_PATH)
  message(FATAL_ERROR
    "MACA_PATH is not set. Export MACA_PATH or pass -DMACA_PATH=<sdk-root>.")
endif()

set(_MACA_COMPILER "${MACA_PATH}/mxgpu_llvm/bin/mxcc")
if(NOT EXISTS "${_MACA_COMPILER}")
  message(FATAL_ERROR "The MACA compiler was not found at ${_MACA_COMPILER}.")
endif()

# A compiler can only be selected before the first project() call. Respect an
# explicit toolchain/compiler supplied by an embedding project; otherwise use
# the compiler shipped with the selected MACA SDK.
if(NOT CMAKE_C_COMPILER)
  set(CMAKE_C_COMPILER "${_MACA_COMPILER}" CACHE FILEPATH "MACA C compiler")
endif()
if(NOT CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "${_MACA_COMPILER}" CACHE FILEPATH "MACA C++ compiler")
endif()
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# FindOpenMP cannot reliably infer mxcc's OpenMP runtime. Seed the variables it
# uses to construct OpenMP::OpenMP_CXX so MACA builds compile with OpenMP and
# link the SDK's ABI-compatible mxomp runtime instead of the system libgomp.
find_library(_MACA_OPENMP_LIBRARY
  NAMES mxomp omp iomp5
  HINTS
    "${MACA_PATH}/mxgpu_llvm/lib"
    "${MACA_PATH}/mxgpu_llvm/lib64"
    "${MACA_PATH}/lib"
  NO_DEFAULT_PATH
  NO_CACHE
)
if(NOT _MACA_OPENMP_LIBRARY)
  message(FATAL_ERROR "The MACA OpenMP runtime was not found under ${MACA_PATH}.")
endif()
set(OpenMP_CXX_FLAGS "-fopenmp")
set(OpenMP_CXX_LIB_NAMES "mxomp")
set(OpenMP_mxomp_LIBRARY "${_MACA_OPENMP_LIBRARY}")

# mxcc cannot reliably run the feature probes used by glog and FindThreads.
# Keep these as directory variables: the InfiniTrain subtree inherits them,
# while an embedding project's cache and sibling directories remain untouched.
# Force FindThreads to select libpthread; claiming libc support clears its link interface.
set(CMAKE_HAVE_LIBC_PTHREAD OFF)
set(CMAKE_HAVE_PTHREADS_CREATE OFF)
set(CMAKE_HAVE_PTHREAD_CREATE ON)
set(HAVE_SYS_TYPES_H 1)
set(HAVE_UNISTD_H 1)
set(HAVE_DLFCN_H 1)
set(HAVE_GLOB_H 1)
set(HAVE_PWD_H 1)
set(HAVE_SYS_TIME_H 1)
set(HAVE_SYS_UTSNAME_H 1)
set(HAVE_SYS_WAIT_H 1)
set(HAVE_SYS_SYSCALL_H 1)
set(HAVE_SYSLOG_H 1)
set(HAVE_UCONTEXT_H 1)
set(HAVE_MODE_T 4)
set(HAVE_HAVE_MODE_T TRUE)
set(HAVE_SSIZE_T 8)
set(HAVE_HAVE_SSIZE_T TRUE)
set(HAVE_PREAD 1)
set(HAVE_PWRITE 1)
set(HAVE_POSIX_FADVISE 1)
set(HAVE_SIGACTION 1)
set(HAVE_SIGALTSTACK 1)
set(HAVE_FCNTL 1)
set(HAVE_DLADDR 1)
set(HAVE___CXA_DEMANGLE 1)

# Configure only the InfiniTrain subtree selected by this provider. Normal
# variables are sufficient for option() with modern CMake and do not override
# an embedding project's global cache entries.
set(USE_CUDA OFF)
set(USE_NCCL OFF)
set(USE_OMP ON)
set(BUILD_SHARED_LIBS OFF)

unset(_MACA_COMPILER)
unset(_MACA_OPENMP_LIBRARY)
