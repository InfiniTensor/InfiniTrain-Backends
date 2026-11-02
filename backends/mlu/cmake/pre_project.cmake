# MLU provider pre-project setup.
#
# Unlike MACA (whose mxcc serves as the whole-build compiler), the InfiniTrain
# core and this provider's host-side code compile with the system C++20
# compiler. The vendor compiler cncc is reserved for device kernels (.mlu),
# which will be introduced together with the kernels.
set(NEUWARE_HOME "$ENV{NEUWARE_HOME}" CACHE PATH "Path to the Neuware SDK")
if(NOT NEUWARE_HOME)
  set(NEUWARE_HOME "/usr/local/neuware" CACHE PATH "Path to the Neuware SDK" FORCE)
endif()
if(NOT EXISTS "${NEUWARE_HOME}/include/cnrt.h")
  message(FATAL_ERROR
    "cnrt.h was not found under ${NEUWARE_HOME}/include. Export NEUWARE_HOME "
    "or pass -DNEUWARE_HOME=<sdk-root>.")
endif()

set(USE_CUDA OFF)
set(USE_NCCL OFF)
set(USE_OMP ON)
set(BUILD_SHARED_LIBS OFF)
