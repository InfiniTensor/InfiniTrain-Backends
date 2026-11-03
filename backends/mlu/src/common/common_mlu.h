#pragma once

#include <cnnl.h>
#include <cnrt.h>

#include "glog/logging.h"

namespace infini_train::common::mlu {

// Runs a CNRT call and aborts with the error string on failure.
#define MLU_CHECK(expr)                                                                                                \
    do {                                                                                                               \
        cnrtRet_t _mlu_ret = (expr);                                                                                   \
        CHECK_EQ(_mlu_ret, cnrtSuccess) << "CNRT call failed: " #expr " -> " << cnrtGetErrorStr(_mlu_ret);             \
    } while (0)

// Runs a CNNL call and aborts with the status code on failure.
#define CNNL_CHECK(expr)                                                                                               \
    do {                                                                                                               \
        cnnlStatus_t _cnnl_status = (expr);                                                                            \
        CHECK_EQ(_cnnl_status, CNNL_STATUS_SUCCESS) << "CNNL call failed: " #expr " -> " << _cnnl_status;              \
    } while (0)

} // namespace infini_train::common::mlu
