#ifndef _CUSPARSE_HELPER_H_
#define _CUSPARSE_HELPER_H_


//// CUDA headers
#include "cuda_runtime.h"
#include "cusparse_v2.h"

//// CUDA headers (home-made)
#include "cu_helper.h"

static const char *_cusparseGetErrorEnum(cusparseStatus_t error)
{
    switch (error)
    {
        case CUSPARSE_STATUS_SUCCESS:
            return "CUSPARSE_STATUS_SUCCESS";
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "CUSPARSE_STATUS_NOT_INITIALIZED";
        case CUSPARSE_STATUS_ALLOC_FAILED:
            return "CUSPARSE_STATUS_ALLOC_FAILED";
        case CUSPARSE_STATUS_INVALID_VALUE:
            return "CUSPARSE_STATUS_INVALID_VALUE";
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return "CUSPARSE_STATUS_ARCH_MISMATCH";
        case CUSPARSE_STATUS_MAPPING_ERROR:
            return "CUSPARSE_STATUS_MAPPING_ERROR";
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return "CUSPARSE_STATUS_EXECUTION_FAILED";
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            return "CUSPARSE_STATUS_INTERNAL_ERROR";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case CUSPARSE_STATUS_ZERO_PIVOT:
            return "CUSPARSE_STATUS_ZERO_PIVOT";
    }
    return "<unknown>";
}

int _cusparse_error_msg(cusparseStatus_t cusparse_status, const char *file_name, int line_number) {
  return _cu_error_msg_format(cusparse_status, file_name, line_number, _cusparseGetErrorEnum(cusparse_status));
}
#define cusparse_error_msg(cusparse_status) _cusparse_error_msg(cusparse_status, __FILE__, __LINE__)


#endif // _CUSPARSE_HELPER_H_
