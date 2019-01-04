#ifndef _CU_HELPER_H_
#define _CU_HELPER_H_


//// CUDA headers (standard)
#include "cuda_runtime.h"
#include "cusparse_v2.h"

//// Error handling function
int _cu_error_msg(cudaError_t cu_status, const char *file_name, int line_number) {
  fprintf(stderr, "[ERROR] during `cudaMemcpy()` at file `%s` line `%d` with error `%s`\n", 
      file_name, line_number, cudaGetErrorString(cu_status));
  return cu_status;
}
#define cu_error_msg(cu_status) _cu_error_msg(cu_status, __FILE__, __LINE__)


#endif // _CU_HELPER_H_
