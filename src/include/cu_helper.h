#ifndef _CU_HELPER_H_
#define _CU_HELPER_H_


//// CUDA headers (standard)
#include "cuda_runtime.h"
#include "cusparse_v2.h"

//// Error handling function
template <class T>
int _cu_error_msg_format(T cu_status, const char *file_name, int line_number, const char *error_content) {
  fprintf(stderr, "[ERROR] during `cudaMemcpy()` at file `%s` line `%d` with error `%s`\n", 
      file_name, line_number, error_content);
  return cu_status;
}

int _cu_error_msg(cudaError_t cu_status, const char *file_name, int line_number) {
  return _cu_error_msg_format(cu_status, file_name, line_number, cudaGetErrorString(cu_status));
}
#define cu_error_msg(cu_status) _cu_error_msg(cu_status, __FILE__, __LINE__)


#endif // _CU_HELPER_H_
