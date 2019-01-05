//// C/C++ standard headers
#include <iostream>
#include <cstdlib>
#include <cstdio>

//// CUDA headers (standard)
#include "cuda_runtime.h"
#include "cusparse_v2.h"

//// Home-made helper CUDA headers
#include "cusparse_helper.h"
#include "cu_helper.h"
#include "tridiag_kernel.h"
#include "tridiag.h"

//// extra home-made headers

//// Define macro function
#define MIN(x,y) ((x<y)?x:y)

//// Define my type
typedef double m_t;


//// main program start
int main(int argc, char *argv[]) {
  

  //// Configuration
  long N = 11;

  // forward tridiagonal multiplication configuration
  int num_of_thread_per_block = 128;
  int num_of_blocks_max = 32;
  int num_of_blocks = MIN((N+num_of_thread_per_block-1)/num_of_thread_per_block, num_of_blocks_max);

  // Define grid and block dimension
  dim3 grid_dim3(num_of_blocks), block_dim3(num_of_thread_per_block);

  // time iteration configuration
  long num_of_time_steps = 3;
  long time_index_start = 0;
  long time_index_max=time_index_start+num_of_time_steps;



  //// logging for configuration
  fprintf(stdout, "[ LOG ] Launching kernel `tridiag_forward` with `%d` blocks with `%d` threads each\n",
      num_of_blocks, num_of_thread_per_block);
  


  //// Variable declaration with initialization if needed
  // some useful variables
  long i;
  size_t
    size_of_m_t = sizeof(m_t),
    size_of_arr = N*size_of_m_t,
    size_of_augmented_arr = (N+2) * size_of_m_t;
  long num_of_arrays_in_tridiags = 3;

  // pointers to host arrays
  m_t
    *h_tridiags_forward[num_of_arrays_in_tridiags], 
    *h_tridiags_backward[num_of_arrays_in_tridiags], 
    *h_x_aug=NULL, *h_x=NULL; 



  //// Allocate memory to host arrays
  for (i=0; i<num_of_arrays_in_tridiags; ++i) {
    h_tridiags_forward[i] = (m_t *) malloc(size_of_arr);
    if (h_tridiags_forward[i] == NULL) { 
      fprintf(stderr,"[ERROR] during `malloc()`"); return -1; 
    }
    h_tridiags_backward[i] = (m_t *) malloc(size_of_arr);
    if (h_tridiags_backward[i] == NULL) { 
      fprintf(stderr,"[ERROR] during `malloc()`"); return -1; 
    }
  }
  h_x_aug = (m_t *) malloc(size_of_augmented_arr);
  if (h_x_aug == NULL) {
    fprintf(stderr,"[ERROR] during `malloc()`"); return -1;
  }
  h_x=h_x_aug+1; 

  

  //// Fill out tridiagonals and `x` vector (associated with `h_x` array)
  for (i=0; i<N; ++i) {

    // for forward tridiagonals
    h_tridiags_forward[i_ld][i] = 1.0; 
    h_tridiags_forward[i_d][i] = 1.0; 
    h_tridiags_forward[i_ud][i] = 0.0;

    // for backward tridiagonals
    h_tridiags_backward[i_ld][i] = 0.0; 
    h_tridiags_backward[i_d][i] = 1.0; 
    h_tridiags_backward[i_ud][i] = 0.0;

    // for the target vector, associated by `h_x`
    h_x[i] = 0.1 * i;
  }

  // requirement of tridiagonal home-made forward and `cuSPARSE`backward routine
  h_tridiags_forward[i_ld][0] = 0.0; 
  h_tridiags_forward[i_ud][N-1] = 0.0; 
  h_tridiags_backward[i_ld][0] = 0.0; 
  h_tridiags_backward[i_ud][N-1] = 0.0; 

  // fill both ends with zeros for augmented arrays 
  // .. (but it is not mandatory since `d_ld[0]==d_ud[N-1]==0`)
  h_x_aug[0] = 0.0; h_x_aug[N+1] = 0.0; 

  
  
  //// Print arrays before the calculation
  std::cout << "h_x (before): ";
  for (i=0; i<N; ++i) {
    std::cout << h_x[i] << " ";
  } std::cout << std::endl;



  //// Run repeated forward-backward routine
  int return_status = -1;
  return_status = tridiag_forward_backward (
    N, h_tridiags_forward, h_tridiags_backward, h_x_aug, 
    time_index_start, time_index_max,
    block_dim3, grid_dim3 );
  if (return_status != 0) { 
    fprintf(stderr, "[ERROR] Abnormal exit from `tridiag_forward_backward()`\n"); 
    return return_status; 
  }



  //// Print arrays after the calculation
  std::cout << "h_x (after): ";
  for (i=0; i<N; ++i) {
    std::cout << h_x[i] << " ";
  } std::cout << std::endl;



  //// Free allocated memory
  for (i=0; i<num_of_arrays_in_tridiags; ++i) {
    free(h_tridiags_forward[i]);
    free(h_tridiags_backward[i]);
  }
  free(h_x_aug);



  //// End program
  return 0;
}

