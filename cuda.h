#ifndef _CUDA_H
#define _CUDA_H

#include "ppcg_options.h"
#include "ppcg.h"
#include "gpu.h"

int generate_cuda(isl_ctx *ctx, struct ppcg_options *options,
	const char *input);

__isl_give isl_printer *print_kernel_arguments(__isl_take isl_printer *p,
  struct gpu_prog *prog, struct ppcg_kernel *kernel, int types);

enum stencil_access_type {
  stencil_access_write,
  stencil_access_read
};

struct stencil_access {
  enum stencil_access_type type;

  isl_val_list *vlist;

  isl_id *ref_id;

  struct stencil_access *next;
};

struct stencil_info {
  int dim;

  struct gpu_local_array_info *array;

  isl_val_list *shift;

  struct stencil_access *access;

  int *halo;

  // Command-line option
  int option_bt;
  int option_bs[4];
  int option_sl;
  int option_ds;
  int option_nakata;
  int option_sm_vec;
  char *option_opt;

  // Execution parameter
  int *side;

  int stream;

  int dimstart;

  int nondiag;

  int associative;
};

#endif
