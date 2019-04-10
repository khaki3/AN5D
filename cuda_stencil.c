#include <assert.h>
#include <stdarg.h>
#include <isl/aff.h>
#include <isl/ast.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include "cuda_common.h"
#include "cuda.h"
#include "gpu.h"
#include "gpu_print.h"
#include "print.h"
#include "util.h"

#include "pet/expr.h"

static isl_bool is_stencil_schedule(__isl_keep isl_schedule_node *node)
{
  node = isl_schedule_node_copy(node);
  node = isl_schedule_node_child(node, 0);

  int ret = (ppcg_ht_parent_has_input_pattern(node) > 0)
    ? isl_bool_true
    : isl_bool_false;

  isl_schedule_node_free(node);
  return ret;
}

static struct gpu_stmt_access *get_access(struct __isl_keep isl_ast_node *node)
{
  isl_id *id = isl_ast_node_get_annotation(node);

  struct ppcg_kernel_stmt *stmt = isl_id_get_user(id);

  struct gpu_stmt_access *access =
    (stmt->type == ppcg_kernel_domain) ?
    stmt->u.d.stmt->accesses : NULL;

  isl_id_free(id);
  return access;
}

static __isl_give isl_ast_node *get_node_user_if_singleton(
  struct __isl_keep isl_ast_node *node)
{
  struct gpu_stmt_access *access;
  int n_write = 0;
  isl_ast_node *tmp, *ret;

  switch (isl_ast_node_get_type(node)) {
  case isl_ast_node_for:
    tmp = isl_ast_node_for_get_body(node);
    ret = get_node_user_if_singleton(tmp);
    isl_ast_node_free(tmp);
    return ret;

  case isl_ast_node_if:
  case isl_ast_node_block:
    return NULL;

  case isl_ast_node_mark:
    access = get_access(node);

    if (!access)
      return NULL;

    for (; access; access = access->next)
      n_write += access->write;

    if (n_write != 0) return NULL;
    tmp = isl_ast_node_mark_get_node(node);
    ret = get_node_user_if_singleton(tmp);
    isl_ast_node_free(tmp);
    return ret;

  case isl_ast_node_user:
    return isl_ast_node_copy(node);
  }

  return NULL;
}

__isl_give isl_val_list *get_relative_access(__isl_keep isl_map *index,
    __isl_keep isl_map *base)
{
  isl_pw_multi_aff *pma_base  = isl_pw_multi_aff_from_map(isl_map_copy(base));
  isl_pw_multi_aff *pma_index = isl_pw_multi_aff_from_map(isl_map_copy(index));
  pma_index = isl_pw_multi_aff_sub(pma_index, pma_base);

  int n              = isl_pw_multi_aff_dim(pma_index, isl_dim_out);
  isl_ctx *ctx       = isl_pw_multi_aff_get_ctx(pma_index);
  isl_val_list *list = isl_val_list_alloc(ctx, n);

  for (int i = 0; i < n; i++) {
    isl_map *map = isl_map_from_pw_aff(isl_pw_multi_aff_get_pw_aff(pma_index, i));
    isl_val *val = isl_map_plain_get_val_if_fixed(map, isl_dim_out, 0);
    isl_map_free(map);

    list = isl_val_list_add(list, val);

    if (i != 0 && !isl_val_is_int(val)) {
      list = isl_val_list_free(list);
      break;
    }
  }

  isl_pw_multi_aff_free(pma_index);

  return list;
}

static struct gpu_local_array_info *get_local_array_info(
  struct ppcg_kernel *kernel, __isl_keep isl_map *map)
{
  const char *name = get_outer_array_name(map);
  int i = find_array_index(kernel, name);
  if (i < 0) {
    printf("ERROR: get_local_array_info\n");
    return NULL;
  }

  return &(kernel->array[i]);
}

static __isl_give isl_set *get_domain(__isl_keep isl_schedule *schedule)
{
  isl_union_set *union_domain = isl_schedule_get_domain(schedule);
  isl_set *domain = isl_set_from_union_set(union_domain);
  return domain;
}

static isl_val_list *get_shift(__isl_keep isl_map *index)
{
  /*
    Here assuming both AST and schedule T[] are having the same order
  */

  isl_map *map = isl_set_identity(isl_map_domain(isl_map_copy(index)));
  const char *outname = isl_map_get_tuple_name(index, isl_dim_out);

  map = isl_map_set_tuple_name(map, isl_dim_out, outname);

  isl_val_list *list = get_relative_access(index, map);

  isl_map_free(map);

  if (!list)
    return NULL;

  isl_bool flag = isl_bool_true;

  /*
    T[a, b, c] -> X[a, b, c] => True
    T[a, b, c] -> X[a, c, b] => False
  */
  for (int i = 0; i < isl_val_list_n_val(list); i++) {
    isl_val *val = isl_val_list_get_val(list, i);

    if ((i == 0 && !isl_val_is_nan(val)) || (i != 0 && !isl_val_is_int(val))) {
      flag = isl_bool_false;
    }

    isl_val_free(val);
  }

  if(flag != isl_bool_true)
    list = isl_val_list_free(list);

  return list;
}

static __isl_null struct stencil_access *stencil_access_free(
  struct stencil_access *sa)
{
  if (sa == NULL)
    return;

  struct stencil_access *next = sa->next;

  isl_val_list_free(sa->vlist);
  isl_id_free(sa->ref_id);
  free(sa);

  stencil_access_free(next);
}

static __isl_null struct stencil_info *stencil_info_free(
  struct stencil_info *si)
{
  if (si == NULL)
    return;

  isl_val_list_free(si->shift);
  stencil_access_free(si->access);
  free(si->halo);
  free(si->side);
  free(si);
}

static int isl_val_list_get_val_int(isl_val_list *list, int i)
{
  isl_val *val = isl_val_list_get_val(list, i);

  assert(isl_val_is_int(val));

  int ret = isl_val_get_num_si(val);

  isl_val_free(val);

  return ret;
}

void extract_halo(struct stencil_info *si)
{
  for (int i = 1; i < si->dim; i++) {
    int max = 0;

    for (struct stencil_access *sa = si->access; sa; sa = sa->next) {
      if (sa->type == stencil_access_write)
        continue;

      int tmp = isl_val_list_get_val_int(sa->vlist, i);

      if (abs(tmp) > max)
        max = abs(tmp);
    }

    si->halo[i] = max;
  }
}

static int bitcount(int n)
{
  int sum = 0;

  for (int i = 0; i < 32; i++)
    if (n & (1 << i))
      sum++;

  return sum;
}

//  "i"th-bit : found "i"+1 th plane access (i >= 0)
//  0         : no reference to array
//  1         : out of plane access
// -1         : conflicting access; both "i"th and "j"th (i != j)
static int pet_expr_plane_access(__isl_keep pet_expr *expr, __isl_keep isl_id_to_ast_expr *ref2expr)
{
  isl_ast_expr *ast_expr;
  isl_val *val;
  int plane;
  int last;

  switch (expr->type) {
  case pet_expr_error:
    return 0;
  case pet_expr_int:
    return 1;
  case pet_expr_double:
    return 1;

  case pet_expr_call:
    plane = 0;

    for (int i = 0; i < expr->n_arg; ++i) {
      plane |= pet_expr_plane_access(expr->args[0], ref2expr);
    }

    if (bitcount(plane) >= 2)
      plane = -1;
    return plane;


  case pet_expr_cast:
    return pet_expr_plane_access(expr->args[0], ref2expr);


  case pet_expr_access:
    if (!isl_id_to_ast_expr_has(ref2expr, expr->acc.ref_id))
      return 1;

    ast_expr = isl_id_to_ast_expr_get(ref2expr,
                                      isl_id_copy(expr->acc.ref_id));
    val = isl_ast_expr_get_val(ast_expr);
    plane = 1 << (1 + isl_val_get_num_si(val));

    isl_val_free(val);
    isl_ast_expr_free(ast_expr);
    return plane;


  case pet_expr_op:
    switch (expr->n_arg) {
    case 1:
      return pet_expr_plane_access(expr->args[pet_un_arg], ref2expr);

    case 2:
      switch (expr->op) {
      case pet_op_assign:
        return pet_expr_plane_access(expr->args[pet_bin_rhs], ref2expr);

      case pet_op_add:
      case pet_op_sub:
        plane = pet_expr_plane_access(expr->args[pet_bin_lhs], ref2expr);
        last  = pet_expr_plane_access(expr->args[pet_bin_rhs], ref2expr);
        if ((plane & 1 || last & 1) && bitcount(plane | last) >= 2)
          // Currently, out of plane accesses are performed for all plane computation
          return -1;
        return plane | last;

      case pet_op_mul:
      case pet_op_div:
      case pet_op_mod:
      default:
        plane = pet_expr_plane_access(expr->args[pet_bin_lhs], ref2expr);
        last  = pet_expr_plane_access(expr->args[pet_bin_rhs], ref2expr);
        if (bitcount((plane | last) & ~1) >= 2 && plane > 1 && last > 1)
             return -1;
        return (plane | last) & ~1;
      }

    case 3:
      plane  = pet_expr_plane_access(expr->args[pet_ter_cond], ref2expr);
      plane |= pet_expr_plane_access(expr->args[pet_ter_true], ref2expr);
      plane |= pet_expr_plane_access(expr->args[pet_ter_false], ref2expr);
      if (bitcount(plane) >= 2)
        plane = -1;
      return plane;
    }
  }

  return 0;
}

static isl_bool is_associative(
  struct stencil_info *si, __isl_keep isl_ast_node *node)
{
  isl_bool ret = isl_bool_true;

  isl_id *id = isl_ast_node_get_annotation(node);
  struct ppcg_kernel_stmt *stmt = isl_id_get_user(id);

  assert(stmt->type == ppcg_kernel_domain);

  isl_ctx *ctx = isl_id_get_ctx(id);
  isl_id_to_ast_expr *ref2expr = isl_id_to_ast_expr_alloc(ctx, 1);

  for (struct stencil_access *sa = si->access; sa; sa = sa->next) {
    isl_ast_expr *expr = 
      isl_ast_expr_from_val(isl_val_int_from_si(ctx, isl_val_list_get_val_int(sa->vlist, 1) + si->halo[1]));
    ref2expr = isl_id_to_ast_expr_set(ref2expr, isl_id_copy(sa->ref_id), expr);
  }

  {
    struct pet_tree *tree = stmt->u.d.stmt->stmt->body;

    assert(pet_tree_get_type(tree) == pet_tree_expr);

    pet_expr *expr = pet_tree_expr_get_expr(tree);

    int result = pet_expr_plane_access(expr, ref2expr);

    if (result == -1 || result == 1) {
      ret = isl_bool_false;
    }

    pet_expr_free(expr);
  }

  isl_id_to_ast_expr_free(ref2expr);
  isl_id_free(id);

  return ret;
}

static struct stencil_info *extract_stencil_info(
  struct ppcg_kernel *kernel, __isl_keep isl_ast_node *node)
{
  struct gpu_stmt_access *head = get_access(node);

  if (!head)
    return NULL;

  struct gpu_stmt_access *write = NULL;

  for (struct gpu_stmt_access *access = head; access; access = access->next)
    if (access->write) {
      write = access;
      break;
    }

  isl_val_list *list;

  if (!(list = get_shift(write->access)))
    return NULL;

  isl_ctx *ctx = isl_val_list_get_ctx(list);

  struct stencil_info *si = isl_calloc_type(ctx, struct stencil_info);

  if (!si) {
    isl_val_list_free(list);
    return NULL;
  }

  si->array = get_local_array_info(kernel, write->access);
  si->shift = list;
  si->access = NULL;
  si->nondiag = 1;

  const char *dest = si->array->array->name;

  for (struct gpu_stmt_access *access = head; access; access = access->next) {
    isl_map *map = access->access;
    const char *src = get_local_array_info(kernel, map)->array->name;

    if (strcmp(dest, src) != 0)
      continue;

    struct stencil_access *sa = isl_calloc_type(ctx, struct stencil_access);

    if (!sa) {
      si = stencil_info_free(si);
      break;
    }

    sa->type = access->write ? stencil_access_write : stencil_access_read;
    sa->vlist = get_relative_access(map, write->access);
    sa->ref_id = isl_id_copy(access->ref_id);
    sa->next = si->access;
    si->access = sa;

    if (!(sa->vlist)) {
      si = stencil_info_free(si);
      break;
    }

    if (si->nondiag) {
      int n = isl_val_list_n_val(sa->vlist);

      for (int i = 2; i < n; i++) {
        if (isl_val_list_get_val_int(sa->vlist, i) != 0 &&
            isl_val_list_get_val_int(sa->vlist, 1) != 0) {
          si->nondiag = 0;
          break;
        }
      }
    }
  }

  si->dim = isl_val_list_n_val(si->access->vlist);
  si->halo = isl_calloc(ctx, int, sizeof(int) * si->dim);
  si->side = isl_calloc(ctx, int, sizeof(int) * si->dim);

  extract_halo(si);

  si->associative = is_associative(si, node);

  si->option_bt     = kernel->options->an5d_bt;
  si->option_bs[1]  = kernel->options->an5d_bs1;
  si->option_bs[2]  = kernel->options->an5d_bs2;
  si->option_bs[3]  = kernel->options->an5d_bs3;
  si->option_sl     = kernel->options->an5d_sl;
  si->option_ds     = kernel->options->an5d_ds;
  si->option_nakata = kernel->options->an5d_nakata;
  si->option_sm_vec = kernel->options->an5d_sm_vec;
  si->option_opt    = kernel->options->an5d_opt;

  return si;
}

#define P_BLOCK_START(p) do { p = ppcg_start_block(p); } while (0)
#define P_BLOCK_END(p) do { p = ppcg_end_block(p); } while (0)
#define P_LINE_START(p) do { p = isl_printer_start_line(p); } while (0)
#define P_LINE_END(p) do { p = isl_printer_end_line(p); } while (0)
#define P_FMT(p, ...) do { p = isl_printer_printf(p, __VA_ARGS__); } while (0)
#define P_APP(p, fun, ...) do { p = fun(p, __VA_ARGS__); } while(0)
#define P_FMTLINE(p, ...) do {                  \
    P_LINE_START(p);                            \
    P_FMT(p, __VA_ARGS__);                      \
    P_LINE_END(p);                              \
  } while (0)

static __isl_take isl_printer *isl_printer_print_char(__isl_take isl_printer *p,
  char c)
{
  char buf[2];

  buf[0] = c;
  buf[1] = '\0';

  return isl_printer_print_str(p, buf);
}

static __isl_take isl_printer *isl_printer_printf(__isl_take isl_printer *p,
  const char *fmt,  ...)
{
  va_list ap;
  int d;
  char *s, c;
  isl_ast_expr *a;

  va_start(ap, fmt);
  while (*fmt) {
    if (*fmt == '%') {
      fmt++;
      switch (*fmt) {
      case 'd':
        d = va_arg(ap, int);
        p = isl_printer_print_int(p, d);
        break;
      case 's':
        s = va_arg(ap, char *);
        p = isl_printer_print_str(p, s);
        break;
      case 'c':
        c = va_arg(ap, int);
        p = isl_printer_print_char(p, c);
        break;
      case '@':
        a = va_arg(ap, isl_ast_expr *);
        p = isl_printer_print_ast_expr(p, a);
        break;
      case '<':
        p = isl_printer_start_line(p);
        break;
      case '>':
        p = isl_printer_end_line(p);
        break;
      case '%':
        p = isl_printer_print_char(p, '%');
        break;
      }
    }
    else {
      p = isl_printer_print_char(p, *fmt);
    }
    fmt++;
  }
  va_end(ap);

  return p;
}

__isl_give isl_ast_expr *create_array_access(struct gpu_local_array_info *local_array_info,
  __isl_keep isl_val_list *vlist)
{
  isl_ctx *ctx = isl_val_list_get_ctx(vlist);

  isl_id *id = isl_id_alloc(ctx, local_array_info->array->name, NULL);

  isl_ast_expr *array = isl_ast_expr_from_id(id);

  int dim = isl_val_list_n_val(vlist);
  
  isl_ast_expr_list *alist = isl_ast_expr_list_alloc(ctx, dim);

  for (int i = 0; i < dim; i++) {
    char itname[5];

    itname[0] = '_';
    itname[1] = '_';
    itname[2] = 'c';
    itname[3] = '0' + i;
    itname[4] = '\0';

    isl_id *id = isl_id_alloc(ctx, itname, NULL);

    isl_ast_expr *iter = isl_ast_expr_from_id(id);

    if (i >= 1 && isl_val_list_get_val_int(vlist, i) != 0)
      iter = isl_ast_expr_add(iter, isl_ast_expr_from_val(isl_val_list_get_val(vlist, i)));

    if (i == 0) {
      iter = isl_ast_expr_pdiv_r(iter, isl_ast_expr_from_val(isl_val_int_from_si(ctx, 2)));
    }

    alist = isl_ast_expr_list_add(alist, iter);
  }

  isl_ast_expr *access = isl_ast_expr_access(array, alist);

  return gpu_local_array_info_linearize_index(local_array_info, access);
}

__isl_give isl_ast_expr *create_sm_access(int plane, __isl_keep isl_val_list *vlist,
  struct stencil_info *si)
{
  isl_ctx *ctx = isl_val_list_get_ctx(vlist);

  if (si->associative &&
      plane != isl_val_list_get_val_int(vlist, 1) + si->halo[1]) {
    return isl_ast_expr_from_id(isl_id_alloc(ctx, "__pet_none", NULL));
  }

  int n = isl_val_list_n_val(vlist);

  int nondiag = 1;

  for (int i = si->dimstart; i < n; i++) {
    if (isl_val_list_get_val_int(vlist, i) != 0) {
      nondiag = 0;
      break;
    }
  }

  isl_id *id = isl_id_alloc(ctx, nondiag ? "__REGREF" : "__SBREF", NULL);

  isl_ast_expr *function = isl_ast_expr_from_id(id);
  
  isl_ast_expr_list *alist = isl_ast_expr_list_alloc(ctx, (!si->associative ? n - 1 : n));

  char sbname[10] = "__a_sb\0";
  char regname[10] = "__a\0";

  if (!si->associative) {
    sbname[2] = (!si->stream) ? 'a' :
      'a' + isl_val_list_get_val_int(vlist, 1) + si->halo[1];

    regname[2] = (!si->stream) ? 'a' :
      'a' + isl_val_list_get_val_int(vlist, 1) + si->halo[1];
  }

  id = isl_id_alloc(ctx, nondiag ? regname : sbname, NULL);

  isl_ast_expr *sb = isl_ast_expr_from_id(id);

  alist = isl_ast_expr_list_add(alist, sb);

  for (int i = si->dimstart; i < n; i++) {
    isl_ast_expr *index = isl_ast_expr_from_val(isl_val_list_get_val(vlist, i));
    alist = isl_ast_expr_list_add(alist, index);
  }

  return isl_ast_expr_call(function, alist);
}

__isl_give isl_ast_expr *replace_array_access(int plane, isl_ast_expr *expr,
  struct stencil_info *si, struct stencil_access *sa)
{
  expr = isl_ast_expr_free(expr);

  if (sa->type == stencil_access_write) {
    isl_ctx *ctx = isl_val_list_get_ctx(sa->vlist);
    return isl_ast_expr_from_id(isl_id_alloc(ctx, "__out", NULL));
  }

  expr = create_sm_access(plane, sa->vlist, si);

  return expr;
}

__isl_give isl_printer *print_accessor(__isl_take isl_printer *p, int plane,
  struct __isl_keep isl_ast_node *node, struct stencil_info *si)
{
  isl_id *id = isl_ast_node_get_annotation(node);

  struct ppcg_kernel_stmt *stmt = isl_id_get_user(id);

  assert(stmt->type == ppcg_kernel_domain);

  isl_id_to_ast_expr *ref2expr = stmt->u.d.ref2expr;

  for (struct stencil_access *sa = si->access; sa; sa = sa->next) {
    if (sa->type == stencil_access_write) {
      isl_ast_expr *expr = isl_id_to_ast_expr_get(ref2expr, isl_id_copy(sa->ref_id));
      P_FMTLINE(p, "#define __DEST (%@)", expr);
      isl_ast_expr_free(expr);
    }
  }

  P_FMT(p, "%<#define __REGREF(reg");
  for (int i = si->dimstart; i < si->dim; i++) {
    P_FMT(p, ", i%d", i);
  }
  P_FMT(p, ") reg%>");

  P_FMT(p, "%<#define __SBREF(sb");
  for (int i = si->dimstart; i < si->dim; i++) {
    P_FMT(p, ", i%d", i);
  }
  P_FMT(p, ") __sbref_wrap(sb, (int)__tid");
  for (int i = si->dimstart; i < si->dim; i++) {
    P_FMT(p, " + i%d", i);
    for (int j = i + 1; j < si->dim; j++)
      P_FMT(p, " * (int)__side%dLenOl", j);
  }
  P_FMT(p, ")%>");

  ref2expr = isl_id_to_ast_expr_copy(stmt->u.d.ref2expr);
  isl_ctx *ctx = isl_id_to_ast_expr_get_ctx(ref2expr);

  for (struct stencil_access *sa = si->access; sa; sa = sa->next) {
    isl_ast_expr *expr = isl_id_to_ast_expr_get(ref2expr, isl_id_copy(sa->ref_id));
    expr = replace_array_access(plane, expr, si, sa);
    ref2expr = isl_id_to_ast_expr_set(ref2expr, isl_id_copy(sa->ref_id), expr);
  }

  int planenum = (!si->stream) ? 1 : si->halo[1] * 2 + 1;

  if (!si->associative) {
    P_FMT(p, "%<#define __CALCEXPR(__rn0");
    for (int l = 0; l < planenum; l++)
      P_FMT(p, ", __%c", 'a' + l);
    P_FMT(p, ")");
  }
  else {
    P_FMT(p, "%<#define __CALCEXPR_%d_wrap(__rn0, __a)", plane);
  }

  P_FMT(p, " do ");
  {
    pet_expr *expr = pet_tree_expr_get_expr(stmt->u.d.stmt->stmt->body);

    p = print_pet_expr_ctrl(p, expr, ref2expr,
                            si->array->array->type, si->option_nakata);

    pet_expr_free(expr);
  }
  P_FMT(p, " while (0)%>");

  P_FMT(p, "%<#define __DB_SWITCH() do {");
  if (si->nondiag || si->associative) {
    int center = 'a' + (si->nondiag ? planenum / 2 : 0);
    P_FMT(p, " __%c_sb = &__%c_sb_double[(__%c_sb == __%c_sb_double) ? __blockSize : 0];",
          center, center, center, center);
  }
  else {
    for (int l = 0; l < planenum; l++) {
      int sym = 'a' + l;
      P_FMT(p, " __%c_sb = &__%c_sb_double[(__%c_sb == __%c_sb_double) ? __blockSize : 0];",
            sym, sym, sym, sym);
    }
  }
  P_FMT(p, " } while (0)%>");

  if (!si->associative) {
    P_FMT(p, "%<#define __CALCSETUP(a");
    for (int l = 1; l < planenum; l++)
      P_FMT(p, ", %c", 'a' + l);
    P_FMT(p, ") do {");
  }
  else {
    P_FMT(p, "%<#define __CALCSETUP(a) do {");
  }
  P_FMT(p, " __DB_SWITCH();");

  if (si->nondiag || si->associative) {
    int center = 'a' + (si->nondiag ? planenum / 2 : 0);
    P_FMT(p, " __%c_sb[__tid] = %c;", center, center);
  }
  else {
    for (int l = 0; l < planenum; l++) {
      int sym = 'a' + l;
      P_FMT(p, " __%c_sb[__tid] = %c;", sym, sym);
    }
  }
  P_FMT(p, " __syncthreads();");
  P_FMT(p, " } while (0)%>");

  if (si->associative) {
    P_FMT(p, "%<#define __CALCEXPR_%d(out, a)", plane);
    P_FMT(p, " do {");
    if (plane == 0) {
      P_FMT(p, " __CALCEXPR_%d_wrap(out, a); ", plane);
    }
    else {
      P_FMT(p, " %s etmp;", si->array->array->type);
      P_FMT(p, " __CALCEXPR_%d_wrap(etmp, a); ", plane);
      P_FMT(p, "out += etmp;");
    }
    P_FMT(p, " } while (0);%>");
  }

  isl_id_to_ast_expr_free(ref2expr);
  isl_id_free(id);

  return p;
}

static isl_printer *print_loop_size(__isl_take isl_printer *p,
  __isl_keep struct isl_ast_node *node)
{
  assert(isl_ast_node_get_type(node) == isl_ast_node_for);

  isl_ast_expr *iter = isl_ast_node_for_get_iterator(node);
  isl_ast_expr *init = isl_ast_node_for_get_init(node);
  isl_ast_expr *cond = isl_ast_node_for_get_cond(node);
  isl_ast_expr *inc  = isl_ast_node_for_get_inc(node);

  isl_ast_expr *arg0, *arg1;
  isl_val *val;

  assert(isl_ast_expr_get_type(cond) == isl_ast_expr_op);
  assert(isl_ast_expr_get_op_type(cond) == isl_ast_op_lt ||
         isl_ast_expr_get_op_type(cond) == isl_ast_op_le);
  assert(isl_ast_expr_get_op_n_arg(cond) == 2);
  assert(arg0 = isl_ast_expr_get_op_arg(cond, 0));
  assert(arg1 = isl_ast_expr_get_op_arg(cond, 1));
  assert(isl_ast_expr_is_equal(arg0, iter));
  assert(isl_ast_expr_get_type(inc) == isl_ast_expr_int);
  assert(val = isl_ast_expr_get_val(inc));
  assert(isl_val_get_num_si(val) == 1);

  switch (isl_ast_expr_get_op_type(cond)) {
  case isl_ast_op_lt:
    P_FMT(p, "(%@ - %@)", arg1, init);
    break;

  case isl_ast_op_le:
    P_FMT(p, "(%@ - %@ + 1)", arg1, init);
    break;
  }

  isl_ast_expr_free(iter);
  isl_ast_expr_free(init);
  isl_ast_expr_free(cond);
  isl_ast_expr_free(inc);
  isl_ast_expr_free(arg0);
  isl_ast_expr_free(arg1);
  isl_val_free(val);

  return p;
}

static isl_printer *print_loop_init(__isl_take isl_printer *p,
  __isl_keep struct isl_ast_node *node)
{
  assert(isl_ast_node_get_type(node) == isl_ast_node_for);

  isl_ast_expr *init = isl_ast_node_for_get_init(node);

  P_FMT(p, "(%@)", init);

  isl_ast_expr_free(init);

  return p;
}

static isl_printer *print_loop_info(__isl_take isl_printer *p,
  __isl_keep struct isl_ast_node *node, struct stencil_info *si)
{
  int nfor = 0;
  isl_ast_node *tmp;
  isl_ast_expr *expr;

  node = isl_ast_node_copy(node);

  while (node) {
    switch (isl_ast_node_get_type(node)) {
    case isl_ast_node_for:
      P_FMT(p, "%<const AN5D_TYPE __c%dLen = ", nfor);
      P_APP(p, print_loop_size, node);
      P_FMT(p, ";%>");

      P_FMT(p, "%<const AN5D_TYPE __c%dPad = ", nfor);
      P_APP(p, print_loop_init, node);
      P_FMT(p, ";%>");

      expr = isl_ast_node_for_get_iterator(node);
      P_FMTLINE(p, "#define __c%d %@", nfor, expr);
      isl_ast_expr_free(expr);
      
      nfor++;
      tmp = node;
      node = isl_ast_node_for_get_body(tmp);
      isl_ast_node_free(tmp);
      break;

    case isl_ast_node_if:
    case isl_ast_node_block:
    case isl_ast_node_mark:
    case isl_ast_node_user:
      node = isl_ast_node_free(node);
      break;
    }
  }

  for (int i = 1; i < si->dim; i++)
    P_FMTLINE(p, "const AN5D_TYPE __halo%d = %d;", i, si->halo[i]);

  return p;
}

static isl_printer *print_param(__isl_take isl_printer *p,
  struct stencil_info *si)
{
  for (int i = 0; i < si->dim; i++)
    P_FMTLINE(p, "const AN5D_TYPE __side%dLen = %d;", i, si->side[i]);

  for (int i = 1; i < si->dim; i++)
    P_FMTLINE(p, "const AN5D_TYPE __OlLen%d = (__halo%d * __side0Len);", i, i);

  for (int i = 1; i < si->dim; i++)
    P_FMTLINE(p, "const AN5D_TYPE __side%dLenOl = (__side%dLen + 2 * __OlLen%d);", i, i, i);

  P_FMT(p, "%<const AN5D_TYPE __blockSize = 1");
  for (int i = si->dimstart; i < si->dim; i++)
    P_FMT(p, " * __side%dLenOl", i);
  P_FMT(p, ";%>");

  return p;
}

static void set_param(struct stencil_info *si, int side0)
{
  // Temporal Dimension
  if (side0 >= 1)
    si->side[0] = side0;
  else
    si->side[0] = (si->option_bt > 0) ? si->option_bt : 4;

  si->stream = !si->option_ds;

  if (si->dim == 3) {
    if (si->stream) {
      si->side[1] = si->option_sl;
      si->side[2] = si->option_bs[1] - si->side[0] * si->halo[2] * 2;
    }
    else {
      si->side[1] = si->option_bs[2] - si->side[0] * si->halo[1] * 2;
      si->side[2] = si->option_bs[1] - si->side[0] * si->halo[2] * 2;
    }

    assert(si->side[1] >= 1 && si->side[2] >= 1 &&
           "Larger spatial blocking size required");
  }

  else if (si->dim == 4) {
    if (si->stream) {
      si->side[1] = si->option_sl;
      si->side[2] = si->option_bs[2] - si->side[0] * si->halo[2] * 2;
      si->side[3] = si->option_bs[1] - si->side[0] * si->halo[3] * 2;
    }
    else {
      si->side[1] = si->option_bs[3] - si->side[0] * si->halo[1] * 2;
      si->side[2] = si->option_bs[2] - si->side[0] * si->halo[2] * 2;
      si->side[3] = si->option_bs[1] - si->side[0] * si->halo[3] * 2;
    }

    assert(si->side[1] >= 1 && si->side[2] >= 1 && si->side[3] >= 1 &&
           "Larger spatial blocking size required");
  }
  
  si->dimstart = (si->stream) ? 2 : 1;

  if (!si->stream) {
    si->nondiag = 0;
    si->associative = 0;
  }

  char *alg = si->option_opt;

  if (alg != NULL) {
    if (strncmp(alg, "nondiag", 8) == 0) {
      assert(si->nondiag && "Stencil pattern must have no diagonal access");
      si->associative = 0;
    }
    if (strncmp(alg, "assoc", 6) == 0) {
      assert(si->associative && "Stencil must be associative");
      si->nondiag = 0;
    }
    if (strncmp(alg, "none", 5) == 0) {
      si->nondiag = 0;
      si->associative = 0;
    }
  }
  else {
    // Default: nondiag > assoc > none
    if (si->nondiag) si->associative = 0;
  }
}

static isl_printer *print_kernel_call(__isl_take isl_printer *p,
  struct gpu_prog *prog, struct ppcg_kernel *kernel, int t)
{
  P_FMT(p, "%<kernel%d_%d<<<k%d_dimGrid, k%d_dimBlock>>> (",
        kernel->id, t, kernel->id, kernel->id);
  P_APP(p, print_kernel_arguments, prog, kernel, 0);
  P_FMT(p, ");%>");

  return p;
}

static isl_printer *print_kernel_env(__isl_take isl_printer *p,
  struct ppcg_kernel *kernel, struct stencil_info *si)
{
  P_APP(p, print_param, si);

  if (si->stream)
    P_FMTLINE(p, "assert((__side1Len >= 2 * __side0Len * __halo1) && \
(__c1Len %% __side1Len == 0 ||\
 __c1Len %% __side1Len >= 2 * __side0Len * __halo1) && \
\"[AN5D ERROR] Too short stream\");");

  P_FMTLINE(p,
    "dim3 k%d_dimBlock(__blockSize, 1, 1);",
    kernel->id);

  P_FMT(p, "%<dim3 k%d_dimGrid(1", kernel->id);
  for (int i = 1; i< si->dim; i++)
    P_FMT(p, " * ((__c%dLen + __side%dLen - 1) / __side%dLen)", i, i, i);
  P_FMT(p, ", 1, 1);%>");

  return p;
}

static isl_printer *print_an5d_type(__isl_take isl_printer *p)
{
  P_FMT(p, "#ifndef AN5D_TYPE%>");
  P_FMT(p, "#define AN5D_TYPE unsigned%>");
  P_FMT(p, "#endif%>");
}

static void print_host(struct gpu_prog *prog,
  struct ppcg_kernel *kernel, struct cuda_info *cuda,
  struct stencil_info *si)
{
  isl_printer *p;

  p = isl_printer_to_file(prog->ctx, cuda->host_c);
  p = isl_printer_set_output_format(p, ISL_FORMAT_C);
  p = isl_printer_indent(p, 4);

  P_BLOCK_START(p);

  isl_ast_node *tree = kernel->tree;

  isl_ast_expr *iter = isl_ast_node_for_get_iterator(tree);

  p = print_an5d_type(p);

  P_APP(p, print_loop_info, tree, si);
  P_FMTLINE(p, "AN5D_TYPE %@;", iter);
  P_FMTLINE(p, "AN5D_TYPE __side0LenMax;");

  // For
  P_BLOCK_START(p);
  P_APP(p, print_kernel_env, kernel, si);
  P_FMT(p, "%<AN5D_TYPE __c0Padr =");
  P_FMT(p, " (__c0Len %% 2) != (((__c0Len + __side0Len - 1) / __side0Len) %% 2)");
  P_FMT(p, " && __c0Len %% __side0Len < 2");
  P_FMT(p, " ? 1 : 0;%>");
  P_FMTLINE(p, "__side0LenMax = __side0Len;");
  P_FMTLINE(p, "for (%@ = __c0Pad; %@ < __c0Pad + __c0Len / __side0Len - __c0Padr; %@ += 1)",
            iter, iter, iter);
  P_BLOCK_START(p);
  P_APP(p, print_kernel_call, prog, kernel, si->side[0]);
  P_BLOCK_END(p);
  P_BLOCK_END(p);

  // Residue
  P_FMTLINE(p, "if ((__c0Len %% 2) != (((__c0Len + __side0LenMax - 1) / __side0LenMax) %% 2))");
  P_BLOCK_START(p);
  for (int i = 0; i < si->side[0]; i++) {
    if (i == 0)
      P_FMT(p, "%<");
    else
      P_FMT(p, "%<else ");

    P_FMT(p, "if (__c0Len %% __side0LenMax == %d)%>", i);
    P_BLOCK_START(p);

    int r = i + (i < 2 ? si->side[0] : 0);
    int tmp = si->side[0];

    if (i == 1 && r / 3 >= 1) {
      P_BLOCK_START(p);
      set_param(si, r - (r / 3) * 2);
      P_APP(p, print_kernel_env, kernel, si);
      P_APP(p, print_kernel_call, prog, kernel, r - (r / 3) * 2);
      P_BLOCK_END(p);

      P_FMTLINE(p, "%@ += 1;", iter);

      P_BLOCK_START(p);
      set_param(si, r / 3);
      P_APP(p, print_kernel_env, kernel, si);
      P_APP(p, print_kernel_call, prog, kernel, r / 3);
      P_BLOCK_END(p);

      P_FMTLINE(p, "%@ += 1;", iter);

      P_BLOCK_START(p);
      set_param(si, r / 3);
      P_APP(p, print_kernel_env, kernel, si);
      P_APP(p, print_kernel_call, prog, kernel, r / 3);
      P_BLOCK_END(p);
    }
    else if  (r / 2 >= 1) {
      P_BLOCK_START(p);
      set_param(si, r - r / 2);
      P_APP(p, print_kernel_env, kernel, si);
      P_APP(p, print_kernel_call, prog, kernel, r - r / 2);
      P_BLOCK_END(p);

      P_FMTLINE(p, "%@ += 1;", iter);

      P_BLOCK_START(p);
      set_param(si, r / 2);
      P_APP(p, print_kernel_env, kernel, si);
      P_APP(p, print_kernel_call, prog, kernel, r / 2);
      P_BLOCK_END(p);
    }

    P_BLOCK_END(p);

    si->side[0] = tmp;
  }
  P_BLOCK_END(p);
  P_FMTLINE(p, "else if (__c0Len %% __side0LenMax)");
  P_BLOCK_START(p);
  for (int i = 1; i < si->side[0]; i++) {
    if (i == 1)
      P_FMT(p, "%<");
    else
      P_FMT(p, "%<else ");

    int tmp = si->side[0];

    P_FMT(p, "if (__c0Len %% __side0LenMax == %d)%>", i);
    P_BLOCK_START(p);
    set_param(si, i);
    P_APP(p, print_kernel_env, kernel, si);
    P_APP(p, print_kernel_call, prog, kernel, i);
    P_BLOCK_END(p);

    si->side[0] = tmp;
  }
  P_BLOCK_END(p);

  P_BLOCK_END(p);

  P_FMTLINE(p, "cudaCheckKernel();");

  isl_printer_free(p);
  isl_ast_expr_free(iter);
}

static isl_printer *print_iter(__isl_take isl_printer *p,
  struct stencil_info *si)
{
  for (int i = 1; i < si->dim; i++) {
    P_FMTLINE(p, "const AN5D_TYPE __side%dNum = (__c%dLen + __side%dLen - 1) / __side%dLen;", i, i, i, i);
  }

  P_FMTLINE(p, "const AN5D_TYPE __tid = threadIdx.y * blockDim.x + threadIdx.x;");

  for (int i = si->dimstart; i < si->dim; i++) {
    P_FMT(p, "%<const AN5D_TYPE __local_c%d = __tid", i);

    for (int j = i + 1; j < si->dim; j++)
      P_FMT(p, " / __side%dLenOl", j);

    if (i != si->dimstart)
      P_FMT(p, " %% __side%dLenOl", i);
    P_FMT(p, ";%>");
  }

  if (si->stream) {
    P_FMT(p, "%<const AN5D_TYPE __c1Id = blockIdx.x");
    for (int j = 2; j < si->dim; j++)
      P_FMT(p, " / __side%dNum", j);
    P_FMT(p, ";%>");
  }

  for (int i = si->dimstart; i < si->dim; i++) {
    P_FMT(p, "%<const AN5D_TYPE __c%d = (blockIdx.x", i);

    for (int j = i + 1; j < si->dim; j++)
      P_FMT(p, " / __side%dNum", j);

    P_FMT(p, " %% __side%dNum", i);

    P_FMT(p, ") * __side%dLen + __local_c%d + __c%dPad - __OlLen%d;%>", i, i, i, i);
  }

  return p;
}

static void print_kernel(struct gpu_prog *prog, struct ppcg_kernel *kernel,
  struct cuda_info *cuda, struct stencil_info *si,
  __isl_keep isl_ast_node *node)
{
  isl_printer *p;

  p = isl_printer_to_file(prog->ctx, cuda->kernel_c);
  p = isl_printer_set_output_format(p, ISL_FORMAT_C);
  p = isl_printer_indent(p, 4);

  p = print_an5d_type(p);

  P_APP(p, print_loop_info, kernel->tree, si);
  P_APP(p, print_param, si);
  P_APP(p, print_iter, si);

  int planenum = (!si->stream) ? 1 : si->halo[1] * 2 + 1;

  if (si->associative)
    P_FMTLINE(p, "%s __reg_0;", si->array->array->type);
  for (int i = 0; i < si->side[0]; i++) {
    for (int l = 0; l < planenum; l++) {
      P_FMTLINE(p, "%s __reg_%d_%d;",
                si->array->array->type, i + (si->associative ? 1 : 0), l);
    }
  }

  if (si->nondiag || si->associative) {
    int center = 'a' + (si->nondiag ? planenum / 2 : 0);
    P_FMTLINE(p, "__shared__ %s __%c_sb_double[__blockSize * 2];",
              si->array->array->type, center);
    P_FMTLINE(p, "%s *__%c_sb = __%c_sb_double;", si->array->array->type, center, center);
  }
  else {
    for (int l = 0; l < planenum; l++) {
      P_FMTLINE(p, "__shared__ %s __%c_sb_double[__blockSize * 2];",
                si->array->array->type, 'a' + l);
      P_FMTLINE(p, "%s *__%c_sb = __%c_sb_double;", si->array->array->type, 'a' + l, 'a' + l);
    }
  }

  P_FMT(p, "%<const AN5D_TYPE __loadValid = 1");
  for (int i = si->dimstart; i < si->dim; i++) {
    P_FMT(p, " && __c%d >= __c%dPad - __halo%d", i, i, i);
    P_FMT(p, " && __c%d < __c%dPad + __c%dLen + __halo%d", i, i, i, i);
  }
  P_FMT(p, ";%>");

  P_FMT(p, "%<const AN5D_TYPE __updateValid = 1");
  for (int i = si->dimstart; i < si->dim; i++) {
    P_FMT(p, " && __c%d >= __c%dPad", i, i);
    P_FMT(p, " && __c%d < __c%dPad + __c%dLen", i, i, i);
  }
  P_FMT(p, ";%>");

  for (int i = 1; i <= si->side[0]; i++) {
    P_FMT(p, "%<const AN5D_TYPE __writeValid%d = __updateValid", i);
    for (int j = si->dimstart; j < si->dim; j++) {
      P_FMT(p, " && __local_c%d >= (__halo%d * %d)", j, j, i);
      P_FMT(p, " && __local_c%d < __side%dLenOl - (__halo%d * %d)", j, j, j, i);
    }
    P_FMT(p, ";%>");
  }

  P_FMTLINE(p, "const AN5D_TYPE __storeValid = __writeValid%d;", si->side[0]);

  struct stencil_access *sa;

  for (sa = si->access; sa; sa = sa->next)
    if (sa->type == stencil_access_write)
      break;

  assert(sa);

  isl_ast_expr *access = create_array_access(si->array, sa->vlist);

  if (!si->stream) {
    P_FMTLINE(p, "#define __LOAD(reg) do { if (__loadValid) {\
 reg = %@; }} while (0)", access);
  } else {
    P_FMTLINE(p, "AN5D_TYPE __c1;");
    P_FMTLINE(p, "AN5D_TYPE __h;");
    P_FMTLINE(p, "const AN5D_TYPE __c1Pad2 = __c1Pad + __side1Len * __c1Id;");

    P_FMTLINE(p, "#define __LOAD(reg, h) do { if (__loadValid) {\
 __c1 = __c1Pad2 - __halo1 + h; reg = %@; }} while (0)", access);
  }

  if (!si->associative) {
    P_APP(p, print_accessor, -1, node, si);

    for (int i = 1; i < si->side[0]; i++) {
      P_FMT(p, "%<#define __CALC%d(out, reg0", i);
      for (int l = 1; l < planenum; l++)
        P_FMT(p, ", reg%d", l);
      P_FMT(p, ") do { __CALCSETUP(reg0");
      for (int l = 1; l < planenum; l++)
        P_FMT(p, ", reg%d", l);
      P_FMT(p, "); if (__writeValid%d) __CALCEXPR(out, reg0", i);
      for (int l = 1; l < planenum; l++)
        P_FMT(p, ", reg%d", l);
      P_FMT(p, "); else out = reg%d; } while (0)%>", planenum / 2);
    }
  }
  else {
    for (int plane = 0; plane <= 2 * si->halo[1]; plane++)
      P_APP(p, print_accessor, plane, node, si);

    P_FMT(p, "%<#define __CALCEXPR(out0");
    for (int l = 1; l < planenum; l++)
      P_FMT(p, ", out%d", l);
    P_FMT(p, ", reg) do { ");
    for (int l = 0; l < planenum; l++)
      P_FMT(p, "__CALCEXPR_%d(out%d, reg); ", l, l);
    P_FMT(p, "} while (0);%>");

    for (int i = 1; i <= si->side[0]; i++) {
      P_FMT(p, "%<#define __CALC%d(out0", i);
      for (int l = 1; l < planenum; l++)
        P_FMT(p, ", out%d", l);
      P_FMT(p, ", reg) do { __CALCSETUP(reg); if (__writeValid%d) { __CALCEXPR(out0", i);
      for (int l = 1; l < planenum; l++)
        P_FMT(p, ", out%d", l);
      P_FMT(p, ", reg); } else out%d = reg; } while (0)%>", planenum / 2);
    }
  }

  if (!si->stream) {
    P_FMT(p, "%<#define __STORE(reg0");
    for (int l = 1; l < planenum; l++)
      P_FMT(p, ", reg%d", l);
    P_FMT(p, ") do { __CALCSETUP(reg0");
    for (int l = 1; l < planenum; l++)
      P_FMT(p, ", reg%d", l);
    P_FMT(p, "); if (__storeValid) __CALCEXPR(__DEST, reg0");
    for (int l = 1; l < planenum; l++)
      P_FMT(p, ", reg%d", l);
    P_FMT(p, "); } while (0)%>");
  }
  else if (si->associative) {
    P_FMT(p, "%<#define __STORE(h, out) do { if (__storeValid) { \
__c1 = __c1Pad2 - __halo1 + h; __DEST = out; }} while (0)%>");
  }
  else {
    P_FMT(p, "%<#define __STORE(h");
    for (int l = 0; l < planenum; l++)
      P_FMT(p, ", reg%d", l);
    P_FMT(p, ") do { __CALCSETUP(reg0");
    for (int l = 1; l < planenum; l++)
      P_FMT(p, ", reg%d", l);
    P_FMT(p, "); if (__storeValid) { __c1 = __c1Pad2 - __halo1 + h; \
__CALCEXPR(__DEST, reg0");
    for (int l = 1; l < planenum; l++)
      P_FMT(p, ", reg%d", l);
    P_FMT(p, "); } } while (0)%>");
  }

  // No stream
  if (!si->stream) {
    P_BLOCK_START(p);
    for (int i = 0; i <= si->side[0]; i++) {
      if (i == 0)
        P_FMTLINE(p, "__LOAD(__reg_%d_0);", 0);
      else if (i < si->side[0])
        P_FMTLINE(p, "__CALC%d(__reg_%d_0, __reg_%d_0);", i, i, i - 1);
      else
        P_FMTLINE(p, "__STORE(__reg_%d_0);", i - 1);
    }
    P_BLOCK_END(p);

    isl_ast_expr_free(access);
    isl_printer_free(p);
    return;
  }

  int z[100], z2[100];

  if (!si->associative) {
    P_FMTLINE(p, "if (__c1Id == 0)");
    P_BLOCK_START(p);

    for (int i = 0; i < si->halo[1]; i++)
      P_FMTLINE(p, "__LOAD(__reg_%d_%d, %d);", si->side[0] - 1, i, i);

    for (int i = 0; i <= si->side[0]; i++) {
      z[i] = si->halo[1] - 1;
    }

    int double_buffer_switching_count1 = 0;

    while (z[si->side[0]] < si->halo[1] * si->side[0]) {
      for (int i = 0; i <= si->side[0]; i++) {
        if (i == 0 || z[i-1] == z[i] + 1 + si->halo[1]) {
          z[i]++;

          if (i == 0)
            P_FMTLINE(p, "__LOAD(__reg_%d_%d, %d);", 0, z[i] % planenum, z[i]);

          else {
            double_buffer_switching_count1++;

            if (i < si->side[0])
              P_FMT(p, "%<__CALC%d(__reg_%d_%d", i, i, z[i] % planenum);
            else
              P_FMT(p, "%<__STORE(%d", z[i]);

            for (int j = - si->halo[1]; j <= si->halo[1]; j++)
              if (z[i] + j < si->halo[1])
                P_FMT(p, ", __reg_%d_%d", si->side[0] - 1, z[i] + j);
              else
                P_FMT(p, ", __reg_%d_%d", i - 1, (z[i] + j) % planenum);

            P_FMT(p, ");%>");
          }
        }
      }
    }

    P_BLOCK_END(p);

    P_FMTLINE(p, "else");
    P_BLOCK_START(p);

    for (int i = 0; i <= si->side[0]; i++) {
      z2[i] = -1 + si->halo[1] * i;
    }

    int double_buffer_switching_count2 = 0;

    while (z2[si->side[0]] < si->halo[1] * si->side[0]) {
      for (int i = 0; i <= si->side[0]; i++) {
        if (i == 0 || z2[i-1] == z2[i] + 1 + si->halo[1]) {
          z2[i]++;

          if (i == 0)
            P_FMTLINE(p, "__LOAD(__reg_%d_%d, %d);", 0, z2[i] % planenum, z2[i]);

          else {
            double_buffer_switching_count2++;
            
            if (i < si->side[0])
              P_FMT(p, "%<__CALC%d(__reg_%d_%d", i, i, z2[i] % planenum);
            else
              P_FMT(p, "%<__STORE(%d", z2[i]);

            for (int j = - si->halo[1]; j <= si->halo[1]; j++)
              P_FMT(p, ", __reg_%d_%d", i - 1, (z2[i] + j) % planenum);

            P_FMT(p, ");%>");
          }
        }
      }
    }

    if (double_buffer_switching_count1 % 2 != double_buffer_switching_count2 % 2)
      P_FMTLINE(p, "__DB_SWITCH(); __syncthreads();");

    P_BLOCK_END(p);

    // Validate implementation
    for (int i = 0; i <= si->side[0]; i++) {
      assert(z[i] == z2[i]);
    }

    if (si->nondiag) {
      int center = 'a' + (si->nondiag ? planenum / 2 : 0);
      P_FMTLINE(p, "__%c_sb = __%c_sb_double + __blockSize * %d;",
                center, center, double_buffer_switching_count1 % 2);
    }
    else {
      for (int l = 0; l < planenum; l++) {
        P_FMTLINE(p, "__%c_sb = __%c_sb_double + __blockSize * %d;",
                  'a' + l, 'a' + l, double_buffer_switching_count1 % 2);
      }
    }

    P_FMTLINE(p, "if (__c1Id == __side1Num - 1)");
    P_BLOCK_START(p);

    P_FMTLINE(p, "for (__h = %d; __h <= __c1Len - __side1Len * __c1Id + __halo1 * 2 - %d;)",
              z[0] + 1, planenum);
    P_BLOCK_START(p);
    for (int l = 0; l < planenum; l++) {
      for (int i = 0; i <= si->side[0]; i++) {
        z[i]++;

        if (i == 0)
          P_FMTLINE(p, "__LOAD(__reg_%d_%d, __h);", 0, z[i] % planenum);

        else {
          if (i < si->side[0])
            P_FMT(p, "%<__CALC%d(__reg_%d_%d", i, i, z[i] % planenum);
          else
            P_FMT(p, "%<__STORE(__h - %d", z[0] - z[i]);

          for (int j = - si->halo[1]; j <= si->halo[1]; j++)
            P_FMT(p, ", __reg_%d_%d", i - 1, (z[i] + j) % planenum);

          P_FMT(p, ");%>");
        }
      }
      P_FMTLINE(p, "__h++;");
    }
    if (si->side[0] % 2)
      P_FMTLINE(p, "__DB_SWITCH(); __syncthreads();");
    P_BLOCK_END(p);

    P_FMTLINE(p, "if (0) {}");
    for (int l = 0; l < planenum; l++) {
      P_FMTLINE(p, "else if (__h + %d == __c1Len - __side1Len * __c1Id + __halo1 * 2)", l);
      P_BLOCK_START(p);

      for (int zs = 1; zs <= l + si->side[0] * si->halo[1]; zs++) {

        for (int i = 0; i <= si->side[0]; i++) {

          if ((i == 0 && zs <= l) ||
              (i != 0 && zs <= l + (i - 1) * si->halo[1])) {
            if (i == 0)
              P_FMTLINE(p, "__LOAD(__reg_%d_%d, __h + %d);", 0, (z[i] + zs) % planenum, zs - 1);

            else {
              if (i < si->side[0])
                P_FMT(p, "%<__CALC%d(__reg_%d_%d", i, i, (z[i] + zs) % planenum);
              else {
                int diff = zs - (z[0] - z[i]) - 1;
                P_FMT(p, "%<__STORE(__h %c %d", (diff < 0) ? '-' : '+', abs(diff));
              }

              for (int j = - si->halo[1]; j <= si->halo[1]; j++)
                if (l + i * si->halo[1] - (zs + j) < si->halo[1])
                  P_FMT(p, ", __reg_%d_%d", 0, (z[i] + zs + j) % planenum);

                else
                  P_FMT(p, ", __reg_%d_%d", i - 1, (z[i] + zs + j) % planenum);

              P_FMT(p, ");%>");
            }
          }
        }
      }

      P_BLOCK_END(p);
    }
    P_BLOCK_END(p);

    P_FMTLINE(p, "else");
    P_BLOCK_START(p);
    P_FMTLINE(p, "for (__h = %d; __h <= __side1LenOl - %d;)", z2[0] + 1, planenum);
    P_BLOCK_START(p);
    for (int l = 0; l < planenum; l++) {
      for (int i = 0; i <= si->side[0]; i++) {
        z2[i]++;

        if (i == 0)
          P_FMTLINE(p, "__LOAD(__reg_%d_%d, __h);", 0, z2[i] % planenum);

        else {
          if (i < si->side[0])
            P_FMT(p, "%<__CALC%d(__reg_%d_%d", i, i, z2[i] % planenum);
          else
            P_FMT(p, "%<__STORE(__h - %d", z2[0] - z2[i]);

          for (int j = - si->halo[1]; j <= si->halo[1]; j++)
            P_FMT(p, ", __reg_%d_%d", i - 1, (z2[i] + j) % planenum);

          P_FMT(p, ");%>");
        }
      }
      P_FMTLINE(p, "__h++;");
    }
    if (si->side[0] % 2)
      P_FMTLINE(p, "__DB_SWITCH();  __syncthreads();");
    P_BLOCK_END(p);

    for (int l = 0; l < planenum; l++) {
      P_FMTLINE(p, "if (__h == __side1LenOl) return;");
      for (int i = 0; i <= si->side[0]; i++) {
        z2[i]++;

        if (i == 0)
          P_FMTLINE(p, "__LOAD(__reg_%d_%d, __h);", 0, z2[i] % planenum);

        else {
          if (i < si->side[0])
            P_FMT(p, "%<__CALC%d(__reg_%d_%d", i, i, z2[i] % planenum);
          else
            P_FMT(p, "%<__STORE(__h - %d", z2[0] - z2[i]);

          for (int j = - si->halo[1]; j <= si->halo[1]; j++)
            P_FMT(p, ", __reg_%d_%d", i - 1, (z2[i] + j) % planenum);

          P_FMT(p, ");%>");
        }
      }
      P_FMTLINE(p, "__h++;");
    }
  
    P_BLOCK_END(p);
  }
  else {  /* Associative stencil optimization */

    P_FMTLINE(p, "if (__c1Id == 0)");
    P_BLOCK_START(p);

    int double_buffer_switching_count1 = 0;

    for (int l = si->halo[1]; l <= si->halo[1] * 2 - 1; l++) {
      P_FMTLINE(p, "__LOAD(__reg_0, %d);", l - si->halo[1]);

      for (int i = 1; i <= si->side[0]; i++) {
        double_buffer_switching_count1++;

        P_FMT(p, "%<__CALC%d(__reg_%d_%d", i, i, l);
        for (int j = 1; j < planenum; j++)
          P_FMT(p, ", __reg_%d_%d", i, (planenum - j + l) % planenum);
        P_FMT(p, ", __reg_0);%>");
      }
    }

    z[0] = si->halo[1] - 1;
    for (int i = 1; i <= si->side[0]; i++)
      z[i] = si->halo[1] * 2 - 1;

    while (z[si->side[0]] < si->halo[1] * si->side[0] + si->halo[1] * 2) {
      for (int i = 0; i <= si->side[0]; i++) {
        if (i == 0) {
          z[i]++;
          P_FMTLINE(p, "__LOAD(__reg_0, %d);", z[i]);
        }
        else if ((i == 1) ||
                 (i >= 2 && (z[i] + 1) - (z[i-1] - si->halo[1] * 2) == si->halo[1])) {
          z[i]++;
          double_buffer_switching_count1++;

          P_FMT(p, "%<__CALC%d(__reg_%d_%d", i, i, z[i] % planenum);
          for (int j = 1; j < planenum; j++)
            P_FMT(p, ", __reg_%d_%d", i, (z[i] - j + planenum) % planenum);
          if (i == 1)
            P_FMT(p, ", __reg_0");
          else
            P_FMT(p, ", __reg_%d_%d", i - 1, (z[i-1] - si->halo[1] * 2) % planenum);
          P_FMT(p, ");%>");

          if (i == si->side[0] && (z[i] - si->halo[1] * 2) >= si->halo[1])
            P_FMTLINE(p, "__STORE(%d, __reg_%d_%d);",
                      z[i] - si->halo[1] * 2, i, (z[i] - si->halo[1] * 2) % planenum);
        }
      }
    }
    P_BLOCK_END(p);

    P_FMTLINE(p, "else");
    P_BLOCK_START(p);

    for (int i = 0; i <= si->side[0]; i++) {
      z2[i] = si->halo[1] * i - 1;
    }

    int double_buffer_switching_count2 = 0;

    while (z2[si->side[0]] < si->halo[1] * si->side[0] + si->halo[1] * 2) {
      for (int i = 0; i <= si->side[0]; i++) {
        if (i == 0) {
          z2[i]++;
          P_FMTLINE(p, "__LOAD(__reg_0, %d);", z2[i]);
        }
        else if ((i == 1) ||
                 (i >= 2 && (z2[i] + 1) - (z2[i-1] - si->halo[1] * 2) == si->halo[1])) {
          z2[i]++;
          double_buffer_switching_count2++;

          P_FMT(p, "%<__CALC%d(__reg_%d_%d", i, i, z2[i] % planenum);
          for (int j = 1; j < planenum; j++)
            P_FMT(p, ", __reg_%d_%d", i, (z2[i] - j + planenum) % planenum);
          if (i == 1)
            P_FMT(p, ", __reg_0");
          else
            P_FMT(p, ", __reg_%d_%d", i - 1, (z2[i-1] - si->halo[1] * 2) % planenum);
          P_FMT(p, ");%>");

          if (i == si->side[0] && (z2[i] - si->halo[1] * 2) >= si->halo[1] * si->side[0])
            P_FMTLINE(p, "__STORE(%d, __reg_%d_%d);",
                      z2[i] - si->halo[1] * 2, i, (z2[i] - si->halo[1] * 2) % planenum);
        }
      }
    }

    if (double_buffer_switching_count1 % 2 != double_buffer_switching_count2 % 2)
      P_FMTLINE(p, "__DB_SWITCH(); __syncthreads();");

    P_BLOCK_END(p);

    // Validate implementation
    for (int i = 0; i <= si->side[0]; i++) {
      assert(z[i] == z2[i]);
    }

    P_FMTLINE(p, "__a_sb = __a_sb_double + __blockSize * %d;",
              double_buffer_switching_count1 % 2);

    P_FMTLINE(p, "if (__c1Id == __side1Num - 1)");
    P_BLOCK_START(p);

    P_FMTLINE(p, "for (__h = %d; __h <= __c1Len - __side1Len * __c1Id + __halo1 * 2 - %d;)",
              z[0] + 1, planenum + si->halo[1]);
    P_BLOCK_START(p);
    for (int l = 0; l < planenum; l++) {
      for (int i = 0; i <= si->side[0]; i++) {
        if (i == 0) {
          z[i]++;
          P_FMTLINE(p, "__LOAD(__reg_0, __h);");
        }
        else if ((i == 1) ||
                 (i >= 2 && (z[i] + 1) - (z[i-1] - si->halo[1] * 2) == si->halo[1])) {
          z[i]++;

          P_FMT(p, "%<__CALC%d(__reg_%d_%d", i, i, z[i] % planenum);
          for (int j = 1; j < planenum; j++)
            P_FMT(p, ", __reg_%d_%d", i, (z[i] - j) % planenum);
          if (i == 1)
            P_FMT(p, ", __reg_0");
          else
            P_FMT(p, ", __reg_%d_%d", i - 1, (z[i-1] - si->halo[1] * 2) % planenum);
          P_FMT(p, ");%>");

          if (i == si->side[0] && (z[i] - si->halo[1] * 2) >= si->halo[1])
            P_FMTLINE(p, "__STORE(__h - %d, __reg_%d_%d);", z[0] - (z[i] - si->halo[1] * 2),
                      i, (z[i] - si->halo[1] * 2) % planenum);
        }
      }
      P_FMTLINE(p, "__h++;");
    }
    if (si->side[0] % 2)
      P_FMTLINE(p, "__DB_SWITCH(); __syncthreads();");
    P_BLOCK_END(p);

    for (int i = 0; i <= si->side[0]; i++)
      z[i]++;

    P_FMTLINE(p, "if (0) {}");
    for (int l = si->halo[1]; l < planenum + si->halo[1]; l++) {
      P_FMTLINE(p, "else if (__h + %d == __c1Len - __side1Len * __c1Id + __halo1 * 2)", l);
      P_BLOCK_START(p);

      for (int i2 = 1; i2 <= si->side[0]; i2++) {

        for (int zs = ((i2 == 1) ? 0 : l - si->halo[1]); zs < l; zs++) {
          int pad = zs + (i2 - 1) * si->halo[1];

          for (int i = i2; i <= si->side[0]; i++) {
            if (i == 1) {
              P_FMTLINE(p, "__LOAD(__reg_0, __h + %d);", pad);
            }

            if (zs >= l - si->halo[1] && i == i2) {
              P_FMT(p, "%<__CALC%d(__reg_%d_%d", i, i, (z[i] + pad) % planenum);
              int j;
              for (j = 1; j <= si->halo[1] + (zs - (l - si->halo[1])); j++)
                P_FMT(p, ", __reg_%d_%d", i, (z[i] + pad) % planenum);
              for (; j < planenum; j++)
                P_FMT(p, ", __reg_%d_%d", i, (z[i] + pad - j) % planenum);
            }
            else {
              P_FMT(p, "%<__CALC%d(__reg_%d_%d", i, i, (z[i] + pad) % planenum);
              for (int j = 1; j < planenum; j++)
                P_FMT(p, ", __reg_%d_%d", i, (z[i] + pad - j) % planenum);
            }
            if (i == 1)
              P_FMT(p, ", __reg_0");
            else
              P_FMT(p, ", __reg_%d_%d", i - 1, (z[i-1] + pad - si->halo[1] * 2) % planenum);
            P_FMT(p, ");%>");

            if (i == si->side[0]) {
              int diff = z[0] - (z[i] - si->halo[1] * 2) - pad;
              P_FMTLINE(p, "__STORE(__h %c %d, __reg_%d_%d);",
                        (diff > 0) ? '-' : '+', abs(diff),
                        i, (z[i] + pad - si->halo[1] * 2) % planenum);
            }
          }

          if (zs >= l - si->halo[1] && i2 < si->side[0]) {
            P_FMT(p, "%<__reg_%d_%d = ", i2, (z[i2] + pad - si->halo[1]) % planenum);
            if (i2 == 1)
              P_FMT(p, "__reg_0");
            else
              P_FMT(p, "__reg_%d_%d", i2 - 1, (z[i2 - 1] + pad - si->halo[1] * 2) % planenum);
            P_FMT(p, ";%>");
          }
        }
      }

      P_BLOCK_END(p);
    }
    P_BLOCK_END(p);

    P_FMTLINE(p, "else");
    P_BLOCK_START(p);
    P_FMTLINE(p, "for (__h = %d; __h <= __side1LenOl - %d;)", z2[0] + 1, planenum);
    P_BLOCK_START(p);
    for (int l = 0; l < planenum; l++) {
      for (int i = 0; i <= si->side[0]; i++) {
        if (i == 0) {
          z2[i]++;
          P_FMTLINE(p, "__LOAD(__reg_0, __h);");
        }
        else if ((i == 1) ||
                 (i >= 2 && (z2[i] + 1) - (z2[i-1] - si->halo[1] * 2) == si->halo[1])) {
          z2[i]++;

          P_FMT(p, "%<__CALC%d(__reg_%d_%d", i, i, z2[i] % planenum);
          for (int j = 1; j < planenum; j++)
            P_FMT(p, ", __reg_%d_%d", i, (z2[i] - j) % planenum);
          if (i == 1)
            P_FMT(p, ", __reg_0");
          else
            P_FMT(p, ", __reg_%d_%d", i - 1, (z2[i-1] - si->halo[1] * 2) % planenum);
          P_FMT(p, ");%>");

          if (i == si->side[0] && (z2[i] - si->halo[1] * 2) >= si->halo[1])
            P_FMTLINE(p, "__STORE(__h - %d, __reg_%d_%d);", z2[0] - (z2[i] - si->halo[1] * 2),
                      i, (z2[i] - si->halo[1] * 2) % planenum);
        }
      }
      P_FMTLINE(p, "__h++;");
    }
    if (si->side[0] % 2)
      P_FMTLINE(p, "__DB_SWITCH(); __syncthreads();");
    P_BLOCK_END(p);

    for (int l = 0; l < planenum; l++) {
      P_FMTLINE(p, "if (__h == __side1LenOl) return;");
      for (int i = 0; i <= si->side[0]; i++) {
        if (i == 0) {
          z2[i]++;
          P_FMTLINE(p, "__LOAD(__reg_0, __h);");
        }
        else if ((i == 1) ||
                 (i >= 2 && (z2[i] + 1) - (z2[i-1] - si->halo[1] * 2) == si->halo[1])) {
          z2[i]++;

          P_FMT(p, "%<__CALC%d(__reg_%d_%d", i, i, z2[i] % planenum);
          for (int j = 1; j < planenum; j++)
            P_FMT(p, ", __reg_%d_%d", i, (z2[i] - j) % planenum);
          if (i == 1)
            P_FMT(p, ", __reg_0");
          else
            P_FMT(p, ", __reg_%d_%d", i - 1, (z2[i-1] - si->halo[1] * 2) % planenum);
          P_FMT(p, ");%>");

          if (i == si->side[0] && (z2[i] - si->halo[1] * 2) >= si->halo[1])
            P_FMTLINE(p, "__STORE(__h - %d, __reg_%d_%d);", z2[0] - (z2[i] - si->halo[1] * 2),
                      i, (z2[i] - si->halo[1] * 2) % planenum);
        }
      }
      P_FMTLINE(p, "__h++;");
    }

    P_BLOCK_END(p);
  }

  isl_ast_expr_free(access);
  isl_printer_free(p);
}

static __isl_give isl_printer *print_kernel_header_t(__isl_take isl_printer *p,
  struct gpu_prog *prog, struct ppcg_kernel *kernel, int t)
{
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "__global__ void kernel");
  p = isl_printer_print_int(p, kernel->id);
  p = isl_printer_print_str(p, "_");
  p = isl_printer_print_int(p, t);
  p = isl_printer_print_str(p, "(");
  p = print_kernel_arguments(p, prog, kernel, 1);
  p = isl_printer_print_str(p, ")");

  return p;
}

static void print_kernel_headers_t(struct gpu_prog *prog,
  struct ppcg_kernel *kernel, struct cuda_info *cuda, int t)
{
  isl_printer *p;

  p = isl_printer_to_file(prog->ctx, cuda->kernel_h);
  p = isl_printer_set_output_format(p, ISL_FORMAT_C);
  p = print_kernel_header_t(p, prog, kernel, t);
  p = isl_printer_print_str(p, ";");
  p = isl_printer_end_line(p);
  isl_printer_free(p);

  p = isl_printer_to_file(prog->ctx, cuda->kernel_c);
  p = isl_printer_set_output_format(p, ISL_FORMAT_C);
  p = print_kernel_header_t(p, prog, kernel, t);
  p = isl_printer_end_line(p);
  isl_printer_free(p);
}

static void print_sbref_wrap(struct gpu_prog *prog, struct cuda_info *cuda,
  struct stencil_info *si)
{
  isl_printer *p;

  p = isl_printer_to_file(prog->ctx, cuda->kernel_c);
  p = isl_printer_set_output_format(p, ISL_FORMAT_C);

  P_FMTLINE(p, "__device__%s %s __sbref_wrap(%s *sb, size_t index) \
{ return sb[index]; }", si->option_sm_vec ? " __forceinline__" : "",
            si->array->array->type, si->array->array->type);

  p = isl_printer_end_line(p);
  isl_printer_free(p);
}

static void add_kernel_argument(struct ppcg_kernel *kernel)
{
  isl_ast_node *tree = kernel->tree;

  isl_ast_expr *iter = isl_ast_node_for_get_iterator(tree);

  kernel->space = isl_space_insert_dims(kernel->space, isl_dim_set, 0, 1);

  isl_space_set_dim_id(kernel->space, isl_dim_set, 0,
                       isl_id_alloc(kernel->ctx, isl_ast_expr_to_C_str(iter), NULL));

  isl_ast_expr_free(iter);
}

isl_bool try_print_stencil_kernel(struct gpu_prog *prog,
  struct ppcg_kernel *kernel, struct cuda_info *cuda)
{
  if (is_stencil_schedule(prog->original_schedule_node) <= 0)
    return isl_bool_error;

  assert(isl_ast_node_get_type(kernel->tree) == isl_ast_node_for);

  isl_ast_node *node = get_node_user_if_singleton(kernel->tree);

  if (!node)
    return isl_bool_error;

  struct stencil_info *si = extract_stencil_info(kernel, node);

  if (!si) {
    isl_ast_node_free(node);
    return isl_bool_error;
  }

  add_kernel_argument(kernel);

  set_param(si, -1);
  print_host(prog, kernel, cuda, si);

  print_sbref_wrap(prog, cuda, si);

  for (int i = si->side[0]; i >= 1; i--) {
    set_param(si, i);
    print_kernel_headers_t(prog, kernel, cuda, i);
    fprintf(cuda->kernel_c, "{\n");
    print_kernel(prog, kernel, cuda, si, node);
    fprintf(cuda->kernel_c, "}\n");
  }

  stencil_info_free(si);
  isl_ast_node_free(node);

  return isl_bool_true;
}
