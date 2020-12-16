#include "FwdBwdCHMM_var.hpp"

List FwdBwdCHMM_var(mat J_node, mat J_edge, uword N, uword P) {
  mat J_fwd(N,P);
  mat S_d(N,P);
  mat S_od(N-1,P);
  rowvec Jtmp, sum_V_diff;
  bool is_pd = true;
  
  int i;
  rowvec mtmp, J_bwd, J_fwd_msg, J_bwd_msg;
  
  // compute forward messages
  J_fwd.row(0) = J_node.row(0);
  for (i = 1; i < N; i++) {
    mtmp = J_edge.row(i-1) / J_fwd.row(i-1);
    J_fwd_msg = - mtmp % J_edge.row(i-1);
    J_fwd.row(i) = J_node.row(i) + J_fwd_msg;
    if (accu(J_fwd.row(i) <= 0) > 0) {
      is_pd = false;
      break;
    }
  }
  
  // compute backward messegs & marginal and pairwise densities
  if (is_pd) {
    S_d.row(N-1) = 1 / J_fwd.row(N-1);
    J_bwd = J_node.row(N-1);
    for (i = N-2; i >= 0; i--) {
      mtmp = J_edge.row(i) / J_bwd;
      J_bwd_msg = - mtmp % J_edge.row(i);
      Jtmp = J_bwd_msg + J_fwd.row(i);
      if (accu(Jtmp <= 0) > 0) {
        is_pd = false;
        break;
      }
      S_d.row(i) = 1 / Jtmp;
      S_od.row(i) = J_edge.row(i) / J_fwd.row(i) % S_d.row(i+1);
      if (i > 0) {
        J_bwd = J_node.row(i) + J_bwd_msg;
        if (accu(J_bwd <= 0) > 0) {
          is_pd = false;
          break;
        }
      } 
    }
  }
  
  if (is_pd) {
    sum_V_diff = sum(S_d.head_rows(N - 1)) + sum(S_d.tail_rows(N - 1)) - 2 * sum(S_od);
    is_pd = prod(sum_V_diff > 1e-300) > 0;
  }
  
  
  
  return List::create(Named("S_d") = S_d, Named("S_od") = S_od, Named("sum_V_diff") = sum_V_diff, Named("is_pd") = is_pd);
}
