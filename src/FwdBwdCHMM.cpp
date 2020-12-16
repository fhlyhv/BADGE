#include "FwdBwdCHMM.hpp"


/*
 Forward-Backward Message Passing in Gaussian Markov Chains to compute the marginal and pairwise stats
 written in RcppArmadillo
 Yu Hang, NTU, Jul 2019
 Inputs:
           h_node and J_node: N x P matrix of node potentials in the information form, where N is length of 
                              the chains and P is the number of chains. We deal P chains in parallel
           J_edge:            (N-1) x P matrix of NEGATIVE edge potnetials 
                              (add - to the potentials to simplify the main script)
 Outputs:
           mu:                N x P matrix of marginal mean
           S_d:               N x P matrix of marginal variance
           S_od:              (N-1) x P matrix of covariance between every two consecutive time points

*/


List FwdBwdCHMM(mat h_node, mat J_node, mat J_edge, uword N, uword P) {
  mat h_fwd(N,P);
  mat J_fwd(N,P);
  mat mu(N,P);
  mat S_d(N,P);
  mat S_od(N-1,P);
  rowvec Jtmp, sum_E2_diff;
  bool is_pd = true;
  
  int i;
  rowvec mtmp, h_bwd, J_bwd, h_fwd_msg, J_fwd_msg, h_bwd_msg, J_bwd_msg;
  
  // compute forward messages
  h_fwd.row(0) = h_node.row(0);
  J_fwd.row(0) = J_node.row(0);
  for (i = 1; i < N; i++) {
    mtmp = J_edge.row(i-1) / J_fwd.row(i-1);
    h_fwd_msg = mtmp % h_fwd.row(i-1);
    J_fwd_msg = - mtmp % J_edge.row(i-1);
    h_fwd.row(i) = h_node.row(i) + h_fwd_msg;
    J_fwd.row(i) = J_node.row(i) + J_fwd_msg;
    if (accu(J_fwd.row(i) <= 0) > 0) {
      is_pd = false;
      break;
    }
    
  }
  
  // compute backward messegs & marginal and pairwise densities
  if (is_pd) {
    S_d.row(N-1) = 1 / J_fwd.row(N-1);
    mu.row(N-1) = h_fwd.row(N-1) % S_d.row(N-1);
    h_bwd = h_node.row(N-1);
    J_bwd = J_node.row(N-1);
    for (i = N-2; i >= 0; i--) {
      mtmp = J_edge.row(i) / J_bwd;
      h_bwd_msg = mtmp % h_bwd;
      J_bwd_msg = - mtmp % J_edge.row(i);
      Jtmp = J_bwd_msg + J_fwd.row(i);
      if (accu(Jtmp <= 0) > 0) {
        is_pd = false;
        break;
      }
      S_d.row(i) = 1 / Jtmp;
      mu.row(i) = (h_bwd_msg + h_fwd.row(i)) / Jtmp;
      S_od.row(i) = J_edge.row(i) / J_fwd.row(i) % S_d.row(i+1);
      if (i > 0) {
        h_bwd = h_node.row(i) + h_bwd_msg;
        J_bwd = J_node.row(i) + J_bwd_msg;
        if (accu(J_bwd <= 0) > 0) {
          is_pd = false;
          break;
        }
      } 
    }
  }
  
  if (is_pd) {
    sum_E2_diff = sum(square(mu.head_rows(N - 1) - mu.tail_rows(N - 1))) + sum(S_d.head_rows(N - 1)) + sum(S_d.tail_rows(N - 1)) - 2 * sum(S_od);
    is_pd = prod(sum_E2_diff > 1e-300) > 0;
  }
    
  
  return List::create(Named("mu") = mu, Named("S_d") = S_d, Named("S_od") = S_od, Named("sum_E2_diff") = sum_E2_diff, Named("is_pd") = is_pd);
}
