#include "FwdBwdDHMM.hpp"


/*
 Forward-Backward Message Passing in Discrete Markov Chains to compute the marginal and pairwise stats
 written in RcppArmadillo
 Yu Hang, NTU, Jul 2019
 Inputs:
 log_A              n_states x n_states  logarithm of the Normalized transition matrix
 log_B              N x n_states  logarithm of the observation matrix
 N                  length of the time series
 n_states           number of states
 Outputs:
 marg_density:      N x 1 matrix of marginal mean
 sum_pair_density:  n_states x n_states matrix of the sum of the pairwise density across time
 entropy:           entropy of the HMM
 
 */



List FwdBwdDHMM(rowvec log_s_init, mat log_A, mat log_B, uword N, uword n_states) {
  mat log_alpha(N, n_states), log_beta(N, n_states, fill::zeros), tmp, marg_density, log_marg_density, sum_pair_density;
  cube pair_density(n_states, n_states, N - 1), log_pair_density(n_states, n_states, N - 1), log_diff, entropy_cube;
  rowvec tmp_rowmax;
  vec tmp_colmax;
  double entropy, entropy_max;
  
  log_alpha.row(0) = log_B.row(0) + log_s_init;
  log_alpha.row(0) -= log_alpha.row(0).max();
  // log_alpha.row(0) -= accu(exp(log_alpha.row(0)));
  
  // tmp = log_A;
  // tmp_rowmax = max(tmp);
  // tmp.each_row() -= tmp_rowmax;
  // log_alpha.row(0) = log(sum(exp(tmp))) + log_B.row(0) + tmp_rowmax;
  // log_alpha.row(0) -= log_alpha.row(0).max();
  
  
  int i, j;
  // log_alpha.row(0).zeros();
  for (i = 1; i < N; i ++) {
    tmp = log_A + repmat(log_alpha.row(i - 1).t(), 1, n_states);
    tmp_rowmax = max(tmp);
    tmp.each_row() -= tmp_rowmax;
    log_alpha.row(i) = log(sum(exp(tmp))) + log_B.row(i) + tmp_rowmax;
    log_alpha.row(i) -= log_alpha.row(i).max();
    //log_alpha.row(i) -= accu(exp(log_alpha.row(i)));
  }
  
  
  
  for (i = N - 2; i >= 0; i --) {
    tmp = log_A + repmat(log_B.row(i + 1) + log_beta.row(i + 1), 
                         n_states, 1);
    tmp_colmax = max(tmp, 1);
    tmp.each_col() -= tmp_colmax;
    log_beta.row(i) = (log(sum(exp(tmp),1)) + tmp_colmax).t();
    log_beta.row(i) -= log_beta.row(i).max();
    //log_beta.row(i) -= accu(exp(log_beta.row(i)));
  } 
  
  
  
  log_marg_density = log_alpha + log_beta;
  log_marg_density.each_col() -= max(log_marg_density, 1);
  log_marg_density.each_col() -= log(sum(exp(log_marg_density), 1));
  marg_density = exp(log_marg_density);
  
  for (i = 0; i < n_states; i ++) {
    for (j = 0; j < n_states; j ++) {
      log_pair_density.tube(i, j) = log_alpha.col(i).head(N - 1) +
        log_beta.col(j).tail(N - 1) + log_B.col(j).tail(N - 1) +
        log_A(i, j);
    }
  }
  
  for (i = 0; i < N - 1; i ++) {
    log_pair_density.slice(i) -= log_pair_density.slice(i).max();
    log_pair_density.slice(i) -= log(accu(exp(log_pair_density.slice(i))));
  }
  pair_density = exp(log_pair_density);
  sum_pair_density = sum(pair_density, 2);
  
  log_diff = log_pair_density.tail_slices(N - 2);
  for (i = 0; i < n_states; i ++) {
    for (j = 0; j < n_states; j ++) {
      log_diff.tube(i, j) -= log_marg_density(span(1, N - 2), i);
    }
  }
  
  entropy_cube = log_pair_density + log(abs(join_slices(log_pair_density.slice(0), log_diff)));
  entropy_max = entropy_cube.max();
  entropy = accu(exp(entropy_cube - entropy_max)) * exp(entropy_max);
  
  // entropy = accu(exp(log_pair_density + log(abs(log_pair_density)))) -
  //   accu(exp(log_marg_density + log(abs(log_marg_density))));
  
  // entropy = - accu(pair_density % log_pair_density) +
  //   accu(marg_density.rows(1, N - 2) %
  //   log_marg_density.rows(1, N - 2));
  
  
  return List::create(Named("marg_density") = marg_density.col(1), Named("sum_pair_density") = sum_pair_density, Named("entropy") = entropy);
  
  
  
}
