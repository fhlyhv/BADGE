#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

List FwdBwdDHMM(rowvec log_s_init, mat log_A, mat log_B, uword N, uword n_states);
