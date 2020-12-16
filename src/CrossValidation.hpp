#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]


using namespace Rcpp;
using namespace arma;


double CrossValidation(double bandwidth_int, mat data, uword n_fold, uword N, uword P, bool is_nonzero_mean);
