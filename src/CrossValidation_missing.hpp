#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]


using namespace Rcpp;
using namespace arma;


double CrossValidation_missing(double bandwidth_int, mat data, uword n_fold, uword N, uword P, bool is_nonzero_mean, umat is_not_na);
