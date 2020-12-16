#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]


using namespace Rcpp;
using namespace arma;


mat SumEKodMDataUpdate(mat data_mu, mat EKod_mat, uvec idl, uvec idu, umat idl_mat, uword N, uword P);
