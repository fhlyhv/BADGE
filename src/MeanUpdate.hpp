#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]


#include "FwdBwdCHMM.hpp"
using namespace Rcpp;
using namespace arma;


List MeanUpdate(mat EKd_mat, mat EKdinv_mat, mat EKod_mat, mat VKod_mat, mat data, mat Emu_mat, double Elambda, 
                uvec idj, uword N, uword P, uword Pa, uvec idl, uvec idu, uvec ida, umat ida_mat, uvec row_missing, bool is_row_missing);
