#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]


using namespace Rcpp;
using namespace arma;



List MissingUpdate(mat data, mat Vdata, mat COVdata, mat EKod_mat, mat VKod_mat, mat EKd_mat, mat EKdinv_mat, mat Emu_mat, 
                   double eta_thr, uvec idr_missing, umat id_missing, uvec idl, uvec idu,uword N, uword P, uword Pe, uword N_missing, 
                   bool is_nonzero_mean);
