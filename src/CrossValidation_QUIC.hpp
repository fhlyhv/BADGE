#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include "QUICParameterLearning.hpp"

using namespace arma;


field<mat> CrossValidation_QUIC(mat data, uvec idr, uvec idc, uvec idl, uvec idu, uword N, uword P, uword Pe, bool is_nonzero_mean, 
                                bool is_missing, umat is_na, bool is_rnd_missing, uvec id_missing_vec, uvec idr_missing, int N_missing,
                                bool is_row_missing, uvec row_missing);
