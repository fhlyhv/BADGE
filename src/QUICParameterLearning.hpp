#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace arma;



mat QUICParameterLearning(mat K0, mat S, uvec idr, uvec idc, uword max_outer_iter, uword max_inner_iter);
