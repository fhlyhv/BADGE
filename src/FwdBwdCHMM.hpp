#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

List FwdBwdCHMM(mat h_node, mat J_node, mat J_edge, uword N, uword P);
