#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]


using namespace Rcpp;
using namespace arma;

List LogDetTriDiag(mat K_d, mat K_od, uword P, uword N);
