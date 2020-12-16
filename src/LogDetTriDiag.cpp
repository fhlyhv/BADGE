#include "LogDetTriDiag.hpp"



/*
 Compute Log-Determinant of a Tri-diagonal matrix
 Yu Hang, NTU, Jul 2019
 Inputs:
           K_d:  P x 1 vector of the diagonal of the P x P tri-diagonal matrix
           K_od: P-1 x 1 vector of the NEGATIVE first off-diagonals of the P x P tri-diagonal matrix
*/

List LogDetTriDiag(mat K_d, mat K_od, uword P, uword N) {
  rowvec logdetK(N, fill::zeros), C_d2, C_od;
  bool is_pd = true;
  
  
  for (uword j = 0; j < P; j++) {
    if (j == 0) {
      C_d2 = K_d.row(j);
    }
    else {
      C_od = K_od.row(j-1) / sqrt(C_d2);
      C_d2 = K_d.row(j) - square(C_od);
    }
    if (any(C_d2 <= 0)) {
      is_pd = false;
      break;
    } else {
      logdetK += log(C_d2);
    }
  }
  return List::create(Named("logdetK") = logdetK.t(), Named("is_pd") = is_pd);
}
