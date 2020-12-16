#include "SumEKodMDataUpdate.hpp"


/*
 * update the mean and variance (covariance) of the missing data
 * 
 */


mat SumEKodMDataUpdate(mat data_mu, mat EKod_mat, uvec idl, uvec idu, umat idl_mat, uword N, uword P) {
  uword i, j;
  uvec vP = regspace<uvec>(0, P - 1), id_tmp, idj_vec(1);
  mat EKodi, sum_EKod_m_data(N, P);
  
  
  if (P < N) {
    for (j = 0; j < P; j ++) {
      id_tmp = join_cols(vP.head(j), vP.tail(P - 1 - j));
      idj_vec.fill(j);
      sum_EKod_m_data.col(j) = sum(data_mu.cols(id_tmp) % EKod_mat.cols(idl_mat(idj_vec, id_tmp)), 1);
    }
  } else {
    EKodi.set_size(P, P);
    EKodi.zeros();
    for (i = 0; i < N; i ++) {
      EKodi(idl) = EKod_mat.row(i);
      EKodi(idu) = EKod_mat.row(i);
      sum_EKod_m_data.row(i) = data_mu.row(i) * EKodi;
    }
  }
  
  
  return sum_EKod_m_data;
}
