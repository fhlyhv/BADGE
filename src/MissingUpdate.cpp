#include "MissingUpdate.hpp"


/*
 * update the mean and variance (covariance) of the missing data
 * 
 */


List MissingUpdate(mat data, mat Vdata, mat COVdata, mat EKod_mat, mat VKod_mat, mat EKd_mat, mat EKdinv_mat, mat Emu_mat, 
                   double eta_thr, uvec idr_missing, umat id_missing, uvec idl, uvec idu,uword N, uword P, uword Pe, uword N_missing, 
                   bool is_nonzero_mean)  {
  mat K, K_missing, K_old, S(P, P), S_missing, EKodi(P, P, fill::zeros), VKodi(P, P, fill::zeros);
  vec h, h_old, h_missing;
  uvec idc_missing, id1(1, fill::ones), idr(1), id_all(P), idc_obsv;
  uword i;
  double eta;
  
  for (i = 0; i < N_missing; i ++) {
    
    EKodi(idl) = EKod_mat.row(idr_missing(i));
    EKodi(idu) = EKod_mat.row(idr_missing(i));
    VKodi(idl) = VKod_mat.row(idr_missing(i));
    VKodi(idu) = VKod_mat.row(idr_missing(i));
    
    K = (EKodi % repmat(EKdinv_mat.row(idr_missing(i)), P, 1)) * EKodi / 2 + EKodi;
    K += K.t();
    K.diag() += (EKdinv_mat.row(idr_missing(i)) * VKodi + EKd_mat.row(idr_missing(i))).t();
    if (K.has_nan()) {
      printf("idr = %d\n", idr_missing(i));
      K.print();
    }
    idc_missing = id_missing(find(id_missing.col(0) == idr_missing(i)), id1);
    idr.fill(idr_missing(i));
    id_all.ones();
    id_all(idc_missing).zeros();
    idc_obsv = find(id_all == 1);
    if (K(idc_missing, idc_missing).is_sympd()) {
      K_missing = K(idc_missing, idc_missing);
      if (is_nonzero_mean) h_missing = K.rows(idc_missing) * Emu_mat.row(idr_missing(i)).t()
        - K(idc_missing, idc_obsv) * data(idr, idc_obsv).t();
      else h_missing = - K(idc_missing, idc_obsv) * data(idr, idc_obsv).t();
    } else {
      S.zeros();
      S(idl) = COVdata.row(i);
      S(idu) = COVdata.row(i);
      S.diag() = Vdata.row(i).t();
      K_old = inv_sympd(S(idc_missing, idc_missing));
      
      h_old = K_old * data(idr, idc_missing).t();
      
      eta = eta_thr;
      while (true) {
        K_missing = (1 - eta) * K_old + eta * K(idc_missing, idc_missing);
        if (K_missing.is_sympd()) {
          break;
        } else eta /= 2;
        if (eta < 1e-10) {
          eta = 0;
          K_missing = K_old;
          break;
        }
      }

      K_old.clear();
      if (is_nonzero_mean) h_missing = (1 - eta) * h_old + eta * (K.rows(idc_missing) * Emu_mat.row(idr_missing(i)).t()
                                                                    - K(idc_missing, idc_obsv) * data(idr, idc_obsv).t());
      else h_missing = (1 - eta) * h_old - eta * K(idc_missing, idc_obsv) * data(idr, idc_obsv).t();
      h_old.clear();
    }
    
    data(idr, idc_missing) = solve(K_missing, h_missing).t();
    //printf("i = %i, start inverting K_missing\n", i);
    S.zeros();
    S(idc_missing, idc_missing) = inv_sympd(K_missing);
    Vdata.row(i) = S.diag().t();
    COVdata.row(i) = S(idl).t();
    h_missing.clear();
    K_missing.clear();
  }
  
  return List::create(Named("data") = data, Named("Vdata") = Vdata, Named("COVdata") = COVdata);
  
}
