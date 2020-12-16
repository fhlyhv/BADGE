#include "MeanUpdate.hpp"



List MeanUpdate(mat EKd_mat, mat EKdinv_mat, mat EKod_mat, mat VKod_mat, mat data, mat Emu_mat, double Elambda, 
               uvec idj, uword N, uword P, uword Pa, uvec idl, uvec idu, uvec ida, umat ida_mat, uvec row_missing, bool is_row_missing) {
  
  mat pL1pEmu(N, P), pL1pEmu2mn2, tmp1(N, P), tmp2(N, P), Vmu_mat(N, P), COVmu_mat(N - 1, P), EKodi(P, P, fill::zeros), tmp3i, VKodi(P, P, fill::zeros), tmp3(N, Pa);
  vec h_mu, zeta_mu_d, zeta_mu_od(N - 1);
  uword i, j, k;
  List results_CHMM;
  
  
  
  for (i = 0; i < N; i ++) {
    EKodi(idl) = EKod_mat.row(i);
    EKodi(idu) = EKod_mat.row(i);
    VKodi(idl) = VKod_mat.row(i);
    VKodi(idu) = VKod_mat.row(i);
    tmp3i = (EKodi % repmat(EKdinv_mat.row(i), P, 1)) * EKodi;
    tmp1.row(i) = tmp3i.diag().t();
    tmp3i += 2 * EKodi;
    pL1pEmu.row(i) = data.row(i) * tmp3i;
    tmp2.row(i) = EKdinv_mat.row(i) * VKodi;
    tmp3.row(i) = tmp3i(ida).t();
  }
  
  pL1pEmu2mn2 = EKd_mat + tmp2;
  pL1pEmu += pL1pEmu2mn2 % data + tmp1 % Emu_mat;
  pL1pEmu2mn2 += tmp1;
  
  for (k= 0; k < P; k ++) {
    j = idj(k);
    
    
    pL1pEmu.col(j) -= sum(Emu_mat % tmp3.cols(ida_mat.col(j)), 1);
    /*for (i = 0; i < N; i ++) {
      pL1pEmu(i, j) -= accu(Emu_mat.row(i) % tmp3.slice(j).row(i));
    }*/
    
    h_mu = pL1pEmu.col(j);
    zeta_mu_d = pL1pEmu2mn2.col(j);
    if (is_row_missing) {
      h_mu(row_missing).zeros();
      zeta_mu_d(row_missing).zeros();
    }
    zeta_mu_d.head(N - 1) += Elambda;
    zeta_mu_d.tail(N - 1) += Elambda;
    zeta_mu_od.fill(Elambda);
    
    results_CHMM = FwdBwdCHMM(h_mu, zeta_mu_d, zeta_mu_od, N, 1);
    Emu_mat.col(j) = as<vec>(results_CHMM["mu"]);
    Vmu_mat.col(j) = as<vec>(results_CHMM["S_d"]);
    COVmu_mat.col(j) = as<vec>(results_CHMM["S_od"]);
    
  }
  
  
  return List::create(Named("Emu_mat") = Emu_mat, Named("Vmu_mat") = Vmu_mat, 
                      Named("COVmu_mat") = COVmu_mat);
  
}
