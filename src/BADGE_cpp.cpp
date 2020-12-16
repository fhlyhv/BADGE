#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>
#include <progress_bar.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include "CrossValidation.hpp"
#include "CrossValidation_missing.hpp"
#include "CrossValidation_QUIC.hpp"
#include "FwdBwdCHMM_var.hpp"
#include "SumEKodMDataUpdate.hpp"
#include "FwdBwdCHMM.hpp"
#include "FwdBwdDHMM.hpp"
#include "LogDetTriDiag.hpp"
#include "MeanUpdate.hpp"
#include "MissingUpdate.hpp"

using namespace Rcpp;
using namespace arma;


// [[Rcpp::export]]

List BADGE_cpp(arma::mat data, int anneal_iters, double T, int max_iter, double tol_relative, double tol_s, 
               arma::umat is_na, bool normalize_data, bool is_nonzero_mean) {
  
  // declare variables
  auto t_start = std::chrono::high_resolution_clock::now();
  int P = data.n_cols, N = data.n_rows, Pe = P * (P - 1) / 2, Pa = Pe + P, i, j, N_missing, iter;
  uvec idr(Pe), idc(Pe), idd, idl, idu, ida, row_missing, v1_missing, v2_missing, v2_rnd, row_missing_rnd, row_obsv; 
  uvec id_missing_vec, idr_missing, idr_missing_rnd, idr_missing_rnd0, id_sub, id_w, id_rnd, idj, idr_vec(1), idc_vec(1), idj_vec(1), id_tmp, vP = regspace<uvec>(0, P - 1);
  umat id_missing, tmp0, idl_mat(P, P);
  double bandwidth_int = 3.0, bandwidth, Elambda, c_init = 1.0, thr, a, b, c0, d0, c1, d1, Egamma = 1e6, Ealpha_rnd, bandwidth_ub, bandwidth_diff;
  double ELBO0, ELBOh, ELBO_max, ELBO_tmp, Hs_tmp, HJ_tmp, sum_EJ2_diff_tmp, eta_thr = 1.0, eta, eta_max, Hkappa_tmp, Hkappa_max, sum_Ekappa2_diff_tmp;
  bool is_missing, is_row_missing, is_rnd_missing, is_pd;
  mat Vdata, COVdata, Vdata_rnd, COVdata_rnd, w_mat, data_mu, std_data, data_mu2, data_mu_rnd, data_mu_rnd2;
  mat zeta_mu_d, zeta_mu_od, h_mu, Emu_mat, Vmu_mat, COVmu_mat, Vmu_rnd;
  mat Es_mat(N, Pe), S, K, sum_pair_density(Pe, 4), Es_mat_old, log_B(N, 2, fill::zeros), sum_pair_density_tmp;
  vec Es_tmp, pL1pEs_true, cdf_s(1);
  mat22 log_A, log_A_rnd;
  rowvec2 log_s_init, log_s_rnd;
  mat h_J, zeta_J_d(N, Pe), zeta_J_od(N - 1, Pe), EJ_mat(N, Pe), VJ_mat, COVJ_mat, EJ2_mat;
  vec h_J_tmp, zeta_J_d_tmp, zeta_J_od_tmp, h_J_j, zeta_J_d_j, zeta_J_od_j(N - 1), EJ_tmp, VJ_tmp, EJ2_tmp;
  mat EKod_mat, EKod2_mat, VKod_mat, EKod_mat_old, sum_EKod_m_data, sum_EKod_m_data_rnd, EKodi, VKodi, Si;
  vec pL1pEKod_true, pL1pEKod, pL1pVKodmn2_true, pL1pVKodmn2, EKod_tmp, EKod2_tmp;
  mat h_kappa, zeta_kappa_d(N, P), zeta_kappa_od(N - 1, P), Ekappa, Vkappa, COVkappa;
  mat pL1pEkappa, pL1pEkappa2mn2, natgrad_h_kappa, natgrad_zeta_kappa_d, natgrad_zeta_kappa_od;
  mat EKd_mat(N, P), EKdinv_mat, EKd_mat_old, pL1pEKd_true, pL1pEKd, pL1pEKdinv_true, pL1pEKdinv;
  vec h_kappa_tmp, zeta_kappa_d_tmp, zeta_kappa_od_tmp, Ekappa_tmp, Vkappa_tmp, EKd_tmp, EKdinv_tmp;
  vec weights0, w, std_vec, lb, ub, bd_range, eta_max_vec(P, fill::zeros), id_w0, idi;
  rowvec Ealpha(Pe), Hs(Pe), HJ, Hkappa, sum_EJ2_diff, sum_Ekappa2_diff, ELBO0_vec, Egamma_rnd;
  List results_DHMM, results_CHMM, results_CHMM_var, results_logdet, results_MeanUpdate, results_MissingUpdate;
  double diff_s, diff_d_max, diff_od_max, diff_d_r, diff_od_r;
  mat diff_d, diff_od;
  field<mat> results_CV;
  
  
  // preprocessing
  
  j = 0;
  for (i = 0; i < P - 1; i ++) {
    idr(span(j, j + P - i - 2)) = regspace<uvec>(i + 1, P - 1);
    idc(span(j, j + P - i - 2)).fill(i);
    j = j + P - i - 1;
  }
  idd = regspace<uvec>(0, P - 1) * P + regspace<uvec>(0, P - 1);
  idl = idc * P + idr;
  idu = idr * P + idc;
  ida = join_cols(idl, idd);
  
  idl_mat(idl) = regspace<uvec>(0, Pe - 1);
  idl_mat(idu) = idl_mat(idl);
  idl_mat(idd) = regspace<uvec>(Pe, Pa - 1);
  
  
  is_missing = accu(is_na) > 0;
  if (is_missing) {
    row_missing = find(sum(is_na, 1) == P);
    if (row_missing.n_elem > 0) {
      is_row_missing = true;
      v1_missing.set_size(N);
      v1_missing.ones();
      v1_missing(row_missing).zeros();
      row_obsv = find(v1_missing == 1);
      data.rows(row_missing).zeros();
      tmp0 = is_na;
      tmp0.rows(row_missing).zeros();
      v1_missing = 1 - v1_missing;
    } else {
      is_row_missing = false;
      row_obsv = regspace<uvec>(0, N - 1);
    }
    id_missing_vec = find(tmp0 == 1);
    if (id_missing_vec.n_elem > 0) {
      data(id_missing_vec).zeros();
      is_rnd_missing = true;
      id_missing = ind2sub(size(N, P), id_missing_vec).t();
      idr_missing = unique(id_missing.col(0));
      N_missing = idr_missing.n_elem;
      Vdata = conv_to<mat>::from(tmp0.rows(idr_missing));
      COVdata.set_size(N_missing, Pe);
      COVdata.zeros();
      v2_missing.set_size(N);
      v2_missing.zeros();
      v2_missing(idr_missing) = regspace<uvec>(0, N_missing - 1);
    } else is_rnd_missing = false;
    tmp0.clear();
  } else {
    is_na.clear();
    is_row_missing = false;
    is_rnd_missing = false;
    row_obsv = regspace<uvec>(0, N - 1);
  }
  
  if (P >= N) {
    VKodi.set_size(P, P);
    VKodi.zeros();
  }
  if ((is_nonzero_mean && P >= N) || is_rnd_missing) {
    EKodi.set_size(P, P);
    EKodi.zeros();
  }
  if (is_rnd_missing) Si.set_size(P, P);
  
  if (normalize_data) {
    if (is_missing) bandwidth = CrossValidation_missing(bandwidth_int, data, N, N, P, is_nonzero_mean, 1 - is_na);
    else bandwidth = CrossValidation(bandwidth_int, data, N, N, P, is_nonzero_mean);
  } 
  
  // Estimate the mean of data if is_nonzero_mean = true
  if (is_nonzero_mean) {
    
    if (!normalize_data) bandwidth = N / 2.0;
    
    id_w0 = regspace(- ceil(bandwidth) + 1.0, ceil(bandwidth) - 1.0);
    weights0 = 1.0 - square(id_w0 / bandwidth);
    weights0 /= accu(weights0);
    
    Emu_mat.set_size(N, P);
    for (i = 0; i < N; i ++) {
      idi = id_w0 + i;
      id_sub = find((idi >=0) % (idi < N));
      if (id_sub.n_elem < idi.n_elem) {
        w = weights0(id_sub);
        w /= accu(w);
        id_w = conv_to<uvec>::from(idi(id_sub));
      } else {
        w = weights0;
        id_w = conv_to<uvec>::from(idi);
      }
      if (is_missing) Emu_mat.row(i) = sum(data.rows(id_w) % repmat(w, 1, P)) / sum(repmat(w, 1, P) % (1.0 - is_na.rows(id_w)));
      else Emu_mat.row(i) = sum(data.rows(id_w) % repmat(w, 1, P));
    }
    data_mu = data - Emu_mat;
    if (is_row_missing) data_mu.rows(row_missing).zeros();
    if (is_rnd_missing) data_mu(id_missing_vec).zeros();
    
    Elambda = (N - 1.0) * P / accu(square(Emu_mat.head_rows(N - 1) - Emu_mat.tail_rows(N - 1)));
    zeta_mu_d.set_size(N, P);
    zeta_mu_d.fill(c_init);
    zeta_mu_d.row(0) += Elambda;
    zeta_mu_d.row(N - 1) += Elambda;
    zeta_mu_d.rows(span(1, N - 2)) += 2 * Elambda;
    zeta_mu_od.set_size(N - 1, P);
    zeta_mu_od.fill(Elambda);
    h_mu = Elambda * join_cols(Emu_mat.row(0) - Emu_mat.row(1), 2 * Emu_mat.rows(span(1, N - 2)) - Emu_mat.rows(span(0, N - 3)) - Emu_mat.rows(span(2, N - 1)),
                               Emu_mat.row(N - 1) - Emu_mat.row(N - 2)) + c_init * Emu_mat;
    results_CHMM_var = FwdBwdCHMM_var(zeta_mu_d, zeta_mu_od, N, P);
    Vmu_mat = as<mat>(results_CHMM_var["S_d"]);
    COVmu_mat = as<mat>(results_CHMM_var["S_od"]);
    
    
  } else data_mu = data;
  
  
  // normalize data using estimated time-varying mean and variance
  if (normalize_data) {
    id_w0 = regspace(- ceil(bandwidth) + 1.0, ceil(bandwidth) - 1.0);
    weights0 = 1.0 - square(id_w0 / bandwidth);
    weights0 /= accu(weights0);
    std_data.set_size(N, P);
    for (i = 0; i < N; i ++) {
      idi = id_w0 + i;
      id_sub = find((idi >=0) % (idi < N));
      if (id_sub.n_elem < idi.n_elem) {
        w = weights0(id_sub);
        w /= accu(w);
        id_w = conv_to<uvec>::from(idi(id_sub));
      } else {
        w = weights0;
        id_w = conv_to<uvec>::from(idi);
      }
      
      if (is_missing) std_data.row(i) = sqrt(sum(square(data_mu.rows(id_w)) % repmat(w, 1, P)) / sum(repmat(w, 1, P) % (1.0 - is_na.rows(id_w))));
      else std_data.row(i) = sqrt(sum(square(data.rows(id_w)) % repmat(w, 1, P)));
    }
    data /= std_data;
    Emu_mat /= std_data;
    data_mu /= std_data;
    std_data.clear();
  } 
  data_mu2 = square(data_mu);
  
  Rprintf("initialize all parameters ...\n");
  // initialize other variables
  results_CV = CrossValidation_QUIC(data_mu, idr, idc, idl, idu, N, P, Pe, false, 
                                    is_missing, is_na, is_rnd_missing, id_missing_vec, idr_missing, N_missing, 
                                    is_row_missing, row_missing);
  Es_mat = results_CV(0);
  EJ_mat = results_CV(1);
  EKd_mat = results_CV(2);
  results_CV.clear();
  
  //initialize s
  
  
  a = 1.0 + Pe - accu(Es_mat.row(0));
  b = 1.0 + accu(Es_mat.row(0));
  log_s_init(0) = boost::math::digamma(a) - boost::math::digamma(a + b);
  log_s_init(1) = boost::math::digamma(b) - boost::math::digamma(a + b);
  
  sum_pair_density.col(0) = conv_to<vec>::from(sum((Es_mat.head_rows(N - 1) < 0.5) % (Es_mat.tail_rows(N - 1) < 0.5)));
  sum_pair_density.col(1) = conv_to<vec>::from(sum((Es_mat.head_rows(N - 1) < 0.5) % (Es_mat.tail_rows(N - 1) > 0.5)));
  sum_pair_density.col(2) = conv_to<vec>::from(sum((Es_mat.head_rows(N - 1) > 0.5) % (Es_mat.tail_rows(N - 1) < 0.5)));
  sum_pair_density.col(3) = conv_to<vec>::from(sum((Es_mat.head_rows(N - 1) > 0.5) % (Es_mat.tail_rows(N - 1) > 0.5)));
  
  
  c0 = 1.0 + accu(sum_pair_density.col(0));
  d0 = 1.0 + accu(sum_pair_density.col(1));
  c1 = 1.0 + accu(sum_pair_density.col(3));
  d1 = 1.0 + accu(sum_pair_density.col(2));
  
  log_A(0, 0) = boost::math::digamma(c0) - boost::math::digamma(c0 + d0);
  log_A(0, 1) = boost::math::digamma(d0) - boost::math::digamma(c0 + d0);
  log_A(1, 1) = boost::math::digamma(c1) - boost::math::digamma(c1 + d1);
  log_A(1, 0) = boost::math::digamma(d1) - boost::math::digamma(c1 + d1);
  
  for (j = 0; j < Pe; j ++) {
    results_DHMM = FwdBwdDHMM(log_s_init, log_A, join_rows(1 - Es_mat.col(j), Es_mat.col(j)), N, 2);
    Es_mat.col(j) = as<vec>(results_DHMM["marg_density"]);
    sum_pair_density_tmp = as<mat>(results_DHMM["sum_pair_density"]);
    sum_pair_density.row(j) = join_rows(sum_pair_density_tmp.row(0), sum_pair_density_tmp.row(1));
    Hs(j) = as<double>(results_DHMM["entropy"]);
  }
  
  // initialize J
  Ealpha = (N - 1.0) / (sum(square(EJ_mat.head_rows(N - 1) - EJ_mat.tail_rows(N - 1))) + 1e-10); //.fill(1e6);
  //Ealpha.print();
  zeta_J_d.fill(c_init);
  zeta_J_d.row(0) += Ealpha;
  zeta_J_d.row(N - 1) += Ealpha;
  zeta_J_d.rows(1, N - 2).each_row() += 2 * Ealpha;
  zeta_J_od.each_row() = Ealpha;
  h_J = join_cols(EJ_mat.row(0) - EJ_mat.row(1), 2 * EJ_mat.rows(span(1, N - 2)) - EJ_mat.rows(span(0, N - 3)) - EJ_mat.rows(span(2, N - 1)),
                  EJ_mat.row(N - 1) - EJ_mat.row(N - 2));
  h_J.each_row() %= Ealpha;
  h_J += c_init * EJ_mat;
  results_CHMM_var = FwdBwdCHMM_var(zeta_J_d, zeta_J_od, N, Pe);
  VJ_mat = as<mat>(results_CHMM_var["S_d"]);
  COVJ_mat = as<mat>(results_CHMM_var["S_od"]);
  results_logdet = LogDetTriDiag(zeta_J_d, zeta_J_od, N, Pe);
  HJ = -as<rowvec>(results_logdet["logdetK"]) / 2;
  EJ2_mat = square(EJ_mat) + VJ_mat;
  sum_EJ2_diff = sum(square(EJ_mat.head_rows(N - 1) - EJ_mat.tail_rows(N - 1))) + as<rowvec>(results_CHMM_var["sum_V_diff"]);
  
  EKod_mat = EJ_mat % Es_mat;
  EKod2_mat = EJ2_mat % Es_mat;
  // if (is_rnd_missing) VKod_mat = EKod2_mat - square(EKod_mat);
  sum_EKod_m_data = SumEKodMDataUpdate(data_mu, EKod_mat, idl, idu, idl_mat, N, P);
  
  // initialize kappa
  Ekappa = log(EKd_mat);
  Egamma = (N - 1.0) * P / accu(square(Ekappa.head_rows(N - 1) - Ekappa.tail_rows(N - 1))); //
  zeta_kappa_d.fill(c_init);
  zeta_kappa_d.row(0) += Egamma;
  zeta_kappa_d.row(N - 1) += Egamma;
  zeta_kappa_d.rows(1, N - 2) += 2 * Egamma;
  zeta_kappa_od.fill(Egamma);
  h_kappa = Egamma * join_cols(Ekappa.row(0) - Ekappa.row(1), 2 * Ekappa.rows(span(1, N - 2)) - Ekappa.rows(span(0, N - 3)) - Ekappa.rows(span(2, N - 1)),
                               Ekappa.row(N - 1) - Ekappa.row(N - 2)) + c_init * Ekappa;
  results_CHMM_var = FwdBwdCHMM_var(zeta_kappa_d, zeta_kappa_od, N, P);
  Vkappa = as<mat>(results_CHMM_var["S_d"]);
  COVkappa = as<mat>(results_CHMM_var["S_od"]);
  sum_Ekappa2_diff = sum(square(Ekappa.head_rows(N - 1) - Ekappa.tail_rows(N - 1))) + as<rowvec>(results_CHMM_var["sum_V_diff"]);
  results_logdet = LogDetTriDiag(zeta_kappa_d, zeta_kappa_od, N, P);
  Hkappa = - as<rowvec>(results_logdet["logdetK"]) / 2;
  
  EKdinv_mat = 1 / EKd_mat;
  Ekappa = Vkappa / 2 - log(EKdinv_mat);
  EKd_mat = exp(Ekappa + Vkappa / 2);
  // EKdinv_mat = exp(- Ekappa + Vkappa / 2);
  
  pL1pEKd_true = - data_mu2 / 2;
  
  EKd_mat_old = EKd_mat;
  Es_mat_old = Es_mat;
  EKod_mat_old = EKod_mat;
  
  // initialize parameters for simulated annealing
  bandwidth_ub = (N / 2.0 - 1.0) * (1.0 - 1.0 / T) + 1.0;
  bandwidth = bandwidth_ub;
  
  lb = regspace(0, N - 1) - ceil(bandwidth) + 0.5;
  lb(find(lb < -0.5)).fill(-0.5);
  ub = regspace(0, N - 1) + ceil(bandwidth) - 0.5;
  ub(find(ub > N - 0.5)).fill(N - 0.5);
  bd_range = ub - lb;
  bandwidth_diff = (bandwidth_ub - 1.0) / anneal_iters * 10.0;
  
  Rprintf("Start simulated annealing with %i iterations...\n", anneal_iters);
  Progress pb(anneal_iters / 10, true);
  
  // Natural Gradient Variational Inference
  for (iter = 1; iter <= max_iter; iter ++) {
    
    if (T > 1) {  //iter > 1 && 
      id_rnd = conv_to<uvec>::from(round(lb + bd_range % randu(N)));
      if (is_row_missing) {
        row_missing_rnd = find(v1_missing(id_rnd) == 1);
        while (row_missing_rnd.n_elem != 0) {
          id_rnd(row_missing_rnd) = conv_to<uvec>::from(round(lb(row_missing_rnd) + bd_range(row_missing_rnd) % randu(row_missing_rnd.n_elem)));
          row_missing_rnd = row_missing_rnd(find(v1_missing(id_rnd(row_missing_rnd)) == 1));
        }
      }
      
      data_mu_rnd = data_mu.rows(id_rnd);
      data_mu_rnd2 = square(data_mu_rnd);
      if (is_nonzero_mean) Vmu_rnd = Vmu_mat.rows(id_rnd);
      if (is_rnd_missing) {
        
        v2_rnd = v2_missing(id_rnd);
        idr_missing_rnd = find(v2_rnd > 0);
        idr_missing_rnd0 = v2_rnd(idr_missing_rnd);
        
        Vdata_rnd = Vdata.rows(idr_missing_rnd0);
        COVdata_rnd = COVdata.rows(idr_missing_rnd0);
      }
      sum_EKod_m_data_rnd = SumEKodMDataUpdate(data_mu_rnd, EKod_mat, idl, idu, idl_mat, N, P);
      
      
      idj = randperm(Pe);
      
      for (i = 0; i < Pe; i ++){
        j = idj(i);
        
        // compute gradient wrt off-diagonal elements in K
        
        sum_EKod_m_data.col(idr(j)) -= EKod_mat.col(j) % data_mu.col(idc(j));
        sum_EKod_m_data.col(idc(j)) -= EKod_mat.col(j) % data_mu.col(idr(j));
        
        pL1pEKod_true = - EKdinv_mat.col(idr(j)) % data_mu.col(idc(j)) % sum_EKod_m_data.col(idr(j)) -
          EKdinv_mat.col(idc(j)) % data_mu.col(idr(j)) % sum_EKod_m_data.col(idc(j)) -
          2 * data_mu.col(idc(j)) % data_mu.col(idr(j));
        pL1pVKodmn2_true = EKdinv_mat.col(idr(j)) % data_mu2.col(idc(j)) +
          EKdinv_mat.col(idc(j)) % data_mu2.col(idr(j));
        
        sum_EKod_m_data_rnd.col(idr(j)) -= EKod_mat.col(j) % data_mu_rnd.col(idc(j));
        sum_EKod_m_data_rnd.col(idc(j)) -= EKod_mat.col(j) % data_mu_rnd.col(idr(j));
        
        pL1pEKod = - EKdinv_mat.col(idr(j)) % data_mu_rnd.col(idc(j)) % sum_EKod_m_data_rnd.col(idr(j)) -
          EKdinv_mat.col(idc(j)) % data_mu_rnd.col(idr(j)) % sum_EKod_m_data_rnd.col(idc(j)) -
          2 * data_mu_rnd.col(idc(j)) % data_mu_rnd.col(idr(j));
        
        pL1pVKodmn2 = EKdinv_mat.col(idr(j)) % data_mu_rnd2.col(idc(j)) +
          EKdinv_mat.col(idc(j)) % data_mu_rnd2.col(idr(j));
        
        if (is_rnd_missing) {
          if (idr(j) - idc(j) > 1) id_tmp = join_cols(vP.head(idc(j)), vP(span(idc(j) + 1, idr(j) - 1)), vP.tail(P - 1 - idr(j)));
          else id_tmp = join_cols(vP.head(idc(j)), vP.tail(P - 1 - idr(j)));
          
          idr_vec.fill(idr(j));
          idc_vec.fill(idc(j));
          
          pL1pEKod_true(idr_missing) += - EKdinv_mat(idr_missing, idr_vec) % 
            sum(EKod_mat(idr_missing, idl_mat(idr_vec, id_tmp)) % COVdata.cols(idl_mat(idc_vec, id_tmp)), 1) - EKdinv_mat(idr_missing, idc_vec) % 
            sum(EKod_mat(idr_missing, idl_mat(idc_vec, id_tmp)) % COVdata.cols(idl_mat(idr_vec, id_tmp)), 1) - 2 * COVdata.col(j);
          pL1pVKodmn2_true(idr_missing) += EKdinv_mat(idr_missing, idr_vec) % Vdata.col(idc(j)) +
            EKdinv_mat(idr_missing, idc_vec) % Vdata.col(idr(j));
          
          
          idj_vec.fill(j);
          
          pL1pEKod(idr_missing_rnd) += - EKdinv_mat(idr_missing_rnd, idr_vec) % 
            sum(EKod_mat(idr_missing_rnd, idl_mat(idr_vec, id_tmp)) % COVdata_rnd.cols(idl_mat(idc_vec, id_tmp)), 1) - EKdinv_mat(idr_missing_rnd, idc_vec) % 
            sum(EKod_mat(idr_missing_rnd, idl_mat(idc_vec, id_tmp)) % COVdata_rnd.cols(idl_mat(idr_vec, id_tmp)), 1) - 2 * COVdata_rnd.col(j);
          pL1pVKodmn2(idr_missing_rnd) += EKdinv_mat(idr_missing_rnd, idr_vec) % Vdata_rnd.col(idc(j)) +
            EKdinv_mat(idr_missing_rnd, idc_vec) % Vdata_rnd.col(idr(j));
        } 
        
        if (is_nonzero_mean) {
          pL1pVKodmn2_true += EKdinv_mat.col(idr(j)) % Vmu_mat.col(idc(j)) +
            EKdinv_mat.col(idc(j)) % Vmu_mat.col(idr(j));
          pL1pVKodmn2 += EKdinv_mat.col(idr(j)) % Vmu_rnd.col(idc(j)) +
            EKdinv_mat.col(idc(j)) % Vmu_rnd.col(idr(j));
        }
        
        if (is_row_missing) {
          pL1pEKod_true(row_missing).zeros();
          pL1pVKodmn2_true(row_missing).zeros();
        }
        
        // update s
        pL1pEs_true = pL1pEKod_true % EJ_mat.col(j) - pL1pVKodmn2_true % EJ2_mat.col(j) / 2;
        ELBO0 = log_s_init(0) * (1 - Es_mat(0, j)) + log_s_init(1) * Es_mat(0, j) + accu(pL1pEs_true % Es_mat.col(j)) +
          accu(sum_pair_density.row(j) % join_rows(log_A.row(0), log_A.row(1))) + Hs(j);
        
        log_s_rnd(0) = randg(distr_param(a, 1.0));
        log_s_rnd(1) = randg(distr_param(b, 1.0));
        log_s_rnd = log(log_s_rnd) - log(accu(log_s_rnd));
        log_s_rnd *= (1.0 - 1.0 / T);
        log_s_rnd += log_s_init / T;
        
        log_A_rnd(0, 0) = randg(distr_param(c0, 1.0));
        log_A_rnd(0, 1) = randg(distr_param(d0, 1.0));
        log_A_rnd(1, 0) = randg(distr_param(d1, 1.0));
        log_A_rnd(1, 1) = randg(distr_param(c1, 1.0));
        log_A_rnd = mat(log(log_A_rnd)).each_col() -  log(sum(log_A_rnd, 1)); //- repmat(log(sum(log_A_rnd, 1)), 1, 2);
        log_A_rnd *= (1.0 - 1.0 / T);
        log_A_rnd += log_A / T;
        
        log_B.col(1) = pL1pEKod % EJ_mat.col(j) - pL1pVKodmn2 % EJ2_mat.col(j) / 2;
        
        results_DHMM = FwdBwdDHMM(log_s_rnd, log_A_rnd, log_B, N, 2);   //log_A
        Es_tmp = as<vec>(results_DHMM["marg_density"]);
        sum_pair_density_tmp = as<mat>(results_DHMM["sum_pair_density"]);
        Hs_tmp = as<double>(results_DHMM["entropy"]);
        ELBOh = log_s_init(0) * (1 - Es_tmp(0)) + log_s_init(1) * Es_tmp(0) + 
          accu(pL1pEs_true % Es_tmp) + accu(sum_pair_density_tmp % log_A) + Hs_tmp;
        
        if (randu() < exp((ELBOh - ELBO0) / (1 - 1 / T))) { //(ELBOh > ELBO0) { // 
          Es_mat.col(j) = Es_tmp;
          sum_pair_density.row(j) = join_rows(sum_pair_density_tmp.row(0), sum_pair_density_tmp.row(1));
          Hs(j) = Hs_tmp;
          EKod_mat.col(j) = Es_mat.col(j) % EJ_mat.col(j);
          EKod2_mat.col(j) = Es_mat.col(j) % EJ2_mat.col(j);
        }
        
        // update J
        
        ELBO0 = accu(pL1pEKod_true % EKod_mat.col(j)) - accu(pL1pVKodmn2_true % EKod2_mat.col(j)) / 2 -
          Ealpha(j) / 2 * sum_EJ2_diff(j) + HJ(j);
        
        
        zeta_J_d_j = pL1pVKodmn2_true % Es_mat.col(j);
        zeta_J_d_j(0) += Ealpha(j);
        zeta_J_d_j(N - 1) += Ealpha(j);
        zeta_J_d_j(span(1, N - 2)) += 2 * Ealpha(j);
        zeta_J_od_j.fill(Ealpha(j));
        is_pd = FwdBwdCHMM_var(zeta_J_d_j, zeta_J_od_j, N, 1)["is_pd"];
        
        if (is_pd) eta_max = eta_thr;
        else {
          h_J_j = pL1pEKod_true % Es_mat.col(j);
          ELBO_max = ELBO0;
          eta_max = 0;
          eta = eta_thr / 2;
          while (eta > 1e-2) {
            h_J_tmp = (1 - eta) * h_J.col(j) + eta * h_J_j;
            zeta_J_d_tmp = (1 - eta) * zeta_J_d.col(j) + eta * zeta_J_d_j;
            zeta_J_od_tmp = (1 - eta) * zeta_J_od.col(j) + eta * zeta_J_od_j;
            results_CHMM = FwdBwdCHMM(h_J_tmp, zeta_J_d_tmp, zeta_J_od_tmp, N, 1);
            is_pd = results_CHMM["is_pd"];
            if (is_pd) {
              results_logdet = LogDetTriDiag(zeta_J_d_tmp, zeta_J_od_tmp, N, 1);
              HJ_tmp = - as<double>(results_logdet["logdetK"]) / 2;
              
              EJ_tmp = as<vec>(results_CHMM["mu"]);
              VJ_tmp = as<vec>(results_CHMM["S_d"]);
              sum_EJ2_diff_tmp = as<double>(results_CHMM["sum_E2_diff"]);
              
              EJ2_tmp = square(EJ_tmp) + VJ_tmp;
              EKod_tmp = Es_mat.col(j) % EJ_tmp;
              EKod2_tmp = Es_mat.col(j) % EJ2_tmp;
              
              ELBOh = accu(pL1pEKod_true % EKod_tmp) - accu(pL1pVKodmn2_true % EKod2_tmp) / 2 -
                Ealpha(j) / 2 * sum_EJ2_diff_tmp + HJ_tmp;
              if (ELBO_max > ELBO0 && ELBOh <= ELBO_max) break;
              else {
                if (ELBOh > ELBO_max) {
                  eta_max = eta;
                  ELBO_max = ELBOh;
                }
                eta /= 2;
              }
            } else eta /= 2;
          }
        }
        
        if (eta_max > 0) {
          Ealpha_rnd = Ealpha(j) / T + randg(distr_param((N - 1.0) / 2.0, 2.0 / sum_EJ2_diff(j))) * (1.0 - 1.0 / T);
          
          zeta_J_d_j = pL1pVKodmn2 % Es_mat.col(j);
          zeta_J_d_j(0) += Ealpha_rnd;
          zeta_J_d_j(N - 1) += Ealpha_rnd;
          zeta_J_d_j(span(1, N - 2)) += 2 * Ealpha_rnd;
          zeta_J_od_j.fill(Ealpha_rnd);
          h_J_tmp = (1 - eta_max) * h_J.col(j) + eta_max * pL1pEKod % Es_mat.col(j);
          zeta_J_d_tmp = (1 - eta_max) * zeta_J_d.col(j) + eta_max * zeta_J_d_j;
          zeta_J_od_tmp = (1 - eta_max) * zeta_J_od.col(j) + eta_max * zeta_J_od_j;
          
          results_CHMM = FwdBwdCHMM(h_J_tmp, zeta_J_d_tmp, zeta_J_od_tmp, N, 1);
          is_pd = results_CHMM["is_pd"];
          if (is_pd) {
            results_logdet = LogDetTriDiag(zeta_J_d_tmp, zeta_J_od_tmp, N, 1);
            HJ_tmp = - as<double>(results_logdet["logdetK"]) / 2;
            
            EJ_tmp = as<vec>(results_CHMM["mu"]);
            VJ_tmp = as<vec>(results_CHMM["S_d"]);
            sum_EJ2_diff_tmp = as<double>(results_CHMM["sum_E2_diff"]);
            
            EJ2_tmp = square(EJ_tmp) + VJ_tmp;
            EKod_tmp = Es_mat.col(j) % EJ_tmp;
            EKod2_tmp = Es_mat.col(j) % EJ2_tmp;
            
            ELBOh = accu(pL1pEKod_true % EKod_tmp) - accu(pL1pVKodmn2_true % EKod2_tmp) / 2 -
              Ealpha(j) / 2 * sum_EJ2_diff_tmp + HJ_tmp;
            if (randu() < exp((ELBOh - ELBO0) / (1 - 1 / T))) { //(ELBOh > ELBO0) {//
              h_J.col(j) = h_J_tmp;
              zeta_J_d.col(j) = zeta_J_d_tmp;
              zeta_J_od.col(j) = zeta_J_od_tmp;
              EJ_mat.col(j) = EJ_tmp;
              EJ2_mat.col(j) = EJ2_tmp;
              HJ(j) = HJ_tmp;
              sum_EJ2_diff(j) = sum_EJ2_diff_tmp;
              Ealpha(j) = (N - 1.0) / sum_EJ2_diff_tmp;
              
              /*if (sum_EJ2_diff_tmp > 0) {
                Ealpha(j) = (N - 1.0) / sum_EJ2_diff_tmp;
                if (Ealpha(j) > 1e300) {
                  Ealpha(j) = 1e300;
                  sum_EJ2_diff(j) = (N - 1.0) / 1e300;
                }
                else sum_EJ2_diff(j) = sum_EJ2_diff_tmp;
              }*/
              
              EKod_mat.col(j) = EKod_tmp;
              EKod2_mat.col(j) = EKod2_tmp;
              
            }
          }
        }
        
        sum_EKod_m_data.col(idr(j)) += EKod_mat.col(j) % data_mu.col(idc(j));
        sum_EKod_m_data.col(idc(j)) += EKod_mat.col(j) % data_mu.col(idr(j));
        sum_EKod_m_data_rnd.col(idr(j)) += EKod_mat.col(j) % data_mu_rnd.col(idc(j));
        sum_EKod_m_data_rnd.col(idc(j)) += EKod_mat.col(j) % data_mu_rnd.col(idr(j));
      }
      
      VKod_mat = EKod2_mat - square(EKod_mat);
      
      // compute gradient wrt diagonal elements in K
      
      pL1pEKdinv_true = -square(sum_EKod_m_data) / 2;
      
      pL1pEKd = - data_mu_rnd2 / 2;
      pL1pEKdinv = -square(sum_EKod_m_data_rnd) / 2;
      
      if (is_nonzero_mean) {
        pL1pEKd_true -= Vmu_mat / 2;
        pL1pEKd -= Vmu_rnd / 2;
      }
      if (is_rnd_missing) {
        pL1pEKd_true.rows(idr_missing) -= Vdata / 2;
        pL1pEKd.rows(idr_missing_rnd) -= Vdata_rnd / 2;
      }
      
      if (P < N) {
        for (j = 0; j < P; j ++) {
          id_tmp = join_cols(vP.head(j), vP.tail(P - 1 - j));
          idj_vec.fill(j);
          pL1pEKdinv.col(j) += sum(pL1pEKd.cols(id_tmp) % VKod_mat.cols(idl_mat(idj_vec, id_tmp)), 1);
          pL1pEKdinv_true.col(j) += sum(pL1pEKd_true.cols(id_tmp) % VKod_mat.cols(idl_mat(idj_vec, id_tmp)), 1);
          if (is_nonzero_mean) {
            pL1pEKdinv.col(j) -= sum(Vmu_rnd.cols(id_tmp) % square(EKod_mat.cols(idl_mat(idj_vec, id_tmp))), 1) / 2;
            pL1pEKdinv_true.col(j) -= sum(Vmu_mat.cols(id_tmp) % square(EKod_mat.cols(idl_mat(idj_vec, id_tmp))), 1) / 2;
          }
        }
      } else {
        for (i = 0; i < N; i ++) {
          VKodi(idl) = VKod_mat.row(i);
          VKodi(idu) = VKod_mat.row(i);
          
          pL1pEKdinv.row(i) += pL1pEKd.row(i) * VKodi;   //sum(EKodi * nSi % EKodi, 1).t()
          pL1pEKdinv_true.row(i) += pL1pEKd_true.row(i) * VKodi; 
          
          if (is_nonzero_mean) {
            EKodi(idl) = EKod_mat.row(i);
            EKodi(idu) = EKod_mat.row(i);
            pL1pEKdinv.row(i) -= Vmu_rnd.row(i) * square(EKodi) / 2; 
            pL1pEKdinv_true.row(i) -= Vmu_mat.row(i) * square(EKodi) / 2; 
          }
        }
      }
      
      if (is_rnd_missing) {
        for (j = 0; j < idr_missing.n_elem; j ++) {
          i = idr_missing(j);
          EKodi(idl) = EKod_mat.row(i);
          EKodi(idu) = EKod_mat.row(i);
          Si(idl) = COVdata.row(j);
          Si(idu) = COVdata.row(j);
          Si.diag() = Vdata.row(j).t();
          pL1pEKdinv_true.row(i) -= sum(EKodi * Si % EKodi, 1).t();
        }
        
        
        
        for (j = 0; j < idr_missing_rnd.n_elem; j ++) {
          i = idr_missing_rnd(j);
          EKodi(idl) = EKod_mat.row(i);
          EKodi(idu) = EKod_mat.row(i);
          Si(idl) = COVdata_rnd.row(j);
          Si(idu) = COVdata_rnd.row(j);
          Si.diag() = Vdata_rnd.row(j).t();
          pL1pEKdinv.row(i) -= sum(EKodi * Si % EKodi, 1).t();
        }
        
      }
      
      if (is_row_missing) {
        pL1pEKd_true.rows(row_missing).zeros();
        pL1pEKdinv_true.rows(row_missing).zeros();
        pL1pEkappa = pL1pEKd_true % EKd_mat % (1 - Ekappa) -
          pL1pEKdinv_true % EKdinv_mat % (1 + Ekappa);
        pL1pEkappa.rows(row_obsv) += 0.5;
        ELBO0_vec = 0.5 * sum(Ekappa.rows(row_obsv));
      } else {
        pL1pEkappa = 0.5 + pL1pEKd_true % EKd_mat % (1 - Ekappa) -
          pL1pEKdinv_true % EKdinv_mat % (1 + Ekappa);
        ELBO0_vec = 0.5 * sum(Ekappa); 
      }
      pL1pEkappa2mn2 = - pL1pEKd_true % EKd_mat - pL1pEKdinv_true % EKdinv_mat;
      ELBO0_vec += - sum(pL1pEkappa2mn2) - Egamma / 2 * sum_Ekappa2_diff + Hkappa;
      
      // update kappa
      natgrad_h_kappa = pL1pEkappa - h_kappa;
      natgrad_zeta_kappa_d = pL1pEkappa2mn2 - zeta_kappa_d;
      natgrad_zeta_kappa_d.row(0) += Egamma;
      natgrad_zeta_kappa_d.row(N - 1) += Egamma;
      natgrad_zeta_kappa_d.rows(1, N - 2) += 2 * Egamma;
      natgrad_zeta_kappa_od = Egamma - zeta_kappa_od;
      
      
      
      for (j = 0; j < P; j ++) {
        ELBO_max = ELBO0_vec(j);
        eta = eta_thr;
        while (eta > 1e-10) {
          h_kappa_tmp = h_kappa.col(j) + eta * natgrad_h_kappa.col(j);
          zeta_kappa_d_tmp = zeta_kappa_d.col(j) + eta * natgrad_zeta_kappa_d.col(j);
          zeta_kappa_od_tmp = zeta_kappa_od.col(j) + eta * natgrad_zeta_kappa_od.col(j);
          results_logdet = LogDetTriDiag(zeta_kappa_d_tmp, zeta_kappa_od_tmp, N, 1);
          if (results_logdet["is_pd"]) {
            results_CHMM = FwdBwdCHMM(h_kappa_tmp, zeta_kappa_d_tmp, 
                                      zeta_kappa_od_tmp, N, 1);
            Ekappa_tmp = as<vec>(results_CHMM["mu"]);
            Vkappa_tmp = as<vec>(results_CHMM["S_d"]);
            sum_Ekappa2_diff_tmp = as<double>(results_CHMM["sum_E2_diff"]);
            EKd_tmp = exp(Ekappa_tmp + Vkappa_tmp / 2);
            EKdinv_tmp = exp(- Ekappa_tmp + Vkappa_tmp / 2);
            if (is_row_missing) ELBO_tmp = 0.5 * accu(Ekappa_tmp(row_obsv));
            else ELBO_tmp = 0.5 * accu(Ekappa_tmp);
            ELBO_tmp += accu(pL1pEKd_true.col(j) % EKd_tmp) +
              accu(pL1pEKdinv_true.col(j) % EKdinv_tmp) - Egamma / 2 *
              sum_Ekappa2_diff_tmp - as<double>(results_logdet["logdetK"]) / 2;
            if (ELBO_max > ELBO0_vec(j) && ELBO_tmp <= ELBO_max) {
              break;
            } else {
              if (ELBO_tmp > ELBO_max) {
                eta_max_vec(j) = eta;
                ELBO_max = ELBO_tmp;
              }
              eta /= 2;
            }
          }
          else
            eta /= 2;
        }
      }
      
      pL1pEkappa = 0.5 + pL1pEKd % EKd_mat % (1 - Ekappa) - 
        pL1pEKdinv % EKdinv_mat % (1 + Ekappa);
      pL1pEkappa2mn2 = - pL1pEKd % EKd_mat - pL1pEKdinv % EKdinv_mat;
      
      Egamma_rnd = Egamma / T + (1.0 - 1.0 / T) * randg<rowvec>(P, distr_param((N - 1.0) * P / 2.0, 2.0 / accu(sum_Ekappa2_diff)));
      
      natgrad_h_kappa = pL1pEkappa - h_kappa;
      natgrad_zeta_kappa_d = pL1pEkappa2mn2 - zeta_kappa_d;
      natgrad_zeta_kappa_d.row(0) += Egamma_rnd; //Egamma; //
      natgrad_zeta_kappa_d.row(N - 1) += Egamma_rnd; //Egamma;  //
      natgrad_zeta_kappa_d.rows(1, N - 2).each_row() += 2 * Egamma_rnd; //repmat(2 * Egamma_rnd, N - 2, 1); // 2 * Egamma; //
      natgrad_zeta_kappa_od = - zeta_kappa_od;
      natgrad_zeta_kappa_od.each_row() += Egamma_rnd;
      
      for (j = 0; j < P; j ++) {
        h_kappa_tmp = h_kappa.col(j) + eta_max_vec(j) * natgrad_h_kappa.col(j);
        zeta_kappa_d_tmp = zeta_kappa_d.col(j) + eta_max_vec(j) * natgrad_zeta_kappa_d.col(j);
        zeta_kappa_od_tmp = zeta_kappa_od.col(j) + eta_max_vec(j) * natgrad_zeta_kappa_od.col(j);
        results_logdet = LogDetTriDiag(zeta_kappa_d_tmp, zeta_kappa_od_tmp, N, 1);
        if (results_logdet["is_pd"]) {
          results_CHMM = FwdBwdCHMM(h_kappa_tmp, zeta_kappa_d_tmp,
                                    zeta_kappa_od_tmp, N, 1);
          Ekappa_tmp = as<vec>(results_CHMM["mu"]);
          Vkappa_tmp = as<vec>(results_CHMM["S_d"]);
          sum_Ekappa2_diff_tmp = as<double>(results_CHMM["sum_E2_diff"]);;
          EKd_tmp = exp(Ekappa_tmp + Vkappa_tmp / 2);
          EKdinv_tmp = exp(- Ekappa_tmp + Vkappa_tmp / 2);
          Hkappa_tmp = - as<vec>(results_logdet["logdetK"])(0) / 2;
          if (is_row_missing) ELBO_tmp = 0.5 * accu(Ekappa_tmp(row_obsv));
          else ELBO_tmp = 0.5 * accu(Ekappa_tmp);
          ELBO_tmp +=  accu(pL1pEKd_true.col(j) % EKd_tmp) + //(row_obsv_rnd)
            accu(pL1pEKdinv_true.col(j) % EKdinv_tmp) - Egamma / 2 * sum_Ekappa2_diff_tmp + Hkappa_tmp;
          if (randu() < exp((ELBOh - ELBO0_vec(j)) / (1 - 1 / T))) { //(ELBOh > ELBO0_vec(j)) { // 
            h_kappa.col(j) =  h_kappa_tmp;
            zeta_kappa_d.col(j) = zeta_kappa_d_tmp;
            zeta_kappa_od.col(j) = zeta_kappa_od_tmp;
            Ekappa.col(j) = Ekappa_tmp;
            sum_Ekappa2_diff(j) = sum_Ekappa2_diff_tmp;
            EKd_mat.col(j) = EKd_tmp;
            EKdinv_mat.col(j) = EKdinv_tmp;
            Hkappa(j) = Hkappa_tmp;
          }
        }
      }
      
    } else {
      
      
      idj = randperm(Pe);
      for (i = 0; i < Pe; i ++){
        j = idj(i);
        
        sum_EKod_m_data.col(idr(j)) -= EKod_mat.col(j) % data_mu.col(idc(j));
        sum_EKod_m_data.col(idc(j)) -= EKod_mat.col(j) % data_mu.col(idr(j));
        
        
        pL1pEKod_true = - EKdinv_mat.col(idr(j)) % data_mu.col(idc(j)) % sum_EKod_m_data.col(idr(j)) -
          EKdinv_mat.col(idc(j)) % data_mu.col(idr(j)) % sum_EKod_m_data.col(idc(j))-
          2 * data_mu.col(idc(j)) % data_mu.col(idr(j));
        pL1pVKodmn2_true = EKdinv_mat.col(idr(j)) % data_mu2.col(idc(j)) +
          EKdinv_mat.col(idc(j)) % data_mu2.col(idr(j));
        
        
        if (is_rnd_missing) {
          if (idr(j) - idc(j) > 1) id_tmp = join_cols(vP.head(idc(j)), vP(span(idc(j) + 1, idr(j) - 1)), vP.tail(P - 1 - idr(j)));
          else id_tmp = join_cols(vP.head(idc(j)), vP.tail(P - 1 - idr(j)));
          
          idr_vec.fill(idr(j));
          idc_vec.fill(idc(j));
          
          
          pL1pEKod_true(idr_missing) += - EKdinv_mat(idr_missing, idr_vec) % 
            sum(EKod_mat(idr_missing, idl_mat(idr_vec, id_tmp)) % COVdata.cols(idl_mat(idc_vec, id_tmp)), 1) - EKdinv_mat(idr_missing, idc_vec) % 
            sum(EKod_mat(idr_missing, idl_mat(idc_vec, id_tmp)) % COVdata.cols(idl_mat(idr_vec, id_tmp)), 1) - 2 * COVdata.col(j);
          pL1pVKodmn2_true(idr_missing) += EKdinv_mat(idr_missing, idr_vec) % Vdata.col(idc(j)) +
            EKdinv_mat(idr_missing, idc_vec) % Vdata.col(idr(j));
        } 
        
        if (is_nonzero_mean) {
          pL1pVKodmn2_true += EKdinv_mat.col(idr(j)) % Vmu_mat.col(idc(j)) +
            EKdinv_mat.col(idc(j)) % Vmu_mat.col(idr(j));
        }
        
        if (is_row_missing) {
          pL1pEKod_true(row_missing).zeros();
          pL1pVKodmn2_true(row_missing).zeros();
        }
        
        
        
        // update s
        log_B.col(1) = pL1pEKod_true % EJ_mat.col(j) - pL1pVKodmn2_true % EJ2_mat.col(j) / 2;
        
        results_DHMM = FwdBwdDHMM(log_s_init, log_A, log_B, N, 2);
        Es_mat.col(j) = as<vec>(results_DHMM["marg_density"]);
        sum_pair_density_tmp = as<mat>(results_DHMM["sum_pair_density"]);
        sum_pair_density.row(j) = join_rows(sum_pair_density_tmp.row(0), sum_pair_density_tmp.row(1));
        if (iter == 1) Hs(j) = as<double>(results_DHMM["entropy"]);
        
        EKod_mat.col(j) = Es_mat.col(j) % EJ_mat.col(j);
        EKod2_mat.col(j) = Es_mat.col(j) % EJ2_mat.col(j);
        
        
        // update J
        ELBO0 = accu(pL1pEKod_true % EKod_mat.col(j)) - accu(pL1pVKodmn2_true % EKod2_mat.col(j)) / 2 -
          Ealpha(j) / 2 * sum_EJ2_diff(j) + HJ(j);
        
        h_J_j = pL1pEKod_true % Es_mat.col(j);
        zeta_J_d_j = pL1pVKodmn2_true % Es_mat.col(j);
        zeta_J_d_j(0) += Ealpha(j);
        zeta_J_d_j(N - 1) += Ealpha(j);
        zeta_J_d_j(span(1, N - 2)) += 2 * Ealpha(j);
        zeta_J_od_j.fill(Ealpha(j));
        is_pd = FwdBwdCHMM_var(zeta_J_d_j, zeta_J_od_j, N, 1)["is_pd"];
        
        if (is_pd) eta_max = eta_thr;
        else {
          ELBO_max = ELBO0;
          eta_max = 0;
          eta = eta_thr / 2;
          while (eta > 1e-4) {
            h_J_tmp = (1 - eta) * h_J.col(j) + eta * h_J_j;
            zeta_J_d_tmp = (1 - eta) * zeta_J_d.col(j) + eta * zeta_J_d_j;
            zeta_J_od_tmp = (1 - eta) * zeta_J_od.col(j) + eta * zeta_J_od_j;
            results_CHMM = FwdBwdCHMM(h_J_tmp, zeta_J_d_tmp, zeta_J_od_tmp, N, 1);
            is_pd = results_CHMM["is_pd"];
            if (is_pd) {
              results_logdet = LogDetTriDiag(zeta_J_d_tmp, zeta_J_od_tmp, N, 1);
              HJ_tmp = - as<double>(results_logdet["logdetK"]) / 2;
              
              EJ_tmp = as<vec>(results_CHMM["mu"]);
              VJ_tmp = as<vec>(results_CHMM["S_d"]);
              sum_EJ2_diff_tmp = as<double>(results_CHMM["sum_E2_diff"]);
              
              EJ2_tmp = square(EJ_tmp) + VJ_tmp;
              EKod_tmp = Es_mat.col(j) % EJ_tmp;
              EKod2_tmp = Es_mat.col(j) % EJ2_tmp;
              
              ELBOh = accu(pL1pEKod_true % EKod_tmp) - accu(pL1pVKodmn2_true % EKod2_tmp) / 2 -
                Ealpha(j) / 2 * sum_EJ2_diff_tmp + HJ_tmp;
              if (ELBO_max > ELBO0 && ELBOh <= ELBO_max) break;
              else {
                if (ELBOh > ELBO_max) {
                  eta_max = eta;
                  ELBO_max = ELBOh;
                }
                eta /= 2;
              }
            } else eta /= 2;
          }
        }
        
        if (eta_max > 0) {
          h_J.col(j) = (1 - eta_max) * h_J.col(j) + eta_max * h_J_j;
          zeta_J_d.col(j) = (1 - eta_max) * zeta_J_d.col(j) + eta_max * zeta_J_d_j;
          zeta_J_od.col(j) = (1 - eta_max) * zeta_J_od.col(j) + eta_max * zeta_J_od_j;
          
          results_CHMM = FwdBwdCHMM(h_J.col(j), zeta_J_d.col(j), zeta_J_od.col(j), N, 1);
          EJ_mat.col(j) = as<vec>(results_CHMM["mu"]);
          VJ_mat.col(j) = as<vec>(results_CHMM["S_d"]);
          COVJ_mat.col(j) = as<vec>(results_CHMM["S_od"]);
          //sum_EJ2_diff_tmp = as<double>(results_CHMM["sum_E2_diff"]);
          sum_EJ2_diff(j) = as<double>(results_CHMM["sum_E2_diff"]);
          Ealpha(j) = (N - 1.0) / sum_EJ2_diff(j);
          
          EJ2_mat.col(j) = square(EJ_mat.col(j)) + VJ_mat.col(j);
          EKod_mat.col(j) = Es_mat.col(j) % EJ_mat.col(j);
          EKod2_mat.col(j) = Es_mat.col(j) % EJ2_mat.col(j);
          
          
          results_logdet = LogDetTriDiag(zeta_J_d.col(j), zeta_J_od.col(j), N, 1);
          HJ(j) = - as<double>(results_logdet["logdetK"]) / 2;
          
          /*if (sum_EJ2_diff_tmp > 0) {
            Ealpha(j) = (N - 1.0) / sum_EJ2_diff_tmp;
            if (Ealpha(j) > 1e300) {
              Ealpha(j) = 1e300;
              sum_EJ2_diff(j) = (N - 1.0) / 1e300;
            }
            else sum_EJ2_diff(j) = sum_EJ2_diff_tmp;
          }*/
          
        }
        
        sum_EKod_m_data.col(idr(j)) += EKod_mat.col(j) % data_mu.col(idc(j));
        sum_EKod_m_data.col(idc(j)) += EKod_mat.col(j) % data_mu.col(idr(j));
        
      }
      
      VKod_mat = EKod2_mat - square(EKod_mat);
      
      // Update Diagonal elements in K
      
      pL1pEKdinv_true = -square(sum_EKod_m_data) / 2;
      if (is_nonzero_mean) pL1pEKd_true -= Vmu_mat / 2;
      if (is_rnd_missing) pL1pEKd_true.rows(idr_missing) -= Vdata / 2;
      
      if (P < N) {
        for (j = 0; j < P; j ++) {
          id_tmp = join_cols(vP.head(j), vP.tail(P - 1 - j));
          idj_vec.fill(j);
          pL1pEKdinv_true.col(j) += sum(pL1pEKd_true.cols(id_tmp) % VKod_mat.cols(idl_mat(idj_vec, id_tmp)), 1);
          if (is_nonzero_mean) {
            pL1pEKdinv_true.col(j) -= sum(Vmu_mat.cols(id_tmp) % square(EKod_mat.cols(idl_mat(idj_vec, id_tmp))), 1) / 2;
          }
        }
      } else {
        for (i = 0; i < N; i ++) {
          VKodi(idl) = VKod_mat.row(i);
          VKodi(idu) = VKod_mat.row(i);
          
          pL1pEKdinv_true.row(i) += pL1pEKd_true.row(i) * VKodi; 
          
          if (is_nonzero_mean) {
            EKodi(idl) = EKod_mat.row(i);
            EKodi(idu) = EKod_mat.row(i);
            pL1pEKdinv_true.row(i) -= Vmu_mat.row(i) * square(EKodi) / 2; 
          }
        }
      }
      
      
      
      if (is_rnd_missing) {
        
        for (j = 0; j < idr_missing.n_elem; j ++) {
          i = idr_missing(j);
          EKodi(idl) = EKod_mat.row(i);
          EKodi(idu) = EKod_mat.row(i);
          Si(idl) = COVdata.row(j);
          Si(idu) = COVdata.row(j);
          Si.diag() = Vdata.row(j).t();
          pL1pEKdinv_true.row(i) -= sum(EKodi * Si % EKodi, 1).t();
        }
      }
      
      if (is_row_missing) {
        pL1pEKd_true.rows(row_missing).zeros();
        pL1pEKdinv_true.rows(row_missing).zeros();
        pL1pEkappa = pL1pEKd_true % EKd_mat % (1 - Ekappa) -
          pL1pEKdinv_true % EKdinv_mat % (1 + Ekappa);
        pL1pEkappa.rows(row_obsv) += 0.5;
        ELBO0_vec = 0.5 * sum(Ekappa.rows(row_obsv));
      } else {
        pL1pEkappa = 0.5 + pL1pEKd_true % EKd_mat % (1 - Ekappa) -
          pL1pEKdinv_true % EKdinv_mat % (1 + Ekappa);
        ELBO0_vec = 0.5 * sum(Ekappa);
      }
      pL1pEkappa2mn2 = - pL1pEKd_true % EKd_mat - pL1pEKdinv_true % EKdinv_mat;
      
      ELBO0_vec += - sum(pL1pEkappa2mn2) - Egamma / 2 * sum_Ekappa2_diff + Hkappa;
      
      natgrad_h_kappa = pL1pEkappa - h_kappa;
      natgrad_zeta_kappa_d = pL1pEkappa2mn2 - zeta_kappa_d;
      natgrad_zeta_kappa_d.row(0) += Egamma;
      natgrad_zeta_kappa_d.row(N - 1) += Egamma;
      natgrad_zeta_kappa_d.rows(1, N - 2) += 2 * Egamma;
      natgrad_zeta_kappa_od = Egamma - zeta_kappa_od;
      
      for (j = 0; j < P; j ++) {
        ELBO_max = ELBO0_vec(j);
        Hkappa_max = Hkappa(j);
        eta = eta_thr;
        while (eta > 1e-10) {
          h_kappa_tmp = h_kappa.col(j) + eta * natgrad_h_kappa.col(j);
          zeta_kappa_d_tmp = zeta_kappa_d.col(j) + eta * natgrad_zeta_kappa_d.col(j);
          zeta_kappa_od_tmp = zeta_kappa_od.col(j) + eta * natgrad_zeta_kappa_od.col(j);
          results_logdet = LogDetTriDiag(zeta_kappa_d_tmp, zeta_kappa_od_tmp, N, 1);
          if (results_logdet["is_pd"]) {
            results_CHMM = FwdBwdCHMM(h_kappa_tmp, zeta_kappa_d_tmp, 
                                      zeta_kappa_od_tmp, N, 1);
            Ekappa_tmp = as<vec>(results_CHMM["mu"]);
            Vkappa_tmp = as<vec>(results_CHMM["S_d"]);
            sum_Ekappa2_diff_tmp = as<double>(results_CHMM["sum_E2_diff"]);
            EKd_tmp = exp(Ekappa_tmp + Vkappa_tmp / 2);
            EKdinv_tmp = exp(- Ekappa_tmp + Vkappa_tmp / 2);
            Hkappa_tmp = - as<vec>(results_logdet["logdetK"])(0) / 2;
            if (is_row_missing) ELBO_tmp = 0.5 * accu(Ekappa_tmp(row_obsv));
            else ELBO_tmp = 0.5 * accu(Ekappa_tmp);
            ELBO_tmp +=  accu(pL1pEKd_true.col(j) % EKd_tmp) +
              accu(pL1pEKdinv_true.col(j) % EKdinv_tmp) - Egamma / 2 * sum_Ekappa2_diff_tmp + Hkappa_tmp;
            if (ELBO_max > ELBO0_vec(j) && ELBO_tmp <= ELBO_max) {
              eta *= 2;
              h_kappa.col(j) += eta * natgrad_h_kappa.col(j);
              zeta_kappa_d.col(j) += eta * natgrad_zeta_kappa_d.col(j);
              zeta_kappa_od.col(j) += eta * natgrad_zeta_kappa_od.col(j);
              Hkappa(j) = Hkappa_max;
              break;
            } else {
              if (ELBO_tmp > ELBO_max) {
                ELBO_max = ELBO_tmp;
                Hkappa_max = Hkappa_tmp;
              }
              eta /= 2;
            }
          }
          else
            eta /= 2;
        }
      }
      
      
      results_CHMM = FwdBwdCHMM(h_kappa, zeta_kappa_d, zeta_kappa_od, N, P);
      Ekappa = as<mat>(results_CHMM["mu"]);
      Vkappa = as<mat>(results_CHMM["S_d"]);
      COVkappa = as<mat>(results_CHMM["S_od"]);
      sum_Ekappa2_diff = as<rowvec>(results_CHMM["sum_E2_diff"]);
      EKd_mat = exp(Ekappa + Vkappa / 2);
      EKdinv_mat = exp(- Ekappa + Vkappa / 2);
    }
    
    
    
    // update mean parameters
    if (is_nonzero_mean) {
      idj = randperm(P);
      results_MeanUpdate = MeanUpdate(EKd_mat, EKdinv_mat, EKod_mat, VKod_mat, data, Emu_mat, Elambda, 
                                      idj, N, P, Pa, idl, idu, ida, idl_mat, row_missing, is_row_missing);
      Emu_mat = as<mat>(results_MeanUpdate["Emu_mat"]);
      Vmu_mat = as<mat>(results_MeanUpdate["Vmu_mat"]);
      COVmu_mat = as<mat>(results_MeanUpdate["COVmu_mat"]);
      //update Elambda
      Elambda = (N - 1.0) * P / (accu(square(Emu_mat.head_rows(N - 1) - Emu_mat.tail_rows(N - 1))) + accu(Vmu_mat.head_rows(N - 1)) +
        accu(Vmu_mat.tail_rows(N - 1)) - 2 * accu(COVmu_mat));
    }
    
    
    
    // update missing data
    if (is_rnd_missing) {  //&& iter > 1
      
      results_MissingUpdate = MissingUpdate(data, Vdata, COVdata, EKod_mat, VKod_mat, EKd_mat, EKdinv_mat, Emu_mat, eta_thr, 
                                            idr_missing, id_missing, idl, idu, N, P, Pe, N_missing, is_nonzero_mean);
      
      data = as<mat>(results_MissingUpdate["data"]);
      Vdata = as<mat>(results_MissingUpdate["Vdata"]);
      COVdata = as<mat>(results_MissingUpdate["COVdata"]);
      
    }
    
    if (is_nonzero_mean) {
      data_mu = data - Emu_mat;
      data_mu2 = square(data_mu);
      pL1pEKd_true = - data_mu2 / 2;
      sum_EKod_m_data = SumEKodMDataUpdate(data_mu, EKod_mat, idl, idu, idl_mat, N, P);
    } else if (is_rnd_missing) {
      data_mu(id_missing_vec) = data(id_missing_vec);
      data_mu2(id_missing_vec) = square(data_mu(id_missing_vec));
      pL1pEKd_true(id_missing_vec) = - data_mu2(id_missing_vec) / 2;
      sum_EKod_m_data.rows(idr_missing) = SumEKodMDataUpdate(data_mu.rows(idr_missing), EKod_mat.rows(idr_missing), 
                           idl, idu, idl_mat, N_missing, P);
    }
    
    // update s_init and A
    a = 1.0 + Pe - accu(Es_mat.row(0));
    b = 1.0 + accu(Es_mat.row(0));
    log_s_init(0) = boost::math::digamma(a) - boost::math::digamma(a + b);
    log_s_init(1) = boost::math::digamma(b) - boost::math::digamma(a + b);
    
    c0 = 1.0 + accu(sum_pair_density.col(0));
    d0 = 1.0 + accu(sum_pair_density.col(1));
    c1 = 1.0 + accu(sum_pair_density.col(3));
    d1 = 1.0 + accu(sum_pair_density.col(2));
    
    log_A(0, 0) = boost::math::digamma(c0) - boost::math::digamma(c0 + d0);
    log_A(0, 1) = boost::math::digamma(d0) - boost::math::digamma(c0 + d0);
    log_A(1, 1) = boost::math::digamma(c1) - boost::math::digamma(c1 + d1);
    log_A(1, 0) = boost::math::digamma(d1) - boost::math::digamma(c1 + d1);
    
    // update Egamma
    Egamma = (N - 1.0) * P / accu(sum_Ekappa2_diff);
    
    if (iter <= anneal_iters) {
      if (iter % 10 == 0) {
        /*diff_s = max(abs(vectorise(Es_mat - Es_mat_old)));
         diff_d = EKd_mat - EKd_mat_old;
         diff_od = EKod_mat - EKod_mat_old;
         diff_d_max = max(abs(vectorise(diff_d)));
         diff_d_r = sqrt(mean(vectorise(square(diff_d))) / mean(vectorise(square(EKd_mat))));
         diff_od_max = max(abs(vectorise(diff_od)));
         diff_od_r = sqrt(mean(vectorise(square(diff_od))) / mean(vectorise(square(EKod_mat))));
         Rprintf("iteration %3d: diff_s=%e, diff_d_r=%e, diff_d_max=%e, diff_od_r=%e, diff_od_max=%e\n", iter, 
         diff_s, diff_d_r, diff_d_max, diff_od_r, diff_od_max);
         Es_mat_old = Es_mat;
         EKod_mat_old = EKod_mat;
         EKd_mat_old = EKd_mat;*/
        pb.increment();
        
        bandwidth = bandwidth - bandwidth_diff;
        if (bandwidth < 1.0) bandwidth = 1.0;
        lb = regspace(0, N - 1) - ceil(bandwidth) + 0.5;
        lb(find(lb < -0.5)).fill(-0.5);
        ub = regspace(0, N - 1) + ceil(bandwidth) - 0.5;
        ub(find(ub > N - 0.5)).fill(N - 0.5);
        bd_range = ub - lb;
        T = (bandwidth_ub - 1.0) / (bandwidth_ub - bandwidth);
      }
      
      if (iter == anneal_iters) {
        id_rnd.clear();
        data_mu_rnd.clear();
        data_mu_rnd2.clear();
        sum_EKod_m_data_rnd.clear();
        pL1pEKod.clear();
        pL1pVKodmn2.clear();
        pL1pEs_true.clear();
        Hs.clear();
        Es_tmp.clear();
        sum_pair_density_tmp.clear();
        pL1pEKd.clear();
        pL1pEKdinv.clear();
        Egamma_rnd.clear();
        
        if (is_row_missing) v1_missing.clear();
        if (is_rnd_missing) {
          v2_missing.clear();
          v2_rnd.clear();
          idr_missing_rnd.clear();
          idr_missing_rnd0.clear();
          Vdata_rnd.clear();
          COVdata_rnd.clear();
        }
        if (is_nonzero_mean) Vmu_rnd.clear();
        
        T = 1.0;
        Es_mat_old = Es_mat;
        EKod_mat_old = EKod_mat;
        EKd_mat_old = EKd_mat;
      }
      
    } else {
      diff_s = max(abs(vectorise(Es_mat - Es_mat_old)));
      diff_d = EKd_mat - EKd_mat_old;
      diff_od = EKod_mat - EKod_mat_old;
      diff_d_max = max(abs(vectorise(diff_d)));
      diff_d_r = sqrt(mean(vectorise(square(diff_d))) / mean(vectorise(square(EKd_mat))));
      diff_od_max = max(abs(vectorise(diff_od)));
      diff_od_r = sqrt(mean(vectorise(square(diff_od))) / mean(vectorise(square(EKod_mat))));
      Rprintf("iteration %3d: diff_s=%e, diff_d_r=%e, diff_d_max=%e, diff_od_r=%e, diff_od_max=%e\n", iter, 
              diff_s, diff_d_r, diff_d_max, diff_od_r, diff_od_max);
      
      if (diff_od_r < tol_relative && diff_d_r < tol_relative && diff_s < tol_s) break;
      else {
        Es_mat_old = Es_mat;
        EKod_mat_old = EKod_mat;
        EKd_mat_old = EKd_mat;
      }
    }
    
    
  }
  
  auto t_end = std::chrono::high_resolution_clock::now();
  double run_time = std::chrono::duration<double, std::milli>(t_end-t_start).count() / 1e3;
  if (iter < max_iter) Rprintf("BADGE converges, elapsed time is %f seconds.\n",run_time);
  else Rprintf("BADGE reaches the maximum number of iterations, elapsed time is %f seconds.\n",run_time);
  
  return List::create(Named("EKd_mat") = EKd_mat, Named("EKod_mat") = EKod_mat, Named("EJ_mat") = EJ_mat, Named("Es_mat") = Es_mat,
                            Named("data") = data, Named("run_time") = run_time);
}
