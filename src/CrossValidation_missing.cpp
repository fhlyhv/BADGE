#include "CrossValidation_missing.hpp"


/*
 * Use pairwise composite likelihood to do cross validation in order to determine the bandwidth for noise
 * 
 */

double CrossValidation_missing(double bandwidth_int, mat data, uword n_fold, uword N, uword P, bool is_nonzero_mean, umat is_not_na)  {
  
  double cv_score_old = -1e100, cv_score, bandwidth = bandwidth_int, det_S, bd_max, cv_max;
  uword iter, i, j, id_r, id_c;
  vec id_all = regspace(0, N - 1), weights, weights0, id_w0, idj;
  uvec id_validation, id_w, id_sub, id_vec(N), id_wb;
  mat mu_all(N, P, fill::zeros), S_all, data_mu, weights_mat;
  bool is_cv_old = false, is_cv;
  
  for (iter = 0; iter < N - bandwidth_int; iter ++) { //N - 2
    is_cv = true;
    cv_score = 0;
    id_w0 = regspace(- ceil(bandwidth) + 1.0, ceil(bandwidth) - 1.0);
    weights0 = 1.0 - square(id_w0 / bandwidth);
    
    for (i = 0; i < n_fold; i ++) {
      if (!is_cv) break;
      id_vec.fill(1);
      id_validation = regspace<uvec>(i, n_fold, N - 1);
      id_vec(id_validation).zeros();
      
      if (is_nonzero_mean) {
        for (j = 0; j < N; j ++) {
          idj = id_w0 + j;
          id_sub = find((idj >= 0) % (idj < N));
          weights = weights0(id_sub);
          id_w = conv_to<uvec>::from(idj(id_sub));
          id_sub.clear();
          
          id_sub = find(id_vec(id_w) > 0);
          weights_mat = repmat(weights(id_sub), 1, P);
          weights_mat(find(is_not_na.rows(id_w(id_sub)) == 0)).zeros();
          mu_all.row(j) = sum(data.rows(id_w(id_sub)) % weights_mat) / sum(weights_mat);
          
          id_sub.clear();
          id_w.clear();
          weights.clear();
          weights_mat.clear();
        }
        data_mu = data - mu_all;
      } else data_mu = data;
      
      
      data_mu(find(is_not_na == 0)).zeros();
      for (j = 0; j < id_validation.n_elem; j ++) {
        idj = id_w0 + id_validation(j);
        id_sub = find((idj >= 0) % (idj < N));
        weights = weights0(id_sub);
        id_w = conv_to<uvec>::from(idj(id_sub));
        idj.clear();
        id_sub.clear();
        
        
        
        
        if (accu(prod(is_not_na.rows(id_w), 1)) < 2) {
          is_cv = false;
          
          id_sub.clear();
          id_w.clear();
          weights.clear();
          id_wb.clear();
          
          break;
        } else{
          id_wb.clear();
          id_sub.clear();
          id_sub = find(id_vec(id_w) > 0);
          weights_mat = repmat(weights(id_sub), 1, P);
          weights_mat(find(is_not_na.rows(id_w(id_sub)) == 0)).zeros();
          
          S_all = (data_mu.rows(id_w(id_sub)) % weights_mat).t() *
            data_mu.rows(id_w(id_sub)) / (weights_mat.t() * is_not_na.rows(id_w(id_sub)));
          S_all = (S_all + S_all.t()) / 2;
          
          id_sub.clear();
          id_w.clear();
          weights.clear();
          weights_mat.clear();
          
          for (id_r = 0; id_r < P - 1; id_r ++) {
            for (id_c = id_r + 1; id_c < P; id_c ++) {
              if (is_not_na(id_validation(j), id_r) == 1 && is_not_na(id_validation(j), id_c) == 1) {
                det_S = S_all(id_r, id_r) * S_all(id_c, id_c) - pow(S_all(id_r, id_c), 2);
                // if (det_S < 0) printf("bandwidth = %f, det_S = %f\n", bandwidth, det_S);
                if (det_S > 0) cv_score -= (log(det_S) + (pow(data_mu(id_validation(j), id_r), 2) * S_all(id_c, id_c) +
                    pow(data_mu(id_validation(j), id_c), 2) * S_all(id_r, id_r) - 
                    2 * data_mu(id_validation(j), id_r) * data_mu(id_validation(j), id_c) * S_all(id_r, id_c)) / det_S) / 2;
                else cv_score -= (log(S_all(id_r, id_r)) + pow(data_mu(id_validation(j), id_r), 2) / S_all(id_r, id_r) +
                                  log(S_all(id_c, id_c)) + pow(data_mu(id_validation(j), id_c), 2) / S_all(id_c, id_c)) / 2;
              } else if (is_not_na(id_validation(j), id_r) == 1) {
                cv_score -= (log(S_all(id_r, id_r)) + pow(data_mu(id_validation(j), id_r), 2) / S_all(id_r, id_r)) / 2;
              } else if (is_not_na(id_validation(j), id_c) == 1) {
                cv_score -= (log(S_all(id_c, id_c)) + pow(data_mu(id_validation(j), id_c), 2) / S_all(id_c, id_c)) / 2;
              }
            }
          }
        }
        
      }
      id_validation.clear();
    }
    if (is_cv_old && is_cv && cv_score < cv_score_old && bandwidth > P / 2.0) {
      // printf("cv_score = %f, cv_score_old = %f, \n", cv_score, cv_score_old);
      break;
    } else {
      if (is_cv) {
        if (cv_score > cv_score_old) {
          cv_max = cv_score;
          bd_max = bandwidth;
        }
        cv_score_old = cv_score;
        is_cv_old = true;
      }
      
      weights0.clear();
      id_w0.clear();
      
    }
    printf("bandwidth = %f, cv_score = %f, is_cv = %i\n", bandwidth, cv_score, is_cv);
    if (bandwidth < P) bandwidth ++;
    else bandwidth *= 1.5; //+= 0.5;
    
  }
  //printf("bandwidth = %f, cv_score = %f, is_cv = %i\n", bandwidth, cv_score, is_cv);
  
  return bd_max;
  
  
  
}
