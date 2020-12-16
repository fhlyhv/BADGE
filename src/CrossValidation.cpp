#include "CrossValidation.hpp"


/*
* Use pairwise composite likelihood to do cross validation in order to determine the bandwidth for noise
* 
*/


double CrossValidation(double bandwidth_int, mat data, uword n_fold, uword N, uword P, bool is_nonzero_mean)  {
  
  double cv_score_old = -1e100, cv_score, bandwidth = bandwidth_int, det_S, bd_max, cv_max;
  uword iter, i, j, id_r, id_c;
  vec id_all = regspace(0, N - 1), weights, weights0, id_w0, idj;
  uvec id_validation, id_w, id_sub, id_vec(N);
  mat mu_all(N, P, fill::zeros), S_all, data_mu;
  bool is_cv_old = false, is_cv;
  
  for (iter = 0; iter < N - bandwidth_int; iter ++) {
    is_cv = true;
    cv_score = 0;
    id_w0 = regspace(- ceil(bandwidth) + 1.0, ceil(bandwidth) - 1.0);
    weights0 = 1.0 - square(id_w0 / bandwidth);
    
    for (i = 0; i < n_fold; i ++) {
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
          mu_all.row(j) = sum(data.rows(id_w(id_sub)) % repmat(weights(id_sub) / sum(weights(id_sub)), 1, P));
          
          id_sub.clear();
          id_w.clear();
          weights.clear();
        }
        data_mu = data - mu_all;
      } else data_mu = data;
      
      
      for (j = 0; j < id_validation.n_elem; j ++) {
        idj = id_w0 + id_validation(j);
        id_sub = find((idj >= 0) % (idj < N));
        weights = weights0(id_sub);
        id_w = conv_to<uvec>::from(idj(id_sub));
        id_sub.clear();
        
        /*if (id_validation(j) < bandwidth - 1) {
          id_sub = find(idj >= 0);
          weights = weights0(id_sub);
          id_w = conv_to<uvec>::from(idj(id_sub));
          id_sub.clear();
        } else if (id_validation(j) > N - bandwidth) {
          id_sub = find(idj < N);
          weights = weights0(id_sub);
          id_w = conv_to<uvec>::from(idj(id_sub));
          id_sub.clear();
        } else {
          weights = weights0;
          id_w = conv_to<uvec>::from(idj);
        }*/
        id_sub = find(id_vec(id_w) > 0);
        
        if (id_sub.n_elem < 2) {
          is_cv = false;
          
          id_sub.clear();
          id_w.clear();
          weights.clear();
          
          break;
        } else{
          S_all = data_mu.rows(id_w(id_sub)).t() % repmat(weights(id_sub).t() / sum(weights(id_sub)), P, 1) *
            data_mu.rows(id_w(id_sub));
          
          id_sub.clear();
          id_w.clear();
          weights.clear();
          
          for (id_r = 0; id_r < P - 1; id_r ++) {
            for (id_c = id_r + 1; id_c < P; id_c ++) {
              det_S = S_all(id_r, id_r) * S_all(id_c, id_c) - pow(S_all(id_r, id_c), 2);
              cv_score -= (log(det_S) + (pow(data_mu(id_validation(j), id_r), 2) * S_all(id_c, id_c) +
                pow(data_mu(id_validation(j), id_c), 2) * S_all(id_r, id_r) - 
                2 * data_mu(id_validation(j), id_r) * data_mu(id_validation(j), id_c) * S_all(id_r, id_c)) / det_S) / 2;
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
      printf("bandwidth = %f, cv_score = %f, is_cv = %i\n", bandwidth, cv_score, is_cv);
      if (bandwidth < P) bandwidth ++;
      else bandwidth *= 1.5; //+= 0.5;
      weights0.clear();
      id_w0.clear();
      idj.clear();
    }
  }
 
  return bd_max;
  
  
  
}
