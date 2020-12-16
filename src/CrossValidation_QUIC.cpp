#include "CrossValidation_QUIC.hpp"


field<mat> CrossValidation_QUIC(mat data, uvec idr, uvec idc, uvec idl, uvec idu, uword N, uword P, uword Pe, bool is_nonzero_mean, 
                                bool is_missing, umat is_na, bool is_rnd_missing, uvec id_missing_vec, uvec idr_missing, int N_missing, 
                                bool is_row_missing, uvec row_missing) {
  double bandwidth_init = (P / 2.0 < N / 10.0) ? P / 2.0 : N / 10.0;
  double cv_score_old = -1e100, cv_score, bandwidth = bandwidth_init, cv_max = cv_score_old, bd_max, thr;
  vec id_w0, weights0, w, idi, std_vec, cdf_s(1);
  rowvec Es_vec;
  uvec id_sub, id_w, id_nonzero, id_zero, idi_vec(1), n_diff;
  mat mu(N, P), Es_mat(N, Pe), Es_mat0, Es_old, EJ_mat(N, Pe), EKd_mat(N, P), K, S_normalized, w_mat, is_not_na, I(P, P, fill::eye), K_init;
  if (is_missing) is_not_na = 1.0 - conv_to<mat>::from(is_na);
  cube S(P, P, N), K_cube(P, P, N);
  bool is_cv_score;
  field<mat> outputs(4);
  field<uvec> col_obsv, col_missing;
  int i, j;
  cdf_s.fill(1.0 - (double) P / Pe); //(0.7); //
  if (is_rnd_missing) {
    col_obsv.set_size(N_missing);
    col_missing.set_size(N_missing);
    for (i = 0; i < N_missing; i ++) {
      col_obsv(i) = find(is_na.row(idr_missing(i)) == 0);
      col_missing(i) = find(is_na.row(idr_missing(i)));
    }
  }
  
  K_cube.slice(0) = I;
  
  
  while (bandwidth < N) {
    cv_score = 0;
    is_cv_score = true;
    
    id_w0 = join_cols(regspace(- ceil(bandwidth) + 1.0, -1.0), regspace(1.0, ceil(bandwidth) - 1.0));
    weights0 = 1.0 - square(id_w0 / bandwidth);
    weights0 /= accu(weights0);
    
    if (is_nonzero_mean) {
      mu.set_size(N, P);
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
        if (is_missing) {
          if (max(sum(is_na.rows(id_w))) == id_w.n_elem) {
            is_cv_score = false;
            break;
          }
          mu.row(i) = sum(mat(data.rows(id_w)).each_col() % w) / sum(mat(is_not_na.rows(id_w)).each_col() % w);
        }
        else mu.row(i) = sum(mat(data.rows(id_w)).each_col() % w);
      }
      if (is_cv_score) {
        data -= - mu;
        if (is_row_missing) data.rows(row_missing).zeros();
        if (is_rnd_missing) data(id_missing_vec).zeros();
      }
    }
    
    if (is_cv_score) {
      
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
        
        if (is_missing) {
          if (max(sum(is_na.rows(id_w))) == id_w.n_elem) {
            is_cv_score = false;
            break;
          }
          w_mat = mat(is_not_na.rows(id_w)).each_col() % w;
          S.slice(i) = (data.rows(id_w) % w_mat).t() * data.rows(id_w) / (w_mat.t() * is_not_na.rows(id_w));
        } else S.slice(i) = (mat(data.rows(id_w)).each_col() % w).t() * data.rows(id_w);
        
        std_vec = sqrt(S.slice(i).diag());
        
        if (std_vec.min() <= 0) {
          is_cv_score = false;
          break;
        }
        
        S_normalized = S.slice(i) / (std_vec * std_vec.t());
        Es_mat.row(i) = S_normalized(idl).t();
      }
      
      
      
      if (is_cv_score) {
        Es_mat = abs(Es_mat);
        thr = as_scalar(quantile(vectorise(Es_mat), cdf_s));
        //Rprintf("thr = %f, mean_s = %f\n", thr, mean(mean(Es_mat)));
        Es_mat.clean(thr); //(find(Es_mat < thr)).zeros();
        Es_mat(find(Es_mat >= thr)).ones();
        
        
        if (is_rnd_missing) j = 0;
        for (i = 0; i < N; i ++) {
          if (! (is_missing && accu(is_na.row(i)) == P)) {
            id_nonzero = find(Es_mat.row(i));
            //Rprintf("i = %i, n_nonzero = %i\n", i, id_nonzero.n_elem); //, min(eig_sym(S.slice(i))));
            if (id_nonzero.n_elem > 0) {
              if (bandwidth == bandwidth_init) {
                if (i > 0) {
                  K_cube.slice(i) = K_init;
                  id_zero = find(Es_mat.row(i) - Es_vec < 0);
                  if (id_zero.n_elem > 0) {
                    K_cube.slice(i)(idl(id_zero)).zeros();
                    K_cube.slice(i)(idu(id_zero)).zeros();
                  }
                }
              } else {
                id_zero = find(Es_mat.row(i) - Es_old.row(i) < 0);
                if (id_zero.n_elem > 0) {
                  K_cube.slice(i)(idl(id_zero)).zeros();
                  K_cube.slice(i)(idu(id_zero)).zeros();
                }
              }
              
              //if (i == 273) K_init.print();
              
              K = QUICParameterLearning(K_cube.slice(i), S.slice(i), idr(id_nonzero), idc(id_nonzero), 50, 50);
            } else K = diagmat(1 / S.slice(i).diag());
            
            K_cube.slice(i) = K;
            if (bandwidth == bandwidth_init) {
              K_init = K;
              Es_vec = Es_mat.row(i);
            }
            
            if (!is_missing || (is_missing && accu(is_na.row(i)) == 0)) {
              if (id_nonzero.n_elem > 0) cv_score += accu(log(diagvec(chol(K)))) - as_scalar(data.row(i) * K * data.row(i).t()) / 2;
              else cv_score += as_scalar(accu(log(K.diag())) - square(data.row(i)) * K.diag()) / 2;
            } else {
              idi_vec.fill(i);
              if (id_nonzero.n_elem > 0) {
                K = K(col_obsv(j), col_obsv(j)) - K(col_obsv(j), col_missing(j)) * solve(K(col_missing(j), col_missing(j)), K(col_missing(j), col_obsv(j)));
                K = (K + K.t()) / 2;
                cv_score += accu(log(diagvec(chol(K)))) - as_scalar(data(idi_vec, col_obsv(j)) * K * data(idi_vec, col_obsv(j)).t()) / 2;
              } else cv_score += as_scalar(accu(log(vec(K.diag()).elem(col_obsv(j)))) - square(data(idi_vec, col_obsv(j))) * vec(K.diag()).elem(col_obsv(j))) / 2;
              j ++;
            }
          }
          
        }
        
        //Rprintf("bandwidth = %f, cv_score = %f, cv_score_old = %f\n", bandwidth, cv_score, cv_score_old);
        if (cv_score < cv_score_old) break;  //&& bandwidth >= P
        else {
          if (cv_score > cv_max) {
            cv_max = cv_score;
            bd_max = bandwidth;
          }
          Es_old = Es_mat;
          cv_score_old = cv_score;
          if (bandwidth == bandwidth_init) {
            K_init.clear();
            Es_vec.clear();
          }
          bandwidth *= 1.5;
        }
      }
      
      
    }
  }
  
  bandwidth = bd_max;
  //Rprintf("bandwidth = %f\n", bandwidth);
  id_w0 = regspace(- ceil(bandwidth) + 1.0, ceil(bandwidth) - 1.0);
  weights0 = 1.0 - square(id_w0 / bandwidth);
  weights0 /= accu(weights0);
  
  if (is_nonzero_mean) {
    mu.set_size(N, P);
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
      if (is_missing) {
        mu.row(i) = sum(mat(data.rows(id_w)).each_col() % w) / sum(mat(is_not_na.rows(id_w)).each_col() % w);
      }
      else mu.row(i) = sum(mat(data.rows(id_w)).each_col() % w);
    }
    data -= - mu;
    if (is_row_missing) data.rows(row_missing).zeros();
    if (is_rnd_missing) data(id_missing_vec).zeros();
  }
  
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
    
    if (is_missing) {
      w_mat = mat(is_not_na.rows(id_w)).each_col() % w;
      S.slice(i) = (data.rows(id_w) % w_mat).t() * data.rows(id_w) / (w_mat.t() * is_not_na.rows(id_w));
    } else S.slice(i) = (mat(data.rows(id_w)).each_col() % w).t() * data.rows(id_w);
    S.slice(i) = (S.slice(i) + S.slice(i).t()) / 2;
    std_vec = sqrt(S.slice(i).diag());
    S_normalized = S.slice(i) / (std_vec * std_vec.t());
    Es_mat.row(i) = S_normalized(idl).t();
  }
  
  Es_mat = abs(Es_mat);
  Es_mat0 = Es_mat;
  cdf_s.fill(0.7); //(1.0 - (double) P / Pe);
  thr = as_scalar(quantile(vectorise(Es_mat), cdf_s));
  Es_mat.clean(thr); //(find(Es_mat < thr)).zeros();
  Es_mat(find(Es_mat >= thr)).fill(0.7); //ones(); //
  
  
  if(bandwidth > 2.0 * P) {
    K = K_cube.slice(0);
    K_cube.clear();
    for (i = 0; i < N; i ++) {
      if (!is_missing || S.slice(i).is_sympd()) K = inv_sympd(S.slice(i));
      else K = QUICParameterLearning(K, S.slice(i), idr, idc, 50, 50);
      
      EKd_mat.row(i) = K.diag().t();
      EJ_mat.row(i) = K(idl).t();
    }
  } else {
    thr = 1.0 - pow(bandwidth / P / 2.0, 2);
    if (thr <= 0.7) thr = 0.7;
    Rprintf("chosen thr = %f\n", thr);
    cdf_s.fill(thr); //(1.0 - (double) P / Pe);
    thr = as_scalar(quantile(vectorise(Es_mat0), cdf_s));
    Es_mat0.clean(thr); 
    
    // n_diff = sum(Es_mat0.tail_rows(N - 1) - Es_mat0.head_rows(N - 1) < 0, 1);
    // n_diff.t().print();
    if (cdf_s(0) < 1.0 - 2.0 * P / Pe) {
      K_cube.clear();
      
      for (i = 0; i < N; i ++) {
        id_nonzero = find(Es_mat0.row(i));
        //Rprintf("i = %i, id_nonzero = %i\n", i, id_nonzero.n_elem);
        if (id_nonzero.n_elem > 0) {
          if (i > 0) {
            K_init = diagmat(K.diag());
            K_init(idl(id_nonzero)) = K(idl(id_nonzero));
            K_init(idu(id_nonzero)) = K(idu(id_nonzero));
          } else {
            S_normalized = S.slice(i);
            S_normalized *= bandwidth / P / 2.0;
            S_normalized.diag() += (1 - bandwidth / P / 2.0);
            S_normalized = inv_sympd(S_normalized);
            
            K_init = diagmat(S_normalized.diag());
            K_init(idl(id_nonzero)) = S_normalized(idl(id_nonzero));
            K_init(idu(id_nonzero)) = S_normalized(idu(id_nonzero));
          }
          
          
          K = QUICParameterLearning(K_init, S.slice(i), idr(id_nonzero), idc(id_nonzero), 50, 50);
        } else K = diagmat(1 / S.slice(i).diag());
        EKd_mat.row(i) = K.diag().t();
        EJ_mat.row(i) = K(idl).t();
      }
      
    } else {
      if (is_row_missing) {
        for (i = 0; i < row_missing.n_elem; i ++) {
          j = row_missing(i);
          K_cube.slice(j) = K_cube.slice(j - 1);
          id_zero = find(Es_old.row(j) - Es_old.row(j - 1) < 0);
          if (id_zero.n_elem > 0) {
            K_cube.slice(j)(idl(id_zero)).zeros();
            K_cube.slice(j)(idu(id_zero)).zeros();
          }
        }
      }
      
      for (i = 0; i < N; i ++) {
        
        id_nonzero = find(Es_mat.row(i));
        //Rprintf("i = %i, n_nonzero = %i\n", i, id_nonzero.n_elem); //, min(eig_sym(S.slice(i))));
        if (id_nonzero.n_elem > 0) {
            id_zero = find(Es_mat0.row(i) - Es_old.row(i) < 0);
            if (id_zero.n_elem > 0) {
              K_cube.slice(i)(idl(id_zero)).zeros();
              K_cube.slice(i)(idu(id_zero)).zeros();
            }
          K = QUICParameterLearning(K_cube.slice(i), S.slice(i), idr(id_nonzero), idc(id_nonzero), 50, 50);
        } else K = diagmat(1 / S.slice(i).diag());
        
        EKd_mat.row(i) = K.diag().t();
        EJ_mat.row(i) = K(idl).t();
      }
    }
    
    
    
    /*K_init = K_cube.slice(0);
     K_cube.clear();
     for (i = 0; i < N; i ++) {
     id_nonzero = find(Es_mat0.row(i));
     
     if (id_nonzero.n_elem > 0) {
     if (i == 0) id_zero = find((Es_mat0.row(i) == 0) % (K_init(idl).t() != 0) );
     else id_zero = find(Es_mat0.row(i) - Es_mat0.row(i - 1) < 0);
     if (id_zero.n_elem > 0) {
     K_init(idl(id_zero)).zeros();
     K_init(idu(id_zero)).zeros();
     }
     
     K = QUICParameterLearning(K_init, S.slice(i), idr(id_nonzero), idc(id_nonzero), 50, 50);
     } else K = QUICParameterLearning_diag(K_init, S.slice(i), 50);
     
     K_init = K;
     
     EKd_mat.row(i) = K.diag().t();
     EJ_mat.row(i) = K(idl).t();
     }*/
    
  }
  Es_mat.replace(0, 0.3);
  
  outputs(0) = Es_mat;
  outputs(1) = EJ_mat;
  outputs(2) = EKd_mat;
  outputs(3) = mu;
  
  return outputs;
}
