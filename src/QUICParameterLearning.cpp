#include "QUICParameterLearning.hpp"

mat QUICParameterLearning(mat K0, mat S, uvec idr, uvec idc, uword max_outer_iter, uword max_inner_iter) {
  uword p = K0.n_cols, iter_outer, iter_inner, p_od = idr.n_elem, i;
  uvec idl = idc * p + idr, idu = idr * p + idc, idd = linspace<uvec>(0, p - 1, p), ida = join_cols(idd * p + idd, idl);
  vec Sida = S(ida), Sidu = S(idu), Sd = S.diag(), Did(p + p_od), gradK, Kd = K0.diag();
  mat U(p, p), W, Kh(p, p, fill::zeros);
  double a, b, mu, diffD = 0;
  bool is_converge = false;
  S.clear();
  
  double objh, xi0 = 1, xi, sum_grad2;
  while (! K0.is_sympd()) {
    K0 *= 0.9;
    K0.diag() = Kd;
  }
  double obj0 = - 2*accu(log(diagvec(chol(K0)))) + accu(Sd % K0.diag()) + 2 * accu(Sidu % K0(idu));
  //Rprintf("obj0 = %e\n", obj0);
  for (iter_outer = 0; iter_outer < max_outer_iter; iter_outer ++) {
    W = inv_sympd(K0);
    gradK = Sida - W(ida);
    Did.zeros();
    U.zeros();
    for (iter_inner = 0; iter_inner < max_inner_iter; iter_inner ++) {
      for (i = 0; i < p; i++) {
        a = pow(W(i, i), 2);
        b = gradK(i) + accu(W.col(i) % U.col(i));
        mu = - b / a;
        Did(i) += mu;
        U.row(i) += mu * W.row(i);
        diffD += fabs(mu);
      }
      
      for (i = 0; i < p_od; i ++) {
        a = pow(W(idl(i)), 2) + W(idr(i), idr(i)) * W(idc(i), idc(i));
        b = gradK(p + i) + accu(W.col(idr(i)) % U.col(idc(i)));
        mu = - b / a;
        Did(p + i) += mu;
        U.row(idr(i)) += mu * W.row(idc(i));
        U.row(idc(i)) += mu * W.row(idr(i));
        diffD += fabs(mu);
      }
      
      if (diffD < 0.05 * accu(abs(Did))) break;
      else diffD = 0;
    }
    sum_grad2 = accu(Did % gradK);
    xi = xi0;
    while (true) {
      Kh(ida) = K0(ida) + xi * Did;
      Kh(idu) = Kh(idl);
      if (Kh.is_sympd()) {
        objh = - 2*accu(log(diagvec(chol(Kh)))) + accu(Sd % Kh.diag()) + 2 * accu(Sidu % Kh(idu));
        if (xi * abs(Did).max() < 1e-5 && obj0 - objh < 1e-5) {
          is_converge = true;
          break;
        }
        if (objh <= obj0 + 1e-3 * xi * sum_grad2) break;
        else xi /= 2;
      } else xi /=2;
      if (xi < 1e-10 && obj0 == objh) {
        is_converge = true;
        break;
      }
    }
    //Rprintf("xi0 = %e, xi = %e, obj0 - objh = %e\n", xi0, xi, obj0 - objh);
    // printf("xi = %e\n", xi);
    if (is_converge) break; 
    else {
      //if (xi < xi0 / 10) xi0 = 10 * xi;
      obj0 = objh;
      K0(ida) = Kh(ida);
      K0(idu) = K0(idl);
    }
  }
  //Rprintf("objh = %e\n", objh);
  return Kh;
}
