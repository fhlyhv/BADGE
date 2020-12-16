#' @title Variational Bayesian Structure Learning of Dynamic Graphical Models
#' @param data N x P matrix of observed data. N is the sample size and P is the dimension. Missing data can be denoted by NA.
#' @param max_iter maximum number of iterations (1e6 by default).
#' @param tol_s tolerance for the maximum absolute difference between the expected state values in two consecutive iterations to check the convergence of the algorithm (1e-1 by default).
#' @param tol_relative tolerance for the relative difference between the estimated precision matrices in two consecutive iterations to check the convergence (1e-2 by default)
#' @param anneal_iters number of iterations for asymtotic simulated annealing (500 by default)
#' @param is_nonzero_mean boolean value to determine whether the mean of the observations is nonzero across time (FALSE by default)
#' @param normalize_data boolean value to determine whether to normalize the data to have unit variance across time (FALSE by default). 
#' It is recommended to to normalize the data before learning the time-varying graphical models
#' @return EKd_mat N x P matrix of the diagonal of the precision matrices at N time points.
#' @return EKod_mat N x P(P-1)/2 matrix of the vectorized lower-triangular parts of the precision matrices at N time points
#' @return EJ_mat N x P(P-1)/2 matrix of the expectation of the vectorized lower-triangular parts of the J matrices at N time points
#' @return Es_mat N x P(P-1)/2 matrix of the expecation of the vectorized lower-triangular parts of the s matrices at N time points
#' @return data N x P matrix of the imputed data
#' @return run_time overall run time
#' @export

BADGE <- function(data, T = Inf, max_iter = 1e6, tol_s = 1e-1, tol_relative = 1e-2,
                  anneal_iters = 500, is_nonzero_mean = FALSE, normalize_data = FALSE) {
  
  
  
  
  
  # check inputs
  if (!is.matrix(data) && !is.data.frame(data)) {
    stop("The input data should be either a matrix or a data frame")
  } else if (is.data.frame(data)) {
    data = as.matrix(data)
  }
  set.seed(0)
  
  
  cat("BADGE starts...\n")
  is_na = is.na(data);
  
  BADGE_cpp(data, anneal_iters, T, max_iter, tol_relative, tol_s, is_na, normalize_data, is_nonzero_mean)
  
}