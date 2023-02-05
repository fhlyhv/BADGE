# BASS (BAyesian learning of graphical models with Smooth Structural changes, formally named BADGE)

This R package implements the BASS algorithm for learning time-varying graphical models in the following paper:

H. Yu and J. Dauwels, Efficient Variational Bayes Learning of Graphical Models With Smooth Structural Changes, IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, pp. 475 - 488, 2023.

This package is only applicable to the time domain.

The Matlab toolbox for learning graphical models from multivariate stationary time series in the frequency domain can be found at https://github.com/fhlyhv/BADGE_frequency.

## Dependence
Please make sure to install the following package dependencies before using R package `BADGE`. 
```r
install.packages(c("Rcpp", "RcppArmadillo", "BH", "RcppProgress", "KernSmooth", "tictoc", "devtools"))
```

## Installation
The R package `BADGE` can be installed from source files in the GitHub repository (R package `devtools` is needed):
```r
library(devtools)
install_github(repo="fhlyhv/BADGE")
```

If it does not work, please download this repository directly as a zip or tar.gz file and install the package locally.

## Example
Below we provide an example of how to call the funtion BADGE to learn time-varying graphical models from the synthetic data associated with the package. To test the algorithm on your own data, please replace data with your own data when calling BADGE. Note that data is a N x P matrix with N time points and P variables.
```r
library("BADGE")
# Use synthetic data associated with the package
data("data")
data("Ktruev") # the vectorized lower-triangular part of the precision matrices at all time points

# call the function BADGE
results = BADGE(data, anneal_iters = 500)

# extract the zero pattern of the precision matrices
n = dim(data)[1]
p = dim(data)[2]
xy = KernSmooth::bkde(as.vector(results$Es_mat) + 0.01 * rnorm(n * p * (p - 1) / 2),
                      bandwidth =  KernSmooth::dpik(as.vector(results$Es_mat + 0.01 * rnorm(n * p * (p - 1) / 2))))
x = xy$x
y = xy$y
thr = x[y == min(y[x<0.9 & x > 0.1])][1]

# check performance
precision = sum(results$Es_mat>thr & Ktruev!=0)/sum(results$Es_mat>thr)
recall = sum(results$Es_mat>thr & Ktruev!=0)/sum(Ktruev!=0)
f1_score = 2*precision*recall/(precision + recall)
cat("precision = ",precision,", recall = ", recall, ", f1_score = ", f1_score, "run_time = ", results$run_time)
```
