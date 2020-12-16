\name{BADGE-package}
\alias{BADGE-package}
\alias{BADGE}
\docType{package}
\title{
\packageTitle{BADGE}
}
\description{
\packageDescription{BADGE}
}
\details{

The DESCRIPTION file:
\packageDESCRIPTION{BADGE}
\packageIndices{BADGE}
This R package implements the BAGE algorithm proposed in the paper ``Efficient Variational Bayesian Structure Learning of Dynamic Graphical Models''. Please refer to the paper for more details.
}
\author{
\packageAuthor{BADGE}

Maintainer: \packageMaintainer{BADGE}
}
\references{
Yu, H., & Dauwels, J. (2020). Efficient Variational Bayesian Structure Learning of Dynamic Graphical Models. arXiv preprint arXiv:2009.07703.
}
~~ Optionally other standard keywords, one per line, from file KEYWORDS in the R documentation directory ~~
\keyword{ package }
\seealso{
~~ Optional links to other man pages, e.g. ~~
~~ \code{\link[<pkg>:<pkg>-package]{<pkg>}} ~~
}
\examples{
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
}
