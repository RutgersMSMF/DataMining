## NAME: Steven Semeraro
## NETID: ss3403

################
## Code for HW 1 problem 5
##
## INSTRUCTIONS:
## The following file performs non-negative OLS.
## 
## Fill in the code in parts labeled "FILL IN". 
##
################

##
## Performs the following:
##   argmin_{beta} || X * beta - Y ||_2^2
##    subject to: beta[j] >= 0 for all j in S
## 
## S is a subset of {1,...,p}
##
nonneg_OLS <- function(X, Y, S){
  
  THRESH = 1e-22
  p = ncol(X)
  n = nrow(X)
  L = 2*eigen(t(X) %*% X)$values[1]
  
  XtX = t(X) %*% X
  XtY = t(X) %*% Y
  
  beta = rep(0, p)
  
  while (TRUE){
    
    beta_new = solve(XtX, XtY)
      
    if (sum((beta_new - beta)^2) < THRESH){
      break
    } else {
      beta = beta_new
    }
    
  }
  
  return(beta_new)
  
}

## Computes 
##   argmin_{v} || u - v ||_2^2 
##  subject to: v[j] >= 0 for all j in S
##
project <- function(u, S){
  
  ## FILL IN: the projection function
  for (i in 1:length(u)) {
    
    if (i %in% S) {
      u[i] = max(u[i], 0)
    } else {
      u[i] = u[i]
    }
    
  }
  
  return(u)
  
}
  

n = 60
p = 20
num_nonneg = 7

ntrials = 300

nonneg_err = 0
ols_err = 0
ols2_err = 0

print(1:num_nonneg)

for (it in 1:ntrials){
  
  ## The true beta_star has the first num_nonneg coordinates as 0.05
  
  beta_star = c(rep(0.05, num_nonneg), rnorm(p- num_nonneg))
  X = matrix(rnorm(n*p), n, p)
  Y = X %*% beta_star + rnorm(n)
  
  beta_ols = solve(t(X) %*% X, t(X) %*% Y)
  beta_ols2 = project(beta_ols, 1:num_nonneg)
  beta_nonneg = nonneg_OLS(X, Y, 1:num_nonneg)
  
  ols_err = ols_err + sum((beta_ols - beta_star)^2)
  ols2_err = ols2_err + sum((beta_ols2 - beta_star)^2)
  nonneg_err = nonneg_err + sum((beta_nonneg - beta_star)^2)
  
}

print(sprintf("OLS: %.3f  OLS with postprocessing: %.3f   nonneg OLS: %f", 
              ols_err/ntrials, ols2_err/ntrials, nonneg_err/ntrials))






