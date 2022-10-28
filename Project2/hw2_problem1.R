## NAME: Steven Semeraro
## NETID: ss3403

################
## Code for HW 2 problem 1
##
## INSTRUCTIONS:
## The following file implements logistic lasso:
##    min_{beta}  \sum_{i=1}^n - Y_i (X_i^T beta) + log(1 + exp(X_i^T beta))   + lambda * |beta|_1
##
## We use projected gradient descent, i.e., ISTA but with a different gradient.
## 
## Fill in the code in parts labeled "FILL IN". There are FIVE parts that you have to fill in.
##
################

## logistic lasso
## INPUT: X  n--by--p matrix, Y n--by--1 vector, lambda scalar
## OUTPUT: beta p--by--1 vector
##

logisticLasso <- function(X, Y, lambda) {
  
  p = ncol(X)
  n = nrow(X)
  
  beta = rep(0, p)
  stepsize = 1

  ## parameter for backtracking line search
  alpha = 0.05
  gamma = 0.8
  
  it = 0
  
  while (TRUE) {
    
    print(it)
    
    cur_obj = logisticLassoObj(X, Y, beta, lambda)

    ## compute gradient update
    gradient = t(X) %*% (sigmoid(X %*% beta) - Y) # Iterative Re weighted Least Squares
    prox = beta - (stepsize * gradient) # Proximal Gradient Descent Input
    beta_new = softThresh(prox, lambda) # Update Beta Vector

    ## backtracking line search
    ## FILL IN: complete the computation of the backtracking line search criterion
    while(logisticLassoObj(X, Y, beta_new, lambda) > cur_obj + alpha * sum(gradient * (beta_new - beta))) {  
      
      stepsize = stepsize * gamma # Step size update
      prox = beta - (stepsize * gradient) # Proximal Gradient Descent
      beta_new = softThresh(prox, stepsize) # Update Beta Vector
      
    }
    
    it = it+1

    if (it %% 100 == 0) {
      
      print(sprintf("iteration: %d   converg (log10): %.4f   stepsize (log10): %.4f", 
                    it, 
                    log10(sum((beta - beta_new)^2)/sum(beta^2)), 
                    log10(stepsize)
                    )) 
      
    }
    
    if (sum((beta - beta_new)^2)/sum(beta^2) < 1e-10){
      
      return(beta_new)
      
    } else {
      
      beta = beta_new

    }
      
  }
  
}


softThresh <- function(u, lambda){
  
  u[abs(u) <= lambda] = 0
  u[u > lambda] = u[u > lambda] - lambda
  u[u < -lambda] = u[u < -lambda] + lambda
  
  return(u)
  
}

## INPUT: vector x
## OUTPUT: vector of e^x/(1 + e^x)
sigmoid <- function(x) {
  return(exp(x)/(1 + exp(x)))
}

## OUTPUT: objective of the logistic lasso loss, a single scalar

## FILL IN: compute the logistic lasso objective 
##        \sum_{i=1}^n - Y_i (X_i^T beta) + log(1 + exp(X_i^T beta))  + lambda * |beta|_1
##          with respect to X, Y, beta, lambda

logisticLassoObj <- function(X, Y, beta, lambda) {
  
  # Computes the Logistic Lasso Objective Function
  
  temp1 = sum(-t(Y) %*% (X %*% beta))
  temp2 = sum(log(1 + exp(X %*% beta)))
  temp3 = sum(lambda * abs(beta))
  
  LogisticLasso = temp1 + temp2 + temp3

  return(LogisticLasso)
  
}

###############
###############
## 
## Testing our algorithm
## 
###############
###############
set.seed(1)
library(glmnet)

p = 100
n = 500
s = 5

beta = c(rnorm(s), rep(0, p-s))
  
X = matrix(rnorm(p*n), n, p)
Y = sign(X %*% beta + rnorm(n)*0.3)
Y[Y==-1] = 0

lambda = 0.05

# X = 500 rows, 100 cols
# Y = vector of 500 

betahat = logisticLasso(X, Y, n*lambda)  ## NOTE different scaling on lambda

res = glmnet(X, Y, family="binomial", lambda=lambda)
betahat2 = res$beta

print("Glmnet")
print(sum(betahat2^2))

print("This Code")
print(sum(betahat^2))

estimation_err = sum((betahat - beta)^2)/sum(beta^2)
dist_to_glmnet = sum(betahat2 - betahat)^2/sum(betahat2^2)

print(sprintf(
              "Estimation error: %.4f   
              Deviation from Glmnet solution: %.4f", 
              estimation_err, 
              dist_to_glmnet
              ))






