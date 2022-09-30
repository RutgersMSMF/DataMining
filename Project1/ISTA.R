

lassoISTA <- function(X, Y, lambda){
  
  THRESH = 1e-7
  p = ncol(X)
  n = nrow(X)
  L = 2*eigen(t(X) %*% X)$values[1]
  
  XtX = t(X) %*% X
  XtY = t(X) %*% Y
  
  beta = rep(0, p)
  
  while (TRUE){
    
    beta_new = softThresh( beta - (2/L)*(XtX %*% beta - XtY), lambda/L)
    
    if (sum((beta_new - beta)^2) < THRESH) {
      break
    }
    else {
      beta = beta_new
    }
    
  }
  
  return(beta_new)
  
}



softThresh <- function(u, lambda){
  
  u[abs(u) <= lambda] = 0
  u[u > lambda] = u[u > lambda] - lambda
  u[u < -lambda] = u[u < -lambda] + lambda
  
  return(u)
  
}
  