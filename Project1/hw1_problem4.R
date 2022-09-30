## NAME: Steven Semeraro
## NETID: ss3403

################
## Code for HW 1 problem 4
##
## INSTRUCTIONS:
## The following file performs refitted lasso regression with cross-validation
## 
## Fill in the code in parts labeled "FILL IN". 
##
################

# Old Path
# source("ISTA.R")
# cars = read.csv("cars.csv", as.is=TRUE)

# Local Machine Path
source("C:/Users/steve/Documents/GitHub/DataMining/Project1/ISTA.R")
cars = read.csv("C:/Users/steve/Documents/GitHub/DataMining/Project1/cars.csv", as.is=TRUE)
set.seed(1)

Y = as.vector(cars[, "mpg"])
X = as.matrix(cars[, !(names(cars) %in% c("mpg", "name"))])
oldX = scale(X)
old_p = ncol(oldX)

## We create interaction features of the form
## column j * column j' for all (j, j')
## column j * (column j')^2 for all (j, j')
## (column j)^2 * (column j')^2 for all (j, j')

for (j in 1:old_p){
  X = cbind(X, oldX*oldX[, j], oldX*oldX[, j]^2, oldX^2*oldX[, j], oldX^2 * oldX[,j]^2)
}

Y = scale(Y)
X = scale(X)
X = cbind(X, rep(1, nrow(X)))

n = 200 # Training Set
test_ix = sample(nrow(X), nrow(X) - n)

# Training Set 
# Length: 200
X1 = X[-test_ix, ]
Y1 = Y[-test_ix]

# Test Set
# Length: 192
X2 = X[test_ix, ]
Y2 = Y[test_ix]

p = ncol(X1)
n = nrow(X1)

K = 5

## FILL IN: randomly permute the n samples (rows of X1 and entries of Y1).
rand = sample(nrow(X1), size = n, replace = TRUE)
X1 = X1[rand, ]
Y1 = sample(Y1, size = n, replace = TRUE)

lambda_ls = 10^(seq(-2, 1, 0.05))

errs = rep(0, length(lambda_ls))

index_matrix = matrix(1:n, ncol=K)

for (k in 1:K){
  
  valid_ix = index_matrix[, k]
  
  ## FILL IN: create variables Xtrain, Ytrain, Xvalid, Yvalid
  Xtrain = X1[valid_ix, 1:p]
  Ytrain = Y1[valid_ix]
  Xvalid = X2
  Yvalid = Y2
  
  for (il in 1:length(lambda_ls)){
    
    lambda = lambda_ls[il]
    beta_lasso = lassoISTA(Xtrain, Ytrain, lambda)
    
    S = which(abs(beta_lasso) > 1e-10)
    
    if (length(S) == 0) {
      errs[il] = Inf
    }
    else {
      
      XS = Xtrain[, S]
      
      ## For refitting, we use ridge regression with a small penalty instead of 
      ## OLS in the event that the columns of X are not linearly independent
      
      beta_refit = solve(t(XS) %*% XS + 1e-10 * diag(length(S)), t(XS) %*% Ytrain)
      errs[il] = mean((Xvalid[, S] %*% beta_refit - Yvalid)^2)
        
    }
    
  }
  
  print(errs)
  
  plot(
    1:length(lambda_ls), 
    errs, 
    xlab = 'Lambda', 
    ylab = 'Error', 
    main = 'Ordinary Least Squares',
    col = 'blue'
  )
  
}

# Print Results
min_index = which.min(errs)
lambda_star = lambda_ls[min_index]
beta_lasso = lassoISTA(X2, Y2, lambda_star)

S = which(abs(beta_lasso) > 1e-10)

beta_refit = solve(t(X1) %*% X1 + lambda_star * diag(ncol(X1)), t(X1) %*% Y1)
test_error = mean((X2 %*% beta_refit - Y2)^2)

## For comparison, we also compute the OLS
beta_ols = solve(t(X1) %*% X1 + 1e-10 * diag(ncol(X1)), t(X1) %*% Y1)
ols_error = mean((X2 %*% beta_ols - Y2)^2)

## We compute OLS where we only use the first 7 variables and 
## the all 1 constant feature.
S = c(1:7, ncol(X1))
beta_ols2 = solve(t(X1[, S]) %*% X1[, S] + 1e-10 * diag(length(S)), t(X1[, S]) %*% Y1)
ols2_error = mean((X2[, S] %*% beta_ols2 - Y2)^2)

baseline = mean((mean(Y1) - Y2)^2)

print(sprintf("Test error: %.3f  Baseline: %.3f   OLS: %.3f   OLS (with first 7 vars): %.3f", 
              test_error, baseline, ols_error, ols2_error))





