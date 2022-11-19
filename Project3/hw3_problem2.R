################
## Code for HW 3 problem 2
##
## Instructions:
##
## The following file implements Bootstrap aggregation
## with OLS linear regression
##
## Complete the following R program by
## filling in lines labled __FILL IN___
##
##
################

library(dplyr)
library(magrittr)

# Initial Path
# wines = read.csv("wines2_hw3.csv")

# Local Machine Path 
wines = houses = read.csv("C:/Users/steve/Documents/GitHub/DataMining/Project3/wines2_hw3.csv")

set.seed(1)
wines = wines[sample(1:nrow(wines), nrow(wines)), ]


words = c("sweet", "acid", "earthy", "fruit", "tannin", "herb",
          "tart", "spice", "smooth", "full", "intense",
          "soft", "dry", "apple", "pear", "cherry",
          "berry", "aroma", "citrus", "lemon", "lime", "peach", "blossom",
          "sugar", "simple", "cinnamon",
          "crisp", "honey", "brisk", "fresh", "sour", "floral",
          "dark", "complex", "oak", "balance", "caramel", "plum", "mint",
          "apricot", "cream", "vanilla", "butter", "sharp",
          "chocolate", "delicious", "licorice", "mineral", "wood",
          "pale", "vegetable", "bitter", "water", "juice",
          "basic", "harsh", "modest",
          "salty", "clean", "tea", "beautiful", "flavor",
          "ripe", "touch", "fair", "drink", "classic",
          "alcohol", "refine", "bargain", "common",
          "mild", "depth", "delicate", "reserve")




######
## Standardization

n = nrow(wines)

Y = wines$points
X = wines[, c(words, "price")]

X = cbind(rep(1,n), as.matrix(X))

p = ncol(X)

for (j in 2:p){
    X[, j] = (X[, j] - mean(X[, j]))/sd(X[, j])
}

######################
##  Analysis with Ensemble (bagging) OLS
##

set.seed(1)

reorder = sample(1:n, n, replace=FALSE)

X = X[reorder, ]
Y = Y[reorder]

n_learn = 500
n_test = 2000


Y_test = Y[1:n_test]
Y_learn = Y[(n_test+1):(n_test+n_learn)]

X_test = X[1:n_test, ]
X_learn = X[(n_test+1):(n_test+n_learn), ]


## Take a bootstrap samples (samples with replacement)
## of X_learn and Y_learn and use it to construct
## a linear regression

## number of bootstrap datasets
BootStrap = c(1, 2, 5, 10, 20, 50)
for(i in 1:length(BootStrap)) {
  
  B = BootStrap[i]
  
  ## each row of "betas" correspond to a linear
  ## regression coef from a bootstrapped sample
  betas = matrix(0, B, p)
  
  for (b in 1:B){
    
    ## FILL IN; use this optional space
    ## to set up sampling with replacement if needed
    
    # Initialize Arrays
    XBoot = matrix(0, nrow = 500, ncol = 77)
    YBoot = matrix(0, nrow = 500, ncol = 1)
    
    for (i in 1:500) {
      
      # Generate Random Number
      rv = sample(1:500, size = 1, replace = TRUE)
      
      # Sample From Rows
      for (j in 1:77) {
        XBoot[i, j] = X_learn[rv, j]
      }
      
      # Grab Y
      YBoot[i, 1] = Y_learn[rv]
      
    }
    
    ## FILL IN
    # Randomly Sample From Training Set
    Xb = XBoot
    
    ## FILL IN
    # Randomly Sample From Training Set
    Yb = YBoot
    
    ## We add a small ridge penalty in case
    ## t(Xb) %*% Xb is not invertible
    IdentityMatrix = diag(c(0, rep(1, p-1)))
    betas[b, ] = solve(t(Xb) %*% Xb + 0.00001 * IdentityMatrix, t(Xb) %*% Yb)
    
  }
  
  ## beta averaged across all B bootstrap samples
  ## FILL IN
  betas_mean = colMeans(betas)
  
  Y_pred = X_test %*% betas_mean
  
  beta_solo = solve( t(X_learn) %*% X_learn + 0.00001*diag(c(0, rep(1, p-1))), t(X_learn) %*% Y_learn)
  Y_pred2 = X_test %*% beta_solo
  
  ######################
  ## evaluate test error
  test_error = mean( (Y_test - Y_pred)^2 ) 
  
  baseline_error = mean( (Y_test - Y_pred2 )^2 ) 
  
  print("Test MSE of (1) ensemble OLS and (2) vanilla OLS")
  print(c(test_error, baseline_error))
  
}







