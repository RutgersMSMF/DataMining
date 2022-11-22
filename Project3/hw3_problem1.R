################
## Code for HW 3 problem 1
##
## INSTRUCTIONS:
## 
## The following file implements Kernel Ridge Regression with 
## Gaussian (RBF) Kernel.
##
## It uses KRR to predict the housing price on the Boston
## Housing Data-set, where each row is a small neighborhood
## in Boston and where the features describe the environment
## of the neighborhood.
## 
## Fill in the code in parts labeled "FILL IN". 
##
##
################

set.seed(1)

# Initial Path
# houses = read.csv("Boston.csv")

# Local Machine Path
houses = read.csv("C:/Users/steve/Documents/GitHub/DataMining/Project3/Boston.csv")

features = names(houses)
features = features[! features %in% c("X", "medv")]

Y = houses[, "medv"]
X = houses[, features]
X = scale(X)

n = 300
test_ixs = sample(nrow(houses), nrow(houses) - n)

# Training Set
X1 = X[-test_ixs, ]
Y1 = Y[-test_ixs]

# Test Set
Xtest = X[test_ixs, ]
Ytest = Y[test_ixs]

# Use Subsets for of Training Set and Test Set for Cross Validation !!!

nfold = 5
fold_mat = matrix(1:n, nrow=nfold)

## We have to select both bandwidth and 
## regularization parameter
lambda_ls = 2^(seq(-6, 6, by=1))
print(lambda_ls)
bandwidth_ls = 2^(seq(-6, 6, by=1))
print(bandwidth_ls)

# Store Errors Here
valid_errs = array(0, dim=c(nfold, length(bandwidth_ls), length(lambda_ls)))

# This Loop Begins the Cross Validation Iterations 
for (k in 1:nfold){
  
  valid_ix = fold_mat[k, ]
  
  Xtrain = X1[-valid_ix, ]
  Ytrain = Y1[-valid_ix]
  
  Xvalid = X1[valid_ix, ]
  Yvalid = Y1[valid_ix]
  
  ## computes all pairwise distances between
  ## data points in Xtrain
  xtx = Xtrain %*% t(Xtrain)
  xnorms = diag(xtx)
  
  train_dists = matrix(1, nrow(Xtrain), 1) %*% xnorms - 2*xtx + xnorms %*% matrix(1, 1, nrow(Xtrain))  
  ## train_dists[i, i'] is the distance | Xtrain[i, ] - Xtrain[i', ] |^2
  
  ## computes all pairwise distances between data points in 
  ## Xvalid and Xtrain
  xtx2 = Xvalid %*% t(Xtrain)
  xnorms_valid = diag(Xvalid %*% t(Xvalid))
  
  train_valid_dists = matrix(1, nrow(Xvalid), 1) %*% xnorms - 2*xtx2 +  xnorms_valid %*% matrix(1, 1, nrow(Xtrain))
  ## train_valid_dists[i, i'] is the distance | Xvalid[i, ] - Xtrain[i', ] |^2
  
  # This Loop Starts the Bandwidth Selection Iteration
  for (j in 1:length(bandwidth_ls)){
    
    h = bandwidth_ls[j]
    
    ## K[i, i'] contains K( Xtrain[i, ], Xtrain[i', ])
    K = exp( -train_dists / h^2 )
    
    ## K2[i, i'] contains K( Xvalid[i, ], Xtrain[i', ])
    K2 = exp( -train_valid_dists / h^2 )
    
    for (l in 1:length(lambda_ls)){
      
      lambda = lambda_ls[l]

      ## FILL IN; see definition of "K"
      # From Notes: alpha = (K - l*I)^-1 * Y
      IdentityMatrix = diag(nrow(K))
      alpha = solve(K + (lambda * IdentityMatrix)) %*% Ytrain
      
      ## Ypred[i] should be predicted value for Xvalid[i]
      ## FILL IN; see definition of "K2"
      # From Notes: sum[ (K * alpha) ]
      Ypred = K2 %*% alpha
      
      valid_errs[k, j, l] = mean( (Yvalid - Ypred)^2 )
      
    }
    
  }
  
}

mean_valid_errs = apply(valid_errs, 2:3, mean)

min_err = min(mean_valid_errs)
best_ix = which(mean_valid_errs == min_err, arr.ind=TRUE)

hstar = bandwidth_ls[best_ix[1]]
lambda_star = lambda_ls[best_ix[2]]

print(sprintf("Selected bandwidth: %.3f   lambda: %.3f", hstar, lambda_star))

## computes all pairwise distances between
## data points in X1
xtx = X1 %*% t(X1)
x1norms = diag(xtx)
x1_dists = matrix(1, nrow(X1), 1) %*% x1norms - 2*xtx + x1norms %*% matrix(1, 1, nrow(X1))
## x1_dists[i, i'] is the distance | X1[i, ] - X1[i', ] |^2


## computes all pairwise distances between data points in 
## Xtest and X1
xtx2 = Xtest %*% t(X1)
xnorms_test = diag(Xtest %*% t(Xtest))
x1_test_dists = matrix(1, nrow(Xtest), 1) %*% x1norms - 2*xtx2 +  xnorms_test %*% matrix(1, 1, nrow(X1))
## x1_test_dists[i, i'] is the distance | Xtest[i, ] - X1[i', ] |^2

## K[i, i'] contains K( X1[i, ], X1[i', ])
K = exp( - x1_dists/ hstar^2 )

## K2[i, i'] contains K( Xtest[i, ], X1[i', ])
K2 = exp( - x1_test_dists/ hstar^2 )

## FILL IN; see definition of K
# Use Best Fit Lambda
IdentityMatrix = diag(nrow(K))
alpha = solve(K + (lambda_star * IdentityMatrix)) %*% Y1
  
## Ypred[i] should be predicted value of Xtest[i]
## FILL IN; see definition of K2
Ypred = K2 %*% alpha
  
test_err = mean( (Ytest - Ypred)^2 )

## Compute the MSE for OLS for comparison purpose
X1t = cbind(X1, rep(1, nrow(X1)))
Xtestt = cbind(Xtest, rep(1, nrow(Xtest)))

betahat = solve( t(X1t) %*% X1t, t(X1t) %*% Y1)
Ypred2 = Xtestt %*% betahat
test_err2 = mean( (Ytest - Ypred2)^2 )

print( sprintf("KRR MSE: %.3f   OLS MSE: %.3f", test_err, test_err2))

print( sprintf("KRR R-squared: %.3f   OLS R-squared: %.3f", 1-test_err/var(Ytest), 1-test_err2/var(Ytest)))

# Lets Plot Mean Validation Errors
persp(mean_valid_errs, 
      main = "Kernal Ridge Regression",
      xlab = "Bandwidth",
      ylab = "Lambda",
      zlab = "Mean Validation Errors",
      col = 'red',
      )



